import ctypes
import os
import re
import struct
import shutil
import sys
import tempfile
import torch
import torch.distributed as dist
import torch_npu.profiler
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

__all__ = [
    'KernelProfile',
    'bench_msprof',
    'bench',
    'parse_num_bytes',
]


def bench(fn: Callable, num_warmups: int = 30, num_tests: int = 30,
          post_fn: Optional[Callable] = None, barrier: Optional[Callable] = None) -> tuple[float, float, float]:
    """Return mean, minimum and maximum execution time in seconds using NPU events.

    fn must enqueue its complete operation on the current stream, including any
    deferred completion hook. post_fn runs after each timed iteration. The optional
    barrier runs after warmup, outside timing; results are local to this rank.
    """
    if num_warmups < 0 or num_tests <= 0:
        raise ValueError('num_warmups must be nonnegative and num_tests must be positive')
    torch.npu.synchronize()
    for _ in range(num_warmups):
        fn()
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(num_tests)]
    if barrier is not None:
        barrier()
    for start, end in zip(start_events, end_events, strict=True):
        start.record()
        fn()
        end.record()
        if post_fn is not None:
            post_fn()
    torch.npu.synchronize()
    times = [start.elapsed_time(end) / 1000 for start, end in zip(start_events, end_events, strict=True)]
    return sum(times) / len(times), min(times), max(times)


def parse_num_bytes(text: str) -> int:
    """Parse a positive byte count with optional binary suffixes: 1G, 64M, 512K."""
    match = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt])?(?:i?b)?\s*', text, re.IGNORECASE)
    if match is None:
        raise ValueError('Expected a byte count such as 1G, 64M, 512K, or 1048576')
    scale = {None: 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}
    suffix = match.group(2)
    whole, _, fraction = match.group(1).partition('.')
    numerator = int(whole + fraction) * scale[suffix.lower() if suffix else None]
    value = numerator // 10 ** len(fraction)
    if value <= 0:
        raise ValueError('Byte count must be positive')
    return value


@dataclass
class KernelProfile:
    """Per-kernel profiling result with AIC and AIV pipe utilization.
    All pipe fields are ratios in [0, 1] relative to their own total_cycles."""
    dur_ns: float               # duration in nanoseconds
    # AIC pipes
    aic_total_cycles: int = 0
    aic_mad: float = 0.0        # Cube MAD
    aic_scalar: float = 0.0
    aic_mte1: float = 0.0       # L1 -> L0
    aic_mte2: float = 0.0       # GM -> L1
    aic_mte3: float = 0.0       # UB -> GM (on AIC, rarely used)
    aic_fixpipe: float = 0.0    # L0C -> out
    # AIV pipes
    aiv_total_cycles: int = 0
    aiv_vec: float = 0.0        # Vector compute
    aiv_scalar: float = 0.0
    aiv_mte2: float = 0.0       # GM -> UB
    aiv_mte3: float = 0.0       # UB -> GM

    def __add__(self, other: 'KernelProfile') -> 'KernelProfile':
        """Sum two profiles: dur_ns adds, pipe ratios are cycle-weighted averages."""
        aic_c = self.aic_total_cycles + other.aic_total_cycles
        aiv_c = self.aiv_total_cycles + other.aiv_total_cycles
        wa = self.aic_total_cycles / aic_c if aic_c else 0
        wb = other.aic_total_cycles / aic_c if aic_c else 0
        va = self.aiv_total_cycles / aiv_c if aiv_c else 0
        vb = other.aiv_total_cycles / aiv_c if aiv_c else 0
        return KernelProfile(
            dur_ns=self.dur_ns + other.dur_ns,
            aic_total_cycles=aic_c,
            aic_mad=self.aic_mad * wa + other.aic_mad * wb,
            aic_scalar=self.aic_scalar * wa + other.aic_scalar * wb,
            aic_mte1=self.aic_mte1 * wa + other.aic_mte1 * wb,
            aic_mte2=self.aic_mte2 * wa + other.aic_mte2 * wb,
            aic_mte3=self.aic_mte3 * wa + other.aic_mte3 * wb,
            aic_fixpipe=self.aic_fixpipe * wa + other.aic_fixpipe * wb,
            aiv_total_cycles=aiv_c,
            aiv_vec=self.aiv_vec * va + other.aiv_vec * vb,
            aiv_scalar=self.aiv_scalar * va + other.aiv_scalar * vb,
            aiv_mte2=self.aiv_mte2 * va + other.aiv_mte2 * vb,
            aiv_mte3=self.aiv_mte3 * va + other.aiv_mte3 * vb,
        )

    @property
    def dur_us(self) -> float:
        return self.dur_ns / 1000

    @property
    def us(self) -> float:
        return self.dur_us

    @property
    def ns(self) -> float:
        return self.dur_ns

    def tflops(self, flops: float) -> float:
        return flops / self.dur_ns / 1000

    def gbps(self, bytes_moved: float) -> float:
        return bytes_moved / self.dur_ns

    def __repr__(self):
        def _fmt(label, val):
            return f'{label}={val*100:.1f}%' if val > 0.001 else None
        parts = [f'{self.dur_ns/1000:.1f}us']
        for s in [_fmt('mad', self.aic_mad), _fmt('aic_mte2', self.aic_mte2),
                  _fmt('aic_mte1', self.aic_mte1), _fmt('aic_fix', self.aic_fixpipe),
                  _fmt('aiv_vec', self.aiv_vec), _fmt('aiv_mte2', self.aiv_mte2),
                  _fmt('aiv_mte3', self.aiv_mte3)]:
            if s:
                parts.append(s)
        if self.aic_total_cycles:
            parts.append(f'aic_cycles={self.aic_total_cycles}')
        if self.aiv_total_cycles:
            parts.append(f'aiv_cycles={self.aiv_total_cycles}')
        return f'KernelProfile({", ".join(parts)})'

    def __str__(self) -> str:
        def pct(value):
            return f'{value * 100:.1f}%'

        parts = []
        if self.aic_total_cycles:
            parts.append(
                f'AIC[scalar={pct(self.aic_scalar)}, mte1={pct(self.aic_mte1)}, '
                f'mte2={pct(self.aic_mte2)}, mte3={pct(self.aic_mte3)}, '
                f'mad={pct(self.aic_mad)}, fix={pct(self.aic_fixpipe)}]'
            )
        if self.aiv_total_cycles:
            parts.append(
                f'AIV[scalar={pct(self.aiv_scalar)}, mte2={pct(self.aiv_mte2)}, '
                f'mte3={pct(self.aiv_mte3)}, vec={pct(self.aiv_vec)}]'
            )
        return ', '.join(parts) if parts else 'util=N/A'


@contextmanager
def _suppress_stdout_stderr(suppress: bool):
    if not suppress:
        yield
        return
    libc = ctypes.CDLL(None)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    sys.stdout.flush()
    sys.stderr.flush()
    libc.fflush(None)
    with open(os.devnull, 'w') as devnull:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stderr(devnull), redirect_stdout(devnull):
                yield
        finally:
            # Flush C stdio before restoring the original file descriptors.
            libc.fflush(None)
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _parse_ffts_profile(prof_path: Path) -> dict[str, KernelProfile]:
    """Parse ffts_profile.data binary and correlate with kernel names.

    Returns {kernel_name: KernelProfile} with averaged metrics across iterations.
    """
    # Build name lookup: hash -> name
    hash_to_name: dict[int, str] = {}
    for f in prof_path.rglob('*hash_dic.slice_*'):
        if f.name.endswith('.done'):
            continue
        text = f.read_bytes().decode('utf-8', errors='replace')
        for line in text.strip().split('\n'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                try:
                    h = int(parts[0])
                    if h >= 2**63:
                        h -= 2**64
                    hash_to_name[h] = parts[1]
                except ValueError:
                    pass

    # Build seq -> name from task_track (64-byte records)
    seq_to_name: dict[int, str] = {}
    for f in prof_path.rglob('*task_track.slice_*'):
        if f.name.endswith('.done'):
            continue
        data = f.read_bytes()
        for i in range(len(data) // 64):
            vals = struct.unpack_from('<8q', data, i * 64)
            seq = (vals[3] >> 32) & 0xFFFF
            name_hash = vals[5]
            if name_hash != 0:
                seq_to_name[seq] = hash_to_name.get(name_hash, f'<unknown {name_hash}>')

    # Parse ffts_profile records (128 bytes each)
    aic_records: dict[str, list[dict]] = defaultdict(list)
    aiv_records: dict[str, list[dict]] = defaultdict(list)
    for f in prof_path.rglob('ffts_profile*'):
        if f.name.endswith('.done') or f.stat().st_size == 0:
            continue
        data = f.read_bytes()
        for i in range(len(data) // 128):
            vals = struct.unpack_from('<16q', data, i * 128)
            num_total_cycles = vals[1]
            if num_total_cycles == 0:
                continue
            seq = (vals[0] >> 32) & 0xFFFF
            is_aiv = vals[2] >= (1 << 31)
            dur_ns = float(vals[15] - vals[14])
            name = seq_to_name.get(seq, '<unknown>')
            metrics = {
                'dur_ns': dur_ns,
                'total_cycles': num_total_cycles,
                'vec': vals[4] / num_total_cycles,
                'mad': vals[5] / num_total_cycles,
                'scalar': vals[6] / num_total_cycles,
                'mte1': vals[7] / num_total_cycles,
                'mte2': vals[8] / num_total_cycles,
                'mte3': vals[9] / num_total_cycles,
                'fixpipe': vals[12] / num_total_cycles,
            }
            if is_aiv:
                aiv_records[name].append(metrics)
            else:
                aic_records[name].append(metrics)

    # Merge AIC + AIV into KernelProfile per kernel (average across iterations)
    all_names = set(aic_records.keys()) | set(aiv_records.keys())
    result: dict[str, KernelProfile] = {}
    for name in all_names:
        aic_list = aic_records.get(name, [])
        aiv_list = aiv_records.get(name, [])
        dur_list = aic_list or aiv_list
        n_aic = len(aic_list) or 1
        n_aiv = len(aiv_list) or 1
        result[name] = KernelProfile(
            dur_ns=sum(m['dur_ns'] for m in dur_list) / len(dur_list),
            aiv_total_cycles=int(sum(m['total_cycles'] for m in aiv_list) / n_aiv),
            aiv_vec=sum(m['vec'] for m in aiv_list) / n_aiv,
            aiv_scalar=sum(m['scalar'] for m in aiv_list) / n_aiv,
            aiv_mte2=sum(m['mte2'] for m in aiv_list) / n_aiv,
            aiv_mte3=sum(m['mte3'] for m in aiv_list) / n_aiv,
            aic_total_cycles=int(sum(m['total_cycles'] for m in aic_list) / n_aic),
            aic_mad=sum(m['mad'] for m in aic_list) / n_aic,
            aic_scalar=sum(m['scalar'] for m in aic_list) / n_aic,
            aic_mte1=sum(m['mte1'] for m in aic_list) / n_aic,
            aic_mte2=sum(m['mte2'] for m in aic_list) / n_aic,
            aic_mte3=sum(m['mte3'] for m in aic_list) / n_aic,
            aic_fixpipe=sum(m['fixpipe'] for m in aic_list) / n_aic,
        )
    return result


def _device_busy() -> None:
    """
    Keep the device busy by enqueuing a large dummy matmul on the current stream.

    NOTES: Ascend has no `torch.cuda._sleep` equivalent (spin the device for N cycles), so we
    enqueue a sizable compute op instead. This absorbs unbalanced CPU launch overhead between
    profiled iterations, the same role `torch.cuda._sleep` plays in the CUDA benchmark.
    """
    a = torch.randn((8192, 8192), dtype=torch.float, device='npu')
    a @ a


def bench_msprof(
    fn: Callable,
    kernel_names: Union[str, list[str], None] = None,
    num_warmups: int = 10,
    num_tests: int = 30,
    suppress_verbose_output: bool = True,
    return_all_kernels: bool = False,
    flush_l2: bool = True,
    barrier_comm_profiling: bool = False,
    barrier: Optional[Callable] = None
) -> Union[KernelProfile, list[KernelProfile], dict[str, KernelProfile]]:
    """Profile kernel(s) using torch_npu.profiler with pipe utilization.

    Parses raw ffts_profile.data binary directly (skips msprof CSV analyze).
    Overhead: ~0.6-2s.

    Arguments:
        fn: the function to benchmark (called with no args).
        kernel_names: None -> require exactly one kernel, return KernelProfile.
                      str  -> match one kernel by substring, return KernelProfile.
                      list[str] -> match multiple kernels, return list[KernelProfile].
        num_warmups: warmup iterations before profiling.
        num_tests: measurement iterations inside profiler active phase.
        suppress_verbose_output: suppress torch_npu.profiler stdout/stderr spam.
        return_all_kernels: if True, return dict[str, KernelProfile] (ignores kernel_names).
        flush_l2: flush L2 cache between iterations.
        barrier_comm_profiling: insert a device-busy op and a cross-rank barrier before each
            iteration to reduce unbalanced CPU launch overhead. Disabled by the
            `EP_DISABLE_BARRIER_PROFILING` env var.
        barrier: a custom barrier callable to use instead of `dist.all_reduce`.

    Returns:
        KernelProfile, list[KernelProfile], or dict[str, KernelProfile].
    """
    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()

    if barrier is not None:
        barrier()

    flush_l2_size = int(8e9 // 4)
    barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0
    dummy = torch.ones(1, dtype=torch.float, device='npu') if barrier_comm_profiling and barrier is None else None

    old_work_path = os.environ.get('ASCEND_WORK_PATH')
    prof_dir = Path(tempfile.mkdtemp(prefix='dg_bench_quick_'))
    os.environ['ASCEND_WORK_PATH'] = str(prof_dir)

    try:
        with _suppress_stdout_stderr(suppress_verbose_output):
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                    on_trace_ready=lambda _prof: None,  # no-op: skip trace export (fast)
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        data_simplification=False,
                    ),
            ) as prof:
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(flush_l2_size, dtype=torch.int, device='npu').zero_()
                    if barrier_comm_profiling:
                        _device_busy()
                        if barrier is None:
                            dist.all_reduce(dummy)
                        else:
                            barrier()
                    fn()
                torch.npu.synchronize()
                prof.step()

            profiles = _parse_ffts_profile(Path(prof.prof_if.prof_path))
    finally:
        if old_work_path is None:
            os.environ.pop('ASCEND_WORK_PATH', None)
        else:
            os.environ['ASCEND_WORK_PATH'] = old_work_path
        shutil.rmtree(prof_dir, ignore_errors=True)

    if return_all_kernels:
        return profiles

    if kernel_names is None:
        if len(profiles) != 1:
            summary = '\n'.join(f'  - {n}' for n in profiles) or '  (none)'
            raise RuntimeError(f'Expected unique kernel, but found {len(profiles)}:\n{summary}')
        return next(iter(profiles.values()))
    if isinstance(kernel_names, str):
        matched = {n: p for n, p in profiles.items() if kernel_names in n}
        if len(matched) != 1:
            summary = '\n'.join(f'  - {n}' for n in profiles) or '  (none)'
            raise RuntimeError(f"No unique kernel matching '{kernel_names}':\n{summary}")
        return next(iter(matched.values()))
    return [next(p for n, p in profiles.items() if name in n) for name in kernel_names]
