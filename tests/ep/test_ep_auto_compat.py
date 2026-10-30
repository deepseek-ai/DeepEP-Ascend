"""Compare the public EPBuffer contract across official and candidate builds.

WORLD_SIZE counts nodes and RANK is the node index; do not wrap this mp.spawn
launcher in torchrun. --dry-run and --help work without Torch. Copy this harness,
compat_config.py, compat_trace.py and low_latency_reference.py together when
testing the official baseline. The imported deep_ep build is recorded separately.

Correctness runs do not measure performance. Without --profile-paths, device
transport evidence is explicitly unverified, even when the shape is eligible.
"""

from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from compat_config import configure_backend, JsonlReport, parse_args, plan_cases
from compat_trace import TraceCollector

__test__ = False


def agree(action):
    """Make validation/trace failures visible to all ranks before the next call."""
    import torch
    import torch.distributed as dist

    error = None
    try:
        action()
    except Exception as exc:
        error = exc
        print(f'[rank {dist.get_rank()}] {exc}', flush=True)
    status = torch.tensor([int(error is not None)], dtype=torch.int32, device='npu')
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    if status.item():
        raise AssertionError('Compatibility validation failed on at least one rank') from error


def check_native_contract(value, attribute, backend):
    """Candidate-only markers must not become requirements of the public API."""
    if backend == 'official':
        # Fixed official 3b25377 has neither candidate marker. Catch a mislabeled
        # candidate installation rather than silently calling it the baseline.
        assert not hasattr(value, attribute), f'Official run loaded candidate marker {attribute}'
    else:
        assert hasattr(value, attribute), f'Candidate run is missing {attribute}'
        assert getattr(value, attribute) is False, f'{attribute} must remain False'


def launch(buffer, operation, deferred, **kwargs):
    result = getattr(buffer, operation)(defer_epilogue=deferred, **kwargs)
    if deferred:
        event = result
        values = event.wait()
        # A deferred epilogue is consumed exactly once.
        assert event.wait() is None
        return (*values, event)
    result[-1].wait()
    return result


def scale_packs(sources, tokens, hidden=7168, version=0):
    """Finite positive FP32 scale words viewed as the native INT16 transport.

    Distinct mantissa bits exercise both halves, row identity and cached updates.
    This harness verifies opaque scale transport; its synthetic expert function
    consumes the exactly representable FP8 values independently of these scales.
    """
    import torch

    sources = torch.as_tensor(sources, dtype=torch.int64)
    tokens = torch.as_tensor(tokens, dtype=torch.int64)
    bits = (0x3F000001 + sources[:, None] * 65536 + tokens[:, None] * 128
            + torch.arange(hidden // 128) + version * 1048576)
    return bits.to(torch.int32).contiguous().view(torch.int16)


def make_input(rank, bs, dtype, layout, *, payload_scale=1, scale_version=0, device='npu'):
    import torch
    from low_latency_reference import payload

    values = payload([rank] * bs, range(bs), 7168, torch.bfloat16) * payload_scale
    if dtype == 'bf16':
        return values.to(device)
    values = values.to(torch.float8_e4m3fn).to(device)
    scales = scale_packs([rank] * bs, range(bs), version=scale_version).to(device)
    if layout == 'col':
        scales = scales.t().contiguous().t()
    return values, scales


def check_dispatch(recv, weights, handle, routes, rank, world, alignment,
                   deterministic, *, backend, dtype='bf16', scale_layout='row',
                   weight_scale=1, payload_scale=1, scale_version=0, has_weights=True,
                   cpu_sync=False):
    import torch
    from low_latency_reference import expert_entries, payload

    entries = expert_entries(routes, 128, world, rank)
    counts = [len(rows) for rows in entries]
    check_native_contract(handle, 'is_low_latency', backend)
    assert handle.num_unaligned_recv_tokens_per_expert.cpu().tolist() == counts
    valid_aligned = sum((count + alignment - 1) // alignment * alignment for count in counts)
    values, scales = recv if isinstance(recv, tuple) else (recv, None)
    assert (scales is not None) == (dtype == 'fp8')
    assert values.dtype == (torch.float8_e4m3fn if dtype == 'fp8' else torch.bfloat16)
    assert tuple(values.shape) == (handle.num_expanded_tokens, 7168)
    assert handle.expert_alignment == alignment
    if cpu_sync:
        assert handle.num_recv_tokens_per_expert_list == [
            (c + alignment - 1) // alignment * alignment for c in counts]
        assert handle.num_expanded_tokens == valid_aligned
    else:
        assert handle.num_expanded_tokens >= valid_aligned
    assert (weights is not None) == has_weights
    recv_cpu = values[:valid_aligned].cpu()
    weights_cpu = weights[:valid_aligned].cpu() if weights is not None else None
    scales_cpu = scales[:valid_aligned].cpu() if scales is not None else None
    if scales is not None:
        packs = 7168 // 128 * 2
        assert scales.dtype == torch.int16
        assert tuple(scales.shape) == (handle.num_expanded_tokens, packs)
        expected_stride = (1, handle.num_expanded_tokens) if scale_layout == 'col' else (packs, 1)
        # Empty dimensions can have conventional contiguous strides; there are
        # no live elements whose order differs in that case.
        if scales.numel():
            assert scales.stride() == expected_stride, (scales.stride(), expected_stride)
    offset = 0
    expected_ends = []
    for local, rows in enumerate(entries):
        count = len(rows)
        expected_ends.append(offset + count)
        if count:
            block = recv_cpu[offset:offset + count]
            # All row identities are exact integers even after payload scaling.
            ids = (block[:, :4].float() / payload_scale).long()
            sources = ids[:, 0] * 16 + ids[:, 1]
            tokens = ids[:, 2] * 16 + ids[:, 3]
            keys = sources * handle.num_max_tokens_per_rank + tokens
            order = keys.argsort()
            expected_sources, expected_tokens = zip(*rows)
            expected = payload(expected_sources, expected_tokens, 7168, torch.bfloat16) * payload_scale
            torch.testing.assert_close(block.float()[order], expected.float(), atol=0, rtol=0)
            if scales_cpu is not None:
                expected_sf = scale_packs(expected_sources, expected_tokens, version=scale_version)
                torch.testing.assert_close(scales_cpu[offset:offset + count][order], expected_sf, atol=0, rtol=0)
            if deterministic:
                assert torch.equal(keys, keys.sort().values)
        next_offset = offset + (count + alignment - 1) // alignment * alignment
        assert torch.count_nonzero(recv_cpu[offset + count:next_offset].float()).item() == 0
        if weights_cpu is not None:
            assert torch.count_nonzero(weights_cpu[offset + count:next_offset]).item() == 0
        if scales_cpu is not None:
            assert torch.count_nonzero(scales_cpu[offset + count:next_offset]).item() == 0
        offset = next_offset
    assert handle.psum_num_recv_tokens_per_expert.cpu().tolist() == expected_ends
    # Validate reverse-routing slots, deduplicated counts and capacity-strided IDs.
    expected_counts = [sum(any(rank * (128 // world) <= e < (rank + 1) * (128 // world)
                               for e in selected) for selected in source) for source in routes]
    psum, accumulated = [], 0
    for count in expected_counts:
        accumulated += count
        psum.append(accumulated)
    assert handle.psum_num_recv_tokens_per_rank.cpu().tolist() == psum
    metadata = handle.recv_src_metadata[:accumulated].cpu().tolist()
    visited = set()
    seen_tokens = set()
    for row in metadata:
        source, token = divmod(row[-2], handle.num_max_tokens_per_rank)
        assert (source, token) not in seen_tokens
        seen_tokens.add((source, token))
        valid = []
        for topk_slot, slot in enumerate(row[:-2]):
            expert = routes[source][token][topk_slot]
            belongs = rank * (128 // world) <= expert < (rank + 1) * (128 // world)
            assert (slot >= 0) == belongs
            if belongs:
                assert slot not in visited
                visited.add(slot)
                valid.append(topk_slot)
                torch.testing.assert_close(recv_cpu[slot].float(),
                    (payload([source], [token], 7168, torch.bfloat16)[0] * payload_scale).float(),
                    atol=0, rtol=0)
                if weights_cpu is not None:
                    assert weights_cpu[slot].item() == (topk_slot + 1) / 32 * weight_scale
                if scales_cpu is not None:
                    torch.testing.assert_close(scales_cpu[slot],
                        scale_packs([source], [token], version=scale_version)[0], atol=0, rtol=0)
        assert row[-1] == max(valid)
    assert len(visited) == sum(counts)
    if cpu_sync:
        assert handle.num_recv_tokens == sum(expected_counts)


def expected_transport(operation, case, rank, backend, runtime, *, original_b=None, capacity=None):
    """Policy expectation, never an observation of what the device executed."""
    if backend != 'candidate-auto':
        return 'urma'
    b = case['rank_bs'][rank] if original_b is None else original_b
    m = case['M'] if capacity is None else capacity
    world = len(case['rank_bs'])
    eligible = (runtime['arch'] == 3510 and runtime['num_vec_cores'] == 64
                and 0 < b <= m <= 256
                and world == 128 and b <= 32)
    if operation == 'dispatch' and case['dtype'] == 'fp8':
        eligible = False
    # Target acceptance machines must provide >=190 KiB UB. The actual template
    # name/parameters remain the evidence; an unexpected fallback is a failure.
    return 'mte' if eligible else 'urma'


def invoke(buffer, operation, label, case, rank, backend, runtime, collector, **kwargs):
    import torch

    metadata = dict(case_id=case['case_id'], backend=backend, rank=rank,
                    operation=operation, cached=kwargs.get('handle') is not None if operation == 'dispatch' else None,
                    weight_side_channel=kwargs.get('topk_weights') is not None,
                    dtype=case['dtype'] if operation == 'dispatch' else 'bf16',
                    M=case['M'], B=case['rank_bs'][rank], deferred=case['deferred'],
                    barrier_in_prologue=case['barrier_in_prologue'], hidden=7168,
                    num_topk=8, num_experts=128, num_ranks=len(case['rank_bs']),
                    num_vec_cores=runtime['num_vec_cores'])
    handle = kwargs.get('handle')
    if operation == 'dispatch':
        sync = kwargs.get('do_cpu_sync')
        metadata.update(cpu_sync=sync if sync is not None else handle is None,
                        alignment=handle.expert_alignment if handle is not None else case['alignment'],
                        scale_layout=case['scale_layout'])
        data = kwargs['x']
        if isinstance(data, tuple):
            metadata.update(input_scale_shape=list(data[1].shape), input_scale_stride=list(data[1].stride()),
                            expected_num_sf_packs=data[1].size(1))
        else:
            metadata['expected_num_sf_packs'] = 0
    original_b = handle.topk_idx.size(0) if operation == 'combine' else None
    capacity = handle.num_max_tokens_per_rank if handle is not None else case['M']
    expected = expected_transport(operation, case, rank, backend, runtime,
                                  original_b=original_b, capacity=capacity)
    if expected == 'mte':
        metadata['min_num_ub_bytes'] = 190 * 1024
    if original_b is not None:
        metadata.update(B=original_b, M=capacity)
    result, error, entered = None, None, False
    capture = collector.capture(label, operation, expected_transport=expected, metadata=metadata)
    try:
        capture.__enter__()
        entered = True
    except Exception as exc:
        error = exc

    def check():
        if error is not None:
            raise error
    # A rank-local profiler/filesystem error must be agreed before another rank
    # enters the EP communication. Successful peers close their open profiles.
    try:
        agree(check)
    except Exception as setup_error:
        if entered:
            try:
                capture.__exit__(type(setup_error), setup_error, setup_error.__traceback__)
            except Exception:
                pass
        raise
    try:
        result = launch(buffer, operation, case['deferred'], **kwargs)
        torch.npu.synchronize()
    except Exception as exc:
        error = exc
    try:
        capture.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)
    except Exception as exc:
        error = exc
    agree(check)
    return result


def run_case(buffer, case, rank, world, retained, backend, runtime, collector):
    import torch
    from low_latency_reference import make_routes, expert_entries, payload, prepare_experts, expected_combine
    bs, alignment, sync = case['bs'], case['alignment'], case['cpu_sync']
    dtype, scale_layout = case['dtype'], case['scale_layout']
    routes = make_routes(world, bs, 8, 128, case['seed'])
    if case['route'] == 'hotspot':
        routes = [[list(range(8)) for _ in range(bs)] for _ in range(world)]
    if case['edge_routes']:
        # Include empty ranks and both sides of the automatic cutoff in one
        # collective. Keep the normal API's masked and repeated expert slots.
        for source, rows in enumerate(routes):
            routes[source] = rows[:case['rank_bs'][source]]
            for token, selected in enumerate(routes[source]):
                selected[-1] = -1
                selected[1] = selected[0]
                if token % 3 == 0:
                    selected[-2] = -1
        bs = len(routes[rank])
    counts = [len(rows) for rows in expert_entries(routes, 128, world, rank)]
    assert [len(rows) for rows in routes] == case['rank_bs']
    bs = case['rank_bs'][rank]
    x = make_input(rank, bs, dtype, scale_layout)
    indices = torch.tensor(routes[rank], dtype=torch.int64, device='npu').reshape(bs, 8)
    weights = (torch.arange(1, 9, dtype=torch.float32, device='npu') / 32).repeat(bs, 1)
    weights.masked_fill_(indices < 0, 0)
    stats = torch.zeros(128 // world, dtype=torch.int32, device='npu')
    dispatch_args = dict(x=x, topk_idx=indices, topk_weights=weights, num_experts=128,
                         num_max_tokens_per_rank=case['M'],
                         expert_alignment=alignment, num_sms=0, do_expand=True,
                         do_cpu_sync=sync, do_zero_padding=True, async_with_compute_stream=True,
                         use_tma_aligned_col_major_sf=scale_layout == 'col',
                         cumulative_local_expert_recv_stats=stats)

    def call(operation, label, **kwargs):
        return invoke(buffer, operation, label, case, rank, backend, runtime, collector, **kwargs)

    check_options = dict(backend=backend, dtype=dtype, scale_layout=scale_layout, cpu_sync=sync is not False)
    recv, recv_idx, recv_weights, handle, _ = call('dispatch', 'fresh', **dispatch_args)
    agree(lambda: check_dispatch(recv, recv_weights, handle, routes, rank, world, alignment,
                                 buffer.deterministic, **check_options))
    agree(lambda: _check_no_indices(recv_idx))
    agree(lambda: torch.testing.assert_close(stats.cpu(), torch.tensor(counts, dtype=torch.int32), atol=0, rtol=0))
    # Statistics are supported only on fresh dispatch. A second fresh call
    # checks accumulation and, when requested, stable deterministic ordering.
    again, _, again_weights, _, _ = call('dispatch', 'fresh_stats_second', **dispatch_args)
    agree(lambda: torch.testing.assert_close(stats.cpu(), torch.tensor(counts, dtype=torch.int32) * 2, atol=0, rtol=0))
    if buffer.deterministic:
        valid_aligned = sum((c + alignment - 1) // alignment * alignment for c in counts)
        recv_values = recv[0] if isinstance(recv, tuple) else recv
        again_values = again[0] if isinstance(again, tuple) else again
        agree(lambda: torch.testing.assert_close(again_values[:valid_aligned].float(),
                                                 recv_values[:valid_aligned].float(), atol=0, rtol=0))
        agree(lambda: torch.testing.assert_close(again_weights[:valid_aligned], recv_weights[:valid_aligned],
                                                 atol=0, rtol=0))

    unweighted_expert = prepare_experts(recv, counts, rank, 128, world, torch.bfloat16,
                                       expert_alignment=alignment)
    expected_unweighted = expected_combine(routes, rank, 7168, torch.bfloat16,
                                            reduce_in_local=True, experts=128)
    for weighted in (False, True):
        expert = unweighted_expert.clone()
        if weighted:
            start = 0
            for count in counts:
                expert[start:start + count] = (expert[start:start + count].float()
                    * recv_weights[start:start + count, None]).to(torch.bfloat16)
                start += (count + alignment - 1) // alignment * alignment
        saved = expert.clone()
        for num_bias in (0, 1, 2):
            bias = None if not num_bias else torch.full((bs, 7168), .25, dtype=torch.bfloat16, device='npu')
            if num_bias == 2:
                bias = (bias, torch.full_like(bias, -.125))
            expected_biased = expected_combine(routes, rank, 7168, torch.bfloat16, weighted=weighted,
                reduce_in_local=True, experts=128, bias=0 if not num_bias else .25 if num_bias == 1 else .125)
            for send_weights in (False, True):
                output, output_weights, _ = call('combine',
                    f'weighted_{int(weighted)}_bias_{num_bias}_side_{int(send_weights)}',
                    x=expert, handle=handle, topk_weights=recv_weights if send_weights else None,
                    bias=bias, num_sms=0, async_with_compute_stream=True)
                torch.npu.synchronize()
                agree(lambda: torch.testing.assert_close(output.float().cpu(), expected_biased.float(),
                                                           atol=.125, rtol=.02))
                agree(lambda: torch.testing.assert_close(expert, saved, atol=0, rtol=0))
                if send_weights:
                    agree(lambda: torch.testing.assert_close(output_weights, weights, atol=0, rtol=0))
                else:
                    agree(lambda: _check_no_indices(output_weights))

    # Change both data and metadata while reusing the original layout. A cached
    # dispatch must not overwrite its stored slots or retain the old weights.
    metadata_fields = ('psum_num_recv_tokens_per_rank', 'psum_num_recv_tokens_per_expert',
                       'num_unaligned_recv_tokens_per_expert', 'recv_src_metadata',
                       'dst_buffer_slot_idx', 'dst_gsge_idx')
    saved_metadata = [getattr(handle, name).clone() for name in metadata_fields]
    updated = make_input(rank, bs, dtype, scale_layout, payload_scale=2, scale_version=1)
    cached, cached_idx, cached_weights, cached_handle, _ = call('dispatch', 'cached_updated',
        x=updated, handle=handle, topk_weights=weights * 3, do_expand=True, do_cpu_sync=False,
        do_zero_padding=True, async_with_compute_stream=True,
        use_tma_aligned_col_major_sf=scale_layout == 'col')
    agree(lambda: _check_cached_handle(cached_handle, handle, cached_idx))
    agree(lambda: check_dispatch(cached, cached_weights, cached_handle, routes, rank, world, alignment,
                                 buffer.deterministic, backend=backend, dtype=dtype,
                                 scale_layout=scale_layout, weight_scale=3, payload_scale=2,
                                 scale_version=1))
    for name, saved in zip(metadata_fields, saved_metadata):
        agree(lambda: torch.testing.assert_close(getattr(handle, name), saved, atol=0, rtol=0))
    # Reuse the same unweighted expert data: changing route weight metadata must
    # not change combined x, and returned weights must reflect the cached call.
    output, output_weights, _ = call('combine', 'cached_weights_return',
        x=unweighted_expert, handle=cached_handle, topk_weights=cached_weights,
        num_sms=0, async_with_compute_stream=True)
    torch.npu.synchronize()
    agree(lambda: torch.testing.assert_close(output.float().cpu(), expected_unweighted.float(), atol=.125, rtol=.02))
    agree(lambda: torch.testing.assert_close(output_weights, weights * 3, atol=0, rtol=0))

    # Omitted weights are part of the baseline public contract, both fresh and
    # cached. Neither call may reuse the previous call's weight side channel.
    for cached_mode in (False, True):
        no_weight_args = dict(x=updated, topk_weights=None, do_expand=True, do_cpu_sync=False,
                              do_zero_padding=True, async_with_compute_stream=True,
                              use_tma_aligned_col_major_sf=scale_layout == 'col')
        if cached_mode:
            no_weight_args['handle'] = handle
        else:
            no_weight_args.update(topk_idx=indices, num_experts=128,
                                  num_max_tokens_per_rank=case['M'], expert_alignment=alignment)
        no_weight_x, no_idx, no_weights, no_weight_handle, _ = call(
            'dispatch', 'cached_no_weights' if cached_mode else 'fresh_no_weights', **no_weight_args)
        agree(lambda: _check_no_indices(no_idx))
        agree(lambda: check_dispatch(no_weight_x, no_weights, no_weight_handle, routes, rank, world, alignment,
            buffer.deterministic, backend=backend, dtype=dtype, scale_layout=scale_layout,
            payload_scale=2, scale_version=1, has_weights=False))

    if retained is not None:
        old_expert, old_handle, old_expected = retained
        output, output_weights, _ = call('combine', 'old_handle',
            x=old_expert, handle=old_handle, num_sms=0, async_with_compute_stream=True)
        torch.npu.synchronize()
        agree(lambda: torch.testing.assert_close(output.float().cpu(), old_expected.float(), atol=.125, rtol=.02))
        agree(lambda: _check_no_indices(output_weights))
    else:
        retained = unweighted_expert, handle, expected_unweighted
    if rank == 0:
        print(f"PASS {case['case_id']} EP{world} B={case['rank_bs']} M={case['M']} "
              f"{dtype=} alignment={alignment} deferred={case['deferred']} cpu_sync={sync} "
              f"barrier={case['barrier_in_prologue']} path_evidence={'requested' if collector.enabled else 'unverified'}",
              flush=True)
    return retained


def _check_no_indices(value):
    assert value is None


def _check_cached_handle(actual, expected, indices):
    assert actual is expected and indices is None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_identity(deep_ep, torch, torch_npu, args):
    """Record the imported build, not the repository containing the harness."""
    package = Path(deep_ep.__file__).resolve()
    extension = Path(deep_ep._C.__file__).resolve()
    source_root = next((p for p in package.parents if (p / '.git').exists()), None)
    commit, dirty = None, None
    if source_root is not None:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source_root,
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            commit = result.stdout.strip()
        result = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                cwd=source_root, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            dirty = result.stdout.strip().splitlines()
    if args.expected_source_commit:
        assert commit is not None, 'Cannot verify --expected-source-commit from the imported package'
        assert commit.startswith(args.expected_source_commit), (commit, args.expected_source_commit)
    identity = dict(package_path=str(package), extension_path=str(extension),
                    package_sha256=file_hash(package), extension_sha256=file_hash(extension),
                    source_root=str(source_root) if source_root is not None else None,
                    source_commit=commit, tracked_changes=dirty, build_label=args.build_label,
                    expected_source_commit=args.expected_source_commit,
                    python=sys.version, torch=torch.__version__,
                    torch_npu=getattr(torch_npu, '__version__', None),
                    ascend_home_path=os.getenv('ASCEND_HOME_PATH'),
                    harness_sha256=file_hash(__file__),
                    harness_files_sha256={name: file_hash(Path(__file__).parent / name) for name in (
                        'test_ep_auto_compat.py', 'compat_config.py', 'compat_trace.py',
                        'low_latency_reference.py')},
                    runtime_files_sha256={str(relative): file_hash(package.parent / relative)
                                          for relative in (
                                              Path('buffers/ep.py'),
                                              Path('include/deep_ep/impls/ep/dispatch.hpp'),
                                              Path('include/deep_ep/impls/ep/combine.hpp'))
                                          if (package.parent / relative).is_file()})
    return identity


def worker(local_rank, args):
    import torch
    import torch_npu
    import torch.distributed as dist

    configure_backend(args.backend)
    torch.npu.set_device(local_rank)
    import deep_ep

    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    rank = int(os.getenv('RANK', '0')) * args.num_processes + local_rank
    cases = plan_cases(args, world)
    dist.init_process_group('hccl', rank=rank, world_size=world,
        device_id=torch.device(f'npu:{local_rank}'),
        init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8361')}",
        timeout=timedelta(seconds=600))
    runtime = dict(arch=deep_ep._C.get_npu_arch(), num_vec_cores=deep_ep._C.get_num_vec_cores(),
                   num_ai_cores=deep_ep._C.get_num_ai_cores(),
                   device_name=torch.npu.get_device_name(local_rank),
                   local_rank=local_rank, hidden=7168, num_topk=8, num_experts=128)
    identity = {}
    agree(lambda: identity.update(build_identity(deep_ep, torch, torch_npu, args)))
    metadata = dict(backend=args.backend, world=world, runtime=runtime, build=identity,
                    arguments=vars(args), transport_override=os.getenv('EP_DISABLE_LOW_LATENCY_TRANSPORT'),
                    ub_acceptance_min_bytes=190 * 1024)
    fingerprint = hashlib.sha256(json.dumps(
        dict(backend=args.backend, capacity=args.capacity, cases=cases, profile_paths=args.profile_paths,
             arch=runtime['arch'], num_vec_cores=runtime['num_vec_cores']),
        sort_keys=True).encode()).digest()
    fingerprint_tensor = torch.tensor(list(fingerprint), dtype=torch.int32, device='npu')
    fingerprints = [torch.empty_like(fingerprint_tensor) for _ in range(world)]
    dist.all_gather(fingerprints, fingerprint_tensor)
    if any(not torch.equal(value, fingerprint_tensor) for value in fingerprints):
        raise ValueError('Ranks have different test plans, backends, or device configurations')
    report_holder = []
    try:
        agree(lambda: report_holder.append(JsonlReport(args.output_dir, rank, metadata=metadata)))
    except Exception:
        for opened in report_holder:
            opened.close()
        raise
    with report_holder[0] as report, torch.inference_mode():
        agree(lambda: report.record('started', scope='run', planned_case_ids=[case['case_id'] for case in cases]))
        buffer, retained = None, None
        for case in cases:
            collector = None
            try:
                agree(lambda: report.record('started', case, correctness='pending', path_status='pending'))
                collector = TraceCollector(
                    Path(args.output_dir) / 'traces' / f'rank-{rank:05d}' / case['case_id'],
                    enabled=args.profile_paths, backend=args.backend)
                if buffer is None or buffer.deterministic != case['deterministic']:
                    if buffer is not None:
                        buffer.destroy()
                    buffer = deep_ep.EPBuffer(
                        dist.group.WORLD, num_max_tokens_per_rank=args.capacity,
                        hidden=7168, num_topk=8, deterministic=case['deterministic'],
                        explicitly_destroy=True, use_huge1g_page=not args.no_huge1g_page)
                    agree(lambda: check_native_contract(buffer, 'low_latency_mode', args.backend))
                    retained = None
                deep_ep._C.set_barrier_in_prologue(case['barrier_in_prologue'])
                retained = run_case(buffer, case, rank, world, retained, args.backend, runtime, collector)
                agree(lambda: report.record('pass', case, correctness='passed',
                      path_status='verified' if args.profile_paths else 'unverified',
                      operations=collector.records))
            except Exception as exc:
                report.record('fail', case, correctness='failed',
                              error=str(exc), traceback=traceback.format_exc(),
                              operations=[] if collector is None else collector.records)
                # Do not launch another communication protocol or collective
                # destroy after a failure. mp.spawn terminates the other workers.
                raise
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()
        report.record('pass', scope='run', completed_case_count=len(cases))


def main(argv=None):
    args = parse_args(argv)
    try:
        cases = plan_cases(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        print(json.dumps(dict(backend=args.backend, num_cases=len(cases), cases=cases,
                              path_status='unverified', profile_paths=args.profile_paths),
                         indent=2, allow_nan=False))
        return
    configure_backend(args.backend)
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)


if __name__ == '__main__':
    main()
