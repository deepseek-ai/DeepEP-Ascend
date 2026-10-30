# DeepEP-Ascend

**（中文介绍）**[DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) 是面向华为昇腾 NPU 的高性能机器学习训练与推理通信库，提供 MoE dispatch/combine 的专家并行（EP）all-to-all 操作，支持 FP8 dispatch 和延迟 epilogue。此外还提供流水线并行（PP）、面向上下文并行和数据并行（CP/DP）的 Bucket 集合通信，以及 Engram 远端内存访问等通信原语（开发中）。其公开 buffer API 与 NVIDIA 版 [DeepEP](https://github.com/deepseek-ai/DeepEP) 对齐。Ascend C 内核使用 HCCL/HCOMM、UBMEM 和 URMA 完成通信，并通过 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 在运行时编译。

**(English introduction)** [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) is a high-performance communication library for machine learning training and inference on Huawei Ascend NPUs. It provides expert-parallel (EP) all-to-all operations for MoE dispatch and combine, including FP8 dispatch and deferred epilogues. It also offers communication primitives (work in progress) for pipeline parallelism (PP), context and data parallelism through Bucket collectives (CP/DP), and remote memory access through Engram. Its public buffer APIs are aligned with the NVIDIA version of [DeepEP](https://github.com/deepseek-ai/DeepEP). Ascend C kernels use HCCL/HCOMM, UBMEM and URMA, and are compiled at runtime through [DeepJIT](https://github.com/deepseek-ai/DeepJIT).

## Performance

The measurements below and the earlier low-latency BS/EP matrix predate the
current automatic MTE transport integration and removal of weighting from the
legacy LL combine kernels. They are historical results, not validation of this
revision. See [Automatic decode transport in EPBuffer](#automatic-decode-transport-in-epbuffer)
for the current native EP128 policy and its validation scope; the historical
tables do not establish a speedup of the current integration.

Measured on Ascend 950DT NPUs with CANN 9.2.0 and the manually configured PoC HDK described under [Recommended HDK and firmware](#recommended-hdk-and-firmware). All measurements use netlayer 1, the supernode's external Clos network.

The following runs use 16,384 tokens/rank capacity, hidden size 7168, top-6 routing over 256 experts, 32 AI cores / 64 AIVs, expert alignment 128 and no bias. Dispatch uses FP8 with row-major scales; combine uses BF16. Each rank uses 10 warmups and 50 samples with cache flushing.

| EP size | Dispatch (GB/s) | Combine (GB/s) |
| --- | ---: | ---: |
| EP8 | 373–375 | 345–347 |
| EP16 | 348–352 | 338–341 |
| EP32 | 335–340 | 320–324 |
| EP64 | 323–327 | 294–298 |
| EP128 | 313–320 | 272–278 |

Bandwidth ranges cover all ranks, using timings that include issue and drain but exclude final epilogues. Sustained dispatch reaches roughly 90–95% of the physical payload bandwidth limit for EP sizes up to 32. Larger EP sizes and combine remain under optimization; combine has additional local reduction overhead and HBM contention with URMA. Huawei firmware improvements are also planned to reduce this contention.

## Quick start

### Requirements

- Linux on an Ascend host with supported NPU drivers and firmware.
- Ascend 950 (A5) with UBMEM connectivity and UBC_CTP/URMA channels between the participating ranks.
- An even number of ranks for multi-rank communication.
- CANN with Ascend C, the Bisheng compiler, HCCL and HCOMM headers/libraries.
- Python 3.10 or newer, matching CPU-build PyTorch/`torch_npu` packages, and NumPy for the tests.
- A C++20 compiler and standard library supporting `std::format`. For GCC/libstdc++, use GCC 13 or newer with matching development headers and runtime libraries ([GCC 13 release notes](https://gcc.gnu.org/gcc-13/changes.html)).
- elfutils development headers, including `elfutils/libdwfl.h`, for DeepJIT's host-side exception handling.

The validated stack is Ascend 950DT, CANN 9.2.0, Python 3.12, PyTorch 2.13.0+cpu and torch_npu 2.13.0rc1, with the PoC HDK configuration described below. Kernel support on other Ascend generations or CANN versions has not been established by these measurements.

Installation builds the host extension; device kernels are compiled on first use. Keep the CANN toolkit and Bisheng available at runtime. Set `ASCEND_HOME_PATH` to the active CANN installation, or use `ASCEND_TOOLKIT_HOME`.

Use the CPU build of PyTorch with `torch_npu`. In the Python 3.11 build check,
PyTorch `2.12.0+cu130` and `torch_npu 2.12.0.post1` failed during import because
both registered the same allocator configuration keys. Using `2.12.0+cpu`
resolved that failure; disabling backend autoload alone did not.

The build selects the installed CANN directories for the host CPU (`x86_64` or
`aarch64`). The pinned DeepJIT revision assumes an ARM toolkit path, so the
build creates a version-checked header copy under its temporary build directory
with a host-aware toolkit lookup. The source submodule and CANN installation
are not changed. Updating DeepJIT requires reviewing this compatibility copy.

### Recommended HDK and firmware

Use Huawei's Q3 commercial HDK release for Atlas 850E when it becomes publicly available. Huawei has advised that this release includes the configuration required for full-bandwidth operation. Public availability is currently planned for mid-October 2026, around October 15, through the [Huawei Atlas 850E software download page](https://support.huawei.com/enterprise/zh/atlas-computing/atlas-850e-pid-265102853/software), where users can apply for access when the release becomes available. The date is a vendor release plan; availability is subject to Huawei's publication schedule.

The performance results in this README were collected using a PoC HDK supplied to DeepSeek, with additional manual configuration. That configuration is not a publicly distributed release. Users on earlier PoC versions may observe lower bandwidth and should first check their HDK version and configuration. Use the Q3 commercial release, once available, as the recommended public deployment baseline; these measurements are not results from that unreleased commercial version.

### Installation

Install the elfutils development packages in the host or container where you build the extension. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libdw-dev libelf-dev
```

On openEuler/Fedora/RHEL, use the corresponding packages:

```bash
sudo dnf install -y elfutils-devel elfutils-libelf-devel
```

On openEuler 24.03 LTS SP4, enable [GCC Toolset 14](https://docs.openeuler.org/zh/docs/25.03/server/development/gcc/gcc-14-secondary-version-compilation-toolchain-user-guide.html) for C++20 `<format>` support. The C++ package in the [SP4 aarch64 repository](https://repo.openeuler.org/openEuler-24.03-LTS-SP4/everything/aarch64/Packages/) is named `gcc-toolset-14-gcc-c++`:

```bash
sudo dnf install -y gcc-toolset-14-gcc gcc-toolset-14-gcc-c++ \
    gcc-toolset-14-binutils gcc-toolset-14-libstdc++-devel
source /opt/openEuler/gcc-toolset-14/enable
export CC=/opt/openEuler/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/openEuler/gcc-toolset-14/root/usr/bin/g++
```

Activate the CANN environment and install the matching PyTorch/`torch_npu` packages first, then:

```bash
git clone https://github.com/deepseek-ai/DeepEP-Ascend.git
cd DeepEP-Ascend

# Fetch the required DeepJIT submodule over HTTPS.
git submodule update --init --recursive third-party/deep_jit

python -m pip install --no-build-isolation .
```

For development, use `bash develop.sh` to build and link the extension into the source tree. The `clangd-ascend` submodule and CMake configuration are for IDE/debugging support.

## Features and API compatibility

Training, inference prefill and decoding share the same `EPBuffer` interface and the `deep_ep` Python package name. The implementation follows DeepEP's EPBuffer-based V2.5 APIs; supported modes and stream behavior are specific to Ascend.

| Interface | Ascend support |
| --- | --- |
| `EPBuffer` | Expanded BF16/FP8 dispatch, BF16 combine, cached handles, deterministic layouts, expert padding and deferred epilogues; automatic MTE transport for eligible decode shapes in the same buffer |
| `DecodeBuffer` | BF16 convenience wrapper over one `EPBuffer`; exposes received routing weights and leaves weighting to the expert computation |
| `PPBuffer` | Send/receive between adjacent pipeline ranks |
| `EngramBuffer` | NPU-backed BF16/FP8 tables and per-layer fetch hooks |
| `BucketBuffer` | Batched all-gather, registered storage, multiple groups and sessions for ordinary tensors |
| `BufferAllocator`, `BufferBase`, `deep_ep.comm` | Allocation planning, buffer lifecycle, communicator queries, barriers and stream access |

PP, Engram and Bucket are experimental; remaining work is listed under [Ongoing](#ongoing). EP requires `do_expand=True` and `allow_multiple_reduction=True`; `do_handle_copy` must remain `False`.

Routing indices use `deep_ep.topk_idx_t`, currently fixed to `torch.int64`. Explicit RDMA service levels and QP counts are unsupported; leave `sl_idx=None` and QP arguments at their defaults. The NV-compatible `num_sms` argument selects AI cores on Ascend EP, with `0` selecting the default; cached dispatch and combine reuse the count saved in the handle.

## Ongoing

The Ascend backend is under active development:

- Bucket collectives: batched all-gather is available; Ascend reduce-scatter and all-reduce kernels are still being built.
- Expert load balancing: `EPBuffer.lb_prefetch_weights` and `EPBuffer.lb_reduce_grads` expose the aligned APIs, but their Ascend communication kernels are not implemented yet.
- PP and Engram: these interfaces remain experimental, with further correctness and performance validation ongoing.

Hybrid communication, CPU-backed Engram storage and graph capture remain unsupported.

## EP usage in training and inference

### Buffer initialization

Create one long-lived buffer per EP group and reuse it across layers or microbatches. All ranks must agree on the token capacity. Complete outstanding operations before resizing, reusing or destroying a buffer.

```python
import torch
import torch.distributed as dist
import torch_npu

from deep_ep import EPBuffer

# First select the local NPU and initialize torch.distributed with backend='hccl'.
ep_group = dist.group.WORLD
max_tokens_per_rank = 4096
hidden = 7168
num_topk = 6
num_experts = 256

buffer = EPBuffer(
    ep_group,
    num_max_tokens_per_rank=max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=False,  # Reserve enough storage for BF16 and FP8 dispatch.
    explicitly_destroy=True,
)
```

Use `EPBuffer.get_buffer_size_hint()` to check capacity when managing a buffer pool. After all uses and their waits have finished, call `buffer.destroy()` before tearing down the process group.

### Dispatch and combine

Training, prefill and decoding use the same `EPBuffer` dispatch and combine APIs. The following helpers use the expanded expert layout and defer epilogues with `defer_epilogue=True` and `async_with_compute_stream=True`.

```python
def dispatch_forward(buffer, x, topk_idx, topk_weights, num_experts,
                     expert_alignment=1, do_cpu_sync=True):
    """Route BF16 or FP8 inputs to their selected experts."""
    return buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        expert_alignment=expert_alignment,  # Match the grouped GEMM's token alignment.
        do_cpu_sync=do_cpu_sync,  # False when expert kernels use device-side counts.
        do_expand=True,
        do_zero_padding=True,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )


def dispatch_backward(buffer, grad_recv_x, grad_recv_weights, handle, bias=None):
    """The backward of dispatch is a combine."""
    return buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_weights,
        bias=bias,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )


def combine_forward(buffer, expert_output, handle, bias=None):
    """Reduce BF16 expert outputs back to their original ranks."""
    return buffer.combine(
        expert_output,
        handle=handle,
        bias=bias,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )


def combine_backward(buffer, grad_output, handle):
    """The backward of combine reuses the forward handle for dispatch."""
    return buffer.dispatch(
        grad_output,
        handle=handle,
        do_expand=True,
        do_zero_padding=True,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )
```

Launch communication first, run independent computation, then call `.wait()` when its result is needed. EP kernels and their epilogues run on the current NPU stream:

```python
pending = dispatch_forward(
    buffer, x, topk_idx, topk_weights, num_experts,
    expert_alignment=expert_alignment,
    do_cpu_sync=need_cpu_expert_counts,
)

# Overlap application computation with communication.
shared_output = shared_experts(x)
recv_x, _, recv_weights, handle = pending.wait()

# Expert GEMMs apply recv_weights once and produce BF16 outputs.
expert_output, expert_ctx = local_experts_forward(recv_x, recv_weights, handle)

pending = combine_forward(buffer, expert_output, handle, bias=shared_output)
# ... other independent computation ...
output, _ = pending.wait()
```

Training saves `handle` for `combine_backward` and `dispatch_backward`; wait on their returned events in the same way. See [EP tests](tests/ep/test_ep.py) for complete examples.

## Automatic decode transport in EPBuffer

Use the standard `EPBuffer` for both decode and larger batches. For eligible
EP128 decode shapes it selects direct MTE peer writes inside the normal dispatch and
combine kernels, using the existing symmetric peer mappings and the same
communication arena. No `low_latency_mode` flag or second buffer is needed.
Larger batches and other configurations use the existing transport.

This integrates the LL MTE transfer mechanism into the standard kernels. It
does **not** replace them with the historical fused, server-deduplicated LL
kernels. Dispatch and combine retain their standard metadata, handles and
separate epilogues. Cached dispatch, expert alignment, deterministic ordering,
deferred epilogues, bias and the optional combine weight-return side channel
retain their existing supported semantics.

The weight contract is the standard EPBuffer contract: dispatch transports
`topk_weights` and returns `recv_topk_weights`; the expert computation applies
the weights when required by the model. **Combine sums the supplied expert
outputs and does not multiply them by routing weights.** Its optional
`topk_weights` argument communicates a separate weight tensor, rather than
scaling the hidden-state tensor.

The automatic policy is limited to EP128, A5 (`dav-3510`), 64 AIVs, at least
190 KiB of usable UB, BF16, H=7168, K=8, E=128 and `0 < B <= capacity <= 256`:

| EP | Actual tokens per rank considered for MTE transport | Otherwise |
| --- | --- | --- |
| 128 | 1–32, subject to the conditions above | Existing URMA transport |
| Other EP sizes or model shapes | None | Existing transport |

EP128 retains the candidate policy used in the native transport comparison:
small eligible batches use MTE, while larger batches retain URMA. Other EP sizes
always select URMA in the default native path. This does not change the explicit
experimental `low_latency_mode=True` interface described below.

The 2026-10-21 comparison of candidate `53e8370` with official `3b25377` covered
EP128 B=2/8/16/32/48/64/96/128/256/512/1024/2048/4096, with BF16, H7168/K8/E128,
M=B, alignment=1, no bias and `do_cpu_sync=False`. Actual kernels used MTE only
at B=2/8/16/32 and URMA at B>=48. Timings included the standard epilogues;
the paired runs and raw profiler data were verified. These measurements support
retaining that combined policy, not forcing MTE at every EP128 batch size or
claiming that all API configurations have been validated. Dispatch and combine
decide independently: FP8 dispatch uses URMA, but its BF16 combine may use MTE
if the original B and handle capacity satisfy the policy.

Set `EP_DISABLE_LOW_LATENCY_TRANSPORT=1` consistently on all ranks to
disable the new transport for regression comparisons. Default operation is
automatic; this switch is for validation.

On 2026-10-13, the host extension built and imported on x86_64 Ubuntu 22.04 with
CANN 9.3.0 (W1013), Python 3.11, GCC 14, PyTorch 2.12.0+cpu and
torch_npu 2.12.0.post1. Host API, layout, policy and build-path tests passed.
Offline Bisheng checks compiled eight MTE specializations (EP8/128, cached or
uncached dispatch, combine with or without weight metadata), plus the unweighted
legacy combine kernel and its ABI/layout checks. On the available 28-AI/56-AIV
device, EP1 BF16 dispatch/combine and all six EP2 BF16 cases passed on the
production fallback path with `do_cpu_sync=False`. An isolated diagnostic build
allowing EP2/56-AIV MTE also passed all six BF16 cases in
`test_ep.py` (expert alignment 128/1 and zero/one/two biases), including cached
dispatch and weight metadata checks. Three additional cases passed for deferred
epilogues, deterministic ordering, and their combination. Its logs confirmed
`dispatch_mte` and `combine_mte` ran. These diagnostic EP/core-count overrides are
not enabled in the production policy; those checks did not validate the original
EP8+ automatic MTE regions. The current policy and later EP128 evidence are
described above. Single-layer HCCL rank graphs are accepted while
retaining all required UBMEM/UBC_CTP link checks.

CPU-synchronized receive-count notification did not complete in the earlier
validation environment; the official `3b25377` baseline also reproduced the
dispatch timeout with `do_cpu_sync=True`. The unvalidated receive-count repair
is excluded from this revision. Do not interpret `do_cpu_sync=False` results as
coverage of that path; validate both `true` and the API default before merging.

After checking every participating node is idle, run the compatibility matrix
with both transports:

```bash
python tests/ep/test_ep_auto_compat.py --backend candidate-auto --num-processes 8 --bs 2 32 33 64 65 128 256 2 --output-dir compat-results/candidate-auto
python tests/ep/test_ep_auto_compat.py --backend candidate-disabled --num-processes 8 --bs 2 32 33 64 65 128 256 2 --output-dir compat-results/candidate-disabled
# Host-only policy checks.
c++ -std=c++17 -I. tests/ep/low_latency_policy_test.cpp -o /tmp/ep-policy
/tmp/ep-policy
```

Use a new output directory for each run. See the [compatibility test guide](tests/ep/README.compat.md) for official-baseline runs, CPU-sync isolation, device-path profiling and all-rank result aggregation.

The matrix covers cached weights, bias, padding, deterministic ordering, deferred
epilogues, old-handle reuse, bounded masked/duplicate route fixtures and unequal
local token counts. The duplicate fixtures do not prove arbitrary repeated-expert
capacity safety; see the inherited allocation boundary in the
[offline review](docs/pr/ep128-offline-review.md).
It checks compatibility rather than asserting a speedup. Use the existing
`test_ep.py` suite as well when validation resumes.
See the [EP128 pre-merge checklist](docs/pr/ep128-premerge-checklist.md) for
build, compatibility, device-path and release-evidence requirements.

### Optional BF16 convenience wrapper

`DecodeBuffer` provides a smaller return signature over one native `EPBuffer`.
It does not make its own BS/EP decision, allocate a second arena or apply
routing weights. Use `EPBuffer` directly when the extra layout and epilogue
options are needed.

```python
from deep_ep import DecodeBuffer

buffer = DecodeBuffer(ep_group, num_max_tokens_per_rank=256,
                      hidden=7168, num_topk=8, num_experts=128)
recv_x, handle, event = buffer.dispatch(x, topk_idx, topk_weights)
event.wait()
# Apply handle.recv_topk_weights once in the expert computation.
expert_output = local_experts_forward(
    recv_x, handle.expert_recv_counts, handle.recv_topk_weights)
output, event = buffer.combine(expert_output, handle)
event.wait()
buffer.destroy()  # Collective; before destroying the process group.
```

The wrapper exposes contiguous expert-major rows with alignment 1, int64 device
counts and a `DecodeHandle`. Only the leading sum(counts) rows
are valid; use device counts for grouped GEMM. `do_cpu_sync=True` additionally
returns exact-sized output and populates `handle.num_recv_tokens_per_expert_list`.
`handle.recv_topk_weights` contains the per-row FP32 routing weights returned
by the underlying dispatch. Repeated combine does not modify expert outputs or
these weights. `handle.backend` is always `'ep'`, not a claim about the selected
transport. Handles continue to belong to the same buffer after batch changes.

All ranks must use identical shapes/options and collective call order on
homogeneous devices. Routes must be valid, distinct expert IDs per token, with
contiguous int64 indices and FP32 weights on the current NPU. The adapter does
not add a host agreement collective or inspect device route values. These
requirements apply to every call. Input readiness across
streams remains the caller's responsibility; the adapter orders its own work.

One arena is allocated collectively at construction using EPBuffer's normal
size hint and retained until destruction. Warm up each required shape before
timing. The wrapper does not support graph capture. A communication failure is
not retried using another transport; restart the distributed job.

Validation commands (not yet run for this revision; run the NPU test only after
checking all participating nodes idle):

```bash
python tests/ep/test_decode_host.py
python tests/ep/test_decode.py --num-processes 8 --bs 2 32 64 128 2
```

The NPU test covers both count modes, random/concentrated routes, explicit
caller-side weighting, unweighted combine, repeated combine, stream changes,
batch transitions and old-handle reuse with the single buffer. For multiple nodes, use the same
`MASTER_ADDR`/`MASTER_PORT`, set `WORLD_SIZE` to the **node count**, and run once
per node with its node-index `RANK`, as for `test_ep.py`.

## Legacy explicit low-latency experiment (Ascend 950/A5)

The `low_latency_mode=True` interface remains available for isolated experiments
with the decode kernels synchronized with
[Ascend/DeepEP `8ffafeb`](https://gitcode.com/Ascend/DeepEP/commit/8ffafebc0bf812e8479fda5139333bdd71c38a34)
(2026-10-08). The kernels use this repository's
HCCL/UBMEM peer mappings and DeepJIT; SHMEM and the source repository are not
runtime dependencies. This legacy interface has its own restricted layout and
8-GiB arena; it is separate from default EPBuffer's automatic MTE transport.
Use the default EPBuffer for the complete standard API.

The migration adds EP8 support: an eight-rank group uses the direct paths within
one logical server. EP16/32/64/128/256 retain the source protocol. Dispatch and
combine select the direct path when `num_max_tokens_per_rank <= 32`, and server
deduplication above 32. This decision uses the dispatch capacity saved in its
handle, not the actual batch size or a later dispatch's capacity.
The sync includes compact packet receives, 512-byte count slots, rank-biased
peer buffers and interleaved server traversal at EP128 and above.
The previous weighted-combine revision was exercised on A5 at
EP8/16/32/64/128, including the historical balanced BS/EP matrix. Those results
do not validate this revision: multiplication by dispatch weights has now been
removed from all legacy combine paths, and its UB layout has changed.
Correctness and performance tests must be rerun when the servers are available.

```python
buffer = EPBuffer(
    ep_group, num_max_tokens_per_rank=16, hidden=7168, num_topk=8,
    low_latency_mode=True, explicitly_destroy=True,
)
recv_x, _, _, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=128,
    do_expand=True, do_cpu_sync=False,
)
event.wait()
# Only the leading sum(handle.expert_recv_counts) rows are valid. Rows are
# grouped by local expert; device counts let a grouped GEMM skip unused rows.
# This example computes an unweighted sum of expert outputs. If weighting is
# required, the caller must supply already-weighted outputs in the same layout.
expert_output = local_experts(recv_x, handle.expert_recv_counts)
output, _, event = buffer.combine(expert_output, handle)
event.wait()
# After all uses have completed, collectively destroy before the process group.
buffer.destroy()
```

Legacy explicit-mode contracts:

- Ascend 950/A5 (`dav-3510`), 64 AIVs, EP8/16/32/64/128/256, direct UBMEM connectivity
  between every pair of ranks. All ranks use identical B/M/H/K/E and collective order.
- At least 190 KiB usable UB per vector core. The combine receive arena uses
  the CANN-reported capacity, aligned down to 32 bytes and capped at 240 KiB.
  Host validation and device tiling use that same budget; smaller arenas reduce
  buffering lanes and can reject larger hidden sizes. Rebuild the host extension
  together with the packaged JIT headers when updating this layout.
- `0 < B <= num_max_tokens_per_rank <= 256`, `1 <= K <= 32`, `E <= 1024`, and E
  divisible by EP. Native layout checks additionally enforce UB/window limits.
  Worst-case received rows `EP * B * min(K, E / EP)` must not exceed 10000.
- Exactly 8 GiB of logical communication storage per rank, plus a 2 MiB barrier
  workspace and 2 MiB padding for the `rank * 1024` byte arena offset.
  `EPBuffer.get_buffer_size_hint(..., low_latency_mode=True)` returns
  the communication-storage size. The allocator may round huge-page allocations.
- Expanded layout, expert alignment 1, contiguous NPU tensors. Routes are int64,
  must be in `[0, E)`, and have no repeated expert within a token's top-k.
  Masked (`-1`) and duplicate routes are unsupported by these LL kernels.
  Dispatch still requires FP32 weights for its transported metadata. Combine
  ignores those values and sums the supplied expert outputs without multiplying
  by weights. Returned dispatch indices and weights are still `None` in this
  legacy interface; use standard EPBuffer to receive weights for expert-side
  application. Combine has no weight-return side channel or bias.
- BF16/FP16 inputs, or `(FP8 E4M3FN, scales)` with scales shaped `[B, ceil(H/128)]`:
  INT32 packs of four E8M0 per-32 scales, or FP32 per-128 scales. Scale bytes are
  transported unchanged. These scale packs differ from the throughput path's
  INT16 format. FP8 dispatch combines BF16 or FP16 expert outputs.
- `do_cpu_sync=True` (the default) reads kernel-produced expert counts and trims
  output to the exact received rows. With `False`, output retains worst-case
  capacity based on actual B, and counts remain on the device in
  `handle.expert_recv_counts`. `do_zero_padding=True` clears unused output rows.
- Complete kernels run on the current stream. An internal event serializes
  calls when the current stream changes. Handles can be reused for combine;
  cached dispatch, deterministic sorting, expert padding, deferred epilogues,
  handle copying and load-balancing stats/storage are unsupported in this mode.
  `num_sms` is 0 or 32. No unsupported option silently falls back to throughput.

### Single-host, 8-NPU test commands

Run on the Linux Ascend host, after activating its CANN environment and matching
PyTorch/torch_npu installation. Build and test from this repository:

```bash
cd /path/to/DeepEP-Ascend
git submodule update --init --recursive third-party/deep_jit
# Build and link the matching native extension into this source checkout.
# Select the host compiler as described under Installation first.
bash -e develop.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Check for a stale native extension from another installation.
python - <<'PY'
from pathlib import Path
import torch_npu
import deep_ep
from deep_ep import _C

root = Path.cwd().resolve()
for module in (deep_ep, _C):
    path = Path(module.__file__).resolve()
    print(f'{module.__name__}: {path}')
    assert path.is_relative_to(root), f'Loaded another DeepEP installation: {path}'
PY

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=48360
export HCCL_IF_BASE_PORT=48361 TASK_QUEUE_ENABLE=0
export EP_JIT_LAUNCH_TIMEOUT=300

# Legacy EP8 smoke test: BF16, three dispatch rounds and unweighted combine.
python tests/ep/test_low_latency.py --num-processes 8 \
    --num-tokens 16 --hidden 7168 --num-topk 8 --num-experts 128 \
    --dtype bf16 --test-first-only --skip-perf-test

# Correctness matrix: both count modes, random/hotspot routes and FP8 scale formats.
python tests/ep/test_low_latency.py --num-processes 8 \
    --num-tokens 8 16 32 --hidden 7168 --num-topk 8 --num-experts 128 \
    --dtype bf16 fp16 fp8 --fp8-scale-dtype int32 float32 \
    --do-cpu-sync 0 1 --skip-perf-test

# Keep B fixed and switch direct/dedup paths in the same buffer, including M=256.
python tests/ep/test_low_latency.py --num-processes 8 \
    --num-tokens 16 --num-max-tokens-per-rank 32 33 256 \
    --hidden 7168 --num-topk 8 --num-experts 128 \
    --dtype bf16 fp16 fp8 --fp8-scale-dtype int32 float32 \
    --do-cpu-sync 0 1 --skip-perf-test

# Public-call latency on the NPU stream, after warmup; slowest rank mean.
python tests/ep/test_low_latency.py --num-processes 8 \
    --num-tokens 16 --hidden 7168 --num-topk 8 --num-experts 128 \
    --dtype bf16 fp8 --route random --do-cpu-sync 0 --warmup 10 --num-iters 50
```

`WORLD_SIZE` is the number of nodes and `RANK` is the node index for this launcher;
it spawns eight local workers itself. Do not wrap the command in `torchrun`.
The first call compiles device kernels. If the host has no 1-GiB huge-page pool,
add `--no-huge1g-page`. Timing reports `dispatch_api_us` and `combine_api_us`:
these include the Python adapter's queued work and are not isolated kernel timings.
The test defaults M to each case's B; pass `--num-max-tokens-per-rank` to test
different capacities. Add `--plan` to inspect any matrix without importing Torch.

Host-only checks can run before installing Torch or CANN:

```bash
python tests/ep/test_low_latency_host.py
python tests/ep/test_low_latency.py --num-processes 8 --plan
c++ -std=c++17 -Ideep_ep/include tests/ep/low_latency_layout_test.cpp -o /tmp/deepep-ll-layout
/tmp/deepep-ll-layout
```

Imported headers retain their notices. The synchronized source's BSD license is
in [third-party/ascend-deepep.LICENSE](third-party/ascend-deepep.LICENSE); the earlier
CANN notice is retained at [third-party/cann.LICENSE](third-party/cann.LICENSE).

### Upstream test cases at `8ffafeb`

These are source-repository test entry points; they use its ElasticBuffer/SHMEM
runtime. Use `tests/ep/test_low_latency.py` above for this port's HCCL/DeepJIT path.

| Upstream entry | Cases |
| --- | --- |
| `tests/elastic/test_ep_low_latency.py` | LL valid unique random routes (`normal`); BF16/FP16/FP8, both count modes, INT32/FP32 FP8 scales, repeated combine and profiler kernel timing. Its launcher accepts EP16/32/64/128/256 and `0 < B <= M <= 256`; EP8 is provided by this port. |
| `tests/elastic/config_ll.example.json` | 144 combinations: B=8/16/32, dtype=BF16/FP16/FP8, E=64/128, H=4096/7168, K=4/8, CPU sync=0/1; default INT32 scales and two hosts of eight ranks. |
| `tests/elastic/test_notify.py`, `test_dispatch.py` | HT normal/random/duplicates/invalid/empty/hotspot routes; dispatch checks BF16/FP8, metadata and cached calls. |
| `tests/elastic/test_combine.py` | HT normal/random/duplicates/invalid/empty/local_duplicates/local_experts; optional `sq_pressure`, repeated calls, weight return, token-count and chunk-size matrices (256/512/1024). |
| `tests/elastic/test_low_latency_host.py`, `test_ll_st_contract.py` | LL host API, launch/capacity contracts, independent references and profiler statistics. |
| `tests/api`, `tests/runtime`, `tests/integration`, `tests/test_reference.py`, `tests/test_precision.py` | API/build/lifecycle checks, CPU/NPU reference adapters, Golden and precision tests. |
| `experiments/megamoe/tests`, `experiments/ep_gmm_fused/tests` | Experimental fused-compute API, reference, packaging and measurement tests. |

The upstream JSON is a matrix example, not evidence that every case passed on
this port. See the [upstream test guide](https://gitcode.com/Ascend/DeepEP/blob/8ffafebc0bf812e8479fda5139333bdd71c38a34/tests/README.md)
and [LL launcher](https://gitcode.com/Ascend/DeepEP/blob/8ffafebc0bf812e8479fda5139333bdd71c38a34/tests/elastic/test_ep_low_latency.py).

## Other buffer interfaces

### Bucket all-gather

Bucket communication runs on the comm stream; its epilogue runs on the caller's current stream. Set `EP_AVOID_RECORD_STREAM=1` before use. A session permits ordinary contiguous NPU inputs:

```python
from deep_ep import BucketBuffer, get_num_allocation_alignment

shard = torch.full((1024,), ep_group.rank(), dtype=torch.float32, device='npu')
alignment = get_num_allocation_alignment()
gathered_bytes = ep_group.size() * shard.nbytes
num_bytes = (gathered_bytes + alignment - 1) // alignment * alignment
bucket = BucketBuffer(
    ep_group, num_bytes, explicitly_destroy=True)
with bucket.session():
    pending = bucket.all_gather(shard)
    gathered = pending.wait()  # Flat, rank-ordered view of session storage.
    # Consume or copy gathered before the session storage is reused.
bucket.destroy()
```

All ranks must supply equally sized shards, and the bucket must have enough space for the gathered output; sessions do not grow its storage. Outside a session, output storage must belong to the bucket; without explicit `dsts`, each input must be the local rank's shard of a registered gathered tensor. Use `BufferAllocator` to plan persistent registered storage. Batches support up to 64 tensors, and multi-group buffers select a group with `group=`. Session exit adds no barrier. See [all-gather tests](tests/bucket/test_all_gather.py) and [session tests](tests/bucket/test_session.py).

### Pipeline parallelism and Engram

`PPBuffer` reserves send/receive storage with `num_max_tensor_bytes` and `num_max_inflight_tensors`, then provides `send(x, dst_rank_idx)` and `recv(x, src_rank_idx)` for adjacent pipeline ranks. See [PP tests](tests/pp/test_pp.py).

`EngramBuffer` uses `get_storage_size_hint()`, `set_config()` and `write()` to prepare NPU-backed tables. `fetch(indices)` returns per-layer completion hooks; invoke each hook before consuming its result. See [Engram tests](tests/engram/test_engram.py), including FP8 mode. PP and Engram run entirely on the current stream.

## Environment and tests

Set environment variables consistently on every rank, before importing the package or constructing buffers.

| Variable | Use |
| --- | --- |
| `EP_AVOID_RECORD_STREAM` | Required for Bucket all-gather; EP retains URMA sources through epilogue captures independently of this flag |
| `EP_BUFFER_DEBUG` | Print EP buffer initialization diagnostics; unset to disable |
| `EP_DISABLE_BARRIER_PROFILING` | Disable the extra per-iteration synchronization used by benchmark helpers |

DeepJIT settings use `EP_JIT_*` for DeepEP, falling back to the corresponding `DJ_JIT_*` process-wide setting and then the built-in default. For example, `EP_JIT_CACHE_DIR` takes precedence over `DJ_JIT_CACHE_DIR`. The Ascend backend supports:

| Variable | Default | Use |
| --- | --- | --- |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | Cache root or colon-separated roots; search all roots in order and compile misses into the first |
| `EP_JIT_DEBUG` | `0` | Print compiler commands, binary paths and load timings |
| `EP_JIT_DUMP_ASM` | `0` | Save Ascend assembly under the cache entry's `asm/` directory on a cache miss |
| `EP_JIT_KERNEL_DEBUG_INFO` | `0` | Include kernel source and line information for debugging |
| `EP_JIT_LAUNCH_TIMEOUT` | `300` | Kernel launch timeout in seconds; `0` disables it |
| `EP_JIT_PRINT_COMPILER_COMMAND` | `0` | Print Bisheng and linker commands |
| `EP_JIT_PRINT_LOAD_TIME` | `0` | Print kernel binary load timings |

See [DeepJIT's environment reference](https://github.com/deepseek-ai/DeepJIT#environment-variables) for details.

To reproduce the default EP performance cases after activating the CANN environment:

```bash
export EP_AVOID_RECORD_STREAM=1
export TASK_QUEUE_ENABLE=0
export HCCL_IF_BASE_PORT=48361
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH=/tmp/deepep-ascend-logs
mkdir -p "$ASCEND_PROCESS_LOG_PATH"

python tests/ep/test_ep.py --num-processes 8 --num-ai-cores 32 \
    --num-tokens 16384 --hidden 7168 --num-topk 6 --num-experts 256 \
    --dispatch-dtype bf16 --test-first-only
python tests/ep/test_ep.py --num-processes 8 --num-ai-cores 32 \
    --num-tokens 16384 --hidden 7168 --num-topk 6 --num-experts 256 \
    --dispatch-dtype fp8 --test-first-only
```

For multiple nodes, launch the same command on each node with a shared `MASTER_ADDR` and `MASTER_PORT`, setting `WORLD_SIZE` to the number of nodes and `RANK` to the node index. Omit `--test-first-only` and the dtype filter for the full mode matrix. Additional checks include:

```bash
python tests/buffer/test_allocation.py
python tests/comm/test_stream.py
python tests/comm/test_barrier.py
python tests/bucket/test_all_gather.py
python tests/bucket/test_session.py
python tests/pp/test_pp.py
python tests/engram/test_engram.py
python tests/engram/test_engram.py --use-fp8
python tests/utils/test_gate.py
```

The test helpers in [envs.py](deep_ep/utils/envs.py) interpret `WORLD_SIZE` and `RANK` as node count and node index, and spawn `--num-processes` local workers. Multi-node invocations also require a shared `MASTER_ADDR` and `MASTER_PORT`; these launch conventions do not imply that every topology is supported by the current backend.

## Acknowledgement

DeepEP-Ascend is built on top of Huawei's HCCL/HCOMM, UBMEM and URMA communication stack. Thanks to the asc-comm team for their support!

## Citation

```bibtex
@misc{deepepascend2026,
    title = {{DeepEP-Ascend}: an efficient expert-parallel communication library for {Ascend} {NPUs}},
    author = {Chenggang Zhao and Shangyan Zhou and Kexing Zhou and Rui Tian and Chenqi Zhao and Chenhao Xu and Yizhi Wang and Kuai Yu},
    year = {2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/deepseek-ai/DeepEP-Ascend}},
}
```
