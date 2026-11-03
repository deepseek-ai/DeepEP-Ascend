# DeepEP-Ascend

**（中文介绍）**[DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) 是面向华为昇腾 NPU 的高性能机器学习训练与推理通信库，提供 MoE dispatch/combine 的专家并行（EP）all-to-all 操作，支持 FP8 dispatch 和延迟 epilogue。此外还提供流水线并行（PP）、面向上下文并行和数据并行（CP/DP）的 Bucket 集合通信，以及 Engram 远端内存访问等通信原语（开发中）。其公开 buffer API 与 NVIDIA 版 [DeepEP](https://github.com/deepseek-ai/DeepEP) 对齐。Ascend C 内核使用 HCCL/HCOMM、UBMEM 和 URMA 完成通信，并通过 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 在运行时编译。

**(English introduction)** [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) is a high-performance communication library for machine learning training and inference on Huawei Ascend NPUs. It provides expert-parallel (EP) all-to-all operations for MoE dispatch and combine, including FP8 dispatch and deferred epilogues. It also offers communication primitives (work in progress) for pipeline parallelism (PP), context and data parallelism through Bucket collectives (CP/DP), and remote memory access through Engram. Its public buffer APIs are aligned with the NVIDIA version of [DeepEP](https://github.com/deepseek-ai/DeepEP). Ascend C kernels use HCCL/HCOMM, UBMEM and URMA, and are compiled at runtime through [DeepJIT](https://github.com/deepseek-ai/DeepJIT).

## Performance

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
- Python 3.10 or newer, a matching PyTorch/`torch_npu` installation, and NumPy for the tests.
- A C++20 compiler and standard library supporting `std::format`.

The validated stack is Ascend 950DT, CANN 9.2.0, Python 3.12, PyTorch 2.13.0+cpu and torch_npu 2.13.0rc1, with the PoC HDK configuration described below. Kernel support on other Ascend generations or CANN versions has not been established by these measurements.

Installation builds the host extension; device kernels are compiled on first use. Keep the CANN toolkit and Bisheng available at runtime. Set `ASCEND_HOME_PATH` to the active CANN installation, or use `ASCEND_TOOLKIT_HOME`.

### Recommended HDK and firmware

Use Huawei's Q3 commercial HDK release for Atlas 850E when it becomes publicly available. Huawei has advised that this release includes the configuration required for full-bandwidth operation. Public availability is currently planned for mid-October 2026, around October 15, through the [Huawei Atlas 850E software download page](https://support.huawei.com/enterprise/zh/atlas-computing/atlas-850e-pid-265102853/software), where users can apply for access when the release becomes available. The date is a vendor release plan; availability is subject to Huawei's publication schedule.

The performance results in this README were collected using a PoC HDK supplied to DeepSeek, with additional manual configuration. That configuration is not a publicly distributed release. Users on earlier PoC versions may observe lower bandwidth and should first check their HDK version and configuration. Use the Q3 commercial release, once available, as the recommended public deployment baseline; these measurements are not results from that unreleased commercial version.

### Installation

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
| `EPBuffer` | Expanded BF16/FP8 dispatch, BF16 combine, cached handles, deterministic layouts, expert padding and deferred epilogues |
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

The `bench_msprof` helper keeps its historical `flush_l2=True` behavior for
compatibility: an oversized `zero_()` sweep evicts the preceding working set
but leaves the flush buffer's resident cache lines Dirty. Benchmarks that need a
Clean L2 precondition should request the two-stage path explicitly:

```python
profile = bench_msprof(
    fn,
    flush_l2=True,
    flush_l2_clean=True,
)
```

The second stage revisits the same buffer with an AscendC
`NOTALLOC_CLEAN` store kernel. Misses bypass L2, while resident lines are
retained and marked Clean. Setting `flush_l2_clean=True` while disabling
`flush_l2` is invalid because the normal write sweep is what evicts the old
working set.

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
