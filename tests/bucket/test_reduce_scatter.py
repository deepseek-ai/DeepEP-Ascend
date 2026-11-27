import argparse
import warnings

warnings.filterwarnings('ignore', message='Permission mismatch')
warnings.filterwarnings('ignore', message=r'Warning: The .* owner does not match the current owner\.')

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, parse_num_bytes


def reduce_scatter_ref(src: torch.Tensor, group: dist.ProcessGroup, scale: float = 1.0) -> torch.Tensor:
    # Communicate into separate storage; the tested operation overwrites its input shard.
    hccl_src = src.view(group.size(), -1).clone()
    received = torch.empty_like(hccl_src)
    dist.all_to_all_single(received, hccl_src, group=group)
    # Match cyclic submission order, starting with the local contribution.
    result = received[group.rank()].clone()
    for offset in range(1, group.size()):
        result.add_(received[(group.rank() + offset) % group.size()])
    if scale != 1.0:
        result.mul_(scale)
    return result


def check_result(result: torch.Tensor, ref: torch.Tensor) -> None:
    if not torch.equal(result.view(torch.int32), ref.view(torch.int32)):
        mismatches = (result.view(torch.int32) != ref.view(torch.int32)).sum().item()
        max_error = (result - ref).abs().max().item()
        raise AssertionError(f'Reduction differs bitwise from the self-first cyclic reference: '
                             f'{result.numel()=}, {mismatches=}, {max_error=}')


def test_performance(buffer: deep_ep.BucketBuffer, src: torch.Tensor,
                     group: dist.ProcessGroup, args: argparse.Namespace) -> None:
    rank_idx, num_ranks = group.rank(), group.size()
    hccl_src = torch.randn(args.num_bytes // 4, dtype=torch.float32, device='npu')
    hccl_dst = torch.empty(hccl_src.numel() // num_ranks, dtype=torch.float32, device='npu')
    ref = reduce_scatter_ref(hccl_src, group, args.scale)
    src.copy_(hccl_src)
    src_shard = src.view(num_ranks, -1)[rank_idx]
    original_shard = hccl_src.view(num_ranks, -1)[rank_idx]

    def run_hccl():
        # Baseline measures only the HCCL collective, without scaling.
        dist.reduce_scatter_tensor(hccl_dst, hccl_src, group=group)

    def run_deep_ep():
        # Include the deferred CQ drain, barrier and optional local scaling.
        buffer.reduce_scatter(src, group=group, num_sms=args.num_sms, scale=args.scale).wait()

    def restore_src():
        src_shard.copy_(original_shard)

    run_hccl()
    run_deep_ep()
    check_result(src_shard, ref)
    restore_src()

    bench_kwargs = dict(num_tests=args.num_tests,
                        barrier=lambda: buffer.barrier(group=group, with_cpu_sync=True))
    hccl_t, _, _ = bench(run_hccl, num_warmups=args.num_warmups, **bench_kwargs)
    for _ in range(args.num_warmups):
        run_deep_ep()
        restore_src()
    t, _, _ = bench(run_deep_ep, num_warmups=0, post_fn=restore_src, **bench_kwargs)
    durations = torch.tensor([hccl_t, t], dtype=torch.float32, device='npu')
    dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
    hccl_t, t = durations.tolist()
    run_deep_ep()
    check_result(src_shard, ref)
    max_error = (src_shard - ref).abs().max()
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX, group=group)

    num_remote_bytes = src.nbytes // num_ranks * (num_ranks - 1)
    dist_print(f'Performance (pull-reduce, 1 bucket, 1 issuing block):\n'
               f' > Ranks: {num_ranks}, resident SMs: {args.num_sms}, scale: {args.scale}\n'
               f' > Input bytes per rank: {src.nbytes} ({src.nbytes / (1 << 30):g} GiB), '
               f'bytes per peer: {src.nbytes // num_ranks}\n'
               f' > Receive scratch bytes: 0; max absolute error: {max_error.item():.8g}\n'
               f' > Timing: NPU events around complete collectives, source restore excluded, '
               f'max rank mean; {args.num_warmups} warmups, {args.num_tests} iterations\n'
               f' > DeepEP: {t * 1e6:.1f} us | {num_remote_bytes / t / 1e9:.1f} GB/s remote | '
               f'{hccl_t / t:.2f}x HCCL\n'
               f' > HCCL: {hccl_t * 1e6:.1f} us | {num_remote_bytes / hccl_t / 1e9:.1f} GB/s remote',
               once_in_node=True)


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2 and num_ranks % 2 == 0
    assert args.num_bytes % (4 * num_ranks) == 0, '--num-bytes must be divisible by 4 * world size'
    subgroup = None
    for rank_begin_idx in range(0, num_ranks, 2):
        item = dist.new_group(list(range(rank_begin_idx, rank_begin_idx + 2)))
        if rank_begin_idx <= rank_idx < rank_begin_idx + 2:
            subgroup = item

    shard_sizes = (1, 7, 127, 128, 129, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 16385, 262145, 1048577)
    plan = deep_ep.BufferAllocator()
    source_storage = plan.allocate((1 + num_ranks * max(shard_sizes),), torch.float32)
    buckets = [plan.allocate((num_ranks, 16 if idx % 2 == 0 else 257 + idx), torch.float32)
               for idx in range(64)]
    perf_src = None if args.skip_perf_test else plan.allocate((args.num_bytes // 4,), torch.float32)
    buffer = deep_ep.BucketBuffer([group, subgroup], plan, num_gpu_timeout_secs=30, explicitly_destroy=True)

    for item in (group, subgroup):
        for num_elems in shard_sizes:
            src = source_storage[1:1 + item.size() * num_elems]
            original = torch.randn(src.shape, dtype=torch.float32, device='npu')
            src.copy_(original)
            ref = reduce_scatter_ref(original, item, args.scale)
            handle = buffer.reduce_scatter(src, group=item, num_sms=args.num_sms, scale=args.scale)
            # Exercise deferred completion while independent work runs.
            _ = torch.ones(32, dtype=torch.float32, device='npu') + 1
            result = handle.wait()
            assert result.ndim == 1
            assert result.data_ptr() == src.data_ptr() + item.rank() * num_elems * src.element_size()
            check_result(result, ref)
            # Other ranks read these shards; they must remain unchanged.
            begin, end = item.rank() * num_elems, (item.rank() + 1) * num_elems
            assert torch.equal(src[:begin], original[:begin])
            assert torch.equal(src[end:], original[end:])

    # All 64 buckets: distinct, exactly representable inputs catch lost/duplicated peers.
    for iteration in range(2):
        for idx, src in enumerate(buckets):
            values = torch.arange(src.numel(), dtype=torch.float32, device='npu').view(src.shape) % 17
            values.add_(rank_idx + 100 * idx + iteration * 1000)
            src.copy_(values)
        outputs = buffer.reduce_scatter(buckets, group=group, num_sms=args.num_sms, scale=args.scale).wait()
        for idx, (output, src) in enumerate(zip(outputs, buckets, strict=True)):
            elems = src.shape[1]
            local_values = torch.arange(rank_idx * elems, (rank_idx + 1) * elems,
                                        dtype=torch.float32, device='npu') % 17
            ref = num_ranks * (local_values + 100 * idx + iteration * 1000) + num_ranks * (num_ranks - 1) // 2
            if args.scale != 1.0:
                ref.mul_(args.scale)
            assert output.data_ptr() == src[rank_idx].data_ptr()
            torch.testing.assert_close(output, ref, rtol=0, atol=0)

    # Sessions stage unregistered sources without allocating receive scratch.
    external = torch.randn(num_ranks * 257, dtype=torch.float32, device='npu')
    ref = reduce_scatter_ref(external, group, args.scale)
    with buffer.session():
        result = buffer.reduce_scatter(external, group=group, scale=args.scale).wait()
        check_result(result, ref)
    buffer.barrier(group=group, with_cpu_sync=True)

    dist_print('Pull-reduce bitwise correctness, source-shard placement, untouched remote-source shards, '
               '64-bucket batches, session staging and scaling passed', once_in_node=True)
    if not args.skip_perf_test:
        test_performance(buffer, perf_src, group, args)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test pull-reduce scatter')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=0, choices=(0,))
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30,
                        help='Total performance input bytes PER RANK; default: 1G (1 GiB)')
    parser.add_argument('--num-warmups', type=int, default=10)
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--skip-perf-test', action='store_true')
    args = parser.parse_args()
    if args.num_bytes % 4:
        parser.error('--num-bytes must be divisible by 4 (float32)')
    if args.num_warmups < 0 or args.num_tests <= 0:
        parser.error('--num-warmups must be nonnegative and --num-tests must be positive')
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
