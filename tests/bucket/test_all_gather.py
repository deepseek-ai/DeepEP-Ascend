import argparse
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, parse_num_bytes


def test_performance(buffer: deep_ep.BucketBuffer, gathered: torch.Tensor,
                     group: dist.ProcessGroup, args: argparse.Namespace) -> None:
    rank_idx, num_ranks = group.rank(), group.size()
    gathered.fill_(float('nan'))
    src = gathered[rank_idx]
    # HCCL requires torch_npu format metadata absent from the registered from_blob view.
    hccl_src = torch.empty(src.shape, dtype=src.dtype, device='npu').normal_().add_(rank_idx)
    src.copy_(hccl_src)
    hccl_dst = torch.empty(gathered.numel(), dtype=gathered.dtype, device='npu')

    def run_hccl():
        dist.all_gather_into_tensor(hccl_dst, hccl_src, group=group)

    def run_deep_ep():
        return buffer.all_gather(src, group=group, num_sms=args.num_sms).wait()

    run_hccl()
    result = run_deep_ep()
    assert torch.equal(result, hccl_dst), 'Performance bucket mismatch'

    bench_kwargs = dict(num_warmups=args.num_warmups, num_tests=args.num_tests,
                        barrier=lambda: buffer.barrier(group=group, with_cpu_sync=True))
    hccl_t, _, _ = bench(run_hccl, **bench_kwargs)
    t, _, _ = bench(run_deep_ep, **bench_kwargs)
    durations = torch.tensor([hccl_t, t], dtype=torch.float32, device='npu')
    dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
    hccl_t, t = durations.tolist()
    assert torch.equal(gathered.view(-1), hccl_dst), 'Performance bucket mismatch after benchmarking'

    num_gathered_bytes = gathered.nbytes
    num_remote_bytes = src.nbytes * (num_ranks - 1)
    dist_print(f'Performance (1 bucket, in-place):\n'
               f' > Ranks: {num_ranks}, resident SMs: {args.num_sms}\n'
               f' > Bytes per rank: {src.nbytes} ({src.nbytes / (1 << 30):g} GiB), '
               f'total gathered bytes: {num_gathered_bytes}\n'
               f' > Timing: NPU events around the complete operation, max rank mean; '
               f'{args.num_warmups} warmups, {args.num_tests} iterations\n'
               f' > DeepEP: {t * 1e6:.1f} us | {num_gathered_bytes / t / 1e9:.1f} GB/s gathered | '
               f'{num_remote_bytes / t / 1e9:.1f} GB/s remote | {hccl_t / t:.2f}x HCCL\n'
               f' > HCCL:   {hccl_t * 1e6:.1f} us | {num_gathered_bytes / hccl_t / 1e9:.1f} GB/s gathered | '
               f'{num_remote_bytes / hccl_t / 1e9:.1f} GB/s remote', once_in_node=True)


def check_bucket_batch(buffer: deep_ep.BucketBuffer, targets: list, item: dist.ProcessGroup,
                       args: argparse.Namespace, value_scale: int, iterations: int = 1) -> None:
    """All-gather one tensor per bucket into its registered destination and compare bitwise."""
    for iteration in range(iterations):
        srcs = [torch.full(target[0].shape, item.rank() + value_scale * idx + 1000 * iteration,
                           dtype=torch.float32, device='npu')
                for idx, target in enumerate(targets)]
        dsts = [target[:item.size()] for target in targets]
        for dst in dsts:
            dst.fill_(float('nan'))
        outputs = buffer.all_gather(srcs, dsts=dsts, group=item, num_sms=args.num_sms).wait()
        assert len(outputs) == len(targets)
        for idx, (output, target) in enumerate(zip(outputs, targets, strict=True)):
            ref = torch.stack([torch.full_like(srcs[idx], rank + value_scale * idx + 1000 * iteration)
                               for rank in range(item.size())])
            assert output.data_ptr() == target.data_ptr()
            assert torch.equal(output, ref.view(-1)), \
                f'group of {item.size()} ranks: bucket {idx} of the {value_scale}x batch, iteration {iteration}'


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2 and num_ranks % 2 == 0
    assert args.num_bytes % (4 * num_ranks) == 0, '--num-bytes must be divisible by 4 * world size'
    num_scaleout_ranks, num_scaleup_ranks = deep_ep.comm.get_physical_domain_size(group)
    mode = 'scaleup' if num_scaleout_ranks == 1 else ('scaleout' if num_scaleup_ranks == 1 else 'hybrid')
    dist_print(f'All-gather topology: {num_scaleout_ranks} x {num_scaleup_ranks} ({mode})', once_in_node=True)
    subgroup = None
    for rank_begin_idx in range(0, num_ranks, 2):
        item = dist.new_group(list(range(rank_begin_idx, rank_begin_idx + 2)))
        if rank_begin_idx <= rank_idx < rank_begin_idx + 2:
            subgroup = item
    groups = [group, subgroup]
    rail_group = None
    if num_scaleout_ranks > 1 and num_scaleup_ranks > 1:
        # Same local index in every domain: exercise the pure scale-out kernel too.
        for local_idx in range(num_scaleup_ranks):
            ranks = [domain_idx * num_scaleup_ranks + local_idx for domain_idx in range(num_scaleout_ranks)]
            item = dist.new_group(ranks)
            if rank_idx in ranks:
                rail_group = item
        assert rail_group is not None, 'every rank belongs to exactly one rail'
        assert deep_ep.comm.get_physical_domain_size(rail_group) == (num_scaleout_ranks, 1)
        groups.append(rail_group)

    plan = deep_ep.BufferAllocator()
    gathered = plan.allocate((num_ranks, 2, 8), torch.float32)
    shard_sizes = (7, 127, 128, 129, 1025, 16385)
    unaligned = plan.allocate((1 + num_ranks * max(shard_sizes),), torch.float32)
    buckets = [plan.allocate((num_ranks, 16 if idx % 2 == 0 else 257 + idx), torch.float32)
               for idx in range(64)]
    perf_gathered = None if args.skip_perf_test else plan.allocate((num_ranks, args.num_bytes // 4), torch.float32)
    # Two 40 MiB buckets per rank publish cumulative progress at 32, 72 and 80 MiB.
    # The first publication covers only a prefix of the first bucket, not the second bucket.
    marker_buckets = [plan.allocate((num_ranks, (40 << 20) // 4), torch.float32) for _ in range(2)]
    # Allow initial AICPU module deployment to finish on every rank.
    buffer = deep_ep.BucketBuffer(groups, plan, num_gpu_timeout_secs=600, explicitly_destroy=True)

    for item in groups:
        for iteration in range(8):
            # Exercise epoch reuse with unequal rank progress through successive barriers.
            if item.rank() == iteration % item.size():
                time.sleep(0.01)
            src = torch.arange(16, dtype=torch.float32, device='npu').view(2, 8)
            src += item.rank() * 100 + iteration * 1000
            ref = torch.stack([src - item.rank() * 100 + rank * 100 for rank in range(item.size())])
            dst = gathered[:item.size()]
            # Poison every shard, including our own, to detect missing local copies.
            dst.fill_(float('nan'))
            handle = buffer.all_gather(src, dsts=dst, group=item, num_sms=args.num_sms)
            # Simulate some work to be done while the all-gather is in progress.
            _ = torch.ones(32, dtype=torch.float32, device='npu') + 1
            result = handle.wait()
            assert result.ndim == 1
            assert torch.equal(result, ref.view(-1))

            src += 10000
            ref += 10000
            dst.fill_(float('nan'))
            dst[item.rank()].copy_(src)
            result = buffer.all_gather(dst[item.rank()], group=item, num_sms=args.num_sms).wait()
            assert torch.equal(result, ref.view(-1))
            buffer.barrier(group=item, with_cpu_sync=True)

    # Replay with fresh inputs and consume the result on the capture stream.
    src = torch.empty_like(gathered[0])
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = buffer.all_gather(src, dsts=gathered, group=group, num_sms=args.num_sms).wait().clone()
    ranks = torch.arange(num_ranks, dtype=src.dtype, device=src.device).view(-1, 1, 1)
    for offset in (0, num_ranks):
        src.fill_(rank_idx + offset)
        graph.replay()
        assert torch.equal(result.view_as(gathered), (ranks + offset).expand_as(gathered)), 'Graph replay mismatch'
    graph.reset()

    # Cover different shard sizes and unaligned addresses/tails in each context.
    for item in groups:
        for num_elems in shard_sizes:
            src = torch.arange(num_elems + 1, dtype=torch.float32, device='npu')[1:]
            src += item.rank() * 100
            dst = unaligned[1:1 + item.size() * num_elems].view(item.size(), num_elems)
            dst.fill_(float('nan'))
            ref = torch.stack([src - item.rank() * 100 + rank * 100 for rank in range(item.size())])
            result = buffer.all_gather(src, dsts=dst, group=item, num_sms=args.num_sms).wait()
            assert torch.equal(result, ref.view(-1))

    # Exercise the complete BucketList with distinct inputs and registered destinations.
    for item in groups:
        check_bucket_batch(buffer, buckets, item, args, value_scale=100)

    # Exercise cumulative readiness across bucket boundaries: the 32 and 72 MiB markers
    # cover only prefixes of the 80 MiB shard, so neither permits forwarding the entire batch.
    for item in groups:
        check_bucket_batch(buffer, marker_buckets, item, args, value_scale=1000, iterations=3)

    dist_print('All-gather and 64-bucket batches passed', once_in_node=True)
    if not args.skip_perf_test:
        for item in groups:
            test_performance(buffer, perf_gathered[:item.size()], item, args)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all-gather')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=0)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30,
                        help='Performance input size PER RANK, with binary suffixes: 1G (default), 64M, 512K, or bytes')
    parser.add_argument('--num-warmups', type=int, default=10)
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--skip-perf-test', action='store_true', help='Run correctness tests only')
    args = parser.parse_args()
    if args.num_sms < 0:
        parser.error('--num-sms must be nonnegative')
    if args.num_bytes % 4:
        parser.error('--num-bytes must be divisible by 4 (float32)')
    if args.num_warmups < 0 or args.num_tests <= 0:
        parser.error('--num-warmups must be nonnegative and --num-tests must be positive')
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
