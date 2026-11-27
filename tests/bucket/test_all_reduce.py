import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, parse_num_bytes


def all_reduce_ref(src: torch.Tensor, group: dist.ProcessGroup, scale: float) -> torch.Tensor:
    ref = src.clone()
    dist.all_reduce(ref, group=group)
    if scale != 1.0:
        ref.mul_(scale)
    return ref


def test_performance(buffer: deep_ep.BucketBuffer, src: torch.Tensor,
                     group: dist.ProcessGroup, args: argparse.Namespace) -> None:
    # HCCL requires torch_npu format metadata absent from the registered from_blob view.
    original = torch.empty(src.shape, dtype=src.dtype, device='npu').normal_().add_(group.rank())
    hccl_src = original.clone()
    src.copy_(original)

    def run_hccl():
        dist.all_reduce(hccl_src, group=group)
        if args.scale != 1.0:
            hccl_src.mul_(args.scale)

    def run_deep_ep():
        return buffer.all_reduce(src, group=group, num_sms=args.num_sms, scale=args.scale).wait()

    run_hccl()
    result = run_deep_ep()
    torch.testing.assert_close(result, hccl_src, rtol=1e-5, atol=1e-5)

    # Both operations overwrite their inputs. Restore outside timing, including warmups;
    # bench's post_fn only runs after timed iterations.
    for _ in range(args.num_warmups):
        hccl_src.copy_(original)
        src.copy_(original)
        run_hccl()
        run_deep_ep()
    hccl_src.copy_(original)
    src.copy_(original)
    bench_kwargs = dict(num_warmups=0, num_tests=args.num_tests,
                        barrier=lambda: buffer.barrier(group=group, with_cpu_sync=True))
    hccl_t, _, _ = bench(run_hccl, post_fn=lambda: hccl_src.copy_(original), **bench_kwargs)
    t, _, _ = bench(run_deep_ep, post_fn=lambda: src.copy_(original), **bench_kwargs)
    durations = torch.tensor([hccl_t, t], dtype=torch.float32, device='npu')
    dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
    hccl_t, t = durations.tolist()
    run_hccl()
    result = run_deep_ep()
    torch.testing.assert_close(result, hccl_src, rtol=1e-5, atol=1e-5)

    bus_bytes = 2 * src.nbytes * (group.size() - 1) / group.size()
    dist_print(f'Performance (1 bucket, in-place):\n'
               f' > Ranks: {group.size()}, AICPU threads: 1, scale: {args.scale}\n'
               f' > Input bytes per rank: {src.nbytes} ({src.nbytes / (1 << 30):g} GiB)\n'
               f' > Timing: NPU events around complete operations including scaling, source restore excluded, '
               f'max rank mean; {args.num_warmups} warmups, {args.num_tests} iterations\n'
               f' > DeepEP: {t * 1e6:.1f} us | {src.nbytes / t / 1e9:.1f} GB/s algorithm | '
               f'{bus_bytes / t / 1e9:.1f} GB/s bus | {hccl_t / t:.2f}x HCCL\n'
               f' > HCCL:   {hccl_t * 1e6:.1f} us | {src.nbytes / hccl_t / 1e9:.1f} GB/s algorithm | '
               f'{bus_bytes / hccl_t / 1e9:.1f} GB/s bus', once_in_node=True)


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_scaleout_ranks, num_scaleup_ranks = deep_ep.comm.get_physical_domain_size(group)
    mode = 'scaleup' if num_scaleout_ranks == 1 else ('scaleout' if num_scaleup_ranks == 1 else 'hybrid')
    dist_print(f'All-reduce topology: {num_scaleout_ranks} x {num_scaleup_ranks} ({mode})', once_in_node=True)
    groups = [group]
    if num_ranks > 2:
        for rank_begin_idx in range(0, num_ranks, 2):
            item = dist.new_group(list(range(rank_begin_idx, min(rank_begin_idx + 2, num_ranks))))
            if rank_begin_idx <= rank_idx < rank_begin_idx + 2:
                groups.append(item)

    # Include unaligned addresses, short slices and uneven bucket tails.
    sizes = (1, 127, 128, 129, 8191, 8192, 8193, (1 << 20) - 1, 1 << 20, (1 << 20) + 1)
    plan = deep_ep.BufferAllocator()
    # Reuse one allocation for single-tensor correctness and performance tests.
    num_storage_elems = max(max(sizes) + 2, 0 if args.skip_perf_test else args.num_bytes // 4)
    storage = plan.allocate((num_storage_elems,), torch.float32)
    buckets = [plan.allocate((2, sizes[idx % len(sizes)]), torch.float32) for idx in range(64)]
    # Allow initial AICPU module deployment to finish on every rank.
    buffer = deep_ep.BucketBuffer(groups, plan, num_gpu_timeout_secs=600, explicitly_destroy=True)

    for item in groups:
        # Test each single bucket.
        for num_elems in sizes:
            storage[:num_elems + 2].fill_(float('nan'))
            src = storage[1:1 + num_elems]
            original = torch.randn(src.shape, dtype=src.dtype, device='npu').add_(item.rank())
            src.copy_(original)

            handle = buffer.all_reduce(src, group=item, num_sms=args.num_sms, scale=args.scale)
            # Simulate some work to be done while the all-reduce is in progress.
            ref = all_reduce_ref(original, item, args.scale)
            _ = torch.ones(32, dtype=torch.float32, device='npu') + 1
            result = handle.wait()

            assert result.shape == src.shape and result.data_ptr() == src.data_ptr()
            torch.testing.assert_close(result, ref, rtol=1e-5, atol=1e-5)
            assert torch.isnan(storage[0]).item() and torch.isnan(storage[num_elems + 1]).item()

        # Test multiple buckets with two-dimensional shapes.
        originals = [torch.randn(src.shape, dtype=src.dtype, device='npu').add_(item.rank()) for src in buckets]
        refs = [all_reduce_ref(src, item, args.scale) for src in originals]
        for src, original in zip(buckets, originals, strict=True):
            src.copy_(original)
        outputs = buffer.all_reduce(buckets, group=item, scale=args.scale, num_sms=args.num_sms).wait()
        assert len(outputs) == len(buckets)
        for output, src, ref in zip(outputs, buckets, refs, strict=True):
            assert output.shape == src.shape and output.data_ptr() == src.data_ptr()
            torch.testing.assert_close(output, ref, rtol=1e-5, atol=1e-5)

        # Test sessions with external inputs.
        external = originals[-1].clone()
        with buffer.session():
            result = buffer.all_reduce(external, group=item, num_sms=args.num_sms, scale=args.scale).wait()
            torch.testing.assert_close(result, refs[-1].view(-1), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(external, originals[-1])
        buffer.barrier(group=item, with_cpu_sync=True)

    dist_print('All-reduce and 64-bucket batches passed', once_in_node=True)
    if not args.skip_perf_test:
        test_performance(buffer, storage[:args.num_bytes // 4], group, args)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all-reduce')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=0, choices=(0,), help='AICPU uses num_sms=0')
    parser.add_argument('--scale', type=float, default=1.0, help='FP32 scale applied in the AIV epilogue')
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30,
                        help='Total performance input bytes PER RANK; default: 1G (1 GiB)')
    parser.add_argument('--num-warmups', type=int, default=10)
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--skip-perf-test', action='store_true', help='Run correctness tests only')
    args = parser.parse_args()
    if args.num_bytes % 4:
        parser.error('--num-bytes must be divisible by 4 (float32)')
    if args.num_warmups < 0 or args.num_tests <= 0:
        parser.error('--num-warmups must be nonnegative and --num-tests must be positive')
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
