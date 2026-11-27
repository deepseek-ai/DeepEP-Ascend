import os
import warnings
os.environ['TORCH_NPU_DISABLED_WARNING'] = '1'
warnings.filterwarnings('ignore', message='Permission mismatch')
warnings.filterwarnings('ignore', message=r'Warning: The .* owner does not match the current owner\.')
warnings.filterwarnings('ignore', message='Cannot create tensor with')

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, parse_num_bytes


def hccl_ring_send_recv(src: torch.Tensor, dst: torch.Tensor, shift: int, cyclic: bool, group: dist.ProcessGroup) -> None:
    rank, size = group.rank(), group.size()
    dst_rank = (rank + shift) % size if cyclic else rank + shift
    src_rank = (rank - shift) % size if cyclic else rank - shift
    # HCCL requires torch_npu format metadata absent from registered from_blob views.
    hccl_src = torch.empty(src.shape, dtype=src.dtype, device='npu')
    hccl_src.copy_(src)
    hccl_dst = torch.empty(dst.shape, dtype=dst.dtype, device='npu')
    ops = []
    if 0 <= dst_rank < size:
        ops.append(dist.P2POp(dist.isend, hccl_src, dist.get_global_rank(group, dst_rank), group))
    if 0 <= src_rank < size:
        ops.append(dist.P2POp(dist.irecv, hccl_dst, dist.get_global_rank(group, src_rank), group))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    if 0 <= src_rank < size:
        dst.copy_(hccl_dst)


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2
    num_scaleout_ranks, num_scaleup_ranks = deep_ep.comm.get_physical_domain_size(group)
    assert num_scaleout_ranks == 1, 'ring_send_recv currently supports scale-up only'
    dist_print(f'RingSendRecv topology: {num_scaleout_ranks} x {num_scaleup_ranks}', once_in_node=True)

    # Allocate registered test buffers
    test_specs = [
        ((3,), torch.uint8),                  # 3 bytes (unaligned)
        ((100,), torch.uint8),                # 100 bytes (unaligned)
        ((1025,), torch.bfloat16),            # 2050 bytes (unaligned)
        ((32, 128), torch.bfloat16),          # 8 KiB
        ((512, 1024), torch.bfloat16),        # 1 MiB
        ((4096, 1024), torch.bfloat16),       # 8 MiB
    ]
    allocator = deep_ep.BufferAllocator()
    reg_srcs = [allocator.allocate(shape, dtype) for shape, dtype in test_specs]
    reg_dsts = [allocator.allocate(shape, dtype) for shape, dtype in test_specs]
    perf_numel = args.num_bytes // 2
    perf_src = allocator.allocate((perf_numel,), torch.bfloat16)
    perf_dst = allocator.allocate((perf_numel,), torch.bfloat16)
    buffer = deep_ep.BucketBuffer(group, allocator, num_gpu_timeout_secs=600, explicitly_destroy=True)

    # Separate buffer for BucketSession tests
    session_bytes = deep_ep.get_num_allocation_alignment()
    while session_bytes < 64 * 1024 * 1024:
        session_bytes += deep_ep.get_num_allocation_alignment()
    session_buffer = deep_ep.BucketBuffer(group, session_bytes, num_gpu_timeout_secs=600, explicitly_destroy=True)

    # 1. Direct BucketBuffer.ring_send_recv: open-chain & cyclic, positive/negative shifts, single & batched
    shifts = [1, -1]
    if num_ranks > 2:
        shifts += [2, -2]

    for cyclic in (False, True):
        for shift in shifts:
            has_src = cyclic or (0 <= rank_idx - shift < num_ranks)
            # Single tensor per spec (including unaligned byte counts and boundary sentinel check)
            for idx, (src, dst) in enumerate(zip(reg_srcs, reg_dsts, strict=True)):
                if src.dtype == torch.uint8:
                    src.random_(0, 255)
                    dst.fill_(77)
                else:
                    src.normal_().add_(rank_idx + idx)
                    dst.fill_(-42.0)
                sentinel = dst.clone()
                ref = torch.zeros_like(dst)
                hccl_ring_send_recv(src, ref, shift, cyclic, group)
                out = buffer.ring_send_recv(src, dsts=dst, shift=shift, cyclic=cyclic, group=group).wait()
                assert out.data_ptr() == dst.data_ptr()
                assert out.shape == dst.shape and out.dtype == dst.dtype
                if has_src:
                    assert torch.equal(out, ref), f'Mismatch on spec {idx}, shift={shift}, cyclic={cyclic}'
                else:
                    assert torch.equal(out, sentinel), f'Boundary rank dst modified on spec {idx}, shift={shift}'

            # Batched tensor list (all specs together)
            refs = []
            sentinels = []
            for idx, (src, dst) in enumerate(zip(reg_srcs, reg_dsts, strict=True)):
                if src.dtype == torch.uint8:
                    src.random_(0, 255)
                    dst.fill_(99)
                else:
                    src.normal_().add_(rank_idx * 10 + idx)
                    dst.fill_(-7.0)
                sentinels.append(dst.clone())
                ref = torch.zeros_like(dst)
                hccl_ring_send_recv(src, ref, shift, cyclic, group)
                refs.append(ref)

            outs = buffer.ring_send_recv(reg_srcs, dsts=reg_dsts, shift=shift, cyclic=cyclic, group=group).wait()
            assert len(outs) == len(reg_dsts)
            for idx, (out, dst, ref, sentinel) in enumerate(zip(outs, reg_dsts, refs, sentinels, strict=True)):
                assert out.data_ptr() == dst.data_ptr()
                if has_src:
                    assert torch.equal(out, ref), f'Batch mismatch on spec {idx}, shift={shift}, cyclic={cyclic}'
                else:
                    assert torch.equal(out, sentinel), f'Batch boundary dst modified on spec {idx}, shift={shift}'

    # 2. BucketSession: unregistered external tensors, non-contiguous slices, with & without dsts, mixed with all_gather
    for cyclic in (False, True):
        for shift in (1, -1):
            has_src = cyclic or (0 <= rank_idx - shift < num_ranks)
            k_full = torch.randn(1, 512, 16, 64, dtype=torch.bfloat16, device='npu').add_(rank_idx)
            v_full = torch.randn(1, 512, 16, 64, dtype=torch.bfloat16, device='npu').add_(rank_idx + 100)
            k_slice = k_full[:, -128:, :, :]  # non-contiguous slice
            v_slice = v_full[:, -128:, :, :]
            k_ref = torch.zeros_like(k_slice.contiguous())
            v_ref = torch.zeros_like(v_slice.contiguous())
            hccl_ring_send_recv(k_slice.contiguous(), k_ref, shift, cyclic, group)
            hccl_ring_send_recv(v_slice.contiguous(), v_ref, shift, cyclic, group)

            with session_buffer.session():
                # Without dsts (returns registered views)
                k_out, v_out = session_buffer.ring_send_recv(
                    [k_slice, v_slice], shift=shift, cyclic=cyclic, group=group).wait()
                if has_src:
                    assert torch.equal(k_out, k_ref)
                    assert torch.equal(v_out, v_ref)

                # With external unregistered dsts
                ext_dst = torch.full_like(k_ref, -1.0)
                ext_sentinel = ext_dst.clone()
                out = session_buffer.ring_send_recv(
                    k_slice, dsts=ext_dst, shift=shift, cyclic=cyclic, group=group).wait()
                assert out.data_ptr() == ext_dst.data_ptr()
                if has_src:
                    assert torch.equal(out, k_ref)
                else:
                    assert torch.equal(out, ext_sentinel)

                # Follow up with all_gather in the same session to verify num_allocated_bytes stayed in sync
                ag_src = torch.randn(1024, dtype=torch.float32, device='npu').add_(rank_idx)
                ag_ref = torch.empty(num_ranks * 1024, dtype=torch.float32, device='npu')
                dist.all_gather_single(ag_ref, ag_src, group=group)
                ag_out = session_buffer.all_gather(ag_src, group=group).wait()
                assert torch.equal(ag_out, ag_ref)

    # 2b. BucketSession with omitted default kwargs (`shift=1`, `cyclic=False`)
    with session_buffer.session():
        def_src = torch.randn(64, 128, dtype=torch.bfloat16, device='npu').add_(rank_idx)
        def_ref = torch.zeros_like(def_src)
        hccl_ring_send_recv(def_src, def_ref, 1, False, group)
        def_out = session_buffer.ring_send_recv(def_src).wait()
        if rank_idx > 0:
            assert torch.equal(def_out, def_ref)

        cyc_ref = torch.zeros_like(def_src)
        hccl_ring_send_recv(def_src, cyc_ref, 1, True, group)
        cyc_out = session_buffer.ring_send_recv(def_src, cyclic=True).wait()
        assert torch.equal(cyc_out, cyc_ref)

    dist_print('Eager correctness check passed.', once_in_node=True)

    # 3. ACL Graph capture & replay (both direct BucketBuffer and BucketSession, open-chain & cyclic)
    capture_stream = torch.npu.Stream()
    for shift, cyclic in [(1, False), (-1, False), (1, True)]:
        has_src = cyclic or (0 <= rank_idx - shift < num_ranks)
        # 3a. Direct BucketBuffer.ring_send_recv in ACL Graph (8 MiB + unaligned 100B)
        graph_srcs = [reg_srcs[-1], reg_srcs[1]]
        graph_dsts = [reg_dsts[-1], reg_dsts[1]]
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(capture_stream):
            buffer.ring_send_recv(graph_srcs, dsts=graph_dsts, shift=shift, cyclic=cyclic, group=group).wait()
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=capture_stream):
            buffer.ring_send_recv(graph_srcs, dsts=graph_dsts, shift=shift, cyclic=cyclic, group=group).wait()
        for step in range(4):
            graph_srcs[0].normal_().add_(rank_idx + step)
            graph_srcs[1].random_(0, 255)
            graph_dsts[0].fill_(-5.0)
            graph_dsts[1].fill_(55)
            sentinel0, sentinel1 = graph_dsts[0].clone(), graph_dsts[1].clone()
            ref0 = torch.zeros_like(graph_dsts[0])
            ref1 = torch.zeros_like(graph_dsts[1])
            hccl_ring_send_recv(graph_srcs[0], ref0, shift, cyclic, group)
            hccl_ring_send_recv(graph_srcs[1], ref1, shift, cyclic, group)
            graph.replay()
            torch.npu.synchronize()
            if has_src:
                assert torch.equal(graph_dsts[0], ref0), f'ACL Graph mismatch (8M) at shift={shift}, cyclic={cyclic}, step={step}'
                assert torch.equal(graph_dsts[1], ref1), f'ACL Graph mismatch (100B) at shift={shift}, cyclic={cyclic}, step={step}'
            else:
                assert torch.equal(graph_dsts[0], sentinel0)
                assert torch.equal(graph_dsts[1], sentinel1)

        # 3b. BucketSession ring_send_recv + all_gather in ACL Graph (unregistered external tensors)
        ext_src = torch.randn(1, 64, 128, dtype=torch.bfloat16, device='npu').add_(rank_idx)
        ext_dst = torch.zeros_like(ext_src)
        ag_src = torch.randn(1024, dtype=torch.float32, device='npu').add_(rank_idx)
        ag_dst = torch.zeros(num_ranks * 1024, dtype=torch.float32, device='npu')
        session_graph = torch.npu.NPUGraph()
        with torch.npu.stream(capture_stream):
            with session_buffer.session():
                session_buffer.ring_send_recv(ext_src, dsts=ext_dst, shift=shift, cyclic=cyclic, group=group).wait()
                session_buffer.all_gather(ag_src, dsts=ag_dst, group=group).wait()
        torch.npu.synchronize()
        with torch.npu.graph(session_graph, stream=capture_stream):
            with session_buffer.session():
                session_buffer.ring_send_recv(ext_src, dsts=ext_dst, shift=shift, cyclic=cyclic, group=group).wait()
                session_buffer.all_gather(ag_src, dsts=ag_dst, group=group).wait()
        for step in range(4):
            ext_src.normal_().add_(rank_idx + step)
            ext_dst.fill_(-9.0)
            ext_sentinel = ext_dst.clone()
            ag_src.normal_().add_(rank_idx + step)
            ag_dst.zero_()
            ring_ref = torch.zeros_like(ext_dst)
            hccl_ring_send_recv(ext_src, ring_ref, shift, cyclic, group)
            ag_ref = torch.empty_like(ag_dst)
            dist.all_gather_single(ag_ref, ag_src, group=group)
            session_graph.replay()
            torch.npu.synchronize()
            if has_src:
                assert torch.equal(ext_dst, ring_ref), f'Session ACL Graph ring mismatch at shift={shift}, cyclic={cyclic}, step={step}'
            else:
                assert torch.equal(ext_dst, ext_sentinel)
            assert torch.equal(ag_dst, ag_ref), f'Session ACL Graph all_gather mismatch at step={step}'

    dist_print('ACL Graph correctness check passed.', once_in_node=True)

    # 4. Performance benchmark (open-chain shift=1)
    if not args.skip_perf_test:
        hccl_src = torch.empty(perf_src.shape, dtype=perf_src.dtype, device='npu').normal_().add_(rank_idx)
        perf_src.copy_(hccl_src)
        hccl_dst = torch.empty_like(hccl_src)

        def run_hccl():
            dst_rank = rank_idx + 1
            src_rank = rank_idx - 1
            ops = []
            if dst_rank < num_ranks:
                ops.append(dist.P2POp(dist.isend, hccl_src, dist.get_global_rank(group, dst_rank), group))
            if src_rank >= 0:
                ops.append(dist.P2POp(dist.irecv, hccl_dst, dist.get_global_rank(group, src_rank), group))
            for req in dist.batch_isend_irecv(ops):
                req.wait()

        def run_deep_ep():
            return buffer.ring_send_recv(perf_src, dsts=perf_dst, shift=1, cyclic=False, group=group).wait()

        run_hccl()
        out = run_deep_ep()
        if rank_idx > 0:
            assert torch.equal(out, hccl_dst), 'Performance mismatch'

        bench_kwargs = dict(num_warmups=args.num_warmups, num_tests=args.num_tests,
                            barrier=lambda: buffer.barrier(group=group, with_cpu_sync=True))
        hccl_t, _, _ = bench(run_hccl, **bench_kwargs)
        ep_t, _, _ = bench(run_deep_ep, **bench_kwargs)
        durations = torch.tensor([hccl_t, ep_t], dtype=torch.float32, device='npu')
        dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
        hccl_t, ep_t = durations.tolist()

        dist_print(f' > #Bytes: {args.num_bytes}\n'
                   f' > DeepEP (shift=1, cyclic=False): {ep_t * 1e6:7.1f} us | {args.num_bytes / ep_t / 1e9:6.1f} GB/s | {hccl_t / ep_t:.2f}x HCCL\n'
                   f' > HCCL   (shift=1, cyclic=False): {hccl_t * 1e6:7.1f} us | {args.num_bytes / hccl_t / 1e9:6.1f} GB/s',
                   once_in_node=True)

    session_buffer.destroy()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer.ring_send_recv')
    parser.add_argument('--num-processes', type=int, default=16, help='Number of processes per node')
    parser.add_argument('--num-bytes', type=parse_num_bytes, default='16M', help='Bytes per rank for performance test')
    parser.add_argument('--num-warmups', type=int, default=20, help='Number of warmup iterations')
    parser.add_argument('--num-tests', type=int, default=30, help='Number of timed iterations')
    parser.add_argument('--skip-perf-test', action='store_true', help='Skip performance benchmark')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
