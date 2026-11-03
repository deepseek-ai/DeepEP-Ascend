import warnings
warnings.filterwarnings('ignore', message='Permission mismatch')
warnings.filterwarnings('ignore', message='Cannot create tensor with')

import argparse
import math
import random
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.testing import bench_msprof


def generate_stress_ops(rank_idx: int, num_ranks: int, num_sends: int, shape: tuple,
                        num_max_inflight_tensors: int):
    assert num_max_inflight_tensors > 0
    send_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}
    recv_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}

    for _ in range(num_sends):
        src_rank_idx = random.randint(0, num_ranks - 1)
        dst_rank_idx = (src_rank_idx + (1 if random.randint(0, 1) else -1)) % num_ranks
        st = random.randint(0, 10 ** 8)
        rt = st + random.randint(1, 3 * 10 ** 6)
        send_times[(src_rank_idx, dst_rank_idx)].append(st)
        recv_times[(src_rank_idx, dst_rank_idx)].append(rt)

    ops = []
    for (src_rank_idx, dst_rank_idx) in send_times:
        n = len(send_times[(src_rank_idx, dst_rank_idx)])
        sorted_send = sorted(send_times[(src_rank_idx, dst_rank_idx)])
        sorted_recv = sorted(recv_times[(src_rank_idx, dst_rank_idx)])
        for i in range(n):
            # Keep each send after its previous `num_max_inflight_tensors` recv to avoid deadlock
            if i >= num_max_inflight_tensors:
                sorted_send[i] = max(
                    sorted_send[i], sorted_recv[i - num_max_inflight_tensors] + 1)
            sorted_recv[i] = max(sorted_recv[i], sorted_send[i] + 1)
            tensor = torch.randn(shape, dtype=torch.bfloat16, device='npu')
            if src_rank_idx == rank_idx:
                ops.append(('send', sorted_send[i], dst_rank_idx, i, tensor))
            if dst_rank_idx == rank_idx:
                ops.append(('recv', sorted_recv[i], src_rank_idx, i, tensor))
    ops.sort(key=lambda x: (x[1], x[3]))
    return ops


def test_performance(buffer: deep_ep.PPBuffer, shape: tuple):
    rank_idx, num_ranks = buffer.rank_idx, buffer.num_ranks
    next_rank_idx = (rank_idx + 1) % num_ranks
    prev_rank_idx = (rank_idx + num_ranks - 1) % num_ranks
    send_tensor = torch.full(shape, rank_idx, dtype=torch.bfloat16, device='npu')
    recv_tensor = torch.empty_like(send_tensor)

    def send_recv_once():
        buffer.send(send_tensor, next_rank_idx)
        buffer.recv(recv_tensor, prev_rank_idx)

    send_profile, recv_profile = bench_msprof(
        fn=send_recv_once,
        kernel_names=['pp_send_impl', 'pp_recv_impl'],
        num_warmups=10,
        num_tests=50,
        flush_l2=False,
        barrier_comm_profiling=True,
        barrier=buffer.barrier,
    )
    expected = torch.full_like(recv_tensor, prev_rank_idx)
    assert torch.equal(recv_tensor, expected), f'Rank {rank_idx}: performance test mismatch'

    num_tensor_bytes = send_tensor.numel() * send_tensor.element_size()
    dist_print(f'   * PP: {rank_idx:3}/{num_ranks:3} | '
               f'send: {send_profile.us:.3f} us, {send_profile.gbps(num_tensor_bytes):.0f} GB/s | '
               f'recv: {recv_profile.us:.3f} us, {recv_profile.gbps(num_tensor_bytes):.0f} GB/s')


def check_receive_rings(buffer: deep_ep.PPBuffer, shape: tuple, num_max_inflight_tensors: int):
    tensor_bytes = math.prod(shape) * 2
    rdma_alignment = deep_ep.get_num_rdma_alignment()
    slot_bytes = (tensor_bytes + rdma_alignment - 1) // rdma_alignment * rdma_alignment
    allocation_alignment = deep_ep.get_num_allocation_alignment()
    ring_bytes = 2 * num_max_inflight_tensors * slot_bytes
    storage_bytes = (ring_bytes + allocation_alignment - 1) // allocation_alignment * allocation_alignment
    if buffer.storage.nbytes != storage_bytes:
        raise RuntimeError(f'PP storage has {buffer.storage.nbytes} bytes, expected {storage_bytes}')

    rank_idx, num_ranks = buffer.rank_idx, buffer.num_ranks
    peers = sorted({(rank_idx + 1) % num_ranks, (rank_idx + num_ranks - 1) % num_ranks})
    send_tensor = torch.empty(shape, dtype=torch.bfloat16, device='npu')
    recv_tensor = torch.empty_like(send_tensor)
    # Fill each neighbor's receive ring before publishing release credits.
    for slot in range(num_max_inflight_tensors):
        send_tensor.fill_(rank_idx * num_max_inflight_tensors + slot)
        for peer in peers:
            buffer.send(send_tensor, peer)
    for slot in range(num_max_inflight_tensors):
        for peer in peers:
            buffer.recv(recv_tensor, peer)
            expected = torch.full_like(recv_tensor, peer * num_max_inflight_tensors + slot)
            if not torch.equal(recv_tensor, expected):
                raise RuntimeError(f'Rank {rank_idx}: receive ring mismatch from {peer} at slot {slot}')


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    shape = (args.num_tokens, args.hidden)
    num_max_tensor_bytes = math.prod(shape) * 2
    num_max_inflight_tensors = args.num_max_inflight_tensors
    buffer = deep_ep.PPBuffer(
        group, num_max_tensor_bytes, num_max_inflight_tensors, explicitly_destroy=True)

    assert num_ranks > 1
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Shape: {shape}\n'
               f' > Max inflight tensors: {num_max_inflight_tensors}\n',
               once_in_node=True)

    check_receive_rings(buffer, shape, num_max_inflight_tensors)

    # Run stress tests
    dist_print('Running stress tests:', once_in_node=True)
    for seed in range(args.num_stress_iterations):
        dist_print(f' > Testing with {seed=} ...', once_in_node=True)
        torch.manual_seed(42 + seed)
        random.seed(42 + seed)
        ops = generate_stress_ops(rank_idx, num_ranks, args.num_sends, shape, num_max_inflight_tensors)

        for j, (op, timestamp, peer, _, tensor) in enumerate(ops):
            if op == 'send':
                buffer.send(tensor, peer)
            else:
                result = torch.empty_like(tensor)
                buffer.recv(result, peer)
                assert torch.equal(result, tensor), f'Rank {rank_idx}: mismatch at op {j}'
    dist_print(' > All stress tests passed', once_in_node=True)
    dist_print(once_in_node=True)

    if not args.skip_perf_test:
        dist_print('Running performance tests:', once_in_node=True)
        test_performance(buffer, shape)
        dist_print(once_in_node=True)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PP send/recv kernels')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-tokens', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-max-inflight-tensors', type=int, default=4)
    parser.add_argument('--num-stress-iterations', type=int, default=4)
    parser.add_argument('--num-sends', type=int, default=128)
    parser.add_argument('--skip-perf-test', action='store_true', help='Whether to skip performance tests')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
