"""Check NPU metadata and HCCL use of externally allocated buffer views."""
import argparse

import torch
import torch.distributed as dist
import torch_npu

import deep_ep
from deep_ep.utils.envs import init_dist


def check_storage(tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
    ordinary = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    assert torch_npu.get_npu_format(tensor) == torch_npu.get_npu_format(ordinary)
    tensor.fill_(group.rank())
    gathered = torch.empty(tensor.numel() * group.size(), dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(gathered, tensor.reshape(-1), group=group)
    expected = torch.cat([torch.full_like(tensor.reshape(-1), rank) for rank in range(group.size())])
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int) -> None:
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    alignment = deep_ep.get_num_allocation_alignment()
    try:
        buffer = deep_ep.BucketBuffer(group, alignment, explicitly_destroy=True, use_huge1g_page=False)
        try:
            check_storage(buffer.storage[:32], group)
        finally:
            buffer.destroy()

        buffer = deep_ep.EPBuffer(
            group, num_bytes=alignment, lb_allocation_plan_or_num_bytes=alignment,
            explicitly_destroy=True, use_huge1g_page=False)
        try:
            check_storage(buffer.runtime.lb_storage[:32], group)
        finally:
            buffer.destroy()

        buffer = deep_ep.PPBuffer(
            group, num_max_tensor_bytes=32, num_max_inflight_tensors=1,
            explicitly_destroy=True, use_huge1g_page=False)
        try:
            check_storage(buffer.storage[:32], group)
        finally:
            buffer.destroy()

        buffer = deep_ep.EngramBuffer(
            group, num_gpu_bytes=alignment, num_rdma_storage_bytes=alignment,
            use_cpu_rdma_storage=False, num_allocated_qps=0, qp_depth=0,
            explicitly_destroy=True, use_huge1g_page=False)
        try:
            buffer.set_config([2], hidden=16, num_max_tokens=1, num_entries_per_token=1)
            buffer.write([torch.zeros((2, 16), dtype=torch.bfloat16, device='npu')])
            indices = torch.full((1, 1, 1), rank_idx * 2, dtype=torch.int32, device='npu')
            check_storage(buffer.fetch(indices)[0](), group)
        finally:
            buffer.destroy()
    finally:
        deep_ep.comm.destroy_all_managed_hccl_comm()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NPU format metadata for registered buffer views')
    parser.add_argument('--num-processes', type=int, default=2)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes,), nprocs=args.num_processes)
