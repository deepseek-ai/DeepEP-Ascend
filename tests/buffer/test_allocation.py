import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2 and num_ranks % 2 == 0
    subgroup = None
    for rank_begin_idx in range(0, num_ranks, 2):
        item = dist.new_group(list(range(rank_begin_idx, rank_begin_idx + 2)))
        if rank_begin_idx <= rank_idx < rank_begin_idx + 2:
            subgroup = item

    # The same storage is registered with both groups.
    plan = deep_ep.BufferAllocator()
    gathered = plan.allocate((num_ranks, 2, 8), torch.float32)
    scratch = plan.allocate((7,), torch.bfloat16)
    assert gathered.is_meta and scratch.is_meta
    buffer = deep_ep.BucketBuffer([group, subgroup], plan, explicitly_destroy=True)
    assert gathered.device.type == 'npu' and scratch.device.type == 'npu'
    assert gathered.data_ptr() == buffer.storage.data_ptr()
    assert len(buffer.contexts) == 2
    assert buffer.storage.numel() == plan.num_bytes

    gathered.fill_(rank_idx)
    scratch.fill_(rank_idx + 1)
    assert (gathered == rank_idx).all()
    assert (scratch == rank_idx + 1).all()
    buffer.destroy()

    buffer = deep_ep.BucketBuffer(group, plan.num_bytes, explicitly_destroy=True)
    assert buffer.storage.device.type == 'npu' and buffer.storage.dtype == torch.uint8
    assert buffer.storage.numel() == plan.num_bytes
    buffer.destroy()

    # LB storage shares the EP allocation; its communication kernels remain unsupported.
    plan = deep_ep.BufferAllocator()
    weights = plan.allocate((4, 8), torch.float32)
    buffer = deep_ep.EPBuffer(group, num_bytes=deep_ep.get_num_allocation_alignment(),
                              lb_allocation_plan_or_num_bytes=plan, explicitly_destroy=True)
    assert weights.data_ptr() == buffer.runtime.lb_storage.data_ptr()
    weights.fill_(rank_idx)
    assert (weights == rank_idx).all()
    buffer.destroy()
    dist_print('Buffer allocation and group registration passed', once_in_node=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test buffer allocation and group registration')
    parser.add_argument('--num-processes', type=int, default=4)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes,), nprocs=args.num_processes)
