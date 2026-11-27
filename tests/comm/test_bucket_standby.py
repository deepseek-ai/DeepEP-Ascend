"""Fault-free stand-by takeover for BucketBuffer.

Three nodes with ``--num-processes`` ranks each:

    node1 + node2   old active group
    node0 + node2   new active group
    node0           standby, allocated without communication groups

With a [2, 1] placement, node0 and node1 share a physical SU, so replacement
preserves the topology seen by node2.

Flow: baseline -> standby allocation -> invalidate -> reconfigure -> verify -> destroy.
Nothing exits early; the retired group stays alive but stops participating.
"""

import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing

# URMA collectives refuse to run without this flag.
os.environ.setdefault('EP_AVOID_RECORD_STREAM', '1')

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import parse_num_bytes


NUM_NODES = 3
# Eight float32 values per rank are enough to tell senders apart.
NUM_VALUES = 8


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fault-free stand-by takeover for BucketBuffer')
    parser.add_argument('--num-processes', type=int, default=8, help='processes per node')
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=8 << 20,
                        help='payload bytes for the buffer; supports binary suffixes such as 8M')
    parser.add_argument('--rounds', type=int, default=3, help='all-gather rounds after takeover')
    parser.add_argument('--timeout', type=int, default=300, help='CPU and HCCL timeout in seconds')
    return parser


def init_groups(local_rank: int, num_local_ranks: int, timeout_secs: int):
    """Initialize old and new communication groups in a stable order."""
    rank_idx, num_ranks, topo = init_dist(local_rank, num_local_ranks)
    num_nodes = num_ranks // num_local_ranks
    assert num_nodes == NUM_NODES, f'this test expects {NUM_NODES} nodes, got {num_nodes}'

    timeout = datetime.timedelta(seconds=timeout_secs)
    node_ranks = [tuple(range(idx * num_local_ranks, (idx + 1) * num_local_ranks)) for idx in range(NUM_NODES)]
    active = {'old': node_ranks[1] + node_ranks[2], 'new': node_ranks[0] + node_ranks[2]}
    # First group: global.
    old_global = dist.new_group(list(active['old']), timeout=timeout)
    new_global = dist.new_group(list(active['new']), timeout=timeout)
    groups = {
        'old': [old_global if rank_idx in active['old'] else None],
        'new': [new_global if rank_idx in active['new'] else None],
    }

    # Second group: rail.
    old_rail = new_rail = None
    for rail_idx in range(num_local_ranks):
        old_members = [num_local_ranks + rail_idx, 2 * num_local_ranks + rail_idx]
        new_members = [rail_idx, 2 * num_local_ranks + rail_idx]
        old_group = dist.new_group(old_members, timeout=timeout)
        new_group = dist.new_group(new_members, timeout=timeout)
        if rail_idx == local_rank:
            old_rail = old_group if rank_idx in old_members else None
            new_rail = new_group if rank_idx in new_members else None
    groups['old'].append(old_rail)
    groups['new'].append(new_rail)
    return rank_idx, num_ranks, topo, groups, active


def make_plan(context_sizes: list[int], num_bytes: int):
    """Allocate one gathered tensor per context plus payload storage."""
    plan = deep_ep.BufferAllocator()
    gathered = [plan.allocate((size, NUM_VALUES), torch.float32) for size in context_sizes]
    plan.allocate((num_bytes // 4,), torch.float32)
    return plan, gathered


def check_round(buffer: deep_ep.BucketBuffer, group: dist.ProcessGroup, gathered: torch.Tensor,
                round_idx: int) -> None:
    """All-gather one value per rank and check that every shard came from the expected rank."""
    rank_idx, num_ranks = group.rank(), group.size()
    gathered.fill_(float('nan'))
    src = torch.full((NUM_VALUES,), rank_idx + 1000.0 * round_idx, dtype=torch.float32, device='npu')
    buffer.all_gather(src, dsts=gathered, group=group, num_sms=0).wait()

    # init_dist sets the default device to NPU, so pin the reference to CPU explicitly.
    expected = torch.arange(num_ranks, dtype=torch.float32, device='cpu') + 1000.0 * round_idx
    actual = gathered[:, 0].cpu()
    assert torch.equal(actual, expected), \
        f'round {round_idx}: rank {rank_idx} gathered {actual.tolist()} but expected {expected.tolist()}'


def check_domains(buffer: deep_ep.BucketBuffer, expected: list[tuple[int, int]]) -> None:
    actual = [tuple(context.get_physical_domain_size()) for context in buffer.contexts]
    assert actual == expected, f'expected domains {expected}, got {actual}'


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, topo, groups, active = init_groups(local_rank, num_local_ranks, args.timeout)
    old, new = groups['old'], groups['new']
    context_sizes = [2 * num_local_ranks, 2]

    plan, gathered = make_plan(context_sizes, args.num_bytes)
    is_old = rank_idx in active['old']
    is_new = rank_idx in active['new']
    is_standby = is_new and not is_old

    def resolve_expected_domains(groups_for_buffer) -> list[tuple[int, int]]:
        # Node boundaries need not coincide with logical scale-up domains.
        return [tuple(deep_ep.comm.get_physical_domain_size(group)) for group in groups_for_buffer]

    dist_print(f'[topology] world={num_ranks} ranks_per_node={num_local_ranks} old={list(active["old"])} '
               f'new={list(active["new"])} contexts=2', once_in_node=True)

    # 1) Baseline on the old active group.
    if is_old:
        buffer = deep_ep.BucketBuffer(old, plan, num_gpu_timeout_secs=args.timeout, explicitly_destroy=True)
        assert buffer.active is True
        for item, dst in zip(old, gathered, strict=True):
            buffer.barrier(group=item, with_cpu_sync=True)
            check_round(buffer, item, dst, 0)
    dist_print('[phase] baseline passed', once_in_node=True)

    # 2) The standby allocates storage only.
    allocation_ptr = None
    if is_standby:
        buffer = deep_ep.BucketBuffer(None, plan, num_gpu_timeout_secs=args.timeout, explicitly_destroy=True)
        assert not buffer.active and not buffer.contexts and buffer.main_context is None
        gathered[0].fill_(17)
        allocation_ptr = buffer.storage.data_ptr()
    dist_print('[phase] standby prepared', once_in_node=True)

    # 3) Survivors drop the retired group. Non-collective: the retired group does not join.
    if is_old:
        start = time.perf_counter()
        buffer.invalidate(num_timeout_secs=args.timeout)
        invalidate_secs = time.perf_counter() - start
        assert buffer.active is False, 'invalidate must leave the buffer inactive'
        print(f'[time] rank={rank_idx} invalidate={invalidate_secs:.3f}s', flush=True)
    dist_print('[phase] invalidate passed', once_in_node=True)

    # 4) Survivors and the standby configure the new group together.
    if is_new:
        pointer_before = buffer.storage.data_ptr()
        start = time.perf_counter()
        buffer.reconfigure(new)
        reconfigure_secs = time.perf_counter() - start
        assert buffer.active is True, 'reconfigure must leave the buffer active'
        check_domains(buffer, resolve_expected_domains(new))
        assert buffer.storage.data_ptr() == pointer_before, 'reconfigure replaced the allocation'
        if allocation_ptr is not None:
            assert allocation_ptr == pointer_before, 'the standby lost its prepared allocation'
            assert torch.all(gathered[0] == 17).item(), 'configuration overwrote the prepared payload'
        print(f'[time] rank={rank_idx} reconfigure={reconfigure_secs:.3f}s', flush=True)
    dist_print('[phase] reconfigure passed', once_in_node=True)

    # 5) The new group must work without the retired SU.
    if is_new:
        for item, dst in zip(new, gathered, strict=True):
            buffer.barrier(group=item, with_cpu_sync=True)
            for round_idx in range(1, args.rounds + 1):
                check_round(buffer, item, dst, round_idx)
    dist_print(f'[phase] takeover verified over {args.rounds} rounds', once_in_node=True)

    # 6) Only the new active group can retire collectively; the retired SU exits as-is.
    if is_new:
        buffer.destroy()
    dist_print('[phase] cleanup passed', once_in_node=True)

    for name in ('new', 'old'):
        for group in groups[name]:
            if group is not None and group is not topo:
                dist.destroy_process_group(group)
    dist_print('PASS', once_in_node=True)
    dist.destroy_process_group()


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_processes <= 0 or args.rounds <= 0 or args.timeout <= 0:
        parser.error('--num-processes, --rounds and --timeout must be positive')
    if args.num_bytes <= 0 or args.num_bytes % 4:
        parser.error('--num-bytes must be positive and divisible by 4')


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
