import warnings
warnings.filterwarnings('ignore', message='Permission mismatch')

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.testing import bench_msprof


def test_barrier(buffer: deep_ep.BucketBuffer, args: argparse.Namespace):
    dist_print('Profiling barrier:', once_in_node=True)
    num_scaleout_ranks, num_scaleup_ranks = buffer.main_context.get_logical_domain_size()
    dist_print(f'Config:\n'
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n',
               once_in_node=True)

    # Profile barrier kernel time
    def loop_barrier(num_tests=1000):
        for i in range(num_tests):
            buffer.barrier()

    profile = bench_msprof(
        fn=lambda: loop_barrier(),
        kernel_names='barrier',
        num_warmups=10,
        num_tests=50,
        flush_l2=False,
        barrier_comm_profiling=True,
        barrier=buffer.barrier,
    )
    group = buffer.groups[0]
    dist_print('Perf:', once_in_node=True)
    dist_print(f' > Rank: {group.rank():3}/{group.size():3}, '
               f'barrier time: {profile.us:.3f} us')
    dist_print(once_in_node=True)


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    for i in range(int(1e9) if args.do_pressure_test else 1):
        buffer = deep_ep.BucketBuffer(
            group, deep_ep.get_num_allocation_alignment(),
            explicitly_destroy=True,
        )

        # Test barrier
        test_barrier(buffer, args)

        # Destroy the runtime and communication group
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test barrier performance')

    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')
    args = parser.parse_args()

    # Launch test processes
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
