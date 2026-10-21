# Filter and disable all annoying warnings
import os
os.environ['TORCH_NPU_DISABLED_WARNING'] = '1'

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.math import ceil_div, per_token_cast_to_fp8
from deep_ep.utils.testing import bench_msprof


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    dtype = torch.float8_e4m3fn if args.use_fp8 else torch.bfloat16
    num_entries_per_layer = [int(x) for x in args.num_entries_per_layer.split(',')]
    num_layers = len(num_entries_per_layer)
    num_sf_packs = 2 * ceil_div(args.hidden, 128) if args.use_fp8 else 0
    num_bytes, num_rdma_storage_bytes = deep_ep.EngramBuffer.get_storage_size_hint(
        group, num_entries_per_layer, args.hidden,
        args.num_tokens, args.num_entries_per_token,
        dtype, num_sf_packs)

    # Allocate buffer
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Entries per rank: {num_entries_per_layer}, hidden: {args.hidden}\n'
               f' > Tokens to fetch: {args.num_tokens} x {args.num_entries_per_token} entries '
               f'x {num_layers} layers\n'
               f' > Storage per rank: '
               f'{sum(num_entries_per_layer) * args.hidden * dtype.itemsize / 1024 / 1024:.1f} MB\n',
               once_in_node=True)
    num_allocated_qps, qp_depth, restrict_rd_atomic = deep_ep.EngramBuffer.get_theoretical_config(
        group, num_layers, args.num_tokens, args.num_entries_per_token)
    buffer = deep_ep.EngramBuffer(
        group, num_gpu_bytes=num_bytes, num_rdma_storage_bytes=num_rdma_storage_bytes,
        use_cpu_rdma_storage=False,
        num_allocated_qps=num_allocated_qps, qp_depth=qp_depth,
        restrict_rd_atomic=restrict_rd_atomic, explicitly_destroy=True)
    buffer.set_config(
        num_entries_per_layer, args.hidden,
        args.num_tokens, args.num_entries_per_token, dtype, num_sf_packs)

    # Write buffer
    local_storages, sfs = [], [] if args.use_fp8 else None
    for num_entries in num_entries_per_layer:
        local_bf16 = torch.randn(
            (num_entries, args.hidden), dtype=torch.bfloat16, device='npu')
        if args.use_fp8:
            local_storage, local_sf = per_token_cast_to_fp8(local_bf16)
            local_sf = local_sf.view(torch.int16)
            sf = torch.empty(
                (num_ranks * num_entries, num_sf_packs),
                dtype=local_sf.dtype, device='npu')
            dist.all_gather_into_tensor(sf, local_sf, group=group)
            sfs.append(sf)
        else:
            local_storage = local_bf16
        local_storages.append(local_storage)
    write_storages = [storage.cpu() for storage in local_storages] if args.cpu_staging else local_storages
    buffer.write(write_storages, sfs=sfs)
    if args.cpu_staging:
        # write() owns a completed byte copy, not a reference to caller staging.
        # Mutate the source immediately; fetch must still match the NPU reference.
        for storage in write_storages:
            storage.view(torch.uint8).zero_()

    # Generate random indices with a per-layer global entry range
    def generate_indices():
        return torch.stack([
            torch.randint(
                0, num_ranks * num_entries,
                (args.num_tokens, args.num_entries_per_token),
                dtype=torch.int, device='npu')
            for num_entries in num_entries_per_layer
        ])

    # Correctness check
    if not args.skip_check:
        global_storage_bits = []
        for local_storage in local_storages:
            global_bits = torch.empty(
                (num_ranks * local_storage.shape[0], args.hidden * dtype.itemsize),
                dtype=torch.uint8, device='npu')
            dist.all_gather_into_tensor(
                global_bits, local_storage.view(torch.uint8), group=group)
            global_storage_bits.append(global_bits)

        for use_tma_aligned_col_major_sf in (False, True) if args.use_fp8 else (False,):
            for layer_order in (range(num_layers), reversed(range(num_layers))):
                indices = generate_indices()
                ref_data_bits = [
                    global_storage_bits[layer_idx][indices[layer_idx].view(-1)].view(args.num_tokens, -1)
                    for layer_idx in range(num_layers)
                ]
                ref_sfs = [
                    sfs[layer_idx][indices[layer_idx].view(-1)].view(args.num_tokens, -1)
                    for layer_idx in range(num_layers)
                ] if args.use_fp8 else None

                hooks = buffer.fetch(
                    indices, num_qps=args.num_qps,
                    use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf)
                for layer_idx in layer_order:
                    fetched = hooks[layer_idx]()
                    data, fetched_sf = fetched if args.use_fp8 else (fetched, None)
                    assert torch.equal(ref_data_bits[layer_idx], data.view(torch.uint8)), \
                        f'data mismatch ({layer_idx=})'
                    if args.use_fp8:
                        assert torch.equal(ref_sfs[layer_idx], fetched_sf), \
                            f'fp8 scaling-factor mismatch ({layer_idx=})'

    if not args.skip_performance:
        # Performance test
        dist_print('Running performance test ...', once_in_node=True)
        indices = generate_indices()

        def fetch_and_wait():
            hooks = buffer.fetch(
                indices, num_qps=args.num_qps,
                use_tma_aligned_col_major_sf=True)
            for hook in hooks:
                hook()

        profile = bench_msprof(
            fetch_and_wait,
            kernel_names='engram_fetch_impl',
            num_warmups=10, num_tests=50, flush_l2=True,
            barrier_comm_profiling=True, barrier=buffer.barrier)
        num_fetched_bytes = num_layers * args.num_tokens * args.num_entries_per_token * args.hidden * dtype.itemsize
        dist_print(
            f'Rank {rank_idx}: {profile.us:.1f} us, '
            f'{profile.gbps(num_fetched_bytes):.2f} GB/s, ')

        dist_print(once_in_node=True)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Engram fetch over UBMEM')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn')
    parser.add_argument('--num-qps', type=int, default=0, help='Must be 0 for Ascend automatic backend configuration')
    parser.add_argument('--num-entries-per-layer', type=str, default='524288,524309',
                        help='Comma-separated number of entries per rank for each layer')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num-tokens', type=int, default=512, help='Number of tokens to fetch')
    parser.add_argument('--num-entries-per-token', type=int, default=24,
                        help='Number of entries concatenated per token')
    parser.add_argument('--skip-check', action='store_true', help='Skip correctness check')
    parser.add_argument('--use-fp8', action='store_true', help='Store entries in FP8 with scaling factors')
    parser.add_argument('--cpu-staging', action='store_true', help='Stage write inputs on CPU; runtime tables remain on NPU')
    parser.add_argument('--skip-performance', action='store_true', help='Run correctness checks without the profiler benchmark')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
