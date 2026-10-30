"""Explicit NPU correctness/performance test; --plan and --help need no Torch.

Uses the same node-count WORLD_SIZE convention as tests/ep/test_ep.py. Do not
wrap this mp.spawn launcher in torchrun. First-use DeepJIT compilation is untimed.
"""

import argparse
import itertools
import json
import os
import statistics
from datetime import timedelta

__test__ = False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--num-tokens', nargs='+', type=int, default=[16])
    p.add_argument('--num-max-tokens-per-rank', nargs='+', type=int,
                   help='Capacity matrix; defaults to each case B. M<=32 is direct, M>32 uses dedup.')
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-experts', type=int, default=128)
    p.add_argument('--dtype', nargs='+', choices=['bf16', 'fp16', 'fp8'], default=['bf16'])
    p.add_argument('--fp8-scale-dtype', nargs='+', choices=['int32', 'float32'], default=['int32'])
    p.add_argument('--do-cpu-sync', nargs='+', type=int, choices=[0, 1], default=[0, 1])
    p.add_argument('--route', nargs='+', choices=['random', 'hotspot'], default=['random', 'hotspot'])
    p.add_argument('--rounds', type=int, default=3, help='Reuse one buffer with changed routes and expert outputs')
    p.add_argument('--repeat-combine', type=int, default=3, help='Reuse each dispatch handle')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--num-iters', type=int, default=20)
    p.add_argument('--skip-perf-test', action='store_true')
    p.add_argument('--test-first-only', action='store_true')
    p.add_argument('--no-huge1g-page', action='store_true')
    p.add_argument('--plan', action='store_true')
    return p


def cases(args):
    for b, dtype, sync, route in itertools.product(args.num_tokens, args.dtype, args.do_cpu_sync, args.route):
        for capacity in args.num_max_tokens_per_rank or [b]:
            for scale in args.fp8_scale_dtype if dtype == 'fp8' else ['int32']:
                yield dict(b=b, m=capacity, dtype=dtype, sync=bool(sync), route=route, scale=scale)


def validate(args, world):
    if world not in (8, 16, 32, 64, 128, 256):
        raise ValueError('Low-latency tests require EP8/16/32/64/128/256')
    if not all(0 < case['b'] <= case['m'] <= 256 for case in cases(args)):
        raise ValueError('Require 0 < B <= M <= 256 on every rank')
    if not (4 <= args.hidden and 1 <= args.num_topk <= 32 and world <= args.num_experts <= 1024
            and args.num_experts % world == 0 and args.num_topk <= args.num_experts):
        raise ValueError('Require H>=4, K in [1,32], and E divisible by EP in [EP,1024]')
    if max(args.num_tokens) * world * min(args.num_topk, args.num_experts // world) > 10000:
        raise ValueError('Worst-case expanded receive capacity must not exceed 10000 rows')
    if min(args.rounds, args.repeat_combine, args.num_iters) < 1 or args.warmup < 0:
        raise ValueError('Rounds, repeats and iterations must be positive; warmup must be nonnegative')


def agree(action):
    """Propagate reference failures to every rank before the next collective."""
    import torch
    import torch.distributed as dist
    error = None
    try:
        action()
    except Exception as exc:
        error = exc
        print(f'[rank {dist.get_rank()}] {exc}', flush=True)
    status = torch.tensor([int(error is not None)], dtype=torch.int32, device='npu')
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    if status.item():
        raise AssertionError('Low-latency validation failed on at least one rank') from error


def verify_dispatch(recv, handle, entries, args, case, world):
    import torch
    from low_latency_reference import payload, fp8_scales
    counts = [len(rows) for rows in entries]
    total = sum(counts)
    capacity = world * case['b'] * min(args.num_topk, args.num_experts // world)
    values = recv[0] if isinstance(recv, tuple) else recv
    assert handle.is_low_latency, 'The test must execute the low-latency backend'
    assert values.shape == (total if case['sync'] else capacity, args.hidden)
    assert handle.num_expanded_tokens == values.shape[0]
    assert handle.expert_recv_counts.cpu().tolist() == counts
    assert handle.send_counts[-1].item() == total
    assert handle.num_recv_tokens_per_expert_list == (counts if case['sync'] else [])
    start = 0
    for rows in entries:
        block = values[start:start + len(rows)].float().cpu()
        if rows:
            ids = (block[:, 0].long() * 16 + block[:, 1].long()) * case['b']
            ids += block[:, 2].long() * 16 + block[:, 3].long()
            order = ids.argsort()
            sources, tokens = zip(*rows)
            torch.testing.assert_close(block[order], payload(sources, tokens, args.hidden, torch.float32),
                                       atol=0, rtol=0)
            if isinstance(recv, tuple):
                actual = recv[1][start:start + len(rows)].cpu()[order].contiguous()
                expected = fp8_scales(sources, tokens, args.hidden, case['scale'])
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        start += len(rows)


def measure(invoke, args):
    """NPU event latency of the public call, including its queued adapter work."""
    import torch
    import torch.distributed as dist
    for _ in range(args.warmup):
        result = invoke()
        torch.npu.synchronize()
    samples = []
    for _ in range(args.num_iters):
        dist.barrier()
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        result = invoke()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    latency = torch.tensor([statistics.mean(samples)], dtype=torch.float32, device='npu')
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return latency.item()


def run_case(buffer, args, case, rank, world):
    import torch
    import torch.distributed as dist
    from low_latency_reference import (make_routes, expert_entries, payload, fp8_scales,
                                       prepare_experts, expected_combine)
    expert_dtype = torch.float16 if case['dtype'] == 'fp16' else torch.bfloat16
    b = case['b']
    for round_idx in range(args.rounds):
        routes = make_routes(world, b, args.num_topk, args.num_experts, round_idx)
        for source in range(world):
            for token in range(b):
                if case['route'] == 'hotspot':
                    routes[source][token] = list(range(args.num_topk))
        entries = expert_entries(routes, args.num_experts, world, rank)
        counts = [len(rows) for rows in entries]
        values = payload([rank] * b, range(b), args.hidden, expert_dtype)
        if case['dtype'] == 'fp8':
            x = (values.to(torch.float8_e4m3fn).npu(),
                 fp8_scales([rank] * b, range(b), args.hidden, case['scale']).npu())
        else:
            x = values.npu()
        indices = torch.tensor(routes[rank], dtype=torch.int64, device='npu')
        weights = (torch.arange(1, args.num_topk + 1, dtype=torch.float32, device='npu') / 32).repeat(b, 1)

        def dispatch():
            return buffer.dispatch(x, topk_idx=indices, topk_weights=weights, num_experts=args.num_experts,
                                   num_max_tokens_per_rank=case['m'], do_expand=True, do_cpu_sync=case['sync'])

        recv, idx, recv_weights, handle, event = dispatch()
        event.wait()
        torch.npu.synchronize()
        assert idx is None and recv_weights is None
        agree(lambda: verify_dispatch(recv, handle, entries, args, case, world))
        # Non-unit dispatch weights are metadata: both an unweighted expert
        # output and an explicitly preweighted one must be reduced unchanged.
        for weighted in (False, True):
            expert_output = prepare_experts(recv, counts, rank, args.num_experts, world, expert_dtype,
                                             weighted=weighted, routes=routes)
            expected = expected_combine(routes, rank, args.hidden, expert_dtype,
                                        fp32_scales=case['dtype'] == 'fp8' and case['scale'] == 'float32',
                                        weighted=weighted)
            # Powers of two preserve BF16/FP16 rounding while detecting stale payloads.
            expert_output *= 2 ** round_idx
            expected *= 2 ** round_idx
            saved = expert_output.clone()
            # Alternate streams to exercise buffer serialization and tensor lifetime.
            producer = torch.npu.current_stream()
            other = torch.npu.Stream()
            other.wait_stream(producer)
            for repeat in range(args.repeat_combine):
                stream = producer if repeat % 2 == 0 else other
                with torch.npu.stream(stream):
                    output, output_weights, event = buffer.combine(expert_output, handle)
                event.wait()
                torch.npu.synchronize()
                assert output_weights is None
                agree(lambda: torch.testing.assert_close(output.float().cpu(), expected.float(),
                                                          atol=0.125, rtol=0.02))
                agree(lambda: torch.testing.assert_close(expert_output, saved, atol=0, rtol=0))
        if round_idx == args.rounds - 1 and not args.skip_perf_test:
            dispatch_us = measure(dispatch, args)
            combine_us = measure(lambda: buffer.combine(expert_output, handle), args)
            if rank == 0:
                print(json.dumps(dict(case, dispatch_api_us=dispatch_us, combine_api_us=combine_us)), flush=True)
        dist.barrier()
    if rank == 0:
        print(f'PASS EP{world} {case}', flush=True)


def worker(local_rank, args):
    import torch
    import torch_npu  # Register the NPU device before importing deep_ep.
    import torch.distributed as dist
    import deep_ep
    from deep_ep import _C
    torch.npu.set_device(local_rank)
    if torch.npu.device_count() < args.num_processes:
        raise RuntimeError('Not enough visible NPUs for --num-processes')
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    rank = int(os.getenv('RANK', '0')) * args.num_processes + local_rank
    dist.init_process_group('hccl', rank=rank, world_size=world,
                            init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8361')}",
                            timeout=timedelta(seconds=600))
    selected = list(cases(args))
    if args.test_first_only:
        selected = selected[:1]
    for case in selected:
        dtype = dict(bf16='bfloat16', fp16='float16', fp8='float8_e4m3fn')[case['dtype']]
        shape = (world, case['b'], args.hidden, args.num_topk, args.num_experts)
        if _C.low_latency_layout(*shape, dtype) is None:
            raise ValueError(f'Unsupported native dispatch layout: {case}')
        _C.low_latency_combine_tiling(*shape, 'float16' if dtype == 'float16' else 'bfloat16', case['m'])
    if rank == 0:
        print(f'Testing {deep_ep.__file__}; EP{world}, 8 GiB communication buffer per rank', flush=True)
    buffer = deep_ep.EPBuffer(dist.group.WORLD, num_max_tokens_per_rank=max(case['m'] for case in selected),
                              hidden=args.hidden, num_topk=args.num_topk, low_latency_mode=True,
                              explicitly_destroy=True, use_huge1g_page=not args.no_huge1g_page)
    # Do not attempt another collective after a device failure: mp.spawn stops peers.
    with torch.no_grad():
        for case in selected:
            run_case(buffer, args, case, rank, world)
    buffer.destroy()
    dist.destroy_process_group()


def main():
    args = parser().parse_args()
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    if args.num_processes <= 0 or not 0 <= int(os.getenv('RANK', '0')) < int(os.getenv('WORLD_SIZE', '1')):
        raise ValueError('Invalid node count, node rank or local process count')
    validate(args, world)
    if args.plan:
        selected = list(cases(args))
        print(json.dumps(dict(world_size=world, cases=selected[:1] if args.test_first_only else selected), indent=2))
        return
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)


if __name__ == '__main__':
    main()
