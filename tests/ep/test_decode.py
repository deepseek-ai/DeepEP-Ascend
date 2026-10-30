"""NPU correctness test of the native EP decode wrapper across batch changes.

Same node-count WORLD_SIZE / node-index RANK convention as test_ep.py.
Run on every participating node; do not wrap this mp.spawn launcher in torchrun.
"""

import argparse
from datetime import timedelta
import os

__test__ = False


def worker(local_rank, args):
    import torch
    import torch_npu
    import torch.distributed as dist
    import deep_ep
    from low_latency_reference import make_routes, expert_entries, payload, prepare_experts, expected_combine
    from test_low_latency import agree

    torch.npu.set_device(local_rank)
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    rank = int(os.getenv('RANK', '0')) * args.num_processes + local_rank
    dist.init_process_group('hccl', rank=rank, world_size=world,
                            device_id=torch.device(f'npu:{local_rank}'),
                            init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8361')}",
                            timeout=timedelta(seconds=600))
    buffer = deep_ep.DecodeBuffer(dist.group.WORLD, max(args.bs), use_huge1g_page=not args.no_huge1g_page)
    retained = None
    with torch.inference_mode():
        for sync in (False, True):
            for step, bs in enumerate(args.bs):
                routes = make_routes(world, bs, 8, 128, step)
                # Include concentrated routes: selection must remain correct
                # even though the performance thresholds came from balanced load.
                if sync:
                    routes = [[list(range(8)) for _ in range(bs)] for _ in range(world)]
                entries = expert_entries(routes, 128, world, rank)
                counts = [len(rows) for rows in entries]
                x = payload([rank] * bs, range(bs), 7168, torch.bfloat16).npu()
                indices = torch.tensor(routes[rank], dtype=torch.int64, device='npu')
                weights = (torch.arange(1, 9, dtype=torch.float32, device='npu') / 32).repeat(bs, 1)
                recv, handle, event = buffer.dispatch(x, indices, weights, do_cpu_sync=sync)
                event.wait()
                torch.npu.synchronize()

                def check_dispatch():
                    assert handle.backend == 'ep'
                    assert not handle._handle.is_low_latency
                    assert handle.recv_topk_weights is not None
                    assert handle.expert_recv_counts.cpu().tolist() == counts
                    assert handle.num_recv_tokens_per_expert_list == (counts if sync else [])
                    start = 0
                    for rows in entries:
                        if rows:
                            block = recv[start:start + len(rows)].float().cpu()
                            ids = (block[:, 0].long() * 16 + block[:, 1].long()) * bs
                            ids += block[:, 2].long() * 16 + block[:, 3].long()
                            sources, tokens = zip(*rows)
                            torch.testing.assert_close(block[ids.argsort()], payload(sources, tokens, 7168, torch.float32),
                                                       atol=0, rtol=0)
                        start += len(rows)
                agree(check_dispatch)
                for weighted in (False, True):
                    expert = prepare_experts(recv, counts, rank, 128, world, torch.bfloat16,
                                             weighted=weighted, routes=routes)
                    expected = expected_combine(routes, rank, 7168, torch.bfloat16, weighted=weighted,
                                                reduce_in_local=True, experts=128)
                    saved = expert.clone()
                    producer = torch.npu.current_stream()
                    other = torch.npu.Stream()
                    other.wait_stream(producer)
                    for repeat in range(2):
                        with torch.npu.stream(producer if repeat == 0 else other):
                            output, event = buffer.combine(expert, handle)
                        event.wait()
                        torch.npu.synchronize()
                        agree(lambda: torch.testing.assert_close(output.float().cpu(), expected.float(), atol=.125, rtol=.02))
                        agree(lambda: torch.testing.assert_close(expert, saved, atol=0, rtol=0))
                if retained is None:
                    retained = expert, handle, expected
                else:
                    # The single native arena must preserve handles across
                    # batch changes and internal transport decisions.
                    output, event = buffer.combine(retained[0], retained[1])
                    event.wait()
                    torch.npu.synchronize()
                    agree(lambda: torch.testing.assert_close(output.float().cpu(), retained[2].float(), atol=.125, rtol=.02))
                if rank == 0:
                    print(f'PASS EP{world} BS={bs} sync={sync} backend={handle.backend}', flush=True)
        buffer.destroy()
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--bs', nargs='+', type=int, default=[2, 32, 64, 128, 256, 2])
    parser.add_argument('--no-huge1g-page', action='store_true')
    args = parser.parse_args()
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    if world not in (8, 16, 32, 64, 128) or min(args.bs) <= 0:
        parser.error('This test requires EP8/16/32/64/128 and positive BS')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)
