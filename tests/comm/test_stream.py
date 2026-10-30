import argparse
import os

import torch
import torch.distributed as dist
import torch_npu  # Register the NPU backend even when backend autoload is disabled.

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


def test_events(producer, consumer, comm_stream):
    source = torch.zeros(4096, dtype=torch.float32, device='npu')
    torch.npu.synchronize()
    with torch.npu.stream(producer):
        torch.npu._sleep(100_000_000)
        source.fill_(3)
        ready = deep_ep.EventHandle()
    with torch.npu.stream(comm_stream):
        ready.current_stream_wait()
        intermediate = source * 2
        event = deep_ep.EventOverlap(deep_ep.EventHandle())

    calls = []

    def hook():
        calls.append(torch.npu.current_stream())
        return intermediate + 1

    event.register_hook_after_wait(hook)
    with torch.npu.stream(consumer):
        result = event.wait()
        assert event.wait() is None
    consumer.synchronize()
    assert calls == [consumer]
    assert torch.equal(result, torch.full_like(result, 7))


def test_bucket(group, producer, consumer):
    rank_idx, num_ranks = group.rank(), group.size()
    plan = deep_ep.BufferAllocator()
    dst = plan.allocate((num_ranks, 4096), torch.float32)
    buffer = deep_ep.BucketBuffer(group, plan, explicitly_destroy=True)

    # Warm the JIT before inserting delayed writes on a different current stream.
    for iteration in range(3):
        with torch.npu.stream(producer):
            dst.fill_(float('nan'))
            src = torch.empty(4096, dtype=torch.float32, device='npu')
            if iteration:
                torch.npu._sleep(100_000_000)
            src.fill_(rank_idx + 10 * iteration)
            event = buffer.all_gather(src, dsts=dst)
            assert torch.npu.current_stream() == producer
        del src

        with torch.npu.stream(consumer):
            result = event.wait().clone()
            assert torch.npu.current_stream() == consumer
            assert event.wait() is None
        # Keep the handle's source references until the asynchronous drain completes.
        consumer.synchronize()
        expected = torch.arange(num_ranks, dtype=torch.float32, device='npu') + 10 * iteration
        assert torch.equal(result.view(num_ranks, -1), expected[:, None].expand_as(dst))

    buffer.destroy()


def test_ep(group, stream):
    rank_idx, num_ranks = group.rank(), group.size()
    num_tokens, hidden = 16, 1024
    buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=num_tokens, hidden=hidden,
                              num_topk=1, explicitly_destroy=True)

    for defer_epilogue in (False, True):
        for iteration in range(2):
            with torch.npu.stream(stream):
                x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
                topk_idx = torch.full((num_tokens, 1), (rank_idx + 1) % num_ranks,
                                      dtype=torch.int64, device='npu')
                if iteration:
                    torch.npu._sleep(100_000_000)
                x.fill_(rank_idx + 10 * iteration)
                result = buffer.dispatch(
                    x, topk_idx=topk_idx, num_experts=num_ranks, expert_alignment=1,
                    do_cpu_sync=False, do_handle_copy=False, do_expand=True,
                    async_with_compute_stream=True, defer_epilogue=defer_epilogue)
                assert torch.npu.current_stream() == stream
                if not defer_epilogue:
                    recv_x, _, _, handle, event = result
                    # Current-stream outputs must be usable without a cross-stream wait.
                    snapshot = recv_x[:num_tokens].clone()
                else:
                    event = result
            del x

            with torch.npu.stream(stream):
                # Reuse pressure between issue and wait must not overwrite captured inputs.
                pressure = torch.full((num_tokens, hidden), -100, dtype=torch.bfloat16, device='npu')
                if defer_epilogue:
                    recv_x, _, _, handle = event.wait()
                    snapshot = recv_x[:num_tokens].clone()
                else:
                    event.wait()
                assert torch.npu.current_stream() == stream
            stream.synchronize()
            assert handle.psum_num_recv_tokens_per_rank[-1].item() == num_tokens
            expected = (rank_idx - 1) % num_ranks + 10 * iteration
            assert torch.equal(snapshot, torch.full_like(snapshot, expected))

    buffer.destroy()


def mte_routes(rank_idx, num_ranks, num_tokens):
    # Eight different destination ranks also imply eight valid, distinct experts.
    # Vary the local expert with the source token so each rank exercises its layout.
    num_local_experts = 128 // num_ranks
    return [[((rank_idx + token + k) % num_ranks) * num_local_experts
             + ((rank_idx + token) // num_ranks) % num_local_experts
             for k in range(8)] for token in range(num_tokens)]


def wait_ep_result(result, defer_epilogue, stream, num_outputs):
    if defer_epilogue:
        event = result
        outputs = event.wait()
    else:
        assert isinstance(result, tuple) and len(result) == num_outputs + 1
        *outputs, event = result
        assert event.wait() is None
    assert len(outputs) == num_outputs
    assert event.wait() is None, 'The deferred epilogue must only run once'
    assert torch.npu.current_stream() == stream
    return outputs


def allocation_pressure(shape, dtype):
    # Keep both allocations alive through the wait; use the input's allocation
    # stream so freeing a captured input too early can result in visible reuse.
    return [torch.full(shape, -100 - i, dtype=dtype, device='npu') for i in range(2)]


def check_mte_dispatch(recv_x, recv_idx, recv_weights, handle,
                       rank_idx, num_ranks, num_tokens, capacity, iteration):
    assert recv_idx is None
    assert recv_x.dtype == torch.bfloat16 and recv_x.shape[1] == 7168
    assert recv_weights is not None and recv_weights.dtype == torch.float32
    assert recv_weights.shape == (recv_x.shape[0],)
    assert torch.equal(handle.topk_idx.cpu(),
                       torch.tensor(mte_routes(rank_idx, num_ranks, num_tokens),
                                    dtype=torch.int64, device='cpu'))

    expected_sources = {}
    expected_rank_counts = [0] * num_ranks
    expected_expert_counts = [0] * (128 // num_ranks)
    for source_rank in range(num_ranks):
        for token, experts in enumerate(mte_routes(source_rank, num_ranks, num_tokens)):
            for k, expert in enumerate(experts):
                if expert // (128 // num_ranks) == rank_idx:
                    global_idx = source_rank * capacity + token
                    expected_sources[global_idx] = (source_rank, token, k)
                    expected_rank_counts[source_rank] += 1
                    expected_expert_counts[expert % (128 // num_ranks)] += 1

    expected_count = 8 * num_tokens
    assert len(expected_sources) == expected_count
    assert torch.equal(handle.psum_num_recv_tokens_per_rank.cpu(),
                       torch.tensor(expected_rank_counts, device='cpu').cumsum(0))
    assert torch.equal(handle.num_unaligned_recv_tokens_per_expert.cpu(),
                       torch.tensor(expected_expert_counts, device='cpu'))
    assert torch.equal(handle.psum_num_recv_tokens_per_expert.cpu(),
                       torch.tensor(expected_expert_counts, device='cpu').cumsum(0))

    # do_cpu_sync=False permits padded output/metadata allocations. Only the
    # prefix sums and metadata identify live rows; padding is not test data.
    metadata = handle.recv_src_metadata[:expected_count].cpu().tolist()
    expected_x = torch.empty((expected_count, 7168), dtype=torch.bfloat16, device='cpu')
    expected_weights = torch.empty(expected_count, dtype=torch.float32, device='cpu')
    seen_sources, seen_slots = set(), set()
    for row in metadata:
        slots, global_idx = row[:8], row[-2]
        assert global_idx in expected_sources and global_idx not in seen_sources
        source_rank, token, k = expected_sources[global_idx]
        assert [i for i, slot in enumerate(slots) if slot >= 0] == [k]
        slot = slots[k]
        assert 0 <= slot < expected_count and slot not in seen_slots
        expected_x[slot].fill_(source_rank + 10 * iteration)
        expected_x[slot, 0] = token
        expected_weights[slot] = k + 1
        seen_sources.add(global_idx)
        seen_slots.add(slot)
    assert seen_sources == set(expected_sources)
    assert seen_slots == set(range(expected_count))
    assert torch.equal(recv_x[:expected_count].cpu(), expected_x)
    assert torch.equal(recv_weights[:expected_count].cpu(), expected_weights)


def test_ep_mte(group, producer, consumer, num_tokens, capacity):
    rank_idx, num_ranks = group.rank(), group.size()
    hidden, num_topk, num_experts = 7168, 8, 128
    buffer = deep_ep.EPBuffer(
        group, num_max_tokens_per_rank=capacity, hidden=hidden,
        num_topk=num_topk, explicitly_destroy=True)

    for defer_epilogue in (False, True):
        # First iteration warms this mode's kernels before delayed production.
        for iteration in range(2):
            with torch.npu.stream(producer):
                x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
                topk_idx = torch.tensor(mte_routes(rank_idx, num_ranks, num_tokens),
                                        dtype=torch.int64, device='npu')
                topk_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='npu')
                if iteration:
                    torch.npu._sleep(100_000_000)
                x.fill_(rank_idx + 10 * iteration)
                x[:, 0] = torch.arange(num_tokens, dtype=torch.float32, device='npu')
                topk_weights.copy_(torch.arange(1, num_topk + 1, dtype=torch.float32, device='npu'))
                dispatch_result = buffer.dispatch(
                    x, topk_idx=topk_idx, topk_weights=topk_weights,
                    num_experts=num_experts, expert_alignment=1,
                    do_cpu_sync=False, do_handle_copy=False, do_expand=True,
                    async_with_compute_stream=True, defer_epilogue=defer_epilogue)
                assert torch.npu.current_stream() == producer
                del x, topk_idx, topk_weights
                dispatch_pressure = allocation_pressure((num_tokens, hidden), torch.bfloat16)
                weight_pressure = allocation_pressure((num_tokens, num_topk), torch.float32)

            with torch.npu.stream(consumer):
                # Both inputs and issue were on producer. Explicitly wait before
                # consuming output on another stream; previous_event is ignored
                # by the official baseline. Deferred epilogues run on this stream.
                recv_x, recv_idx, recv_weights, handle = wait_ep_result(
                    dispatch_result, defer_epilogue, consumer, 4)
                consumer.synchronize()
                check_mte_dispatch(recv_x, recv_idx, recv_weights, handle,
                                   rank_idx, num_ranks, num_tokens, capacity, iteration)
                del dispatch_result

                # Use compact, independent expert outputs so early deletion does
                # not retain the combine input via the dispatch return tuple.
                num_recv_tokens = num_topk * num_tokens
                expert_x = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='npu')
                expert_weights = torch.empty(num_recv_tokens, dtype=torch.float32, device='npu')
                if iteration:
                    torch.npu._sleep(100_000_000)
                expert_x.copy_(recv_x[:num_recv_tokens])
                expert_weights.copy_(recv_weights[:num_recv_tokens])
                combine_result = buffer.combine(
                    expert_x, handle=handle, topk_weights=expert_weights,
                    async_with_compute_stream=True, defer_epilogue=defer_epilogue)
                assert torch.npu.current_stream() == consumer
                del expert_x, expert_weights
                combine_pressure = allocation_pressure((num_recv_tokens, hidden), torch.bfloat16)
                combine_weight_pressure = allocation_pressure((num_recv_tokens,), torch.float32)

            with torch.npu.stream(producer):
                # Reverse the stream hand-off for combine, including its deferred
                # reduce epilogue. A second wait must not repeat the epilogue.
                combined_x, combined_weights = wait_ep_result(
                    combine_result, defer_epilogue, producer, 2)
                snapshot = combined_x.clone()
                weight_snapshot = combined_weights.clone()
            producer.synchronize()
            consumer.synchronize()
            expected = torch.full((num_tokens, hidden), (rank_idx + 10 * iteration) * num_topk,
                                  dtype=torch.bfloat16, device='cpu')
            expected[:, 0] = torch.arange(num_tokens, device='cpu') * num_topk
            expected_weights = torch.arange(1, num_topk + 1, dtype=torch.float32, device='cpu')
            assert snapshot.shape == expected.shape and snapshot.dtype == expected.dtype
            assert weight_snapshot.shape == (num_tokens, num_topk)
            assert weight_snapshot.dtype == torch.float32
            assert torch.equal(snapshot.cpu(), expected)
            # Combine transports weights as a side channel; it does not multiply
            # the hidden states by them.
            assert torch.equal(weight_snapshot.cpu(), expected_weights.expand(num_tokens, -1))
            del (dispatch_pressure, weight_pressure, combine_pressure, combine_weight_pressure,
                 recv_x, recv_weights, handle, combine_result, combined_x, combined_weights,
                 snapshot, weight_snapshot)

    buffer.destroy()


def validate_config(args, num_ranks):
    if args.ep_shape in ('mte', 'both'):
        if num_ranks not in (8, 16, 32, 64, 128):
            raise ValueError('MTE shape requires global EP size 8, 16, 32, 64, or 128')
        max_bs = 64 if num_ranks == 8 else 32
        if not 0 < args.ep_bs <= max_bs:
            raise ValueError(f'MTE shape at EP{num_ranks} requires 1 <= --ep-bs <= {max_bs}')
        if not args.ep_bs <= args.ep_capacity <= 256:
            raise ValueError('MTE shape requires --ep-bs <= --ep-capacity <= 256')


def configure_backend(backend):
    # The launcher/environment selects the installed package; this option never
    # switches repositories. Official accepts the same CLI without new attributes.
    if backend == 'candidate-disabled':
        os.environ['EP_DISABLE_LOW_LATENCY_TRANSPORT'] = '1'
    elif backend == 'candidate-auto':
        os.environ.pop('EP_DISABLE_LOW_LATENCY_TRANSPORT', None)


def require_device_delay():
    if not callable(getattr(getattr(torch, 'npu', None), '_sleep', None)):
        raise RuntimeError(
            'This stream harness requires callable torch.npu._sleep for an asynchronous device delay. '
            'The installed torch_npu backend does not provide it; use a compatible backend or validate '
            'a device-delay adapter before running this test. CPU sleep is not a substitute.')


@torch.inference_mode()
def test(local_rank, num_local_ranks, args):
    require_device_delay()
    configure_backend(args.backend)
    _, _, group = init_dist(local_rank, num_local_ranks)
    validate_config(args, group.size())
    original = torch.npu.current_stream()
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    comm_stream = deep_ep.comm.get_comm_stream(None)
    assert comm_stream not in (original, producer, consumer)
    with torch.npu.stream(producer):
        assert deep_ep.comm.get_comm_stream(None) == comm_stream

    previous_barrier = deep_ep._C.set_barrier_in_prologue(args.barrier_in_prologue == 'true')
    try:
        dist_print(f'Stream test: backend={args.backend}, package={deep_ep.__file__}, '
                   f'EP={group.size()}, shape={args.ep_shape}, B={args.ep_bs}, capacity={args.ep_capacity}, '
                   f'barrier_in_prologue={args.barrier_in_prologue}. '
                   'MTE shape eligibility alone does not prove the selected device transport.',
                   once_in_node=True)
        test_events(producer, consumer, comm_stream)
        if not args.skip_bucket:
            test_bucket(group, producer, consumer)
        if args.ep_shape in ('fallback', 'both'):
            test_ep(group, producer)
        if args.ep_shape in ('mte', 'both'):
            test_ep_mte(group, producer, consumer, args.ep_bs, args.ep_capacity)
        assert torch.npu.current_stream() == original
        dist_print('Stream ordering and EP lifetime checks passed', once_in_node=True)
    finally:
        deep_ep._C.set_barrier_in_prologue(previous_barrier)
        deep_ep.destroy_all_managed_hccl_comm()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NPU stream ordering and tensor lifetime')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--ep-shape', choices=('fallback', 'mte', 'both'), default='fallback',
                        help='Keep the original H1024/K1 case, add H7168/K8/E128, or run both')
    parser.add_argument('--ep-bs', type=int, default=16, help='Local token count for the MTE shape')
    parser.add_argument('--ep-capacity', type=int, default=None,
                        help='MTE shape buffer capacity per rank (default: --ep-bs)')
    parser.add_argument('--backend', choices=('official', 'candidate-disabled', 'candidate-auto'),
                        default='official', help='Label the installed build and configure its native MTE opt-out')
    parser.add_argument('--barrier-in-prologue', choices=('false', 'true'), default='false')
    parser.add_argument('--skip-bucket', action='store_true', help='Skip the BucketBuffer public-layer smoke test')
    args = parser.parse_args()
    if args.ep_capacity is None:
        args.ep_capacity = args.ep_bs
    if args.num_processes <= 0:
        parser.error('--num-processes must be positive')
    # init_dist interprets WORLD_SIZE as node count and --num-processes as the
    # number of local ranks. Reject unsupported MTE cases before HCCL setup.
    try:
        validate_config(args, int(os.getenv('WORLD_SIZE', '1')) * args.num_processes)
        require_device_delay()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    configure_backend(args.backend)
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
