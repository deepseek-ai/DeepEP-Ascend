import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


def test_event_hook_is_released_when_it_fails():
    class FailingEvent:
        def current_stream_wait(self):
            return None

    event = deep_ep.EventOverlap(FailingEvent())

    def fail():
        raise RuntimeError('epilogue failed')

    event.register_hook_after_wait(fail)
    try:
        event.current_stream_wait(release_handle=True)
    except RuntimeError as error:
        assert str(error) == 'epilogue failed'
    else:
        raise AssertionError('the failing epilogue did not raise')
    assert event.hook_after_wait is None
    assert event.event is None


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


@torch.inference_mode()
def test(local_rank, num_local_ranks):
    _, _, group = init_dist(local_rank, num_local_ranks)
    original = torch.npu.current_stream()
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    comm_stream = deep_ep.comm.get_comm_stream(None)
    assert comm_stream not in (original, producer, consumer)
    with torch.npu.stream(producer):
        assert deep_ep.comm.get_comm_stream(None) == comm_stream

    test_events(producer, consumer, comm_stream)
    test_bucket(group, producer, consumer)
    test_ep(group, producer)
    assert torch.npu.current_stream() == original
    dist_print('Stream ordering and current-stream EP passed', once_in_node=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NPU stream ordering and tensor lifetime')
    parser.add_argument('--num-processes', type=int, default=2)
    args = parser.parse_args()
    test_event_hook_is_released_when_it_fails()
    torch.multiprocessing.spawn(test, args=(args.num_processes,), nprocs=args.num_processes)
