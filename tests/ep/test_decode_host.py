"""CPU-only contracts for the single-EPBuffer decode convenience wrapper.

These mocks check ownership and argument forwarding, not NPU correctness.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_low_latency_host import Device, Event, Tensor, fake_modules, load


class DecodeContracts(unittest.TestCase):
    def setUp(self):
        import sys
        modules, self.torch, self.native = fake_modules()
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        load('deep_ep.buffers.base', 'deep_ep/buffers/base.py')
        load('deep_ep.utils.event', 'deep_ep/utils/event.py')
        load('deep_ep.buffers.ep', 'deep_ep/buffers/ep.py')
        self.api = load('deep_ep.buffers.decode', 'deep_ep/buffers/decode.py')
        self.group = SimpleNamespace(size=lambda: 8)
        self.calls = []
        self.arena = SimpleNamespace(dispatch=self.native_dispatch, combine=self.native_combine,
                                     destroy=Mock())
        self.factory = Mock(return_value=self.arena)
        self.factory.capture = Event
        self.backend_patch = patch.object(self.api, 'EPBuffer', self.factory)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.buffer = self.api.DecodeBuffer(self.group, 4096)

    def native_dispatch(self, x, **kwargs):
        rows = x.shape[0] * 8
        recv = Tensor((rows, 7168), device=Device('npu', 0))
        weights = Tensor((rows,), self.torch.float32, Device('npu', 0))
        handle = SimpleNamespace(num_unaligned_recv_tokens_per_expert=Tensor((16,), self.torch.int32),
                                 num_recv_tokens_per_expert_list=[], num_expanded_tokens=rows)
        self.calls.append(('dispatch', x, kwargs, handle, weights))
        return recv, None, weights, handle, self.api.EventOverlap(Event())

    def native_combine(self, x, handle, **kwargs):
        self.calls.append(('combine', x, handle, kwargs))
        return x, None, self.api.EventOverlap(Event())

    def dispatch(self, bs):
        device = Device('npu', 0)
        return self.buffer.dispatch(Tensor((bs, 7168), device=device),
                                    Tensor((bs, 8), self.torch.int64, device),
                                    Tensor((bs, 8), self.torch.float32, device))

    def test_one_native_arena_is_used_for_every_batch(self):
        handles = [self.dispatch(bs)[1] for bs in (2, 128, 256, 2)]
        self.factory.assert_called_once()
        options = self.factory.call_args.kwargs
        self.assertFalse(options.get('low_latency_mode', False))
        self.assertEqual(options['num_max_tokens_per_rank'], 4096)
        self.assertTrue(all(handle._buffer is self.arena for handle in handles))
        self.assertTrue(all(handle.backend == 'ep' for handle in handles))

    def test_dispatch_exposes_received_weight_metadata(self):
        _, handle, _ = self.dispatch(2)
        call = self.calls[-1]
        self.assertIs(handle.recv_topk_weights, call[4])
        self.assertIs(handle._handle, call[3])
        self.assertEqual(handle.num_expanded_tokens, 16)
        self.assertEqual(handle.expert_recv_counts.dtype, self.torch.int64)

    def test_combine_forwards_input_unchanged_and_reuses_old_handle(self):
        expert, handle, _ = self.dispatch(2)
        self.dispatch(128)
        for _ in range(2):
            output, _ = self.buffer.combine(expert, handle)
            self.assertIs(output, expert)
            self.assertIs(self.calls[-1][1], expert)
            self.assertIs(self.calls[-1][2], handle._handle)
            self.assertNotIn('topk_weights', self.calls[-1][3])
        # Tensor intentionally has no float(), multiply or unsqueeze methods:
        # a wrapper-side weight multiply would fail this contract test.
        other = self.api.DecodeBuffer(self.group, 128)
        with self.assertRaisesRegex(ValueError, 'from this'):
            other.combine(expert, handle)

    def test_failed_launch_is_not_retried(self):
        self.arena.dispatch = Mock(side_effect=RuntimeError('launch failed'))
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            self.dispatch(2)
        self.arena.dispatch.assert_called_once()
        self.factory.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            self.dispatch(2)

    def test_destroy_is_idempotent(self):
        self.dispatch(2)
        self.buffer.destroy()
        self.buffer.destroy()
        self.arena.destroy.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'destroyed'):
            self.dispatch(2)


if __name__ == '__main__':
    unittest.main()
