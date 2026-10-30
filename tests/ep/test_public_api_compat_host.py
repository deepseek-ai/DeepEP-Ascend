"""CPU-only checks of the public Python contract against official 3b25377.

Signature fixtures below were transcribed from that public release, not from
the candidate. Calls execute the real Python wrappers with an inert extension
double; they cannot validate NPU ordering, communication or tensor contents.
Run: python -B tests/ep/test_public_api_compat_host.py
"""

import ast
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_low_latency_host import Event, Tensor, fake_modules, load


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REVISION = '3b25377d04b24fc6154698ded78a2bcb2c59afff'
OFFICIAL_SIGNATURES = {
    'EPHandle.__init__': '(self, do_expand, num_experts, expert_alignment, num_max_tokens_per_rank, num_sms, topk_idx, num_recv_tokens, num_expanded_tokens, num_recv_tokens_per_expert_list, psum_num_recv_tokens_per_rank, psum_num_recv_tokens_per_expert, num_unaligned_recv_tokens_per_expert, recv_src_metadata, dst_buffer_slot_idx, dst_gsge_idx)',
    'EPHandle.deterministic_sort': '(self, do_cpu_sync, is_cached_dispatch, recv_x, recv_sf, recv_topk_weights)',
    'EPBuffer.__init__': '(self, group, num_bytes=None, num_max_tokens_per_rank=0, hidden=0, num_topk=0, use_fp8_dispatch=False, lb_allocation_plan_or_num_bytes=0, deterministic=False, allow_hybrid_mode=True, allow_multiple_reduction=True, prefer_overlap_with_compute=True, sl_idx=None, num_allocated_qps=0, num_cpu_timeout_secs=300, num_gpu_timeout_secs=100, explicitly_destroy=False, *, use_huge1g_page=True)',
    'EPBuffer.destroy': '(self)',
    'EPBuffer.get_buffer_size_hint': '(group, num_max_tokens_per_rank, hidden, num_topk=0, use_fp8_dispatch=False, allow_hybrid_mode=True, allow_multiple_reduction=True)',
    'EPBuffer.capture': '()',
    'EPBuffer.get_theoretical_num_sms': '(self, num_experts, num_topk, num_scaleout_topk=0, rdma_gbs=0, nvlink_gbs=0, sm_read_gbs=0, sm_write_gbs=0)',
    'EPBuffer.get_theoretical_num_qps': '(self, num_sms)',
    'EPBuffer.dispatch': '(self, x, topk_idx=None, topk_weights=None, cumulative_local_expert_recv_stats=None, num_experts=None, num_max_tokens_per_rank=None, expert_alignment=None, num_sms=0, num_qps=0, previous_event=None, async_with_compute_stream=False, allocate_on_comm_stream=False, handle=None, do_handle_copy=False, do_cpu_sync=None, do_expand=False, do_zero_padding=False, use_tma_aligned_col_major_sf=False, defer_epilogue=False)',
    'EPBuffer.combine': '(self, x, handle, topk_weights=None, bias=None, num_sms=0, num_qps=0, previous_event=None, async_with_compute_stream=False, allocate_on_comm_stream=False, defer_epilogue=False)',
    'EPBuffer.lb_get_theoretical_num_sms': '(self)',
    'EPBuffer.lb_prefetch_weights': '(self, redundant_expert_weights, expert_weights, redundancy_mapping, num_sms=0, previous_event=None)',
    'EPBuffer.lb_reduce_grads': '(self, redundant_expert_grads, expert_grads, redundancy_mapping, num_sms=0, previous_event=None)',
}


class RouteTensor(Tensor):
    """Only shape/version metadata is needed by the Python handle wrapper."""

    _version = 0

    def size(self, index):
        return self.shape[index]

    def is_inference(self):
        return False


class PublicAPIContracts(unittest.TestCase):
    def setUp(self):
        modules, self.torch, self.native = fake_modules()
        self.native.get_num_allocation_alignment = lambda: 2 << 20
        self.native.get_num_ai_cores = lambda: 32
        self.native.calculate_ep_buffer_size = Mock(return_value=2 << 20)
        self.native.EPBuffer = Mock()
        self.native.LowLatencyBuffer = Mock(side_effect=AssertionError('Unexpected legacy allocation'))
        modules['deep_ep.comm'].get_hccl_comm_handle = lambda group: SimpleNamespace(get=lambda: 'hccl')
        modules['deep_ep.comm'].get_logical_domain_size = lambda *args: (1, 128)
        modules['deep_ep.comm'].get_physical_domain_size = lambda *args: (1, 128)
        self.torch.npu.synchronize = Mock()
        self.torch.distributed.barrier = Mock()
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        load('deep_ep.buffers.base', 'deep_ep/buffers/base.py')
        self.events = load('deep_ep.utils.event', 'deep_ep/utils/event.py')
        self.ep = load('deep_ep.buffers.ep', 'deep_ep/buffers/ep.py')
        self.group = SimpleNamespace(rank=lambda: 0, size=lambda: 128)
        self.runtime = SimpleNamespace(context=object(), destroy=Mock(), dispatch=Mock(), combine=Mock())
        self.native.EPBuffer.return_value = self.runtime
        self.buffer = self.ep.EPBuffer(self.group, num_max_tokens_per_rank=256,
                                       hidden=7168, num_topk=8, explicitly_destroy=True)
        self.x = Tensor((2, 7168))
        self.routes = RouteTensor((2, 8), self.torch.int64)
        self.weights = Tensor((2, 8), self.torch.float32)
        self.recv = Tensor((16, 7168))
        self.recv_weights = Tensor((16,), self.torch.float32)
        self.metadata = [Tensor((1,), self.torch.int32) for _ in range(6)]
        self.result = (self.recv, None, self.recv_weights, 2, 16, [16], *self.metadata)
        self.event = Event()
        self.runtime.dispatch.return_value = (self.result, self.event, None)
        self.runtime.combine.return_value = ((self.x, self.weights), self.event, None)

    def dispatch(self, **overrides):
        kwargs = dict(topk_idx=self.routes, topk_weights=self.weights, num_experts=128, do_expand=True)
        kwargs.update(overrides)
        return self.buffer.dispatch(self.x, **kwargs)

    def test_official_signatures_keep_parameter_order_kinds_and_defaults(self):
        additions = {'EPBuffer.__init__', 'EPBuffer.get_buffer_size_hint'}
        for name, expected in OFFICIAL_SIGNATURES.items():
            cls, method = name.split('.')
            signature = inspect.signature(getattr(getattr(self.ep, cls), method))
            parameters = list(signature.parameters.values())
            if name in additions:
                addition = signature.parameters['low_latency_mode']
                self.assertEqual(addition.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(addition.default, False)
                parameters = [p for p in parameters if p.name != 'low_latency_mode']
            actual = signature.replace(parameters=[p.replace(annotation=inspect.Parameter.empty) for p in parameters],
                                       return_annotation=inspect.Signature.empty)
            with self.subTest(name=name, baseline=OFFICIAL_REVISION):
                self.assertEqual(str(actual), expected)

    def test_official_top_level_exports_remain_available_in_source(self):
        tree = ast.parse((ROOT / 'deep_ep/__init__.py').read_text(encoding='utf-8'))
        imports = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                   for alias in node.names}
        official = {'comm', 'destroy_all_managed_hccl_comm', 'get_physical_domain_size',
                    'get_logical_domain_size', 'BufferAllocator', 'BufferBase', 'EPBuffer', 'EPHandle',
                    'EngramBuffer', 'BucketBuffer', 'BucketSession', 'PPBuffer',
                    'get_num_allocation_alignment', 'get_num_rdma_alignment', 'get_num_tma_alignment',
                    'topk_idx_t', 'EventOverlap', 'EventHandle', 'init_seed', 'get_local_seed',
                    'get_global_seed', 'dist_print', 'init_dist', 'KernelProfile', 'bench_msprof'}
        self.assertTrue(official <= imports, official - imports)
        self.assertTrue({'DecodeBuffer', 'DecodeHandle'} <= imports)

    def test_default_constructor_uses_one_native_arena_and_official_arguments(self):
        self.assertFalse(self.buffer.low_latency_mode)
        self.native.LowLatencyBuffer.assert_not_called()
        self.native.calculate_ep_buffer_size.assert_called_once_with(128, 256, 7168, 8, False)
        self.native.EPBuffer.assert_called_once_with(0, 128, 'hccl', 2 << 20, 0, True, 300, 100, True)
        self.assertIs(self.buffer.context, self.runtime.context)
        self.torch.distributed.barrier.assert_called_once_with(self.group)
        self.assertEqual(self.torch.npu.synchronize.call_count, 2)

    def test_native_fresh_defaults_optional_weights_and_stats_are_forwarded(self):
        stats = Tensor((1,), self.torch.int32)
        for weights in (None, self.weights):
            with self.subTest(weights=weights):
                result = self.dispatch(topk_weights=weights, cumulative_local_expert_recv_stats=stats)
                args = self.runtime.dispatch.call_args.args
                self.assertIs(args[0], self.x)
                self.assertIsNone(args[1])
                self.assertIs(args[2], self.routes)
                self.assertIs(args[3], weights)
                self.assertIs(args[4], stats)
                self.assertEqual(args[5:14], (None,) * 9)
                self.assertEqual(args[14:18], (256, 128, 1, 32))
                self.assertEqual(args[18:], (None, False, False, False, True, True, False, False, False))
                recv, indices, recv_weights, handle, event = result
                self.assertIs(recv, self.recv)
                self.assertIsNone(indices)
                self.assertIs(recv_weights, self.recv_weights)
                self.assertFalse(handle.is_low_latency)
                self.assertIs(handle.topk_idx, self.routes)
                self.assertEqual(handle.num_recv_tokens_per_expert_list, [16])
                self.assertIs(event.event, self.event)

    def test_cached_dispatch_reuses_handle_configuration_and_updated_weights(self):
        handle = self.dispatch(num_max_tokens_per_rank=64, expert_alignment=16, num_sms=16)[3]
        updated = Tensor((2, 8), self.torch.float32)
        result = self.buffer.dispatch(self.x, handle=handle, topk_weights=updated, num_sms=32, do_expand=True)
        args = self.runtime.dispatch.call_args.args
        self.assertIs(result[3], handle)
        self.assertIs(args[2], self.routes)
        self.assertIs(args[3], updated)
        self.assertEqual(args[5:14], self.ep.EPBuffer._unpack_handle(handle))
        self.assertEqual(args[14:18], (64, 128, 16, 16))
        self.assertIs(args[22], False)

    def test_native_sync_return_keeps_event_wrapper_without_inventing_an_event(self):
        self.runtime.dispatch.return_value = (self.result, None, None)
        returned = self.dispatch()
        self.assertIsInstance(returned[-1], self.events.EventOverlap)
        self.assertIsNone(returned[-1].event)
        with self.assertRaises(AssertionError):
            returned[-1].wait()
        self.runtime.combine.return_value = ((self.x, None), None, None)
        combined = self.buffer.combine(self.recv, returned[3])
        self.assertEqual(len(combined), 3)
        self.assertIsNone(combined[1])
        self.assertIsNone(combined[2].event)

    def test_native_dispatch_schedule_is_forwarded_without_python_stream_wait(self):
        previous = Event()
        self.dispatch(previous_event=previous, async_with_compute_stream=True,
                      allocate_on_comm_stream=True, do_cpu_sync=False, do_zero_padding=True)
        args = self.runtime.dispatch.call_args.args
        self.assertEqual(args[18:25], (previous, True, True, False, False, True, True))
        self.assertEqual(previous.waits, 0)

    def test_cached_context_mutation_and_cpu_sync_are_rejected_before_runtime(self):
        handle = self.dispatch()[3]
        for mismatch in (dict(num_max_tokens_per_rank=128), dict(expert_alignment=16),
                         dict(num_experts=256), dict(do_cpu_sync=True), dict(topk_idx=self.routes)):
            self.runtime.dispatch.reset_mock()
            with self.subTest(mismatch=mismatch), self.assertRaises(AssertionError):
                self.buffer.dispatch(self.x, handle=handle, do_expand=True, **mismatch)
            self.runtime.dispatch.assert_not_called()
        self.routes._version += 1
        with self.assertRaisesRegex(AssertionError, 'must not be modified'):
            self.buffer.dispatch(self.x, handle=handle, do_expand=True)
        self.runtime.dispatch.assert_not_called()

    def test_fp8_scale_tensor_is_forwarded_and_returned_without_conversion(self):
        # The native extension, not this wrapper, validates int16 pack layout.
        values = Tensor((2, 7168), self.torch.float8_e4m3fn)
        scales = Tensor((2, 112), 'torch.int16')
        recv_scales = Tensor((16, 112), 'torch.int16')
        result = list(self.result)
        result[1] = recv_scales
        self.runtime.dispatch.return_value = (tuple(result), self.event, None)
        returned = self.buffer.dispatch((values, scales), topk_idx=self.routes, num_experts=128,
                                       do_expand=True, use_tma_aligned_col_major_sf=True)
        self.assertIs(self.runtime.dispatch.call_args.args[0], values)
        self.assertIs(self.runtime.dispatch.call_args.args[1], scales)
        self.assertTrue(self.runtime.dispatch.call_args.args[25])
        self.assertIs(returned[0][0], self.recv)
        self.assertIs(returned[0][1], recv_scales)

    def test_combine_uses_originating_handle_and_preserves_bias_weight_side_channel(self):
        handle = self.dispatch(num_max_tokens_per_rank=64, num_sms=16)[3]
        self.dispatch(num_max_tokens_per_rank=256, num_sms=32)
        bias0, bias1 = Tensor((2, 7168)), Tensor((2, 7168))
        previous = Event()
        for bias, expected_bias in ((None, (None, None)), (bias0, (bias0, None)),
                                    ((bias0, bias1), (bias0, bias1))):
            result = self.buffer.combine(self.recv, handle, topk_weights=self.recv_weights, bias=bias,
                                         num_sms=32, previous_event=previous,
                                         async_with_compute_stream=True, allocate_on_comm_stream=True)
            args = self.runtime.combine.call_args.args
            self.assertIs(args[0], self.recv)
            self.assertIs(args[1], self.recv_weights)
            self.assertEqual(args[2:4], expected_bias)
            self.assertIs(args[4], handle.recv_src_metadata)
            self.assertIs(args[5], self.routes)
            self.assertEqual(args[7:10], (128, 64, 16))
            self.assertEqual(args[10:], (previous, True, True, False))
            self.assertIs(result[0], self.x)
            self.assertIs(result[1], self.weights)
        # Python forwarding does not imply the baseline C++ runtime honors these options.
        self.assertEqual(previous.waits, 0)

    def test_deferred_dispatch_returns_event_then_four_values_and_runs_once(self):
        epilogue = Mock(return_value=self.result)
        self.runtime.dispatch.return_value = (None, self.event, epilogue)
        event = self.dispatch(async_with_compute_stream=True, defer_epilogue=True)
        self.assertIsInstance(event, self.events.EventOverlap)
        epilogue.assert_not_called()
        result = event.wait()
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], self.recv)
        self.assertIsNone(event.wait())
        epilogue.assert_called_once_with()

    def test_deterministic_sort_runs_after_wait_for_async_or_deferred_dispatch(self):
        self.buffer.deterministic = True
        for deferred in (False, True):
            order = []
            event = SimpleNamespace(current_stream_wait=lambda: order.append('wait'))
            epilogue = lambda: order.append('epilogue') or self.result
            self.runtime.dispatch.return_value = ((None, event, epilogue) if deferred
                                                  else (self.result, event, None))
            with patch.object(self.ep.EPHandle, 'deterministic_sort',
                              side_effect=lambda *args: order.append('sort')) as sort:
                returned = self.dispatch(async_with_compute_stream=True, defer_epilogue=deferred)
                overlap = returned if deferred else returned[-1]
                sort.assert_not_called()
                overlap.wait()
                self.assertEqual(order, ['wait', 'epilogue', 'sort'] if deferred else ['wait', 'sort'])
                overlap.wait()
                self.assertEqual(sort.call_count, 1)

    def test_deferred_combine_returns_pair_once_and_event_release_is_explicit(self):
        handle = self.dispatch()[3]
        epilogue = Mock(return_value=(self.x, self.weights))
        self.runtime.combine.return_value = (None, self.event, epilogue)
        event = self.buffer.combine(self.recv, handle, async_with_compute_stream=True, defer_epilogue=True)
        epilogue.assert_not_called()
        self.assertEqual(event.current_stream_wait(release_handle=True), (self.x, self.weights))
        self.assertIsNone(event.event)
        epilogue.assert_called_once_with()
        with self.assertRaises(AssertionError):
            event.wait()

    def test_native_and_explicit_low_latency_handles_cannot_mix(self):
        foreign = SimpleNamespace(is_low_latency=True)
        with self.assertRaisesRegex(ValueError, 'low-latency handle'):
            self.buffer.dispatch(self.x, handle=foreign, do_expand=True)
        with self.assertRaisesRegex(ValueError, 'low-latency handle'):
            self.buffer.combine(self.recv, foreign)
        self.runtime.dispatch.assert_not_called()
        self.runtime.combine.assert_not_called()

    def test_explicit_destroy_remains_idempotent_and_clears_context(self):
        self.buffer.destroy()
        self.buffer.destroy()
        self.runtime.destroy.assert_called_once_with()
        self.assertIsNone(self.buffer.runtime)
        self.assertIsNone(self.buffer.context)
        self.assertIsNone(self.buffer.hccl_comm_handle)


if __name__ == '__main__':
    unittest.main()
