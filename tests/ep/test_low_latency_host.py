"""Dependency-free host checks. Run with python tests/ep/test_low_latency_host.py.

The fake tensor backend checks API selection, scheduling and allocation contracts;
it is not a substitute for the NPU data-correctness test in test_low_latency.py.
"""

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, dtype='torch.bfloat16', device=('npu', 0)):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim = len(shape)
        self.recorded = []

    def is_contiguous(self):
        return True

    def is_inference(self):
        return True

    def to(self, dtype=None, device=None):
        return Tensor(self.shape, dtype or self.dtype, device or self.device)

    def __getitem__(self, index):
        return Tensor((index.stop, *self.shape[1:]), self.dtype, self.device)

    def record_stream(self, stream):
        self.recorded.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        return [2, 1]


class Event:
    def __init__(self):
        self.waits = 0

    def current_stream_wait(self):
        self.waits += 1


def fake_modules():
    torch = ModuleType('torch')
    torch.Tensor = Tensor
    for dtype in ('bfloat16', 'float16', 'float8_e4m3fn', 'int64', 'int32', 'float32', 'uint8'):
        setattr(torch, dtype, 'torch.' + dtype)
    torch.device = lambda kind, index: Device(kind, index)
    torch.npu = SimpleNamespace(current_device=lambda: 0, current_stream=lambda: 'stream0')
    torch.empty = torch.zeros = lambda shape, dtype, device: Tensor(shape, dtype, device)
    torch.frombuffer = lambda data, dtype: Tensor((len(data),), dtype, ('cpu', None))
    torch.distributed = ModuleType('torch.distributed')
    torch.distributed.ProcessGroup = object
    native = ModuleType('deep_ep._C')
    native.EventHandle = Event
    native.low_latency_layout = lambda p, b, h, k, e, dtype: dict(capacity_rows=p * b * min(k, e // p))
    native.low_latency_dispatch_tiling = native.low_latency_combine_tiling = lambda *args: b'\0' * 16
    native.calls = []
    native.dispatch_low_latency = lambda *args: native.calls.append(('dispatch', args))
    native.combine_low_latency = lambda *args: native.calls.append(('combine', args))
    modules = {name: ModuleType(name) for name in (
        'deep_ep', 'deep_ep.buffers', 'deep_ep.comm', 'deep_ep.utils',
        'deep_ep.buffers.allocator', 'deep_ep.utils.math', 'deep_ep.utils.semantic')}
    modules.update({'torch': torch, 'torch.distributed': torch.distributed, 'deep_ep._C': native})
    modules['deep_ep']._C = native
    modules['deep_ep'].comm = modules['deep_ep.comm']
    for name in ('barrier', 'get_comm_stream', 'get_physical_domain_size', 'get_logical_domain_size'):
        setattr(modules['deep_ep.comm'], name, lambda *args: None)
    modules['deep_ep.buffers.allocator'].BufferAllocator = type('BufferAllocator', (), {})
    modules['deep_ep.utils.math'].align = lambda a, b: (a + b - 1) // b * b
    modules['deep_ep.utils.semantic'].value_or = lambda x, default: default if x is None else x
    return modules, torch, native


class Device(tuple):
    def __new__(cls, kind, index):
        return super().__new__(cls, (kind, index))

    @property
    def index(self):
        return self[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HostContracts(unittest.TestCase):
    def setUp(self):
        modules, self.torch, self.native = fake_modules()
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        load('deep_ep.buffers.base', 'deep_ep/buffers/base.py')
        load('deep_ep.utils.event', 'deep_ep/utils/event.py')
        self.ep = load('deep_ep.buffers.ep', 'deep_ep/buffers/ep.py')
        ll = load('deep_ep.buffers.low_latency', 'deep_ep/buffers/low_latency.py')
        self.buffer = self.ep.EPBuffer.__new__(self.ep.EPBuffer)
        self.buffer.low_latency_mode = True
        self.buffer.runtime = object()
        self.buffer.num_max_tokens_per_rank = 16
        self.buffer.num_ranks = 8
        self.buffer._low_latency = ll.LowLatencyState()

    def dispatch(self, **kwargs):
        params = dict(topk_idx=Tensor((2, 2), self.torch.int64),
                      topk_weights=Tensor((2, 2), self.torch.float32), num_experts=16, do_expand=True)
        params.update(kwargs)
        return self.buffer.dispatch(Tensor((2, 128)), **params)

    def test_ep8_selects_low_latency_and_returns_exact_rows(self):
        recv, indices, weights, handle, event = self.dispatch()
        self.assertTrue(handle.is_low_latency)
        self.assertEqual(recv.shape, (3, 128))
        self.assertEqual(handle.num_recv_tokens_per_expert_list, [2, 1])
        self.assertIsNone(indices)
        self.assertIsNone(weights)
        self.assertEqual(self.native.calls[0][0], 'dispatch')
        self.assertTrue(all(t.recorded for t in event.extra_tensors))

    def test_throughput_buffer_keeps_its_native_dispatch(self):
        self.buffer.low_latency_mode = False
        self.buffer.runtime = SimpleNamespace(dispatch=lambda *args: (_ for _ in ()).throw(RuntimeError('HT selected')))
        with self.assertRaisesRegex(RuntimeError, 'HT selected'):
            self.dispatch(num_sms=32)
        self.assertEqual(self.native.calls, [])

    def test_no_host_sync_uses_actual_batch_capacity(self):
        with patch.object(Tensor, 'cpu', side_effect=AssertionError('host sync')):
            recv, _, _, handle, _ = self.dispatch(do_cpu_sync=False)
        self.assertEqual(recv.shape, (32, 128))
        self.assertEqual(handle.num_expanded_tokens, 32)

    def test_repeat_combine_serializes_without_passing_dispatch_weights(self):
        recv, _, _, handle, dispatch_event = self.dispatch()
        self.buffer.combine(recv, handle)
        self.buffer.combine(recv, handle)
        self.assertEqual(dispatch_event.event.waits, 1)
        self.assertEqual([op for op, _ in self.native.calls], ['dispatch', 'combine', 'combine'])
        for op, args in self.native.calls:
            if op == 'combine':
                self.assertNotIn('source_weights', args[1])
                self.assertIs(args[1]['input'], recv)

    def test_rejects_incompatible_options_before_launch(self):
        for options in (dict(do_expand=False), dict(expert_alignment=128), dict(do_handle_copy=True),
                        dict(defer_epilogue=True), dict(num_sms=16), dict(num_qps=1)):
            with self.subTest(options=options), self.assertRaises(NotImplementedError):
                self.dispatch(**options)
        self.assertEqual(self.native.calls, [])

    def test_rejects_wrong_device_dtype_and_capacity(self):
        for options in (dict(topk_idx=Tensor((2, 2), self.torch.int32)),
                        dict(topk_weights=Tensor((2, 2), self.torch.float32, ('cpu', None))),
                        dict(num_max_tokens_per_rank=1), dict(num_max_tokens_per_rank=257)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.dispatch(**options)
        self.assertEqual(self.native.calls, [])

    def test_rejects_foreign_handle_and_unsupported_weight_metadata(self):
        recv, _, _, handle, _ = self.dispatch()
        with self.assertRaises(NotImplementedError):
            self.buffer.combine(recv, handle, topk_weights=Tensor((3,), self.torch.float32))
        handle._buffer_token = object()
        with self.assertRaises(ValueError):
            self.buffer.combine(recv, handle)
        self.assertEqual(len(self.native.calls), 1)

    def test_failed_launch_is_not_retried(self):
        with patch.object(self.native, 'dispatch_low_latency', side_effect=RuntimeError('launch failed')):
            with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                self.dispatch()
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            self.dispatch()

    def test_destroyed_buffer_is_rejected(self):
        self.buffer.runtime = None
        with self.assertRaisesRegex(RuntimeError, 'destroyed'):
            self.dispatch()

    def test_fp8_keeps_opaque_scale_dtype(self):
        for scale_dtype in (self.torch.int32, self.torch.float32):
            result = self.buffer.dispatch(
                (Tensor((2, 128), self.torch.float8_e4m3fn), Tensor((2, 1), scale_dtype)),
                topk_idx=Tensor((2, 2), self.torch.int64), topk_weights=Tensor((2, 2), self.torch.float32),
                num_experts=16, do_expand=True)
            self.assertEqual(result[0][1].dtype, scale_dtype)

    def test_capacity_selects_tiling_and_survives_handle_reuse(self):
        # Equal B with different M must not share tiling across the direct/dedup
        # threshold. Combine must use the originating handle after another dispatch.
        with patch.object(self.native, 'low_latency_dispatch_tiling', return_value=b'\0' * 32) as dispatch_tiling, \
             patch.object(self.native, 'low_latency_combine_tiling', return_value=b'\0' * 20) as combine_tiling:
            small, _, _, direct, _ = self.dispatch(num_max_tokens_per_rank=32)
            large, _, _, dedup, _ = self.dispatch(num_max_tokens_per_rank=33)
            self.buffer.combine(small, direct)
            self.buffer.combine(large, dedup)
            self.buffer.combine(small, direct)
            self.dispatch(num_max_tokens_per_rank=256)
        self.assertEqual([call.args[-1] for call in dispatch_tiling.call_args_list], [32, 33, 256])
        self.assertEqual([call.args[-1] for call in combine_tiling.call_args_list], [32, 33, 32, 33, 256])


class SourceContracts(unittest.TestCase):
    def test_jit_dependency_closure_is_packaged_and_acyclic(self):
        include = ROOT / 'deep_ep/include'
        seen, active = set(), set()

        def visit(name):
            self.assertNotIn(name, active, f'Circular DeepJIT include: {name}')
            if name in seen:
                return
            path = include / name
            self.assertTrue(path.is_file(), f'Missing JIT header: {name}')
            self.assertIn(path.suffix, ('.h', '.hpp'))
            active.add(name)
            code = path.read_text(encoding='utf-8')
            self.assertNotRegex(code, r'#\s*include\s*"')
            for dep in re.findall(r'#\s*include\s*<([^>]+)>', code):
                if dep.startswith('deep_ep/'):
                    visit(dep)
            active.remove(name)
            seen.add(name)

        visit('deep_ep/impls/low_latency/dispatch.hpp')
        visit('deep_ep/impls/low_latency/combine.hpp')
        self.assertGreater(len(seen), 20)

    def test_ep8_test_matrix_is_available_without_torch(self):
        spec = importlib.util.spec_from_file_location('ll_st', ROOT / 'tests/ep/test_low_latency.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parser().parse_args(['--dtype', 'bf16', 'fp16', 'fp8', '--fp8-scale-dtype', 'int32', 'float32'])
        module.validate(args, 8)
        self.assertEqual(len(list(module.cases(args))), 16)
        with self.assertRaises(ValueError):
            module.validate(args, 4)

    def test_capacity_matrix_covers_both_paths_and_rejects_invalid_capacity(self):
        module = load('ll_capacity_st', 'tests/ep/test_low_latency.py')
        args = module.parser().parse_args(['--num-tokens', '16', '--num-max-tokens-per-rank', '32', '33', '256'])
        module.validate(args, 8)
        self.assertEqual({case['m'] for case in module.cases(args)}, {32, 33, 256})
        self.assertEqual({case['route'] for case in module.cases(args)}, {'random', 'hotspot'})
        for capacity in (0, 15, 257):
            args.num_max_tokens_per_rank = [capacity]
            with self.assertRaises(ValueError):
                module.validate(args, 8)


if __name__ == '__main__':
    unittest.main()
