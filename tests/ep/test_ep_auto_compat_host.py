"""CPU tests of the compatibility harness, not validation of NPU kernels."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_ep_auto_compat as suite
from compat_config import parse_args, plan_cases


class ContractTests(unittest.TestCase):
    def test_official_has_no_candidate_only_markers(self):
        suite.check_native_contract(SimpleNamespace(), 'low_latency_mode', 'official')
        suite.check_native_contract(SimpleNamespace(), 'is_low_latency', 'official')
        with self.assertRaises(AssertionError):
            suite.check_native_contract(SimpleNamespace(low_latency_mode=False), 'low_latency_mode', 'official')
        with self.assertRaises(AssertionError):
            suite.check_native_contract(SimpleNamespace(), 'is_low_latency', 'candidate-auto')
        suite.check_native_contract(SimpleNamespace(is_low_latency=False), 'is_low_latency', 'candidate-disabled')
        with self.assertRaises(AssertionError):
            suite.check_native_contract(SimpleNamespace(is_low_latency=True), 'is_low_latency', 'candidate-auto')

    def test_deferred_return_and_hook_once(self):
        class Event:
            calls = 0
            def wait(self):
                self.calls += 1
                return ('x', None, 'weights', 'handle') if self.calls == 1 else None
        event = Event()
        buffer = SimpleNamespace(dispatch=lambda **kw: event)
        self.assertEqual(suite.launch(buffer, 'dispatch', True), ('x', None, 'weights', 'handle', event))
        self.assertEqual(event.calls, 2)

    def test_policy_uses_original_b_and_handle_capacity_for_combine(self):
        case = plan_cases(parse_args(['--bs', '2', '--edge-routes', 'off']), world=128)[0]
        runtime = dict(arch=3510, num_vec_cores=64)
        self.assertEqual(suite.expected_transport('dispatch', case, 0, 'candidate-auto', runtime), 'mte')
        fp8 = dict(case, dtype='fp8')
        self.assertEqual(suite.expected_transport('dispatch', fp8, 0, 'candidate-auto', runtime), 'urma')
        self.assertEqual(suite.expected_transport('combine', fp8, 0, 'candidate-auto', runtime), 'mte')
        self.assertEqual(suite.expected_transport('combine', case, 0, 'candidate-auto', runtime,
                                                  original_b=33, capacity=256), 'urma')
        self.assertEqual(suite.expected_transport('combine', case, 0, 'candidate-auto', runtime,
                                                  original_b=2, capacity=512), 'urma')
        for backend in ('official', 'candidate-disabled'):
            self.assertEqual(suite.expected_transport('combine', case, 0, backend, runtime), 'urma')

    def test_only_ep128_can_use_mte_for_native_operations(self):
        runtime = dict(arch=3510, num_vec_cores=64)
        args = parse_args(['--bs', '2', '--edge-routes', 'off'])
        for world in (1, 2, 4, 8, 16, 32, 64, 128):
            case = plan_cases(args, world=world)[0]
            for operation in ('dispatch', 'combine'):
                for b in (0, 1, 2, 8, 16, 32, 33, 48, 64, 96, 128, 256):
                    with self.subTest(world=world, operation=operation, b=b):
                        expected = 'mte' if world == 128 and 1 <= b <= 32 else 'urma'
                        self.assertEqual(suite.expected_transport(
                            operation, case, 0, 'candidate-auto', runtime,
                            original_b=b, capacity=256), expected)

    def test_ep128_mixed_rank_batches_keep_independent_fallback(self):
        case = plan_cases(parse_args(['--edge-routes', 'only']), world=128)[0]
        runtime = dict(arch=3510, num_vec_cores=64)
        for rank, b in enumerate(case['rank_bs']):
            for operation in ('dispatch', 'combine'):
                with self.subTest(rank=rank, operation=operation):
                    expected = 'mte' if 1 <= b <= 32 else 'urma'
                    self.assertEqual(suite.expected_transport(
                        operation, case, rank, 'candidate-auto', runtime), expected)

    def test_entrypoint_dry_run_does_not_import_torch(self):
        root = Path(__file__).resolve().parent
        code = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'deep_ep'):
            raise AssertionError('Unexpected device dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['test_ep_auto_compat.py', '--dry-run', '--backend', 'official', '--bs', '2',
            '--alignment', '1', '--deferred', 'false', '--deterministic', 'false',
            '--route', 'random', '--edge-routes', 'off']
runpy.run_path('test_ep_auto_compat.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['num_cases'], 1)
        self.assertEqual(plan['backend'], 'official')
        self.assertEqual(plan['path_status'], 'unverified')

    def test_capture_setup_failure_never_submits_communication(self):
        class Capture:
            def __enter__(self):
                raise OSError('trace directory unavailable')
        collector = SimpleNamespace(capture=lambda *args, **kwargs: Capture())
        calls = []
        buffer = SimpleNamespace(dispatch=lambda **kwargs: calls.append(kwargs))
        case = plan_cases(parse_args(['--bs', '2', '--edge-routes', 'off']))[0]
        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        with patch.dict(sys.modules, torch=fake_torch), patch.object(suite, 'agree', lambda check: check()):
            with self.assertRaisesRegex(OSError, 'trace directory'):
                suite.invoke(buffer, 'dispatch', 'fresh', case, 0, 'candidate-auto',
                             dict(arch=3510, num_vec_cores=64), collector, x=None)
        self.assertEqual(calls, [])


class DispatchOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('Torch CPU is needed only for synthetic tensor oracle tests')
        cls.torch = torch

    def fixture(self, dtype='bf16', layout='row', backend='official'):
        torch = self.torch
        from low_latency_reference import payload
        world, rank, alignment, capacity = 8, 0, 16, 256
        # One source sends three slots to rank 0, two to the same expert.
        routes = [[[0, 0, 1, -1, 16, 32, 48, 64]]] + [
            [[16, 17, 18, 19, 20, 21, 22, 23]] for _ in range(7)]
        values = torch.zeros((32, 7168), dtype=torch.bfloat16)
        row = payload([0], [0], 7168, torch.bfloat16)[0]
        for slot in (0, 1, 16):
            values[slot] = row
        weights = torch.zeros(32)
        weights[0], weights[1], weights[16] = 1 / 32, 2 / 32, 3 / 32
        recv = values
        if dtype == 'fp8':
            sf = torch.zeros((32, 112), dtype=torch.int16)
            for slot in (0, 1, 16):
                sf[slot] = suite.scale_packs([0], [0])[0]
            if layout == 'col':
                sf = sf.t().contiguous().t()
            recv = (values.to(torch.float8_e4m3fn), sf)
        handle = SimpleNamespace(
            num_max_tokens_per_rank=capacity, expert_alignment=alignment,
            num_unaligned_recv_tokens_per_expert=torch.tensor([2, 1] + [0] * 14),
            num_expanded_tokens=32, num_recv_tokens=1,
            num_recv_tokens_per_expert_list=[16, 16] + [0] * 14,
            psum_num_recv_tokens_per_expert=torch.tensor([2, 17] + [32] * 14),
            psum_num_recv_tokens_per_rank=torch.ones(8, dtype=torch.int32),
            recv_src_metadata=torch.tensor([[0, 1, 16, -1, -1, -1, -1, -1, 0, 2]]))
        if backend != 'official':
            handle.is_low_latency = False
        options = dict(backend=backend, dtype=dtype, scale_layout=layout, cpu_sync=True)
        return recv, weights, handle, routes, rank, world, alignment, False, options

    def check_fixture(self, fixture):
        *args, options = fixture
        suite.check_dispatch(*args, **options)

    def test_valid_native_metadata_with_masked_duplicate_and_padding(self):
        for backend in ('official', 'candidate-auto'):
            self.check_fixture(self.fixture(backend=backend))

    def test_fp8_row_and_column_scales_are_checked_as_native_int16_packs(self):
        for layout in ('row', 'col'):
            fixture = self.fixture(dtype='fp8', layout=layout)
            self.check_fixture(fixture)
            fixture[0][1][16, 1] += 1
            with self.assertRaises(AssertionError):
                self.check_fixture(fixture)

    def test_padding_route_weight_and_count_corruption_are_rejected(self):
        for field in ('padding', 'weight', 'route', 'counts'):
            fixture = self.fixture()
            if field == 'padding':
                fixture[0][2, 0] = 1
            elif field == 'weight':
                fixture[1][1] = 1
            elif field == 'route':
                fixture[2].recv_src_metadata[0, 1] = 0
            else:
                fixture[2].num_recv_tokens_per_expert_list[0] = 2
            with self.assertRaises(AssertionError):
                self.check_fixture(fixture)

    def test_fp8_input_is_exact_and_cached_scale_bits_change(self):
        torch = self.torch
        original = suite.make_input(127, 256, 'fp8', 'row', device='cpu')
        updated = suite.make_input(127, 256, 'fp8', 'col', payload_scale=2, scale_version=1, device='cpu')
        torch.testing.assert_close(updated[0].float(), original[0].float() * 2, atol=0, rtol=0)
        self.assertEqual(original[1].dtype, torch.int16)
        self.assertEqual(updated[1].stride(), (1, 256))
        self.assertFalse(torch.equal(updated[1], original[1]))
        self.assertTrue(torch.isfinite(original[1].contiguous().view(torch.float32)).all().item())
        empty = suite.make_input(0, 0, 'fp8', 'col', device='cpu')
        self.assertEqual(empty[0].shape, (0, 7168))
        self.assertEqual(empty[1].shape, (0, 112))


if __name__ == '__main__':
    unittest.main()
