"""Host-only checks for the optional MTE stream test's routing and oracle.

Pure helpers are extracted from test_stream.py without importing deep_ep,
Torch NPU, or its distributed runner. The data-oracle test uses CPU Torch when
available; routing and configuration checks require only the standard library.
"""

import argparse
import ast
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_helpers():
    source = Path(__file__).with_name("test_stream.py")
    names = {"mte_routes", "validate_config", "configure_backend", "check_mte_dispatch", "require_device_delay"}
    nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class StreamHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_helpers()

    def test_routes_are_valid_distinct_and_balanced(self):
        for world in (8, 16, 32, 64, 128):
            for bs in (1, 16, 64 if world == 8 else 32):
                with self.subTest(world=world, bs=bs):
                    received = [0] * world
                    for source in range(world):
                        routes = self.helpers["mte_routes"](source, world, bs)
                        self.assertEqual(len(routes), bs)
                        for experts in routes:
                            self.assertEqual(len(experts), 8)
                            self.assertEqual(len(set(experts)), 8)
                            self.assertTrue(all(0 <= expert < 128 for expert in experts))
                            ranks = [expert // (128 // world) for expert in experts]
                            self.assertEqual(len(set(ranks)), 8)
                            for rank in ranks:
                                received[rank] += 1
                    self.assertEqual(received, [8 * bs] * world)

    def test_mte_gates_and_fallback(self):
        validate = self.helpers["validate_config"]
        for world in (8, 16, 32, 64, 128):
            limit = 64 if world == 8 else 32
            for bs in (1, limit):
                for capacity in (bs, 256):
                    validate(SimpleNamespace(ep_shape="mte", ep_bs=bs, ep_capacity=capacity), world)
            for bs, capacity in ((0, 16), (limit + 1, 256), (16, 15), (16, 257)):
                with self.subTest(world=world, bs=bs, capacity=capacity):
                    with self.assertRaises(ValueError):
                        validate(SimpleNamespace(ep_shape="both", ep_bs=bs, ep_capacity=capacity), world)
        for world in (1, 2, 4, 256):
            with self.assertRaises(ValueError):
                validate(SimpleNamespace(ep_shape="mte", ep_bs=16, ep_capacity=16), world)
            validate(SimpleNamespace(ep_shape="fallback", ep_bs=16, ep_capacity=16), world)

    def test_backend_environment_is_explicit(self):
        with patch.dict(os.environ, {"EP_DISABLE_LOW_LATENCY_TRANSPORT": "stale"}):
            self.helpers["configure_backend"]("official")
            self.assertEqual(os.environ["EP_DISABLE_LOW_LATENCY_TRANSPORT"], "stale")
            for backend, expected in (("candidate-disabled", "1"), ("candidate-auto", None), ("official", None)):
                self.helpers["configure_backend"](backend)
                self.assertEqual(os.environ.get("EP_DISABLE_LOW_LATENCY_TRANSPORT"), expected)

    def test_delay_preflight_rejects_missing_or_noncallable_api(self):
        for torch in (SimpleNamespace(), SimpleNamespace(npu=SimpleNamespace()),
                      SimpleNamespace(npu=SimpleNamespace(_sleep=None)),
                      SimpleNamespace(npu=SimpleNamespace(_sleep=42))):
            with self.subTest(torch=torch), patch.dict(self.helpers, torch=torch):
                with self.assertRaisesRegex(RuntimeError, r'callable torch\.npu\._sleep'):
                    self.helpers['require_device_delay']()

    def test_delay_preflight_does_not_submit_device_work(self):
        delay = Mock(side_effect=AssertionError('Preflight must not enqueue a delay'))
        with patch.dict(self.helpers, torch=SimpleNamespace(npu=SimpleNamespace(_sleep=delay))):
            self.helpers['require_device_delay']()
        delay.assert_not_called()

    def test_worker_checks_delay_before_initializing_distributed(self):
        source = Path(__file__).with_name('test_stream.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'test')
        # Execute the real worker body without the Torch inference decorator.
        worker.decorator_list = []
        for available in (False, True):
            init_dist = Mock(side_effect=LookupError('Reached distributed initialization'))
            configure = Mock()
            torch = SimpleNamespace(npu=SimpleNamespace(_sleep=Mock() if available else None))
            namespace = dict(self.helpers, init_dist=init_dist, configure_backend=configure)
            exec(compile(ast.Module(body=[worker], type_ignores=[]), str(source), 'exec'), namespace)
            with self.subTest(available=available), patch.dict(self.helpers, torch=torch):
                error = LookupError if available else RuntimeError
                message = 'Reached distributed initialization' if available else r'callable torch\.npu\._sleep'
                with self.assertRaisesRegex(error, message):
                    namespace['test'](0, 2, SimpleNamespace(backend='official'))
            if available:
                init_dist.assert_called_once_with(0, 2)
            else:
                init_dist.assert_not_called()
                configure.assert_not_called()

    def test_parent_checks_delay_before_spawning_workers(self):
        source = Path(__file__).with_name('test_stream.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "__name__ == '__main__'")
        for available in (False, True):
            spawn, delay, worker = Mock(), Mock(), object()
            torch = SimpleNamespace(npu=SimpleNamespace(_sleep=delay if available else None),
                                    multiprocessing=SimpleNamespace(spawn=spawn))
            namespace = dict(self.helpers, __name__='__main__', argparse=argparse,
                             torch=torch, test=worker)
            errors = io.StringIO()
            with self.subTest(available=available), patch.dict(self.helpers, torch=torch), \
                    patch.object(sys, 'argv', ['test_stream.py']), redirect_stderr(errors):
                if available:
                    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), namespace)
                else:
                    with self.assertRaises(SystemExit) as caught:
                        exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), namespace)
                    self.assertEqual(caught.exception.code, 2)
            if available:
                spawn.assert_called_once_with(worker, args=(2, namespace['args']), nprocs=2)
            else:
                spawn.assert_not_called()
                self.assertIn('callable torch.npu._sleep', errors.getvalue())
            delay.assert_not_called()

    def test_explicit_backend_registration_precedes_extension_import(self):
        source = Path(__file__).with_name('test_stream.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        imports = {alias.name: index for index, node in enumerate(tree.body) if isinstance(node, ast.Import)
                   for alias in node.names}
        self.assertIn('torch_npu', imports)
        self.assertLess(imports['torch'], imports['torch_npu'])
        self.assertLess(imports['torch_npu'], imports['deep_ep'])

    def test_cpu_oracle_accepts_shuffled_padding_and_rejects_corruption(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU Torch is unavailable; routing/configuration tests still run")
        self.helpers["torch"] = torch
        for world in (8, 16, 32, 64, 128):
            with self.subTest(world=world):
                bs, capacity, iteration, rank = 2, 7, 1, world - 1
                local_experts = 128 // world
                entries, rank_counts, expert_counts = [], [0] * world, [0] * local_experts
                for source in range(world):
                    for token in range(bs):
                        for k in range(8):
                            if (source + token + k) % world == rank:
                                entries.append((source, token, k))
                                rank_counts[source] += 1
                                expert_counts[((source + token) // world) % local_experts] += 1
                entries.reverse()
                count = len(entries)
                x = torch.full((count + 4, 7168), -999, dtype=torch.bfloat16, device="cpu")
                weights = torch.full((count + 4,), -999, dtype=torch.float32, device="cpu")
                metadata = torch.full((count + 4, 10), -1, dtype=torch.int32, device="cpu")
                for index, (source, token, k) in enumerate(entries):
                    slot = (index + 3) % count
                    x[slot].fill_(source + 10 * iteration)
                    x[slot, 0] = token
                    weights[slot] = k + 1
                    metadata[index, k] = slot
                    metadata[index, -2] = source * capacity + token
                    metadata[index, -1] = k
                handle = SimpleNamespace(
                    topk_idx=torch.tensor(self.helpers["mte_routes"](rank, world, bs), device="cpu"),
                    psum_num_recv_tokens_per_rank=torch.tensor(rank_counts, device="cpu").cumsum(0),
                    num_unaligned_recv_tokens_per_expert=torch.tensor(expert_counts, device="cpu"),
                    psum_num_recv_tokens_per_expert=torch.tensor(expert_counts, device="cpu").cumsum(0),
                    recv_src_metadata=metadata)
                check = self.helpers["check_mte_dispatch"]
                check(x, None, weights, handle, rank, world, bs, capacity, iteration)
                x[0, 7] = -1
                with self.assertRaises(AssertionError):
                    check(x, None, weights, handle, rank, world, bs, capacity, iteration)


if __name__ == "__main__":
    unittest.main()
