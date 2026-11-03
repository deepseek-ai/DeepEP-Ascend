"""Exercise reference dispatch with CPU tensors and single-rank collective doubles."""

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
import torch.distributed as dist


class ReferenceDispatchTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(__file__).resolve().parents[1]
        package = ModuleType("deep_ep")
        package.__path__ = [str(root / "deep_ep")]
        utils = ModuleType("deep_ep.utils")
        utils.__path__ = [str(root / "deep_ep/utils")]
        stack.enter_context(
            patch.dict(sys.modules, {"deep_ep": package, "deep_ep.utils": utils})
        )
        spec = importlib.util.spec_from_file_location(
            "deep_ep.utils._refs_host", root / "deep_ep/utils/refs.py"
        )
        self.refs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.refs)
        self.exchanges = []

        def exchange(output, input, output_split_sizes=None, input_split_sizes=None):
            self.assertEqual(output_split_sizes, input_split_sizes)
            if input_split_sizes is not None:
                self.assertEqual(input_split_sizes, [input.shape[0]])
            self.exchanges.append(input)
            output.copy_(input)

        stack.enter_context(patch.object(dist, "get_rank", return_value=0))
        stack.enter_context(patch.object(dist, "get_world_size", return_value=1))
        stack.enter_context(
            patch.object(dist, "all_to_all_single", side_effect=exchange)
        )
        self.x = torch.arange(12, dtype=torch.float32).view(3, 4)
        self.indices = torch.tensor([[0, 1], [2, -1], [-1, -1]], dtype=torch.int64)
        self.weights = torch.tensor([[0.25, 0.75], [1.0, 0.0], [0.0, 0.0]])

    def check_dispatch(self, weights, packed=False, empty=False):
        scale = torch.arange(6, dtype=torch.float32).view(3, 2)
        x = (self.x, scale) if packed else self.x
        indices = torch.full_like(self.indices, -1) if empty else self.indices
        data, received_indices, received_weights, source_indices, counts = (
            self.refs.dispatch(x, indices, weights, 3, 4)
        )
        rows = 0 if empty else 2
        if packed:
            torch.testing.assert_close(data[0], self.x[:rows])
            torch.testing.assert_close(data[1], scale[:rows])
        else:
            torch.testing.assert_close(data, self.x[:rows])
        torch.testing.assert_close(received_indices, indices[:rows])
        torch.testing.assert_close(
            source_indices, torch.arange(rows, dtype=torch.int32)
        )
        torch.testing.assert_close(counts, torch.tensor([rows], dtype=torch.int32))
        self.assertEqual(
            len(self.exchanges), 4 + int(packed) + int(weights is not None)
        )
        if weights is None:
            self.assertIsNone(received_weights)
        else:
            torch.testing.assert_close(received_weights, weights[:rows])

    def test_dispatch_without_weights(self):
        self.check_dispatch(None)

    def test_dispatch_with_weights(self):
        self.check_dispatch(self.weights)

    def test_packed_dispatch_without_weights(self):
        self.check_dispatch(None, packed=True)

    def test_packed_dispatch_with_weights(self):
        self.check_dispatch(self.weights, packed=True)

    def test_no_routed_tokens_without_weights(self):
        self.check_dispatch(None, empty=True)

    def test_no_routed_tokens_with_weights(self):
        self.check_dispatch(self.weights, empty=True)


if __name__ == "__main__":
    unittest.main()
