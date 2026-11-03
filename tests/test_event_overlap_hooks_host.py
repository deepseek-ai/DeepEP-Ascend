"""Exercise the real Python wrapper without loading the native NPU extension."""

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


class FakeEvent:
    def __init__(self):
        self.wait_count = 0

    def current_stream_wait(self):
        self.wait_count += 1


class EventOverlapHookTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Load PyTorch before sys.modules isolation; native modules cannot be reloaded.
        __import__("torch")
        root = Path(__file__).resolve().parents[1]
        extension = ModuleType("deep_ep._C")
        extension.EventHandle = FakeEvent
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "deep_ep": ModuleType("deep_ep"),
                    "deep_ep._C": extension,
                },
            )
        )
        spec = importlib.util.spec_from_file_location(
            "_event_host", root / "deep_ep/utils/event.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.event = FakeEvent()
        self.overlap = module.EventOverlap(self.event)

    def test_hook_returns_result_and_runs_once(self):
        hook = Mock(return_value="done")
        self.overlap.register_hook_after_wait(hook)
        self.assertEqual(self.overlap.wait(), "done")
        self.assertIsNone(self.overlap.wait())
        hook.assert_called_once_with()
        self.assertEqual(self.event.wait_count, 2)

    def test_reentrant_wait_does_not_repeat_hook(self):
        calls = []

        def hook():
            calls.append("hook")
            # Guard keeps the unfixed implementation from recursing indefinitely.
            if len(calls) == 1:
                self.overlap.wait()

        self.overlap.register_hook_after_wait(hook)
        self.overlap.wait()
        self.assertEqual(calls, ["hook"])
        self.assertEqual(self.event.wait_count, 2)

    def test_failing_hook_is_not_retried(self):
        error = RuntimeError("partially applied epilogue")
        hook = Mock(side_effect=error)
        self.overlap.register_hook_after_wait(hook)
        with self.assertRaises(RuntimeError) as caught:
            self.overlap.wait()
        self.assertIs(caught.exception, error)
        self.assertIsNone(self.overlap.hook_after_wait)
        self.assertIsNone(self.overlap.wait())
        hook.assert_called_once_with()

    def test_hook_may_register_a_successor(self):
        successor = Mock(return_value="second")

        def first():
            self.overlap.register_hook_after_wait(successor)
            return "first"

        self.overlap.register_hook_after_wait(first)
        self.assertEqual(self.overlap.wait(), "first")
        successor.assert_not_called()
        self.assertEqual(self.overlap.wait(), "second")
        self.assertIsNone(self.overlap.hook_after_wait)

    def test_duplicate_registration_is_still_rejected(self):
        self.overlap.register_hook_after_wait(lambda: None)
        with self.assertRaises(AssertionError):
            self.overlap.register_hook_after_wait(lambda: None)

    def test_successful_wait_can_release_handle(self):
        self.overlap.register_hook_after_wait(lambda: "done")
        self.assertEqual(self.overlap.current_stream_wait(release_handle=True), "done")
        self.assertIsNone(self.overlap.event)

    def test_native_wait_failure_does_not_consume_hook(self):
        hook = Mock()
        self.overlap.register_hook_after_wait(hook)
        with (
            patch.object(
                self.event,
                "current_stream_wait",
                side_effect=RuntimeError("wait failed"),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.overlap.wait()
        self.assertIs(self.overlap.hook_after_wait, hook)
        hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
