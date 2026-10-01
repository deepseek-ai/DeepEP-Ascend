import torch
from typing import Any, Callable, Optional, Tuple

# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle


class EventOverlap:
    """
    Wait for an NPU event and run a deferred epilogue once on the current stream.

    Attributes:
        event: the recorded NPU event.
        extra_tensors: tensors kept alive by this wrapper.
    """

    def __init__(self, event: Optional[EventHandle] = None,
                 extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the recorded NPU event.
            extra_tensors: tensors kept alive by this wrapper.
        """
        self.event = event

        self.extra_tensors = extra_tensors

        # A wrapper for `with event_overlap(release_handle=True)`
        self._release_handle_by_call = False

        # A hook that will be triggered after `current_stream_wait()`
        self.hook_after_wait: Optional[Callable] = None

    def current_stream_wait(self, release_handle: bool = False) -> Any:
        """
        Make the current stream wait for the event, then enqueue the registered epilogue.

        Returns:
            result: the return value of the registered hook, or `None` if no hook is registered.
        """
        assert self.event is not None
        self.event.current_stream_wait()

        # Call epilogue hook
        result = None
        if self.hook_after_wait is not None:
            # Consume the hook before calling it, including on reentry or failure.
            hook_after_wait = self.hook_after_wait
            self.hook_after_wait = None
            result = hook_after_wait()

        # Release event handle
        if release_handle:
            self.event = None
        return result

    def wait(self) -> Any:
        """Wait on the current stream and return the registered epilogue result."""
        return self.current_stream_wait()

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        """
        Register a hook to invoke after `current_stream_wait()`.
        """
        assert self.hook_after_wait is None, 'A hook is already registered on this `EventOverlap`'
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """
        Configures the 'release_handle' behavior for the upcoming context manager usage.
        Usage:
            with event_overlap(release_handle=True):
                ...
        Returns `self` to ensure no new wrapper object is created, keeping the reference count
        of the underlying event managed solely by this instance.
        """
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> Any:
        """
        Enter a scope whose exit waits for the event and runs the epilogue on the current stream.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False
