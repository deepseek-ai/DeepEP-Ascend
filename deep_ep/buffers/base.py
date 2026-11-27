from abc import ABC, abstractmethod
from typing import Optional

from deep_ep.comm.errors import ErrorStatus, RECOVERABLE_MASK


class BufferBase(ABC):
    """Common lifecycle for communication buffers."""

    def __init__(self, explicitly_destroy: bool = False):
        self.explicitly_destroy = explicitly_destroy
        self.runtime = None

    @property
    def main_context(self):
        """The first communication context, or ``None`` before configuration or after destruction."""
        return self.runtime.main_context if self.runtime is not None else None

    def print_memory_usage(self) -> None:
        """Print this rank's workspace, communication buffer, and RDMA storage sizes."""
        assert self.runtime is not None
        self.runtime.print_memory_usage()

    def _read_error_status(self) -> list[ErrorStatus]:
        if self.runtime is None:
            raise RuntimeError('the buffer has been destroyed')
        return [ErrorStatus(code) for code in self.runtime.get_error_status()]

    def get_error_status(self) -> Optional[list[ErrorStatus]]:
        """Return one status per context, ordered by context, or ``None`` when nothing is recorded.

        The list index is the context index. The query is a plain host memory read and does not
        synchronize the device.
        """
        statuses = self._read_error_status()
        return statuses if any(status != ErrorStatus.NONE for status in statuses) else None

    def get_recoverable_error_status(self) -> Optional[list[ErrorStatus]]:
        """Like ``get_error_status``, but unrecoverable entries are reported as ``NONE``.

        Returns ``None`` when no context holds a recoverable error.
        """
        statuses = [status if status & RECOVERABLE_MASK else ErrorStatus.NONE
                    for status in self._read_error_status()]
        return statuses if any(status != ErrorStatus.NONE for status in statuses) else None

    def clear_error_status(self) -> None:
        """Clear the error record of every context."""
        if self.runtime is None:
            raise RuntimeError('the buffer has been destroyed')
        self.runtime.clear_error_status()

    @abstractmethod
    def destroy(self) -> None:
        """Destroy the runtime. Requires ``explicitly_destroy=True`` at construction."""
        assert self.explicitly_destroy
        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None
