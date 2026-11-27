from enum import IntEnum

RECOVERABLE_MASK = 1 << 31


class ErrorStatus(IntEnum):
    """Error recorded by a DeepEP kernel; bit 31 marks a recoverable error."""
    NONE = 0

    CQE_POLL_TIMEOUT = RECOVERABLE_MASK | 1
    CQE_ERROR = RECOVERABLE_MASK | 2
    BARRIER_TIMEOUT = RECOVERABLE_MASK | 3

    INVALID_COMM_METADATA = 4
    SQ_CAPACITY_EXCEEDED = 5
    SQ_ALIGNMENT_ERROR = 6


__all__ = [
    'ErrorStatus',
    'RECOVERABLE_MASK',
]
