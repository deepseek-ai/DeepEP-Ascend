from typing import Any, Optional, Tuple

# noinspection PyUnresolvedReferences
import deep_ep._C as _C


def get_physical_domain_size(group_or_buffer: Any) -> Tuple[int, int]:
    """Return (scale-out, direct UBMEM) domain sizes for a group or buffer.

    All group members must participate in the first topology discovery, through this
    query or buffer initialization. Later queries for that group and all buffer queries are local.
    """
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_physical_domain_size()
    return _C.get_physical_domain_size(group_or_buffer)


def get_logical_domain_size(group_or_buffer: Any,
                            allow_hybrid_mode: Optional[bool] = None) -> Tuple[int, int]:
    """Return the logical domain sizes supported by the Ascend backend.

    Group queries share the first collective discovery and cache with get_physical_domain_size.
    """
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_logical_domain_size()
    return _C.get_logical_domain_size(
        group_or_buffer, True if allow_hybrid_mode is None else allow_hybrid_mode)
