import os

import torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# noinspection PyUnresolvedReferences
from ._C import init_jit as _init_jit


def init_jit():
    """Initialize the JIT compilation runtime."""
    _init_jit(os.path.dirname(os.path.abspath(__file__)))


init_jit()

# Public API
from . import comm
from .comm import destroy_all_managed_hccl_comm, get_physical_domain_size, get_logical_domain_size
from .buffers.allocator import BufferAllocator
from .buffers.base import BufferBase
from .buffers.ep import EPBuffer, EPHandle
from .buffers.decode import DecodeBuffer, DecodeHandle
from .buffers.engram import EngramBuffer
from .buffers.bucket import BucketBuffer, BucketSession
from .buffers.pp import PPBuffer
# noinspection PyUnresolvedReferences
from ._C import get_num_allocation_alignment, get_num_rdma_alignment, get_num_tma_alignment, topk_idx_t
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle
from .utils.envs import init_seed, get_local_seed, get_global_seed, dist_print, init_dist
from .utils.testing import KernelProfile, bench_msprof

__version__ = '1.0.0'
