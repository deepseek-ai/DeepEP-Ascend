import functools
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .base import BufferBase
from .. import comm
from ..utils.math import align


def _accept_legacy_storage_size(init):
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # TODO: Remove the legacy keyword alias once callers migrate to num_rdma_storage_bytes.
        if 'num_rdma_storage_size' in kwargs:
            if 'num_rdma_storage_bytes' in kwargs:
                raise TypeError('Specify only one of num_rdma_storage_bytes and num_rdma_storage_size')
            kwargs['num_rdma_storage_bytes'] = kwargs.pop('num_rdma_storage_size')
        return init(self, *args, **kwargs)
    return wrapper


class EngramBuffer(BufferBase):
    """Engram receive buffer and NPU storage shards in HCCL-registered memory.

    Stream usage:
        All operations run on the current stream.
    """

    barrier = comm.barrier
    get_comm_stream = comm.get_comm_stream
    get_physical_domain_size = comm.get_physical_domain_size
    get_logical_domain_size = comm.get_logical_domain_size

    @_accept_legacy_storage_size
    def __init__(self,
                 group: dist.ProcessGroup,
                 num_gpu_bytes: int,
                 num_rdma_storage_bytes: int,
                 use_cpu_rdma_storage: bool,
                 num_allocated_qps: int,
                 qp_depth: int,
                 restrict_rd_atomic: bool = False,
                 allow_hybrid_mode: bool = True,
                 sl_idx: Optional[int] = None,
                 num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False,
                 *, use_huge1g_page: bool = True):
        """Allocate receive and storage regions using the NV 2.5 size arguments.

        ``num_rdma_storage_size`` is accepted as a legacy keyword alias for
        ``num_rdma_storage_bytes``.

        Both regions live on NPU in one HCCL allocation. CPU storage and explicit
        QP configuration are unsupported; use ``get_theoretical_config()`` to
        obtain the backend defaults. Hybrid mode is a compatibility preference.
        """
        if use_cpu_rdma_storage:
            raise NotImplementedError('CPU Engram storage is not implemented on Ascend')
        if sl_idx is not None or num_allocated_qps != 0 or qp_depth != 0 or restrict_rd_atomic:
            raise NotImplementedError('RDMA service levels and QP configuration are not implemented on Ascend')
        alignment = _C.get_num_allocation_alignment()
        assert num_gpu_bytes > 0 and num_gpu_bytes % alignment == 0
        assert num_rdma_storage_bytes > 0 and num_rdma_storage_bytes % alignment == 0
        assert num_gpu_timeout_secs > 0
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.num_gpu_bytes = num_gpu_bytes
        self.num_rdma_storage_bytes = num_rdma_storage_bytes
        self.use_cpu_rdma_storage = False
        self.allow_hybrid_mode = allow_hybrid_mode
        self.num_allocated_qps = 0

        self.hccl_comm_handle = comm.get_hccl_comm_handle(group)
        super().__init__(explicitly_destroy)
        self.runtime = _C.EngramBuffer(
            self.rank_idx, self.num_ranks, self.hccl_comm_handle.get(),
            num_gpu_bytes, num_rdma_storage_bytes, use_huge1g_page,
            num_gpu_timeout_secs, explicitly_destroy)
        self.context = self.runtime.context
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()
        torch.npu.synchronize()
        dist.barrier(group)
        torch.npu.synchronize()

    def destroy(self) -> None:
        super().destroy()
        self.context = None
        self.hccl_comm_handle = None

    @staticmethod
    def get_storage_size_hint(group: dist.ProcessGroup,
                              num_entries_per_layer: List[int], hidden: int,
                              num_max_tokens: int, num_entries_per_token: int,
                              dtype: torch.dtype = torch.bfloat16,
                              num_sf_packs: int = 0) -> Tuple[int, int]:
        """Return allocation-aligned receive/SF and per-rank storage sizes."""
        assert num_entries_per_layer and all(num_entries > 0 for num_entries in num_entries_per_layer)
        assert hidden > 0 and num_max_tokens > 0 and num_entries_per_token > 0 and num_sf_packs >= 0
        assert dtype in (torch.bfloat16, torch.float8_e4m3fn)
        assert (dtype == torch.float8_e4m3fn) == (num_sf_packs > 0)
        alignment = _C.get_num_allocation_alignment()
        num_recv_bytes = len(num_entries_per_layer) * num_max_tokens * num_entries_per_token * hidden * dtype.itemsize
        num_sf_bytes = sum(num_entries_per_layer) * num_sf_packs * torch.int16.itemsize
        num_storage_bytes = sum(num_entries_per_layer) * hidden * dtype.itemsize
        return align(num_recv_bytes + num_sf_bytes, alignment), align(num_storage_bytes, alignment)

    @staticmethod
    def get_theoretical_config(group: dist.ProcessGroup,
                               num_layers: int,
                               num_max_tokens: int,
                               num_entries_per_token: int,
                               num_max_read_requests_per_qp: int = 16) -> Tuple[int, int, bool]:
        """Return ``(0, 0, False)`` for Ascend's automatic backend configuration."""
        return 0, 0, False

    def set_config(self, num_entries_per_layer: List[int], hidden: int,
                   num_max_tokens: int, num_entries_per_token: int,
                   dtype: torch.dtype = torch.bfloat16,
                   num_sf_packs: int = 0) -> None:
        """Configure local storage shards and the maximum fetch geometry."""
        self.get_storage_size_hint(self.group, num_entries_per_layer, hidden,
                                   num_max_tokens, num_entries_per_token, dtype, num_sf_packs)
        self.runtime.set_config(num_entries_per_layer, hidden, dtype.itemsize,
                                num_max_tokens, num_entries_per_token, num_sf_packs)

    def write(self, storages: List[torch.Tensor],
              sfs: Optional[List[torch.Tensor]] = None) -> None:
        """Copy CPU/NPU local storage into NPU tables, plus each rank's NPU SF shard.

        CPU inputs are checkpoint staging only; runtime storage remains NPU-backed.
        """
        self.runtime.write(storages, sfs)

    def fetch(self, indices: torch.Tensor, num_qps: int = 0,
              use_tma_aligned_col_major_sf: bool = True) -> List[Callable]:
        """Fetch all layers and return one result hook per layer.

        ``indices`` is int32 with shape ``[num_layers, num_tokens, num_entries_per_token]``.
        SF output can be column-major; Ascend does not add CUDA TMA padding.
        """
        if num_qps != 0:
            raise NotImplementedError('Explicit QP counts are not implemented on Ascend')
        return self.runtime.fetch(indices, num_qps, use_tma_aligned_col_major_sf)
