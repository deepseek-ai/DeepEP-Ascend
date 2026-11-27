from typing import Optional

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .base import BufferBase
from .. import comm


class PPBuffer(BufferBase):
    """Own contiguous symmetric storage for pipeline-parallel send/recv.

    Stream usage:
        All operations run on the current stream.
    """

    barrier = comm.barrier
    get_comm_stream = comm.get_comm_stream

    def __init__(self,
                 group: dist.ProcessGroup,
                 num_max_tensor_bytes: int,
                 num_max_inflight_tensors: int,
                 sl_idx: Optional[int] = None,
                 num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False,
                 *, use_huge1g_page: bool = True):
        """Create the storage and its communication group.

        Arguments:
            group: the communication group.
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.
            sl_idx: the service level index for RDMA traffic. ``None`` uses the default.
            num_gpu_timeout_secs: the NPU-side timeout in seconds.
            explicitly_destroy: whether the user must call ``destroy()`` explicitly.
        """
        if sl_idx is not None:
            raise NotImplementedError('RDMA service levels are not implemented on Ascend')
        assert num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0 and num_gpu_timeout_secs > 0
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        super().__init__(explicitly_destroy)

        # Create CPP runtime
        self.runtime = _C.PPBuffer(
            group.rank(), group.size(), group,
            num_max_tensor_bytes, num_max_inflight_tensors,
            use_huge1g_page, num_gpu_timeout_secs, explicitly_destroy)
        self.context = self.runtime.context
        self.storage = self.runtime.storage
        torch.npu.synchronize()
        dist.barrier(group)
        torch.npu.synchronize()

    def send(self, x: torch.Tensor, dst_rank_idx: int, num_sms: Optional[int] = None) -> None:
        """Send a tensor to an adjacent rank in the pipeline-parallel ring.

        Arguments:
            x: the contiguous NPU tensor to send; its size must not exceed ``num_max_tensor_bytes``.
            dst_rank_idx: the destination rank, which must be the previous or next rank in the ring.
            num_sms: the number of vector cores to use on Ascend. ``None`` or 0 selects all vector cores.
        """
        self.runtime.send(x, dst_rank_idx, 0 if num_sms is None else num_sms)

    def recv(self, x: torch.Tensor, src_rank_idx: int, num_sms: Optional[int] = None) -> None:
        """Receive a tensor from an adjacent rank in the pipeline-parallel ring.

        Arguments:
            x: the contiguous NPU output tensor to receive into; its size must not exceed the
                configured ``num_max_tensor_bytes``.
            src_rank_idx: the source rank, which must be the previous or next rank in the ring.
            num_sms: the number of vector cores to use on Ascend. ``None`` or 0 selects all vector cores.
        """
        self.runtime.recv(x, src_rank_idx, 0 if num_sms is None else num_sms)

    def destroy(self) -> None:
        super().destroy()
        self.storage = None
        self.context = None
