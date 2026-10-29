"""BF16 decode convenience wrapper around one native EPBuffer.

Collective contract: all ranks use the same shapes, options and call order on
homogeneous devices. Routes must contain distinct, valid expert IDs per token.
Routing weights are returned as metadata; combine only sums expert outputs.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from .ep import EPBuffer, EPHandle
from ..utils.event import EventOverlap


@dataclass(frozen=True)
class DecodeHandle:
    """Routing state owned by a DecodeBuffer; pass unchanged to combine.

    Outputs are grouped by local expert. ``expert_recv_counts`` is an int64
    device tensor; only the leading sum(counts) rows are valid. CPU counts are
    populated only when dispatch requests ``do_cpu_sync=True``. The public
    ``recv_topk_weights`` tensor contains the weight of each expanded expert
    row; the expert computation applies it if the model requires weighting.
    ``backend`` is always ``'ep'``; native transport selection is internal.
    """

    expert_recv_counts: torch.Tensor
    num_recv_tokens_per_expert_list: list
    num_expanded_tokens: int
    recv_topk_weights: Optional[torch.Tensor]
    backend: str
    _owner: object = field(repr=False)
    _buffer: EPBuffer = field(repr=False)
    _handle: EPHandle = field(repr=False)


class DecodeBuffer:
    """Expose a smaller BF16 decode API backed by one long-lived EPBuffer.

    Native EPBuffer selects its transport while retaining the standard expert
    layout and combine semantics. This wrapper does not select backends or
    apply routing weights. Use EPBuffer directly for cached dispatch, expert
    alignment, deterministic sorting, bias and deferred epilogues.

    The buffer is allocated collectively at construction and reused for all
    batches up to its capacity. Destroy collectively before the process group.
    """

    def __init__(self, group: dist.ProcessGroup, num_max_tokens_per_rank: int,
                 hidden: int = 7168, num_topk: int = 8, num_experts: int = 128,
                 *, num_sms: int = 32,
                 use_huge1g_page: bool = True, num_cpu_timeout_secs: int = 300,
                 num_gpu_timeout_secs: int = 100):
        if (type(num_max_tokens_per_rank) is not int or num_max_tokens_per_rank <= 0
                or hidden <= 0 or hidden % 128 or not 1 <= num_topk <= 32
                or not group.size() <= num_experts <= 1024 or num_experts % group.size()
                or num_topk > num_experts
                or num_sms < 0 or num_cpu_timeout_secs <= 0 or num_gpu_timeout_secs <= 0):
            raise ValueError('Invalid decode capacity, model shape, cores or timeout')
        self.group = group
        self.num_ranks = group.size()
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden, self.num_topk, self.num_experts = hidden, num_topk, num_experts
        self.num_sms = num_sms
        self.device = torch.device('npu', torch.npu.current_device())
        self._owner = object()
        self._destroyed = False
        self._failed = False
        self._last_event = None
        self._buffer = EPBuffer(
            group, num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden, num_topk=num_topk, explicitly_destroy=True,
            use_huge1g_page=use_huge1g_page, num_cpu_timeout_secs=num_cpu_timeout_secs,
            num_gpu_timeout_secs=num_gpu_timeout_secs)

    def _check_alive(self):
        if self._destroyed:
            raise RuntimeError('DecodeBuffer has been destroyed')
        if self._failed:
            raise RuntimeError('A decode launch failed; restart the distributed job')
        if torch.npu.current_device() != self.device.index:
            raise ValueError('Use the NPU on which this DecodeBuffer was constructed')

    def _check_tensor(self, tensor, shape, dtype, name):
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape)
                or tensor.dtype != dtype or tensor.device != self.device or not tensor.is_contiguous()):
            raise ValueError(f'{name} must be a contiguous {shape} {dtype} tensor on {self.device}')

    def _begin(self):
        # Order buffer reuse if the caller changes the current stream.
        if self._last_event is not None:
            self._last_event.current_stream_wait()

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                 *, do_cpu_sync: bool = False):
        """Return ``(expert_x, handle, event)``; no backend-specific layout handling.

        Inputs: BF16 [B,H], int64 [B,K] valid unique expert IDs, FP32 [B,K]
        weights. All ranks must use identical B/H/K/E and options. The call does
        not inspect device route values or add a host agreement collective.
        Keep route tensors unchanged while the handle is in use. All work is
        enqueued on the current stream. First use can compile the kernels;
        warm up each required shape before timing. Graph capture
        is not currently supported by this adapter.
        """
        self._check_alive()
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError('x must have shape [B,H]')
        bs = x.shape[0]
        if not 0 < bs <= self.num_max_tokens_per_rank:
            raise ValueError('Require 0 < B <= num_max_tokens_per_rank')
        self._check_tensor(x, (bs, self.hidden), torch.bfloat16, 'x')
        self._check_tensor(topk_idx, (bs, self.num_topk), torch.int64, 'topk_idx')
        self._check_tensor(topk_weights, (bs, self.num_topk), torch.float32, 'topk_weights')
        self._begin()
        try:
            recv, _, weights, handle, event = self._buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=self.num_experts,
                num_max_tokens_per_rank=self.num_max_tokens_per_rank,
                expert_alignment=1, num_sms=self.num_sms,
                do_expand=True, do_cpu_sync=do_cpu_sync,
                async_with_compute_stream=True)
            counts = handle.num_unaligned_recv_tokens_per_expert.to(torch.int64)
            # The count conversion must be included in the public event.
            event = EventOverlap(EPBuffer.capture(), (recv, counts, weights, handle, event))
            result = DecodeHandle(counts, handle.num_recv_tokens_per_expert_list,
                                  handle.num_expanded_tokens, weights, 'ep', self._owner, self._buffer, handle)
            self._last_event = event.event
        except Exception:
            self._failed = True
            raise
        return recv, result, event

    def combine(self, x: torch.Tensor, handle: DecodeHandle):
        """Return ``(combined_x, event)`` by summing expert outputs unchanged.

        ``x`` is BF16 with the dispatch output's shape/order. Apply any routing
        weights in the expert computation before this call. Handles remain
        valid after dispatching other batch sizes through the same buffer.
        Repeated combine is supported and does not modify x or the weights.
        """
        self._check_alive()
        if not isinstance(handle, DecodeHandle) or handle._owner is not self._owner:
            raise ValueError('combine requires a dispatch handle from this DecodeBuffer')
        self._check_tensor(x, (handle.num_expanded_tokens, self.hidden), torch.bfloat16, 'expert output')
        self._begin()
        try:
            stream = torch.npu.current_stream()
            # PyTorch's allocator needs the consumer stream as well as the
            # creation stream when callers discard a handle after submission.
            x.record_stream(stream)
            for value in vars(handle._handle).values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
            output, _, event = handle._buffer.combine(
                x, handle._handle, num_sms=self.num_sms, async_with_compute_stream=True)
            # Keep the originating buffer and metadata alive through completion.
            event.extra_tensors = (x, handle, event.extra_tensors)
            self._last_event = event.event
        except Exception:
            self._failed = True
            raise
        return output, event

    def destroy(self):
        """Collectively destroy the underlying EPBuffer; safe to call twice."""
        if self._destroyed:
            return
        self._begin()
        self._buffer.destroy()
        self._last_event = None
        self._destroyed = True
