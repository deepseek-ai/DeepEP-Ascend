import functools
import os
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C
# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle

from .allocator import BufferAllocator
from .base import BufferBase
from .. import comm
from ..utils.event import EventOverlap
from ..utils.math import align
from ..utils.semantic import value_or


class EPHandle:
    """
    Communication handle returned by `EPBuffer.dispatch`.
    Can be reused as a cached handle in subsequent `EPBuffer.dispatch` calls to skip layout
    recomputation, and is consumed by `EPBuffer.combine` to reverse the token routing.

    Attributes:
        is_low_latency: whether the handle belongs to the decode kernels. Low-latency
            handles expose int64 device counts as ``expert_recv_counts`` and private
            reverse-routing tables; throughput-only metadata fields remain None.
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        num_experts: the number of all experts.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the AI core count used during dispatch (reused in cached dispatch and combine).
        topk_idx: top-k expert indices from dispatch, `[num_tokens, num_topk]`.
            Must not be modified while this handle is in use. PyTorch version counters detect ordinary
            in-place changes, including through views, but not writes through `.data` or raw pointers.
            Tensors created in inference mode have no version counter and cannot be checked.
        psum_num_recv_tokens_per_rank: inclusive prefix sum of deduplicated received token counts
            per source rank, shape `[num_ranks]`. A token is counted once per rank even if multiple
            top-k experts land on the same rank.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_local_experts]`. `psum[i]` equals the aligned cumulative count of
            experts before `i` plus the actual (unaligned) token count of expert `i`.
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_local_experts]` with `torch.int`.
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        recv_src_metadata: metadata rows [topk slots..., global_idx, master_idx] produced by dispatch epilogue.
        dst_buffer_slot_idx: destination rank-buffer slot indices, shape `[num_tokens, num_topk]`.
        dst_gsge_idx: SGE-pair indices within each vector core, shape `[num_tokens, num_topk]`.
            Local copies and deduplicated or masked entries are `-1`.
        num_recv_tokens: the total number of received tokens.
    """

    def __init__(self,
                 do_expand: bool,
                 num_experts: int, expert_alignment: int,
                 num_max_tokens_per_rank: int,
                 num_sms: int,
                 topk_idx: torch.Tensor,
                 num_recv_tokens: int,
                 num_expanded_tokens: int,
                 num_recv_tokens_per_expert_list: list,
                 psum_num_recv_tokens_per_rank: torch.Tensor,
                 psum_num_recv_tokens_per_expert: torch.Tensor,
                 num_unaligned_recv_tokens_per_expert: torch.Tensor,
                 recv_src_metadata: torch.Tensor,
                 dst_buffer_slot_idx: torch.Tensor,
                 dst_gsge_idx: torch.Tensor):
        assert topk_idx is not None

        self.do_expand = do_expand
        self.is_low_latency = False
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.psum_num_recv_tokens_per_rank = psum_num_recv_tokens_per_rank
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.recv_src_metadata = recv_src_metadata
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        self.dst_gsge_idx = dst_gsge_idx

        # May not be accurate without host sync
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens

        # For deterministic dispatch
        self.cached_recv_src_metadata_before_sort = None

        # `topk_idx` cannot change in the handle
        self._topk_idx = topk_idx
        self._topk_idx_version = None if topk_idx.is_inference() else topk_idx._version

    @property
    def topk_idx(self) -> torch.Tensor:
        assert self._topk_idx_version is None or self._topk_idx._version == self._topk_idx_version, \
            '`topk_idx` must not be modified while the EP handle is in use'
        return self._topk_idx

    def deterministic_sort(self,
                           do_cpu_sync: bool,
                           is_cached_dispatch: bool,
                           recv_x: torch.Tensor,
                           recv_sf: Optional[torch.Tensor],
                           recv_topk_weights: Optional[torch.Tensor]) -> None:
        """
        Sort expanded received tensors by source token to make dispatch deterministic.

        Ascend dispatch is expand-only: `recv_x` and `recv_topk_weights` are grouped by expert,
        while `recv_src_metadata[:, :-2]` points back to those expanded slots. We sort only the
        expanded arrays and rewrite the metadata slots; `recv_src_metadata` itself keeps its
        original token order.
        """
        # `recv_src_metadata` is generated only by the non-cached dispatch. Cache it before
        # rewriting slot indices so cached dispatch can reuse the deterministic layout.
        if not is_cached_dispatch:
            self.cached_recv_src_metadata_before_sort = self.recv_src_metadata.clone()
        assert self.cached_recv_src_metadata_before_sort is not None
        if is_cached_dispatch:
            # Cached copy epilogue already writes into the deterministic slots recorded above.
            return

        # Build expert-major sort keys. Valid tokens get a negative offset plus source token
        # index, so they sort before padding and remain ordered within each expert.
        src_token_global_index_max_x2 = 10000000000
        expert_token_idx_start = self.psum_num_recv_tokens_per_expert - self.num_unaligned_recv_tokens_per_expert
        token_idx2expert_idx = torch.bucketize(
            torch.arange(recv_x.shape[0], device=recv_x.device),
            expert_token_idx_start[1:], right=True, out_int32=False)
        sort_keys = token_idx2expert_idx * src_token_global_index_max_x2

        slots = self.cached_recv_src_metadata_before_sort[:, :-2]
        src_token_global_idx = self.cached_recv_src_metadata_before_sort[:, -2]
        valid_mask = slots >= 0
        if not do_cpu_sync:
            # Without host sync the metadata tensor may contain worst-case trailing rows.
            num_recv_tokens = self.psum_num_recv_tokens_per_rank[-1]
            oob_tokens_mask = torch.arange(slots.shape[0], device=slots.device) >= num_recv_tokens
            valid_mask[oob_tokens_mask] = False
        sort_keys.scatter_add_(
            0, slots[valid_mask].long(),
            -src_token_global_index_max_x2 // 2 +
            src_token_global_idx.unsqueeze(1).expand_as(slots)[valid_mask].to(torch.int64))

        # Apply the permutation in-place and rewrite metadata slots for combine and cached dispatch.
        orig_slot_idx = torch.sort(sort_keys, stable=True).indices
        def permute(tensor: Optional[torch.Tensor]) -> None:
            if tensor is not None:
                data = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
                data.copy_(data[orig_slot_idx])

        permute(recv_x)
        permute(recv_sf)
        permute(recv_topk_weights)

        dst_slot_idx = torch.empty_like(orig_slot_idx)
        dst_slot_idx[orig_slot_idx] = torch.arange(
            orig_slot_idx.shape[0], dtype=orig_slot_idx.dtype, device=orig_slot_idx.device)
        metadata_slots = self.recv_src_metadata[:, :-2]
        metadata_slots[valid_mask] = dst_slot_idx[metadata_slots[valid_mask].long()].to(metadata_slots.dtype)


class EPBuffer(BufferBase):
    """Expert-parallel dispatch/combine on Ascend, using the expanded layout.

    The native runtime selects a small-batch MTE transport where supported,
    using the same communication allocation, layouts, handles and epilogues.
    Dispatch transports routing weights as metadata; combine never multiplies
    them into the expert outputs. Apply model weights in the expert computation.

    Stream usage:
        All operations run on the current stream.
    """

    barrier = comm.barrier
    get_comm_stream = comm.get_comm_stream
    get_physical_domain_size = comm.get_physical_domain_size
    get_logical_domain_size = comm.get_logical_domain_size

    def __init__(self,
                 group: dist.ProcessGroup,
                 num_bytes: Optional[int] = None,
                 num_max_tokens_per_rank: int = 0,
                 hidden: int = 0,
                 num_topk: int = 0,
                 use_fp8_dispatch: bool = False,
                 lb_allocation_plan_or_num_bytes: Union[BufferAllocator, int] = 0,
                 deterministic: bool = False,
                 allow_hybrid_mode: bool = True,
                 allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True,
                 sl_idx: Optional[int] = None,
                 num_allocated_qps: int = 0,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False,
                 *, use_huge1g_page: bool = True, low_latency_mode: bool = False):
        """Create an EP buffer with the NV 2.5 argument order.

        Ascend uses a single HCCL domain. Hybrid and
        overlap preferences are compatibility options. LB storage can be allocated;
        LB kernels, RDMA service levels and explicit QP counts are unsupported. ``num_bytes`` excludes
        workspace and must be allocation-aligned.

        The default path automatically selects its internal transport without
        changing the public API or allocating a second communication buffer.

        ``low_latency_mode=True`` selects the legacy experimental decode kernels (Ascend 950/A5,
        EP8/16/32/64/128/256, 0 < B <= capacity <= 256). This mode reserves 8 GiB
        per rank plus 2 MiB each for barrier workspace and rank-offset padding,
        and has a restricted metadata/feature contract. Its combine also only
        sums expert outputs; it no longer applies dispatch weights.
        """
        lb_allocation_plan = lb_allocation_plan_or_num_bytes if isinstance(lb_allocation_plan_or_num_bytes, BufferAllocator) else None
        num_lb_bytes = lb_allocation_plan_or_num_bytes if lb_allocation_plan is None else lb_allocation_plan.num_bytes
        if lb_allocation_plan is not None:
            assert not lb_allocation_plan.materialized
        assert isinstance(num_lb_bytes, int) and num_lb_bytes >= 0
        assert num_lb_bytes % _C.get_num_allocation_alignment() == 0
        if sl_idx is not None or num_allocated_qps != 0:
            raise NotImplementedError('RDMA service levels and QP configuration are not implemented on Ascend')
        assert allow_multiple_reduction, 'Ascend requires allow_multiple_reduction=True'
        assert num_cpu_timeout_secs > 0 and num_gpu_timeout_secs > 0

        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_allocated_qps = 0
        self.num_lb_bytes = num_lb_bytes
        self.low_latency_mode = low_latency_mode
        if low_latency_mode:
            if deterministic or num_lb_bytes or lb_allocation_plan is not None:
                raise NotImplementedError('Low latency does not support deterministic sorting or LB storage')
            if self.num_ranks not in (8, 16, 32, 64, 128, 256):
                raise ValueError('Low latency requires EP8/16/32/64/128/256')
            if not 0 < num_max_tokens_per_rank <= 256:
                raise ValueError('Low latency requires 0 < num_max_tokens_per_rank <= 256')
            if num_bytes not in (None, 8 << 30):
                raise ValueError('Low latency requires exactly 8 GiB of communication storage')
            num_bytes = 8 << 30
        if num_bytes is None:
            num_bytes = self.get_buffer_size_hint(
                group, num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
                allow_hybrid_mode, allow_multiple_reduction)
        assert num_bytes >= 0 and num_bytes % _C.get_num_allocation_alignment() == 0
        self.num_bytes = num_bytes
        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'Initializing EP buffer with {num_bytes} bytes at rank EP {self.rank_idx}/{self.num_ranks}')

        self.hccl_comm_handle = comm.get_hccl_comm_handle(group)
        super().__init__(explicitly_destroy)
        if low_latency_mode:
            from .low_latency import LowLatencyState
            self._low_latency = LowLatencyState()
            self.runtime = _C.LowLatencyBuffer(
                self.rank_idx, self.num_ranks, self.hccl_comm_handle.get(),
                use_huge1g_page, num_cpu_timeout_secs, num_gpu_timeout_secs, explicitly_destroy)
        else:
            self.runtime = _C.EPBuffer(
                self.rank_idx, self.num_ranks, self.hccl_comm_handle.get(),
                num_bytes, num_lb_bytes, use_huge1g_page, num_cpu_timeout_secs, num_gpu_timeout_secs,
                explicitly_destroy)
        self.context = self.runtime.context
        if lb_allocation_plan is not None:
            lb_allocation_plan.materialize(self.runtime.lb_storage)
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
        if self.low_latency_mode:
            self._low_latency.tilings.clear()
            self._low_latency.last_event = None
            self._low_latency.inflight = None

    @staticmethod
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             allow_hybrid_mode: bool = True,
                             allow_multiple_reduction: bool = True,
                             *, low_latency_mode: bool = False) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing
        the buffer. The returned value is aligned to 2 MB.

        Arguments:
            group: the communication group.
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to use FP8 for dispatch.
            allow_hybrid_mode: accepted for API compatibility; ignored on Ascend.
            allow_multiple_reduction: must be True on Ascend.
            low_latency_mode: return the fixed 8 GiB decode communication size.

        Returns:
            size: the recommended buffer size in bytes (2 MB-aligned).
        """
        assert allow_multiple_reduction
        if low_latency_mode:
            if group.size() not in (8, 16, 32, 64, 128, 256) or not 0 < num_max_tokens_per_rank <= 256:
                raise ValueError('Low latency requires EP8/16/32/64/128/256 and token capacity in [1,256]')
            return 8 << 30
        return _C.calculate_ep_buffer_size(
            group.size(), num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch)

    @staticmethod
    def _unpack_handle(handle: Optional[EPHandle] = None) \
        -> Tuple[Optional[int], Optional[int], Optional[list],
                 Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor]]:
        if handle is None:
            return None, None, None, None, None, None, None, None, None
        return (handle.num_recv_tokens,
                handle.num_expanded_tokens,
                handle.num_recv_tokens_per_expert_list,
                handle.psum_num_recv_tokens_per_rank,
                handle.psum_num_recv_tokens_per_expert,
                handle.num_unaligned_recv_tokens_per_expert,
                handle.dst_buffer_slot_idx,
                handle.dst_gsge_idx,
                handle.recv_src_metadata)

    @staticmethod
    def capture() -> EventHandle:
        """
        Record an event on the current NPU stream.

        Returns:
            event_handle: the recorded NPU event.
        """
        return EventHandle()

    def get_theoretical_num_sms(self, num_experts: int, num_topk: int,
                                num_scaleout_topk: int = 0,
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,
                                sm_read_gbs: float = 0, sm_write_gbs: float = 0) -> int:
        return _C.get_num_ai_cores()

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        return 0

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 num_experts: Optional[int] = None,
                 num_max_tokens_per_rank: Optional[int] = None,
                 expert_alignment: Optional[int] = None,
                 num_sms: int = 0,
                 num_qps: int = 0,
                 previous_event: Optional[EventHandle] = None,
                 async_with_compute_stream: bool = False,
                 allocate_on_comm_stream: bool = False,
                 handle: Optional[EPHandle] = None,
                 do_handle_copy: bool = False,
                 do_cpu_sync: Optional[bool] = None,
                 do_expand: bool = False,
                 do_zero_padding: bool = False,
                 use_tma_aligned_col_major_sf: bool = False,
                 defer_epilogue: bool = False) \
            -> Union[Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           Optional[torch.Tensor], Optional[torch.Tensor],
                           EPHandle, EventOverlap], EventOverlap]:
        """
        Dispatch tokens to different ranks.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`. For the first type, the shape must be
                `[num_tokens, hidden]` with type `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type
                `torch.float8_e4m3fn`; the second contains SF packs with shape
                `[num_tokens, num_sf_packs]`.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each
                token, `-1` means no selections. Must be `None` if `handle` is provided.
                Low-latency mode requires valid, distinct expert IDs per token (no `-1`).
                Stored by reference in the handle; must not be modified until all uses of the handle complete.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token
                to dispatch. Can be provided with a cached `handle` to dispatch updated weights.
            cumulative_local_expert_recv_stats: `[num_local_experts]` with `torch.int`, a cumulative
                expert count tensor for statistics, useful for online EP load balance monitoring.
            num_experts: the number of all experts. Inferred from `handle` if provided.
            num_max_tokens_per_rank: the maximum number of tokens per rank. Inferred from constructor
                default or `handle` if provided.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of AI cores to use (0 selects a default in Python).
                Cached dispatch always uses the core count from `handle`.
            num_qps: must be 0 on Ascend (automatic backend configuration).
            previous_event: compatibility parameter; ignored on Ascend.
            async_with_compute_stream: return an event for the current-stream work.
            allocate_on_comm_stream: compatibility parameter; allocations use the current stream on Ascend.
            handle: an optional cached `EPHandle` from a previous dispatch; if set, the host will reuse
                the layout information to save some time. `topk_idx` must be `None` (reused from handle).
                Keep the handle's routing tensors unchanged; token count must match.
            do_handle_copy: unsupported on Ascend for now; must be `False`.
            do_cpu_sync: whether to synchronize with host to get exact received token counts.
                `None` defaults to `True` unless `handle` is provided.
            do_expand: whether to use the expanding layout (one slot per expert per token).
            do_zero_padding: whether to zero out the alignment padding slots in the expanded output.
                Only valid when `do_expand` is True.
            use_tma_aligned_col_major_sf: whether FP8 SF output uses column-major layout. Ascend does
                not add padding to the token dimension.
            defer_epilogue: whether to defer the host receive-count wait and copy epilogue until
                `event.current_stream_wait()` is called. This requires `async_with_compute_stream=True`.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`.
                Only returned when `defer_epilogue=False`.
            recv_topk_idx: received expert indices. Only returned when `defer_epilogue=False`.
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
                Only returned when `defer_epilogue=False`.
            handle: the returned communication handle. Only returned when `defer_epilogue=False`.
            event: the communication event (valid only if `async_with_compute_stream` is set).
                With `defer_epilogue=True`, this function returns the `EventOverlap` object directly instead
                of the five-item tuple. Call `event.current_stream_wait()` to run the copy epilogue and obtain
                `(recv_x, recv_topk_idx, recv_topk_weights, handle)`.
        """
        if self.low_latency_mode:
            return self._low_latency.dispatch(
                self, x, topk_idx, topk_weights, cumulative_local_expert_recv_stats,
                num_experts, num_max_tokens_per_rank, expert_alignment, num_sms, num_qps,
                previous_event, async_with_compute_stream, allocate_on_comm_stream, handle,
                do_handle_copy, do_cpu_sync, do_expand, do_zero_padding,
                use_tma_aligned_col_major_sf, defer_epilogue)
        if getattr(handle, 'is_low_latency', False):
            raise ValueError('A low-latency handle cannot be used with a throughput buffer')
        if num_qps not in (None, 0):
            raise NotImplementedError('Explicit QP counts are not implemented on Ascend')

        assert not do_handle_copy, '`do_handle_copy` must be False; handle copying is no longer supported'

        # Unpack SF
        x, sf = x if isinstance(x, tuple) else (x, None)

        # Ascend dispatch is expand-only and non-hybrid
        assert do_expand, 'Ascend dispatch only supports do_expand=True'

        # Unpack handles, reuse some values if possible
        if handle is not None:
            assert topk_idx is None
            assert handle.do_expand == do_expand
            assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'
            topk_idx = handle.topk_idx
            num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
            num_experts = value_or(num_experts, handle.num_experts)
            expert_alignment = value_or(expert_alignment, handle.expert_alignment)
            num_sms = handle.num_sms
            do_cpu_sync = False

            # Should be aligned with the handle context
            assert (num_experts, expert_alignment, num_max_tokens_per_rank) == \
                   (handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank)
        elif num_sms == 0:
            num_sms = self.get_theoretical_num_sms(num_experts, topk_idx.size(1))
        (cached_num_recv_tokens, cached_num_expanded_tokens,
         cached_num_recv_tokens_per_expert_list,
         cached_psum_num_recv_tokens_per_rank, cached_psum_num_recv_tokens_per_expert,
         cached_num_unaligned_recv_tokens_per_expert,
         cached_dst_buffer_slot_idx, cached_dst_gsge_idx,
         cached_recv_src_metadata) = self._unpack_handle(handle)

        # Some default values
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)
        expert_alignment = value_or(expert_alignment, 1)
        do_cpu_sync = value_or(do_cpu_sync, True)

        # Do dispatch
        result, event, deferred_epilogue = self.runtime.dispatch(
            x, sf, topk_idx, topk_weights,
            cumulative_local_expert_recv_stats,
            cached_num_recv_tokens,
            cached_num_expanded_tokens,
            cached_num_recv_tokens_per_expert_list,
            cached_psum_num_recv_tokens_per_rank,
            cached_psum_num_recv_tokens_per_expert,
            cached_num_unaligned_recv_tokens_per_expert,
            cached_dst_buffer_slot_idx,
            cached_dst_gsge_idx,
            cached_recv_src_metadata,
            num_max_tokens_per_rank,
            num_experts, expert_alignment,
            num_sms,
            previous_event,
            async_with_compute_stream, allocate_on_comm_stream,
            do_handle_copy, do_cpu_sync, do_expand,
            do_zero_padding, use_tma_aligned_col_major_sf,
            defer_epilogue)
        event_overlap = EventOverlap(event)

        def finalize_dispatch(dispatch_result: tuple, deterministic_by_hook: bool) \
                -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                         Optional[torch.Tensor], Optional[torch.Tensor], EPHandle]:
            (recv_x, recv_sf, recv_topk_weights,
             num_recv_tokens,
             num_expanded_tokens,
             num_recv_tokens_per_expert_list,
             psum_num_recv_tokens_per_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             dst_buffer_slot_idx, dst_gsge_idx) = dispatch_result

            # Create handle
            nonlocal handle
            is_cached_dispatch = handle is not None
            if not is_cached_dispatch:
                handle = EPHandle(do_expand,
                                  num_experts, expert_alignment,
                                  num_max_tokens_per_rank,
                                  num_sms,
                                  topk_idx,
                                  num_recv_tokens,
                                  num_expanded_tokens,
                                  num_recv_tokens_per_expert_list,
                                  psum_num_recv_tokens_per_rank,
                                  psum_num_recv_tokens_per_expert,
                                  num_unaligned_recv_tokens_per_expert,
                                  recv_src_metadata,
                                  dst_buffer_slot_idx,
                                  dst_gsge_idx)
            assert handle is not None

            # Deterministic epilogue
            if self.deterministic:
                deterministic_epilogue = functools.partial(
                    handle.deterministic_sort,
                    do_cpu_sync, is_cached_dispatch,
                    recv_x, recv_sf, recv_topk_weights)
                if deterministic_by_hook:
                    event_overlap.register_hook_after_wait(deterministic_epilogue)
                else:
                    deterministic_epilogue()

            # Repack SF
            recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x
            return recv_x, None, recv_topk_weights, handle

        # Defer epilogue
        if deferred_epilogue is not None:
            event_overlap.register_hook_after_wait(
                lambda: finalize_dispatch(deferred_epilogue(), False))
            return event_overlap

        # Directly return
        assert result is not None
        return *finalize_dispatch(result, async_with_compute_stream), event_overlap

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                num_sms: int = 0,
                num_qps: int = 0,
                previous_event: EventHandle = None,
                async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False,
                defer_epilogue: bool = False) \
            -> Union[Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap], EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its
                original ranks.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            topk_weights: `[num_tokens]` with `torch.float`. The expanded tokens' top-k weights for
                restoring to their original ranks as a separate output. These values do not
                multiply `x`; any model weighting must already be present in the expert outputs.
            bias: `None`, a tensor, or a tuple of 2 `[num_combined_tokens, hidden]` tensors with
                `torch.bfloat16` final bias to the output.
            num_sms: compatibility parameter; combine always uses `handle.num_sms`.
            num_qps: must be 0 on Ascend (automatic backend configuration).
            previous_event: compatibility parameter; ignored on Ascend.
            async_with_compute_stream: return an event for the current-stream work.
            allocate_on_comm_stream: compatibility parameter; allocations use the current stream on Ascend.
            defer_epilogue: whether to defer the reduce epilogue until `event.current_stream_wait()` is called.
                This requires `async_with_compute_stream=True`.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type
                `torch.bfloat16`. Only returned when `defer_epilogue=False`.
            combined_topk_weights: the reduced top-k weights, with shape `[num_combined_tokens, num_topk]`
                and type `torch.float`. Only returned when `defer_epilogue=False`.
            event: the communication event (valid only if `async_with_compute_stream` is set).
                With `defer_epilogue=True`, this function returns the `EventOverlap` object directly instead
                of the three-item tuple. Call `event.current_stream_wait()` to run the reduce epilogue and
                obtain `(combined_x, combined_topk_weights)`.
        """
        if num_qps not in (None, 0):
            raise NotImplementedError('Explicit QP counts are not implemented on Ascend')

        if self.low_latency_mode:
            return self._low_latency.combine(
                self, x, handle, topk_weights, bias, num_sms, num_qps, previous_event,
                async_with_compute_stream, allocate_on_comm_stream, defer_epilogue)
        if getattr(handle, 'is_low_latency', False):
            raise ValueError('A low-latency handle cannot be used with a throughput buffer')
        assert handle.recv_src_metadata is not None
        num_sms = handle.num_sms
        bias_0, bias_1 = EPBuffer._unpack_bias(bias)
        result, event, deferred_epilogue = self.runtime.combine(
            x, topk_weights,
            bias_0, bias_1,
            handle.recv_src_metadata,
            handle.topk_idx,
            handle.psum_num_recv_tokens_per_rank,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
            num_sms,
            previous_event,
            async_with_compute_stream,
            allocate_on_comm_stream,
            defer_epilogue,
        )
        event_overlap = EventOverlap(event)

        # Defer epilogue
        if deferred_epilogue is not None:
            event_overlap.register_hook_after_wait(deferred_epilogue)
            return event_overlap

        # Directly return
        assert result is not None
        combined_x, combined_topk_weights = result
        return combined_x, combined_topk_weights, event_overlap

    def lb_get_theoretical_num_sms(self) -> int:
        return _C.get_num_ai_cores()

    def lb_prefetch_weights(self,
                            redundant_expert_weights: Sequence[torch.Tensor] | torch.Tensor,
                            expert_weights: Sequence[torch.Tensor] | torch.Tensor,
                            redundancy_mapping: torch.Tensor,
                            num_sms: int = 0,
                            previous_event: Optional[EventHandle] = None) -> EventOverlap:
        """Push this rank's expert weights into the redundant expert slots requested by peers.

        Collective: every rank in the LSA domain must call it, and the fused barriers make the redundant weights
        visible to their users by the time the returned event is waited on.
        Tensor sequences must be nonempty and contain no `None` entries.

        Arguments:
            redundant_expert_weights: each tensor is a contiguous `[num_redundant_experts, *shape]`
                destination and must reside in this buffer's LB region.
            expert_weights: each tensor is a contiguous `[num_local_experts, *shape]` source with at least
                one dimension and a positive `num_local_experts`. Bytes per expert must match its
                destination and be a multiple of `deep_ep.get_num_tma_alignment()`; the trailing shapes may differ.
            redundancy_mapping: `[num_nvlink_ranks, num_redundant_experts]` int32, identical on every rank in the LSA domain;
                entry `[r, c]` is the expert id assigned to redundant slot `c` on LSA rank `r`, or -1 if empty.
                Expert ids are local to the LSA domain, in [0, num_nvlink_ranks * num_local_experts).
            num_sms: the number of SMs to use; 0 uses `lb_get_theoretical_num_sms()`.
            previous_event: compatibility parameter; ignored on Ascend.
        """
        redundant_expert_weights = ([redundant_expert_weights] if isinstance(redundant_expert_weights, torch.Tensor)
                                    else list(redundant_expert_weights))
        expert_weights = [expert_weights] if isinstance(expert_weights, torch.Tensor) else list(expert_weights)

        num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)
        return EventOverlap(self.runtime.lb_prefetch_weights(
            redundant_expert_weights, expert_weights, redundancy_mapping, num_sms, previous_event))

    def lb_reduce_grads(self,
                        redundant_expert_grads: torch.Tensor,
                        expert_grads: torch.Tensor,
                        redundancy_mapping: torch.Tensor,
                        num_sms: int = 0,
                        previous_event: Optional[EventHandle] = None) -> EventOverlap:
        """Accumulate redundant expert gradients from peers into this rank's `expert_grads`.

        The mirror of `lb_prefetch_weights`: use the same redundancy mapping to return gradients.
        Collective, and it adds rather than overwrites, so seed `expert_grads` before calling.

        Arguments:
            redundant_expert_grads: `[num_redundant_experts, hidden]` fp32 redundant gradients that must reside in this buffer's LB region.
            expert_grads: `[num_local_experts, hidden]` fp32 destination, accumulated into.
            redundancy_mapping: see `lb_prefetch_weights`.
            num_sms: the number of SMs to use; 0 uses `lb_get_theoretical_num_sms()`.
            previous_event: compatibility parameter; ignored on Ascend.
        """
        num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)
        return EventOverlap(self.runtime.lb_reduce_grads(
            redundant_expert_grads, expert_grads, redundancy_mapping, num_sms, previous_event))
