"""Legacy explicit experimental adapter for the Ascend decode kernels.

The default EPBuffer integrates native transport selection with the standard
metadata and epilogues. This adapter is only used by low_latency_mode=True.

All ranks must use identical B/M/H/K/E and call order. Routing entries must be
valid expert IDs (no masked -1 entries), with no duplicate expert per token.
Dispatch still transports weight metadata. Combine only sums expert outputs;
any routing weights must already have been applied by the expert computation.
"""

import threading

import torch

from deep_ep import _C
from ..utils.event import EventHandle, EventOverlap


class LowLatencyState:
    def __init__(self):
        self.token = object()
        self.device = torch.device('npu', torch.npu.current_device())
        self.lock = threading.Lock()
        self.last_event = None
        self.failed = False
        self.tilings = {}
        self.inflight = None

    def _check_alive(self, buffer):
        if buffer.runtime is None:
            raise RuntimeError('EPBuffer has been destroyed')
        if self.failed:
            raise RuntimeError('A low-latency launch failed; restart the distributed job')
        if torch.npu.current_device() != self.device.index:
            raise ValueError('Use the NPU on which this EPBuffer was constructed')

    def _tensor(self, value, shape, dtypes, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype not in dtypes or value.device != self.device or not value.is_contiguous()):
            raise ValueError(f'{name} must be a contiguous {shape} tensor on {self.device}, dtype {dtypes}')

    def _tiling(self, operation, shape, dtype, capacity):
        key = (operation, *shape, dtype, capacity)
        if key not in self.tilings:
            data = getattr(_C, f'low_latency_{operation}_tiling')(*shape, dtype, capacity)
            source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            self.tilings[key] = source.to(device=self.device)
        return self.tilings[key]

    def _begin(self, previous_event):
        # The device owns persistent per-AIV ping-pong state. Serialize launches
        # even if the application changes its current stream between calls.
        if self.last_event is not None:
            self.last_event.current_stream_wait()
        if previous_event is not None:
            previous_event.current_stream_wait()

    def _retain(self, tensors):
        retained = tuple(t for t in tensors.values() if isinstance(t, torch.Tensor))
        stream = torch.npu.current_stream()
        for tensor in retained:
            tensor.record_stream(stream)
        self.inflight = retained

    def _finish(self):
        self.last_event = EventHandle()
        event = EventOverlap(self.last_event, self.inflight)
        self.inflight = None
        return event

    @staticmethod
    def _schedule(num_sms, num_qps, allocate_on_comm_stream, defer_epilogue):
        if num_sms not in (0, 32) or num_qps not in (None, 0):
            raise NotImplementedError('Low latency uses exactly 32 AI cores / 64 AIVs and automatic QPs')
        if allocate_on_comm_stream or defer_epilogue:
            raise NotImplementedError('Low latency runs the complete operation on the current stream')

    def dispatch(self, buffer, x, topk_idx, topk_weights, cumulative_local_expert_recv_stats,
                 num_experts, num_max_tokens_per_rank, expert_alignment, num_sms, num_qps,
                 previous_event, async_with_compute_stream, allocate_on_comm_stream, handle,
                 do_handle_copy, do_cpu_sync, do_expand, do_zero_padding,
                 use_tma_aligned_col_major_sf, defer_epilogue):
        from .ep import EPHandle

        self._check_alive(buffer)
        self._schedule(num_sms, num_qps, allocate_on_comm_stream, defer_epilogue)
        if (handle is not None or cumulative_local_expert_recv_stats is not None
                or do_handle_copy or use_tma_aligned_col_major_sf or expert_alignment not in (None, 1)):
            raise NotImplementedError('Low latency requires a fresh handle, alignment 1, row-major scales and no stats/copy')
        if not do_expand:
            raise NotImplementedError('Low latency requires do_expand=True')
        values, scales = x if isinstance(x, tuple) and len(x) == 2 else (x, None)
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError('x must be [B,H] or (FP8 [B,H], scales [B,ceil(H/128)])')
        if not isinstance(topk_idx, torch.Tensor) or topk_idx.ndim != 2:
            raise ValueError('topk_idx must be an int64 [B,K] tensor')
        b, h = values.shape
        k = topk_idx.shape[1]
        capacity = buffer.num_max_tokens_per_rank if num_max_tokens_per_rank is None else num_max_tokens_per_rank
        if type(capacity) is not int or not 0 < b <= capacity <= 256:
            raise ValueError('Low latency requires 0 < B <= num_max_tokens_per_rank <= 256')
        if type(num_experts) is not int or num_experts <= 0:
            raise ValueError('num_experts must be a positive integer')
        self._tensor(values, (b, h), (torch.bfloat16, torch.float16, torch.float8_e4m3fn), 'x')
        self._tensor(topk_idx, (b, k), (torch.int64,), 'topk_idx')
        self._tensor(topk_weights, (b, k), (torch.float32,), 'topk_weights')
        if values.dtype == torch.float8_e4m3fn:
            self._tensor(scales, (b, (h + 127) // 128), (torch.int32, torch.float32), 'scales')
        elif scales is not None:
            raise ValueError('scales are only accepted with FP8 inputs')
        dtype = str(values.dtype).removeprefix('torch.')
        shape = (buffer.num_ranks, b, h, k, num_experts)
        plan = _C.low_latency_layout(*shape, dtype)
        if plan is None:
            raise ValueError('Low-latency shape exceeds topology, workspace, UB or 10000-row combine capacity')
        # Admit a complete dispatch/combine pair before entering communication.
        _C.low_latency_combine_tiling(*shape, 'float16' if dtype == 'float16' else 'bfloat16', capacity)
        c = plan['capacity_rows']
        with self.lock:
            self._check_alive(buffer)
            self._begin(previous_event)
            alloc = torch.zeros if do_zero_padding else torch.empty
            tensors = dict(
                input=values, indices=topk_idx.to(torch.int32), weights=topk_weights, scales=scales,
                output=alloc((c, h), dtype=values.dtype, device=self.device),
                output_scales=(alloc((c, (h + 127) // 128), dtype=scales.dtype, device=self.device)
                               if scales is not None else None),
                expert_recv_counts=torch.empty((num_experts // buffer.num_ranks,), dtype=torch.int64, device=self.device),
                send_counts=torch.empty((num_experts,), dtype=torch.int32, device=self.device),
                token_type=torch.empty((c,), dtype=torch.int32, device=self.device),
                destination_index=torch.empty((c, 3), dtype=torch.int32, device=self.device),
                relay_read_index=torch.empty((c, k, 2), dtype=torch.int32, device=self.device),
                source_mask=torch.empty((b,), dtype=torch.int32, device=self.device),
                source_weights=torch.empty((b, k), dtype=torch.float32, device=self.device),
                tiling=self._tiling('dispatch', shape, dtype, capacity),
            )
            self._retain(tensors)
            try:
                _C.dispatch_low_latency(buffer.runtime, tensors, b, h, k, num_experts, c, dtype)
                counts = tensors['expert_recv_counts']
                count_list = counts.cpu().tolist() if do_cpu_sync is not False else []
                rows = sum(count_list) if do_cpu_sync is not False else c
                out = tensors['output'][:rows]
                out_scales = tensors['output_scales'][:rows] if scales is not None else None
                result = EPHandle(True, num_experts, 1, capacity, 32, topk_idx,
                                  None, rows, count_list, None, None, None, None, None, None)
                result.is_low_latency = True
                result._buffer_token = self.token
                result.num_tokens, result.hidden, result.num_topk = b, h, k
                result.expert_recv_counts = counts
                for name in ('send_counts', 'token_type', 'destination_index', 'relay_read_index',
                             'source_mask', 'source_weights'):
                    setattr(result, name, tensors[name])
                event = self._finish()
            except Exception:
                self.failed = True
                # Keep references and existing allocator stream records; do not
                # issue another device operation that could mask the failure.
                raise
        return ((out, out_scales) if scales is not None else out), None, None, result, event

    def combine(self, buffer, x, handle, topk_weights, bias, num_sms, num_qps,
                previous_event, async_with_compute_stream, allocate_on_comm_stream, defer_epilogue):
        self._check_alive(buffer)
        self._schedule(num_sms, num_qps, allocate_on_comm_stream, defer_epilogue)
        if not getattr(handle, 'is_low_latency', False) or getattr(handle, '_buffer_token', None) is not self.token:
            raise ValueError('Low-latency combine requires a dispatch handle from this buffer')
        if topk_weights is not None or bias is not None:
            raise NotImplementedError('Legacy low-latency combine has no weight-return side channel or bias')
        self._tensor(x, (handle.num_expanded_tokens, handle.hidden), (torch.bfloat16, torch.float16), 'expert output')
        b, h, k, e = handle.num_tokens, handle.hidden, handle.num_topk, handle.num_experts
        dtype = str(x.dtype).removeprefix('torch.')
        with self.lock:
            self._check_alive(buffer)
            self._begin(previous_event)
            tensors = {name: getattr(handle, name) for name in (
                'send_counts', 'token_type', 'destination_index', 'relay_read_index', 'source_mask')}
            tensors.update(input=x, output=torch.empty((b, h), dtype=x.dtype, device=self.device),
                           tiling=self._tiling('combine', (buffer.num_ranks, b, h, k, e), dtype,
                                               handle.num_max_tokens_per_rank))
            self._retain(tensors)
            try:
                _C.combine_low_latency(buffer.runtime, tensors, b, h, k, e, dtype)
                event = self._finish()
            except Exception:
                self.failed = True
                raise
        return tensors['output'], None, event
