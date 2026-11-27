# Filter and disable all annoying warnings
import os
import warnings
os.environ['TORCH_NPU_DISABLED_WARNING'] = '1'
# TODO(HUAWEI): fix permission warnings
warnings.filterwarnings('ignore', message='Permission mismatch')
warnings.filterwarnings('ignore', message=r'Warning: The .* owner does not match the current owner\.')
warnings.filterwarnings('ignore', message='Cannot create tensor with')

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing

import deep_ep
from deep_ep.utils.envs import init_dist, init_seed, dist_print
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.math import align, count_bytes, per_token_cast_to_fp8, safe_div
from deep_ep.utils.refs import combine as ref_combine
from deep_ep.utils.refs import dispatch as ref_dispatch
from deep_ep.utils.refs import generate_pre_combine_data
from deep_ep.utils.testing import bench_msprof, flush_l2_cache


def enumerate_dispatch_modes():
    for expert_alignment in (128, 1):
        for use_fp8_dispatch in (True, False):
            for num_bias in (0, 1, 2):
                for use_col_major_sf in ((False, True) if use_fp8_dispatch else (False, )):
                    yield expert_alignment, num_bias, use_fp8_dispatch, use_col_major_sf


def launch(buffer: deep_ep.EPBuffer, name: str, defer_epilogue: int, **params):
    result = getattr(buffer, name)(defer_epilogue=defer_epilogue, **params)

    # Run deferred correctness path
    if defer_epilogue:
        event = result
        return *event.current_stream_wait(release_handle=True), event

    # Run legacy correctness path
    result[-1].current_stream_wait(release_handle=True)
    return result


# noinspection PyShadowingNames
def test_dispatch_combine(buffer: deep_ep.EPBuffer, args: argparse.Namespace):
    assert not args.allow_hybrid_mode
    assert args.allow_multiple_reduction

    num_ranks = buffer.num_ranks
    num_max_tokens_per_rank = args.num_tokens
    num_tokens = max(1, args.num_tokens - dist.get_rank())
    hidden = args.hidden
    combine_dtype = torch.float32 if args.use_fp32_output else torch.bfloat16
    num_topk = args.num_topk
    num_experts = args.num_experts
    num_local_experts = num_experts // num_ranks
    num_ai_cores = args.num_ai_cores or buffer.get_theoretical_num_sms(num_experts, num_topk)

    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Experts: {num_topk}/{num_experts}\n'
               f' > Tokens: {num_tokens} (max: {num_max_tokens_per_rank}), hidden: {hidden}\n'
               f' > Combine output: {combine_dtype}\n'
               f' > #AI cores: {num_ai_cores}\n'
               f' > Modes: do_expand=1, allow_hybrid_mode=0, allow_multiple_reduction={args.allow_multiple_reduction}\n',
               once_in_node=True)

    scores = get_unbalanced_scores(
        num_tokens, num_experts, num_ranks, num_topk,
        args.unbalanced_ratio, args.precise_unbalanced_ratio, args.synthetic_unbalanced_ratio)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(torch.int64)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # Permute top-k positions to avoid positional bias in scores
    topk_perm = torch.rand_like(topk_weights).argsort(dim=-1)
    topk_idx = topk_idx.gather(dim=-1, index=topk_perm)
    topk_weights = topk_weights.gather(dim=-1, index=topk_perm)

    dist_print('Running expand dispatch/combine test cases:', once_in_node=True)
    for expert_alignment, num_bias, use_fp8_dispatch, use_col_major_sf in enumerate_dispatch_modes():
        if args.dispatch_dtype != 'all' and use_fp8_dispatch != (args.dispatch_dtype == 'fp8'):
            continue
        bias_name = ('None', 'Tensor', 'tuple[2]')[num_bias]
        dist_print(f' > Testing with {expert_alignment=}, {use_fp8_dispatch=}, '
                   f'{use_col_major_sf=}, bias={bias_name} ...', once_in_node=True)

        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
        if use_fp8_dispatch:
            x_data, sf = per_token_cast_to_fp8(x)
            sf = sf.view(torch.int16)
            num_sf_packs = sf.size(1)
            if use_col_major_sf:
                sf_storage = torch.empty((num_tokens * num_sf_packs, ), dtype=torch.int16, device='npu')
                input_sf = torch.as_strided(sf_storage, (num_tokens, num_sf_packs), (1, num_tokens))
                input_sf.copy_(sf)
            else:
                input_sf = sf
            x = (x_data, input_sf)
        bias = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu') if num_bias == 1 else None
        if num_bias == 2:
            bias = tuple(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu') for _ in range(num_bias))

        # Test cumulative stats counter
        cumulative_local_expert_recv_stats = torch.zeros(
            (num_local_experts, ), dtype=torch.int, device='npu')
        dispatch_args = dict(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_sms=args.num_ai_cores,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            async_with_compute_stream=True,
            do_handle_copy=False,
            do_cpu_sync=bool(args.do_cpu_sync),
            do_expand=True,
            do_zero_padding=True,
            use_tma_aligned_col_major_sf=use_col_major_sf,
        )

        if not args.skip_check:
            ref_x = (x_data.view(torch.uint8), input_sf) if use_fp8_dispatch else x
            ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, ref_recv_src_token_idx, \
                ref_num_recv_tokens_per_rank = \
                ref_dispatch(ref_x, topk_idx, topk_weights, num_max_tokens_per_rank, num_experts)
            torch.npu.synchronize()

        expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, handle, event = launch(
            buffer, 'dispatch', args.defer_epilogue,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            **dispatch_args)
        torch.npu.synchronize()
        cached_dispatch_args = dict(dispatch_args, handle=handle, num_sms=0, do_cpu_sync=False)
        del cached_dispatch_args['topk_idx']

        expanded_recv_x_data, expanded_recv_sf = expanded_recv_x if use_fp8_dispatch else (expanded_recv_x, None)
        expanded_check_x = expanded_recv_x_data.view(torch.uint8) if use_fp8_dispatch else expanded_recv_x_data

        if use_fp8_dispatch:
            expected_sf_stride = \
                (1, expanded_recv_x_data.size(0)) if use_col_major_sf else (num_sf_packs, 1)
            assert expanded_recv_sf.stride() == expected_sf_stride

        num_recv_tokens = handle.psum_num_recv_tokens_per_rank[-1].item()
        num_expanded_tokens = align(handle.psum_num_recv_tokens_per_expert[-1].item(), expert_alignment)
        expected_num_recv_tokens = num_recv_tokens if args.do_cpu_sync else num_ranks * num_max_tokens_per_rank
        assert handle.num_recv_tokens == expected_num_recv_tokens
        assert handle.recv_src_metadata.size(0) == expected_num_recv_tokens
        assert expanded_recv_topk_idx is None
        assert expanded_recv_topk_weights is not None
        assert handle.recv_src_metadata is not None
        recv_src_metadata = handle.recv_src_metadata[:num_recv_tokens]

        local_y = generate_pre_combine_data(recv_src_metadata[:, -2], num_max_tokens_per_rank, num_topk, hidden)
        input_for_combine = torch.empty(
            (expanded_recv_x_data.size(0) + 1, hidden), dtype=torch.bfloat16, device='npu')
        input_for_combine[recv_src_metadata[:, :-2].long().flatten()] = local_y.view(-1, hidden)
        del local_y
        input_for_combine = input_for_combine[:-1]

        combined_x, combined_topk_weights, combine_event = launch(
            buffer, 'combine', args.defer_epilogue,
            x=input_for_combine,
            handle=handle,
            topk_weights=expanded_recv_topk_weights,
            bias=bias,
            num_sms=num_ai_cores,
            async_with_compute_stream=True,
            use_fp32_output=args.use_fp32_output,
        )
        torch.npu.synchronize()

        assert combined_topk_weights is not None

        if not args.skip_check:
            ref_recv_x_data, ref_recv_sf = ref_recv_x if use_fp8_dispatch else (ref_recv_x, None)
            assert num_recv_tokens == ref_recv_x_data.size(0), \
                f'{num_recv_tokens=}, expected={ref_recv_x_data.size(0)}'

            recv_src_token_idx = recv_src_metadata[:, -2]
            recv_expanded_slots = recv_src_metadata[:, :-2]
            ref_order = torch.argsort(ref_recv_src_token_idx)
            metadata_order = torch.argsort(recv_src_token_idx)
            assert torch.equal(recv_src_token_idx[metadata_order], ref_recv_src_token_idx[ref_order]), \
                f'{recv_src_token_idx[metadata_order]=}, {ref_recv_src_token_idx[ref_order]=}'

            ref_recv_x_by_metadata = torch.empty_like(ref_recv_x_data)
            ref_recv_sf_by_metadata = torch.empty_like(ref_recv_sf) if ref_recv_sf is not None else None
            ref_recv_topk_idx_by_metadata = torch.empty_like(ref_recv_topk_idx)
            ref_recv_topk_weights_by_metadata = torch.empty_like(ref_recv_topk_weights)
            ref_recv_x_by_metadata[metadata_order] = ref_recv_x_data[ref_order]
            if ref_recv_sf_by_metadata is not None:
                ref_recv_sf_by_metadata[metadata_order] = ref_recv_sf[ref_order]
            ref_recv_topk_idx_by_metadata[metadata_order] = ref_recv_topk_idx[ref_order]
            ref_recv_topk_weights_by_metadata[metadata_order] = ref_recv_topk_weights[ref_order]

            recv_valid_mask = recv_expanded_slots >= 0
            ref_valid_mask = ref_recv_topk_idx_by_metadata >= 0
            assert torch.equal(recv_valid_mask, ref_valid_mask)

            valid_expanded_slots = recv_expanded_slots[recv_valid_mask].long()
            assert valid_expanded_slots.numel() == handle.num_unaligned_recv_tokens_per_expert.sum().item()
            if valid_expanded_slots.numel() > 0:
                assert valid_expanded_slots.min().item() >= 0
                assert valid_expanded_slots.max().item() < num_expanded_tokens
                assert torch.unique(valid_expanded_slots).numel() == valid_expanded_slots.numel()

                expected_expanded_x = \
                    ref_recv_x_by_metadata.unsqueeze(1).expand(-1, num_topk, -1)[recv_valid_mask]
                check_expanded_x = expanded_check_x[valid_expanded_slots]
                assert torch.equal(check_expanded_x, expected_expanded_x)
                if ref_recv_sf_by_metadata is not None:
                    expected_expanded_sf = \
                        ref_recv_sf_by_metadata.unsqueeze(1).expand(-1, num_topk, -1)[recv_valid_mask]
                    assert torch.equal(expanded_recv_sf[valid_expanded_slots], expected_expanded_sf)
                assert torch.equal(
                    expanded_recv_topk_weights[valid_expanded_slots],
                    ref_recv_topk_weights_by_metadata[recv_valid_mask],
                )

            # Make sure deterministic mode produces the same expanded layout twice
            if args.deterministic:
                expanded_recv_x_twice, _, expanded_recv_topk_weights_twice, handle_twice, event_twice = launch(
                    buffer, 'dispatch', args.defer_epilogue, **dispatch_args)
                torch.npu.synchronize()
                expanded_recv_x_data_twice, expanded_recv_sf_twice = \
                    expanded_recv_x_twice if use_fp8_dispatch else (expanded_recv_x_twice, None)
                expanded_check_x_twice = \
                    expanded_recv_x_data_twice.view(torch.uint8) if use_fp8_dispatch else expanded_recv_x_data_twice
                assert torch.equal(
                    expanded_check_x[valid_expanded_slots],
                    expanded_check_x_twice[valid_expanded_slots],
                )
                if expanded_recv_sf is not None:
                    assert torch.equal(
                        expanded_recv_sf[valid_expanded_slots],
                        expanded_recv_sf_twice[valid_expanded_slots],
                    )
                assert torch.equal(
                    expanded_recv_topk_weights[valid_expanded_slots],
                    expanded_recv_topk_weights_twice[valid_expanded_slots],
                )
                recv_src_metadata_twice = handle_twice.recv_src_metadata[:num_recv_tokens]
                metadata_order_twice = torch.argsort(recv_src_metadata_twice[:, -2])
                assert torch.equal(
                    recv_src_metadata[metadata_order],
                    recv_src_metadata_twice[metadata_order_twice],
                )

            topk_range = torch.arange(num_topk, device=recv_src_metadata.device, dtype=recv_src_metadata.dtype)
            master_topk_idx = torch.where(
                recv_valid_mask,
                topk_range.view(1, num_topk),
                torch.full_like(recv_expanded_slots, -1),
            ).max(dim=1).values
            assert torch.equal(
                recv_src_metadata[:, -1],
                master_topk_idx,
            )

            psum_num_recv_tokens_per_rank = [0] + handle.psum_num_recv_tokens_per_rank.tolist()
            for rank_idx in range(num_ranks):
                count = psum_num_recv_tokens_per_rank[rank_idx + 1] - psum_num_recv_tokens_per_rank[rank_idx]
                ref_count = ref_num_recv_tokens_per_rank[rank_idx].item()
                assert count == ref_count, f'{rank_idx=}, {count=}, {ref_count=}'

            psum_num_recv_tokens_per_expert = [0] + handle.psum_num_recv_tokens_per_expert.tolist()
            for expert_idx in range(num_local_experts):
                count = (psum_num_recv_tokens_per_expert[expert_idx + 1] -
                         align(psum_num_recv_tokens_per_expert[expert_idx], expert_alignment))
                ref_count = (ref_recv_topk_idx == expert_idx).sum().item()
                assert count == ref_count, f'{expert_idx=}, {count=}, {ref_count=}'
                assert cumulative_local_expert_recv_stats[expert_idx].item() == ref_count
                assert handle.num_unaligned_recv_tokens_per_expert[expert_idx].item() == ref_count
                if args.do_cpu_sync:
                    assert handle.num_recv_tokens_per_expert_list[expert_idx] == align(ref_count, expert_alignment)

                expert_start = align(psum_num_recv_tokens_per_expert[expert_idx], expert_alignment)
                expert_end = psum_num_recv_tokens_per_expert[expert_idx + 1]
                expert_slots = recv_expanded_slots[ref_recv_topk_idx_by_metadata == expert_idx]
                assert expert_slots.numel() == ref_count
                if ref_count > 0:
                    assert torch.equal(
                        torch.sort(expert_slots).values,
                        torch.arange(expert_start, expert_end, device=expert_slots.device, dtype=expert_slots.dtype),
                    )

                # Check zero padding
                padding_end = align(expert_end, expert_alignment)
                padding_x = expanded_recv_x_data[expert_end:padding_end]
                assert torch.count_nonzero(padding_x.view(torch.uint8)).item() == 0, \
                    f'{expert_idx=}, padding range [{expert_end}, {padding_end}) is not bitwise zero'
                if expanded_recv_sf is not None:
                    assert torch.count_nonzero(expanded_recv_sf[expert_end:padding_end]).item() == 0

            ref_y = generate_pre_combine_data(
                dist.get_rank() * num_max_tokens_per_rank + torch.arange(num_tokens, device='npu'),
                num_max_tokens_per_rank, num_topk, hidden)
            ref_y[topk_idx == -1] = 0
            ref_combined_x = ref_combine(
                ref_y, topk_idx, num_experts, bias, reduce_in_local=True, use_fp32_output=args.use_fp32_output)
            del ref_y
            max_diff = (combined_x.float() - ref_combined_x.float()).abs().max().item()
            assert torch.equal(combined_x, ref_combined_x), f'{max_diff=}'
            assert torch.equal(combined_topk_weights, topk_weights)

            if num_bias == 0 and not use_col_major_sf:
                combined_x_without_weights, combined_topk_weights_without_weights, _ = launch(
                    buffer, 'combine', args.defer_epilogue,
                    x=input_for_combine,
                    handle=handle,
                    bias=bias,
                    num_sms=num_ai_cores,
                    async_with_compute_stream=True,
                    use_fp32_output=args.use_fp32_output,
                )
                torch.npu.synchronize()
                assert torch.equal(combined_x_without_weights, ref_combined_x)
                assert combined_topk_weights_without_weights is None

            # Replay after combine and other dispatches have reused the communication workspace.
            cached_tensors = (
                handle.psum_num_recv_tokens_per_rank, handle.psum_num_recv_tokens_per_expert,
                handle.num_unaligned_recv_tokens_per_expert, handle.recv_src_metadata,
                handle.dst_buffer_slot_idx, handle.dst_gsge_idx,
            )
            cached_copies = [tensor.clone() for tensor in cached_tensors]
            padding_mask = torch.ones(num_expanded_tokens, dtype=torch.bool, device='npu')
            padding_mask[valid_expanded_slots] = False
            for send_weights in (True, False):
                updated_x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='npu') * (8 if send_weights else 0.125)
                if use_fp8_dispatch:
                    updated_x_data, updated_sf = per_token_cast_to_fp8(updated_x)
                    updated_sf = updated_sf.view(torch.int16)
                    if use_col_major_sf:
                        updated_sf = updated_sf.t().contiguous().t()
                    updated_x = (updated_x_data, updated_sf)
                updated_weights = torch.randn_like(topk_weights).masked_fill_(topk_idx < 0, 0)
                updated_ref_x = (updated_x_data.view(torch.uint8), updated_sf) if use_fp8_dispatch else updated_x
                ref_cached_x, _, ref_cached_weights, _, _ = ref_dispatch(
                    updated_ref_x, topk_idx, updated_weights, num_max_tokens_per_rank, num_experts)
                cached_x, cached_topk_idx, cached_weights, cached_handle, _ = launch(
                    buffer, 'dispatch', args.defer_epilogue,
                    **dict(cached_dispatch_args, x=updated_x,
                           num_sms=0 if send_weights else handle.num_sms + 1,
                           topk_weights=updated_weights if send_weights else None))
                torch.npu.synchronize()

                assert cached_handle is handle and cached_topk_idx is None
                assert handle.num_sms == num_ai_cores
                for tensor, saved in zip(cached_tensors, cached_copies):
                    assert torch.equal(tensor, saved)
                cached_x_data, cached_sf = cached_x if use_fp8_dispatch else (cached_x, None)
                ref_cached_x_data, ref_cached_sf = ref_cached_x if use_fp8_dispatch else (ref_cached_x, None)
                cached_check_x = cached_x_data.view(torch.uint8) if use_fp8_dispatch else cached_x_data
                ref_cached_x_by_metadata = torch.empty_like(ref_cached_x_data)
                ref_cached_x_by_metadata[metadata_order] = ref_cached_x_data[ref_order]
                assert torch.equal(
                    cached_check_x[valid_expanded_slots],
                    ref_cached_x_by_metadata.unsqueeze(1).expand(-1, num_topk, -1)[recv_valid_mask])
                assert torch.count_nonzero(cached_check_x[:num_expanded_tokens][padding_mask]).item() == 0
                if cached_sf is not None:
                    assert cached_sf.stride() == expected_sf_stride
                    ref_cached_sf_by_metadata = torch.empty_like(ref_cached_sf)
                    ref_cached_sf_by_metadata[metadata_order] = ref_cached_sf[ref_order]
                    assert torch.equal(
                        cached_sf[valid_expanded_slots],
                        ref_cached_sf_by_metadata.unsqueeze(1).expand(-1, num_topk, -1)[recv_valid_mask])
                    assert torch.count_nonzero(cached_sf[:num_expanded_tokens][padding_mask]).item() == 0
                if send_weights:
                    ref_cached_weights_by_metadata = torch.empty_like(ref_cached_weights)
                    ref_cached_weights_by_metadata[metadata_order] = ref_cached_weights[ref_order]
                    assert torch.equal(
                        cached_weights[valid_expanded_slots], ref_cached_weights_by_metadata[recv_valid_mask])
                    assert torch.count_nonzero(cached_weights[:num_expanded_tokens][padding_mask]).item() == 0
                else:
                    assert cached_weights is None

                cached_combined_x, cached_combined_weights, _ = launch(
                    buffer, 'combine', args.defer_epilogue,
                    x=input_for_combine, handle=cached_handle, topk_weights=cached_weights,
                    bias=bias, num_sms=num_ai_cores, async_with_compute_stream=True,
                    use_fp32_output=args.use_fp32_output)
                torch.npu.synchronize()
                assert torch.equal(cached_combined_x, ref_combined_x)
                if send_weights:
                    assert torch.equal(cached_combined_weights, updated_weights)
                else:
                    assert cached_combined_weights is None

        if not args.skip_perf_test:
            num_unaligned_expanded_tokens = handle.num_unaligned_recv_tokens_per_expert.sum().item()
            topk_rank_idx = torch.where(
                topk_idx >= 0,
                topk_idx // num_local_experts,
                torch.full_like(topk_idx, -1),
            )
            num_epilogue_slots = 0
            for slot_idx in range(num_topk):
                master_mask = topk_idx[:, slot_idx] >= 0
                for prev_slot_idx in range(slot_idx):
                    master_mask &= topk_rank_idx[:, prev_slot_idx] != topk_rank_idx[:, slot_idx]
                num_epilogue_slots += master_mask.sum().item()

            # Preserve the original accounting: received tokens, including local tokens, and logical tensor bytes.
            num_bytes_per_dispatch_token = safe_div(count_bytes(x, topk_idx, topk_weights), num_tokens)
            num_dispatch_urma_bytes = num_recv_tokens * num_bytes_per_dispatch_token
            num_bytes_per_combine_token = safe_div(count_bytes(input_for_combine), input_for_combine.size(0))
            num_combine_urma_bytes = num_recv_tokens * num_bytes_per_combine_token

            def benchmark(fn, kernel_names):
                # Measure communication completion and epilogue with the barrier in the
                # prologue, then measure the prologue with the barrier in the epilogue.
                previous = deep_ep._C.set_barrier_in_prologue(True)
                try:
                    urma_profile, epilogue_profile = bench_msprof(
                        fn=fn,
                        kernel_names=kernel_names,
                        num_warmups=10,
                        num_tests=50,
                        flush_l2=True,
                        barrier_comm_profiling=True,
                        barrier=buffer.barrier,
                    )
                    deep_ep._C.set_barrier_in_prologue(False)
                    prologue_profile = bench_msprof(
                        fn=fn,
                        kernel_names=kernel_names[0],
                        num_warmups=10,
                        num_tests=50,
                        flush_l2=True,
                        barrier_comm_profiling=True,
                        barrier=buffer.barrier,
                    )
                finally:
                    deep_ep._C.set_barrier_in_prologue(previous)
                return prologue_profile, urma_profile, epilogue_profile

            def dispatch_once(use_cached=False):
                event = buffer.dispatch(
                    defer_epilogue=True,
                    **(cached_dispatch_args if use_cached else dispatch_args))
                # Use the same cold-L2 epilogue setup in both profiling passes.
                flush_l2_cache()
                event.current_stream_wait(release_handle=True)

            dispatch_profile, dispatch_urma_profile, epilogue_profile = benchmark(
                fn=dispatch_once,
                kernel_names=['dispatch_impl', 'dispatch_copy_epilogue_impl'],
            )
            num_bytes_per_expanded_token = safe_div(
                count_bytes(expanded_recv_x, expanded_recv_topk_weights), expanded_recv_x_data.size(0))
            num_recv_metadata_bytes = (num_topk + 2) * handle.recv_src_metadata.element_size()
            # Count all output slots, including hidden, scales, and weights zeroed in padding.
            num_dispatch_epilogue_bytes = (
                num_recv_tokens * (num_bytes_per_dispatch_token + num_recv_metadata_bytes) +
                expanded_recv_x_data.size(0) * num_bytes_per_expanded_token)
            dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks:3} | '
                       f'dispatch: {dispatch_profile.us:.3f} us | '
                       f'urma: {dispatch_urma_profile.us:.3f} us, '
                       f'{dispatch_urma_profile.gbps(num_dispatch_urma_bytes):.0f} GB/s | '
                       f'epilogue: {epilogue_profile.us:.3f} us, '
                       f'{epilogue_profile.gbps(num_dispatch_epilogue_bytes):.0f} GB/s')

            cached_dispatch_profile, cached_dispatch_urma_profile, cached_epilogue_profile = benchmark(
                fn=lambda: dispatch_once(use_cached=True),
                kernel_names=['dispatch_impl', 'dispatch_copy_epilogue_impl'],
            )
            # Cached epilogues read existing slot indices instead of writing source metadata.
            num_cached_dispatch_epilogue_bytes = (
                num_dispatch_epilogue_bytes - num_recv_tokens * num_recv_metadata_bytes +
                num_unaligned_expanded_tokens * handle.recv_src_metadata.element_size())
            dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks:3} | '
                       f'cached  : {cached_dispatch_profile.us:.3f} us | '
                       f'urma: {cached_dispatch_urma_profile.us:.3f} us, '
                       f'{cached_dispatch_urma_profile.gbps(num_dispatch_urma_bytes):.0f} GB/s | '
                       f'epilogue: {cached_epilogue_profile.us:.3f} us, '
                       f'{cached_epilogue_profile.gbps(num_cached_dispatch_epilogue_bytes):.0f} GB/s')

            def combine_once():
                event = buffer.combine(
                    x=input_for_combine,
                    handle=handle,
                    topk_weights=expanded_recv_topk_weights,
                    bias=bias,
                    num_sms=num_ai_cores,
                    async_with_compute_stream=True,
                    defer_epilogue=True,
                    use_fp32_output=args.use_fp32_output,
                )
                flush_l2_cache()
                event.current_stream_wait(release_handle=True)
                torch.npu.synchronize()

            combine_profile, combine_urma_profile, combine_epilogue_profile = benchmark(
                fn=combine_once,
                kernel_names=['combine_impl', 'combine_reduce_epilogue_impl'],
            )
            num_combine_epilogue_bytes = (
                num_epilogue_slots + (num_bias + 1 + args.use_fp32_output) * num_tokens) * num_bytes_per_combine_token
            dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks:3} | '
                       f'combine: {combine_profile.us:.3f} us | '
                       f'urma: {combine_urma_profile.us:.3f} us, '
                       f'{combine_urma_profile.gbps(num_combine_urma_bytes):.0f} GB/s | '
                       f'epilogue: {combine_epilogue_profile.us:.3f} us, '
                       f'{combine_epilogue_profile.gbps(num_combine_epilogue_bytes):.0f} GB/s')

        if args.test_first_only:
            break
    dist_print('', once_in_node=True)


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)

    def construct_ep_buffer():
        return deep_ep.EPBuffer(
            group,
            num_max_tokens_per_rank=args.num_tokens,
            hidden=args.hidden,
            num_topk=args.num_topk,
            deterministic=args.deterministic,
            allow_hybrid_mode=True,
            allow_multiple_reduction=bool(args.allow_multiple_reduction),
            explicitly_destroy=True,
        )

    buffer = construct_ep_buffer()

    if args.precise_unbalanced_ratio:
        dist_print('\033[33mWarning: Using precise unbalanced ratio mode. '
                   'Test data is manually constructed and may differ from real world distribution.\033[0m',
                   once_in_node=True)

    # Test once
    test_dispatch_combine(buffer, args)

    # Pressure tests
    for seed in range(int(1e9) if args.do_pressure_test else 0):
        if not args.reuse_ep_buffer:
            buffer.destroy()
            buffer = construct_ep_buffer()

        assert not args.skip_check
        dist_print(f'Testing with {seed=} ...', once_in_node=True)
        init_seed(seed)
        test_dispatch_combine(buffer, args)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test EP dispatch and combine kernels')

    # Resource settings
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn')
    parser.add_argument('--num-ai-cores', type=int, default=0, help='Number of AI cores to use (0 means all)')

    # Model settings
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of top-k experts')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    parser.add_argument('--use-fp32-output', action='store_true', help='Return FP32 combine output instead of BF16')

    # Scenario settings
    parser.add_argument('--do-cpu-sync', type=int, default=1, help='Whether to do CPU sync')
    parser.add_argument('--defer-epilogue', type=int, default=1,
                        help='Whether to defer epilogues in correctness tests')
    parser.add_argument('--dispatch-dtype', choices=('all', 'bf16', 'fp8'), default='all',
                        help='Filter dispatch dtype, also applies to --test-first-only')
    parser.add_argument('--allow-hybrid-mode', type=int, default=0, help='Must be 0 on Ascend')
    parser.add_argument('--allow-multiple-reduction', type=int, default=1, help='Must be 1 on Ascend')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic dispatch ordering')

    # Test settings
    parser.add_argument('--seed', type=int, default=0, help='Default seed for pressure tests')
    parser.add_argument('--skip-check', action='store_true', help='Whether to skip correctness checks')
    parser.add_argument('--skip-perf-test', action='store_true', help='Whether to skip performance tests')
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')
    parser.add_argument('--reuse-ep-buffer', action='store_true',
                        help='Whether to reuse EP buffer for each test')
    parser.add_argument('--test-first-only', action='store_true', help='Only test the first case')
    parser.add_argument('--unbalanced-ratio', type=float, default=1.0, help='The MoE unbalanced ratio')
    routing_modes = parser.add_mutually_exclusive_group()
    routing_modes.add_argument('--precise-unbalanced-ratio', action='store_true',
                               help='Generate top-k indices with precise unbalanced ratio')
    routing_modes.add_argument('--synthetic-unbalanced-ratio', action='store_true',
                               help='Generate synthetic top-k routing with calibrated rank load')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
