#pragma once

#include <cstdint>

#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <c_api/scalar_compute/scalar_compute.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/impls/ep/combine_utils.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

namespace deep_ep {

template <int kNumTopk, int kNumTokenElems, int kNumSlots, int kNumHiddenElems>
__simd_vf__ __forceinline__ void restore_combine_weights(
    __ubuf__ uint32_t* ub_tokens, __ubuf__ uint32_t* ub_output,
    const int first_token_idx, const int num_tokens) {
    uint32_t remaining = kNumTopk;
    const auto mask = asc_update_mask_b32(remaining);
    vector_uint32_t result;
    asc_duplicate_scalar(result, static_cast<uint32_t>(0), mask);
    const auto num_wrap_tokens = math::min(kNumSlots - first_token_idx, num_tokens);
    auto ub_input = ub_tokens + (first_token_idx * kNumTokenElems + kNumHiddenElems) / 2;
    #pragma unroll 1
    for (int i = 0; i < num_wrap_tokens; ++ i) {
        vector_uint32_t weights;
        asc_loadalign_postupdate(weights, ub_input, kNumTokenElems / 2);
        asc_or(result, result, weights, mask);
    }
    ub_input = ub_tokens + kNumHiddenElems / 2;
    #pragma unroll 1
    for (int i = num_wrap_tokens; i < num_tokens; ++ i) {
        vector_uint32_t weights;
        asc_loadalign_postupdate(weights, ub_input, kNumTokenElems / 2);
        asc_or(result, result, weights, mask);
    }
    asc_storealign(ub_output, result, mask);
}

template <bool kOutputFP32, int kNumRanks, int kNumExperts, int kNumTopk,
          int kHidden, int kNumMaxTokensPerRank, int kNumUbBytes,
          int kNumVecCores, int64_t kNumTimeoutCycles, bool kDoBarrier = true,
          int kNumHiddenBytes = kHidden * static_cast<int>(sizeof(bfloat16_t))>
__global__ __vector__ void combine_reduce_epilogue_impl(
    __gm__ std::conditional_t<kOutputFP32, float, bfloat16_t>* combined_x,
    __gm__ float* combined_topk_weights,
    __gm__ int64_t* combined_topk_idx,
    __gm__ bfloat16_t* buffer,
    __gm__ bfloat16_t* bias_0,
    __gm__ bfloat16_t* bias_1,
    __gm__ void* workspace,
    __gm__ void* device_context_ptr,
    const int rank_idx,
    const int num_combined_tokens) {
    AscendC::InitSocState();

    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    if constexpr (kDoBarrier)
        comm::scalar::barrier<
            kNumRanks, kNumRanks, kNumTimeoutCycles, true, false, true, kNumVecCores, kNumVecCores>(
                static_cast<__gm__ layout::EPSignals*>(workspace)->barrier_signals, *device_context, vec_core_idx, rank_idx);

    static constexpr int kNumWeightBytes = math::align(kNumTopk * static_cast<int>(sizeof(float)), kNumUbAlignmentBytes);
    static constexpr int kNumHiddenSlots = math::min(32, kNumUbBytes / (kNumHiddenBytes + kNumWeightBytes));
    static constexpr int kNumCombineTokenElems = kHidden + kNumWeightBytes / static_cast<int>(sizeof(bfloat16_t));
    using HiddenReducer = RingReducer<
        kHidden, math::min(kNumRanks, kNumTopk) + 2, kNumHiddenSlots, 0, kNumWeightBytes, kOutputFP32>;

    struct UBLayout {
        typename HiddenReducer::Buffer hidden;
    };

    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks <= kNumMaxEPRanks and kNumExperts % kNumRanks == 0);
    EP_STATIC_ASSERT(kNumExperts <= kNumMaxExperts);
    EP_STATIC_ASSERT(kNumTopk > 0 and kNumTopk <= 32);
    EP_STATIC_ASSERT(kNumMaxTokensPerRank > 0);
    EP_DEVICE_ASSERT(0 <= num_combined_tokens and num_combined_tokens <= kNumMaxTokensPerRank);
    EP_STATIC_ASSERT(math::min(kNumRanks, kNumTopk) + 2 + kOutputFP32 <= kNumHiddenSlots);
    EP_STATIC_ASSERT(sizeof(UBLayout) <= kNumUbBytes);

    const auto num_vec_cores = static_cast<int>(AscendC::GetBlockNum());
    const auto num_biases = (bias_0 != nullptr) + (bias_1 != nullptr);
    const auto has_topk_weights = combined_topk_weights != nullptr;
    __ubuf__ UBLayout* ub = 0;
    HiddenReducer reducer(&ub->hidden);

    const auto num_tokens_per_core = math::ceil_div(num_combined_tokens, num_vec_cores);
    const auto token_begin = vec_core_idx * num_tokens_per_core;
    const auto num_core_tokens = math::min(num_combined_tokens, (vec_core_idx + 1) * num_tokens_per_core) - token_begin;
    // Stagger token reads within each core's contiguous range.
    const auto token_rotation = num_core_tokens > 0 ? vec_core_idx % num_core_tokens : 0;
    for (int local_token_idx = 0; local_token_idx < num_core_tokens; ++ local_token_idx) {
        const auto relative_token_idx = local_token_idx + token_rotation;
        const auto token_idx = token_begin +
            (relative_token_idx < num_core_tokens ? relative_token_idx : relative_token_idx - num_core_tokens);
        // Keep the last top-k slot for each destination rank.
        uint64_t master_slot_mask = 0;
        uint64_t seen_ranks[math::ceil_div(kNumRanks, 64)] = {};
        #pragma unroll
        for (int slot_idx = kNumTopk - 1; slot_idx >= 0; -- slot_idx) {
            const auto expert_idx = combined_topk_idx[token_idx * kNumTopk + slot_idx];
            if (expert_idx < 0)
                continue;
            const auto dst_rank_idx = expert_idx / (kNumExperts / kNumRanks);
            const auto rank_bit = uint64_t{1} << (dst_rank_idx % 64);
            if ((seen_ranks[dst_rank_idx / 64] & rank_bit) == 0)
                master_slot_mask |= uint64_t{1} << slot_idx;
            seen_ranks[dst_rank_idx / 64] |= rank_bit;
        }
        // Load biases and rank partials in accumulation order.
        const auto num_inputs = num_biases + static_cast<int>(asc_popc(master_slot_mask));
        if constexpr (kOutputFP32)
            reducer.prepare_fp32_output();
        const auto first_input_idx = reducer.slot_idx;
        for (int bias_idx = 0; bias_idx < 2; ++ bias_idx) {
            const auto bias = bias_idx == 0 ? bias_0 : bias_1;
            if (bias != nullptr) {
                reducer.load_next_token(bias + static_cast<int64_t>(token_idx) * kHidden,
                                        asc_load_l2_cache_mode::NOTALLOC_KEEP, 0);
            }
        }
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumTopk; ++ slot_idx) {
            if ((master_slot_mask & (uint64_t{1} << slot_idx)) == 0)
                continue;
            auto slot_group_idx = slot_idx;
            if constexpr (kNumRanks < kNumTopk) {
                const auto expert_idx = combined_topk_idx[token_idx * kNumTopk + slot_idx];
                slot_group_idx = expert_idx / (kNumExperts / kNumRanks);
            }
            const auto combine_slot_idx = slot_group_idx * kNumMaxTokensPerRank + token_idx;
            const auto gm_input = buffer + static_cast<int64_t>(combine_slot_idx) * kNumCombineTokenElems;
            reducer.load_next_token(gm_input, asc_load_l2_cache_mode::NOTALLOC_KEEP,
                                    has_topk_weights ? kNumWeightBytes : 0);
        }
        reducer.reduce(num_inputs, [&](const uint32_t output_idx) __aicore__ {
            if (has_topk_weights) {
                restore_combine_weights<kNumTopk, HiddenReducer::kNumTokenElems, kNumHiddenSlots, kHidden>(
                    reinterpret_cast<__ubuf__ uint32_t*>(ub->hidden.buf[0]),
                    reinterpret_cast<__ubuf__ uint32_t*>(ub->hidden.buf[output_idx] + kHidden),
                    (first_input_idx + num_biases) % kNumHiddenSlots, num_inputs - num_biases);
            }
        });
        reducer.store_last_token(combined_x + static_cast<int64_t>(token_idx) * kHidden,
            asc_store_l2_cache_mode::NOTALLOC_CLEAN,
            has_topk_weights ? reinterpret_cast<__gm__ uint8_t*>(combined_topk_weights + token_idx * kNumTopk) : nullptr,
            kNumTopk * sizeof(float));
    }
}

}  // namespace deep_ep
