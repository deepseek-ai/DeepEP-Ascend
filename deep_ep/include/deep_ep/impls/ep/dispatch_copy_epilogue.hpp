#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>
#include <c_api/atomic/atomic.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>
#include <c_api/asc_simd.h>

namespace deep_ep {

using namespace __cce_simd;

// ptr must be 32-byte aligned; num_elems may be arbitrary.
__simd_vf__ void fill_ubuf(__ubuf__ int* ptr, const int num_elems, const int value) {
    static constexpr int kNumElemsPerVector = sizeof(vector_int32_t) / sizeof(int);
    vector_int32_t value_vec;
    asc_duplicate_scalar(value_vec, value);

    uint32_t remaining = static_cast<uint32_t>(num_elems);
    for (int elem_offset = 0; elem_offset < num_elems; elem_offset += kNumElemsPerVector) {
        asc_storealign(ptr + elem_offset, value_vec, asc_update_mask_b32(remaining));
    }
}

/**
 *  for (int i = 0; i < num_i64_elems; ++ i)
 *      if (offset <= data[i] and data[i] < offset + kNumBins)
 *          acc[data[i] - offset] ++;
 */
template <int kNumBins>
__simd_vf__ void histogram_accumulate_u32(__ubuf__ int64_t* data, __ubuf__ uint32_t* acc, const int num_i64_elems, const int16_t offset) {
    EP_STATIC_ASSERT(kNumBins == 256 or kNumBins == 512);
    static constexpr int kNumBinsPerHistogram = sizeof(vector_uint16_t) / sizeof(uint16_t);
    static constexpr int kNumHistogramVectors = kNumBins / kNumBinsPerHistogram;
    vector_bool mask_b16 = asc_create_mask_b16(PAT_ALL);
    vector_bool mask_b32 = asc_create_mask_b32(PAT_ALL);

    vector_int16_t offset_vec;
    asc_duplicate_scalar(offset_vec, offset, mask_b16);

    vector_uint16_t hist[kNumHistogramVectors];
    #pragma unroll
    for (int hist_idx = 0; hist_idx < kNumHistogramVectors; ++ hist_idx)
        asc_duplicate_scalar(hist[hist_idx], static_cast<uint16_t>(0), mask_b16);

    static constexpr int kNumI64ElemsPerLoad = 2 * sizeof(vector_int16_t) / sizeof(int64_t);
    uint32_t remaining = static_cast<uint32_t>(num_i64_elems);
    for (int elem_offset = 0; elem_offset < num_i64_elems; elem_offset += 4 * kNumI64ElemsPerLoad) {

        // int64[N] = int16[N][4], downsample loads int16[N][0,2], which can save number of instructions
        vector_uint16_t sampled_0, sampled_1, sampled_2, sampled_3;
        asc_loadalign_downsample(sampled_0, (__ubuf__ uint16_t*)(data + elem_offset));
        asc_loadalign_downsample(sampled_1, (__ubuf__ uint16_t*)(data + elem_offset + kNumI64ElemsPerLoad));
        asc_loadalign_downsample(sampled_2, (__ubuf__ uint16_t*)(data + elem_offset + 2 * kNumI64ElemsPerLoad));
        asc_loadalign_downsample(sampled_3, (__ubuf__ uint16_t*)(data + elem_offset + 3 * kNumI64ElemsPerLoad));

        // deinterleave to separate int16[N][0,2] => (int16[N][0], int16[N][2]), then ignore int16[N][2]
        vector_int16_t value_0, value_1;
        vector_uint16_t ignored_0, ignored_1;
        asc_deintlv((vector_u16&)value_0, ignored_0, sampled_0, sampled_1);
        asc_deintlv((vector_u16&)value_1, ignored_1, sampled_2, sampled_3);

        asc_sub(value_0, value_0, offset_vec, mask_b16);
        asc_sub(value_1, value_1, offset_vec, mask_b16);

        vector_uint8_t value_byte_0, value_byte_1;
        asc_deintlv(value_byte_0, value_byte_1, (vector_u8&)value_0, (vector_u8&)value_1);

        vector_bool hist_mask_0, hist_mask_1;
        const auto mask = asc_update_mask_b8(remaining);
        asc_eq_scalar(hist_mask_0, value_byte_1, static_cast<uint8_t>(0), mask);
        asc_frequency_histogram_bin0(hist[0], value_byte_0, hist_mask_0);
        asc_frequency_histogram_bin1(hist[1], value_byte_0, hist_mask_0);
        if constexpr (kNumBins == 512) {
            asc_eq_scalar(hist_mask_1, value_byte_1, static_cast<uint8_t>(1), mask);
            asc_frequency_histogram_bin0(hist[2], value_byte_0, hist_mask_1);
            asc_frequency_histogram_bin1(hist[3], value_byte_0, hist_mask_1);
        }
    }

    // Accumulate histogram into counters
    #pragma unroll
    for (int hist_idx = 0; hist_idx < kNumHistogramVectors; ++ hist_idx) {
        const auto histogram_offset = acc + hist_idx * kNumBinsPerHistogram;
        vector_u32 hist_p0, hist_p1, hist_0, hist_1, acc_0, acc_1;
        asc_unpack_lower(hist_p0, hist[hist_idx]);
        asc_unpack_upper(hist_p1, hist[hist_idx]);
        asc_deintlv(hist_0, hist_1, hist_p0, hist_p1);
        asc_loadalign_deintlv(acc_0, acc_1, histogram_offset);
        asc_add(hist_0, hist_0, acc_0, mask_b32);
        asc_add(hist_1, hist_1, acc_1, mask_b32);
        asc_storealign_intlv(histogram_offset, hist_0, hist_1);
    }
}

/**
 *  for (int i = 0; i < kNumBins; ++i)
 *      dst_slot_idx[i] = dst_slot_idx[i] + align(unaligned_expert_start_slot_idx[i], kExpertAlignment)
 */
template <int kNumBins, int kExpertAlignment>
__simd_vf__ void initialize_dst_slot_idx(__ubuf__ uint32_t* dst_slot_idx,
                                         __ubuf__ uint32_t* unaligned_expert_start_slot_idx) {
    EP_STATIC_ASSERT(kNumBins == 256 or kNumBins == 512);
    EP_STATIC_ASSERT(math::is_power_of_2(kExpertAlignment));
    vector_bool mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t alignment_mask;
    asc_duplicate_scalar(alignment_mask, static_cast<uint32_t>(-kExpertAlignment), mask_b32);

    #pragma unroll
    for (int elem_offset = 0; elem_offset < kNumBins; elem_offset += sizeof(vector_uint32_t) / sizeof(uint32_t)) {
        vector_uint32_t dst_slot_idx_vec;
        vector_uint32_t expert_start_slot_idx_vec;
        // dst_slot_idx[i] = dst_slot_idx[i] + (unaligned_expert_start_slot_idx[i] + (alignment - 1)) & (-alignment)
        asc_loadalign(dst_slot_idx_vec, dst_slot_idx + elem_offset);
        asc_loadalign(expert_start_slot_idx_vec, unaligned_expert_start_slot_idx + elem_offset);
        asc_add_scalar(expert_start_slot_idx_vec, expert_start_slot_idx_vec, static_cast<uint32_t>(kExpertAlignment - 1), mask_b32);
        asc_and(expert_start_slot_idx_vec, expert_start_slot_idx_vec, alignment_mask, mask_b32);
        asc_add(dst_slot_idx_vec, dst_slot_idx_vec, expert_start_slot_idx_vec, mask_b32);
        asc_storealign(dst_slot_idx + elem_offset, dst_slot_idx_vec, mask_b32);
    }
}

template <int kNumRanks>
struct PSumIterator {
    __gm__ const int* psum;
    int token_idx;
    int token_end_idx;
    int rank_idx;
    int rank_start_idx;
    int rank_end_idx;

    __aicore__ __forceinline__ PSumIterator(__gm__ const int* psum, const int token_idx, const int token_end_idx):
        psum(psum),
        token_idx(token_idx),
        token_end_idx(token_end_idx),
        rank_idx(0),
        rank_start_idx(0),
        rank_end_idx(psum[0])
    {
        advance_to(token_idx);
    }

    __aicore__ __forceinline__ void advance(const int num_tokens) {
        advance_to(token_idx + num_tokens);
    }

    __aicore__ __forceinline__ void advance_to(const int target_token_idx) {
        token_idx = target_token_idx;
        while (target_token_idx >= rank_end_idx and rank_idx + 1 < kNumRanks) {
            rank_start_idx = rank_end_idx;
            ++ rank_idx;
            rank_end_idx = psum[rank_idx];
        }
    }

    __aicore__ __forceinline__ int get_local_idx() const {
        return token_idx - rank_start_idx;
    }

    __aicore__ __forceinline__ int get_contiguous_count(const int max_tokens) const {
        return math::min(math::min(rank_end_idx - token_idx, token_end_idx - token_idx), max_tokens);
    }

    __aicore__ __forceinline__ bool finished() const {
        return token_idx >= token_end_idx;
    }
};

template <int kNumRanks, int kNumExperts, int kNumTopk, int kExpertAlignment,
          bool kCachedMode, bool kDoZeroPadding,
          int kNumVecCores, int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank, int kNumUBBytes,
          int64_t kNumTimeoutCycles, bool kDoBarrier = true>
__global__ __vector__ void dispatch_copy_epilogue_impl(
    __gm__ void* buffer,
    __gm__ void* workspace,
    __gm__ void* device_context_ptr,
    __gm__ int* psum_num_recv_tokens_per_rank,
    __gm__ int* psum_num_recv_tokens_per_expert,
    __gm__ void* recv_x,
    __gm__ sf_pack_t* recv_sf,
    __gm__ float* recv_topk_weights,
    __gm__ int* recv_src_metadata,
    const int rank_idx,
    const int recv_sf_token_stride,
    const int recv_sf_hidden_stride
) {
    AscendC::InitSocState();

    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace);
    if constexpr (kDoBarrier)
        comm::scalar::barrier<
            kNumRanks, kNumRanks, kNumTimeoutCycles, true, true, true, kNumVecCores, kNumVecCores>(
                workspace_layout.get_barrier_signals(), *device_context, vec_core_idx, rank_idx);

    EP_STATIC_ASSERT(kNumRanks <= kNumMaxEPRanks);
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0);
    EP_STATIC_ASSERT(math::is_power_of_2(kExpertAlignment));
    EP_STATIC_ASSERT(kNumVecCores > 0);
    EP_STATIC_ASSERT(kNumTopk > 0);
    EP_STATIC_ASSERT(kNumHiddenBytes > 0);
    EP_STATIC_ASSERT(kNumMaxTokensPerRank > 0);
    EP_STATIC_ASSERT(kNumUBBytes > 0);

    static constexpr int kNumMaxNotifyWaitStages = 8;

    static constexpr int kNumZeroBytes = sizeof(vector_u8);
    static constexpr int kNumSFBytes = kNumSFPacks * sizeof(sf_pack_t);
    static constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;

    static constexpr int kNumHistogramStages = 4;
    static constexpr int kNumHistogramStageBytes = 32768;
    static constexpr int kNumHistogramBins = kNumExpertsPerRank <= 256 ? 256 : 512;
    static constexpr int kNumHistogramTokenTopkElems = math::align(kNumTopk, kNumUbAlignmentBytes / static_cast<int>(sizeof(int64_t)));
    static constexpr int kNumHistogramMaxElems = kNumHistogramStageBytes / sizeof(int64_t);
    static constexpr int kNumMaxHistogramTokensPerStage = kNumHistogramMaxElems / kNumHistogramTokenTopkElems;

    static constexpr int kNumTokenBytes = static_cast<int>(layout::TokenLayout(kNumHiddenBytes, kNumSFBytes, kNumTopk, not kCachedMode).get_num_bytes());
    static constexpr int kNumTargetTokenStageBytes = 64 * 1024;
    static constexpr int kNumTokensPerStage = math::max(1, kNumTargetTokenStageBytes / kNumTokenBytes);

    // metadata traffic is small, so 2 stage is enough
    static constexpr int kNumMetadataStages = 2;
    static constexpr int kNumMetadataElemsPerToken = 2 + kNumTopk;
    static constexpr int kNumMetadataBytesPerToken = kNumMetadataElemsPerToken * sizeof(int);
    static constexpr int kNumMetadataStageBytes = math::align(kNumTokensPerStage * kNumMetadataBytesPerToken, kNumUbAlignmentBytes);

    // ExpertState: ub_expert_token_count, ub_psum_num_recv_tokens_per_expert, ub_next_dst_slot_idx
    static constexpr int kNumExpertStateBytes = 2 * kNumHistogramBins * sizeof(uint32_t);
    static constexpr int kNumMetadataPipelineBytes = kNumMetadataStages * kNumMetadataStageBytes;

    static constexpr int kNumMTEStages = math::min(
        kNumMaxNotifyWaitStages - kNumMetadataStages,
        (kNumUBBytes - kNumExpertStateBytes - kNumMetadataPipelineBytes -
         (kDoZeroPadding ? kNumZeroBytes : 0)) / (kNumTokenBytes * kNumTokensPerStage));

    EP_STATIC_ASSERT(kNumExpertsPerRank <= 512);
    EP_STATIC_ASSERT(kNumMaxHistogramTokensPerStage * kNumTopk <= 65535);
    EP_STATIC_ASSERT(kNumExpertStateBytes + kNumHistogramStages * kNumHistogramStageBytes <= kNumUBBytes);
    EP_STATIC_ASSERT(kNumExpertStateBytes + kNumMetadataPipelineBytes + (kDoZeroPadding ? kNumZeroBytes : 0) <= kNumUBBytes);
    EP_STATIC_ASSERT(kNumMTEStages + kNumMetadataStages <= kNumMaxNotifyWaitStages, "Insufficient events");
    EP_STATIC_ASSERT(kNumMTEStages > 0);

    const auto ub_expert_token_count = reinterpret_cast<__ubuf__ uint32_t*>(0);
    const auto ub_next_dst_slot_idx = ub_expert_token_count; // reuse the ub_expert_token_count buffer
    const auto ub_psum_num_recv_tokens_per_expert = math::advance_ptr<__ubuf__ uint32_t>(ub_expert_token_count, kNumHistogramBins * sizeof(uint32_t));
    const auto ub_histogram = math::advance_ptr<__ubuf__ int64_t>(ub_psum_num_recv_tokens_per_expert, kNumHistogramBins * sizeof(uint32_t));
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, not kCachedMode);
    const auto recv_buffer = layout::BufferLayout(token_layout, kNumRanks, kNumMaxTokensPerRank, math::advance_ptr<void>(buffer, 0));
    const auto ub_token_buffer = layout::BufferLayout(token_layout, kNumMTEStages, kNumTokensPerStage, math::advance_ptr<void>(ub_histogram, 0));
    const auto ub_metadata_buffers = math::advance_ptr<__ubuf__ int>(ub_token_buffer.get_buffer_end_ptr(), 0);
    const auto ub_zero_buffer = math::advance_ptr<__ubuf__ uint8_t>(ub_metadata_buffers, kNumMetadataStages * kNumMetadataStageBytes);

    if constexpr (not kCachedMode) {
        asc_set_copy_pad_val(static_cast<uint32_t>(0));
        // expert-start[expert_id] = align(expert-end[expert_id - 1], kExpertAlignment)
        // we first load ub_psum_num_recv_tokens_per_expert[expert_id] = expert-end[expert_id - 1]
        if constexpr (kNumExpertsPerRank == 1) {
            *ub_psum_num_recv_tokens_per_expert = 0;
            asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
            asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
        } else {
            asc_copy_gm2ub_align(
                ub_psum_num_recv_tokens_per_expert, reinterpret_cast<__gm__ uint32_t*>(psum_num_recv_tokens_per_expert),
                1, (kNumExpertsPerRank - 1) * sizeof(uint32_t), 1, 0, true,
                static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV), 0, 0
            );
        }
    }

    const auto num_recv_tokens = psum_num_recv_tokens_per_rank[kNumRanks - 1];
    const auto num_tokens_per_vec_core = math::ceil_div(num_recv_tokens, kNumVecCores);
    const auto token_start_idx = math::min(num_recv_tokens, vec_core_idx * num_tokens_per_vec_core);
    const auto token_end_idx = math::min(num_recv_tokens, token_start_idx + num_tokens_per_vec_core);

    constexpr int64_t kExpertCountClearedBarrier = 0;
    constexpr int64_t kExpertCountCumsumBarrier = 1;

    PSumIterator<kNumRanks> issue_iter(psum_num_recv_tokens_per_rank, token_start_idx, token_end_idx);
    int stage_num_tokens[kNumMTEStages] = {};
    int stage_token_base_idx[kNumMTEStages] = {};
    const auto issue_load = [&](int load_stage_idx, bool is_preload) __aicore__ {
        if (issue_iter.finished())
            return false;

        const auto num_stage_tokens = issue_iter.get_contiguous_count(kNumTokensPerStage);
        const auto load_ub_ptr = ub_token_buffer
            .get_token_buffer(load_stage_idx * kNumTokensPerStage)
            .get_base_ptr<__ubuf__ uint8_t>();
        const auto load_gm_ptr = recv_buffer
            .get_rank_buffer(issue_iter.rank_idx)
            .get_token_buffer(issue_iter.get_local_idx())
            .template get_base_ptr<__gm__ uint8_t>();

        // refills wait until the consumer's MTE3 copies that read this stage have completed
        if (not is_preload) {
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kNumMetadataStages + load_stage_idx));
        }
        // we won't use the token data after this copy, so we use NOTALLOC_CLEAN to avoid polluting the L2 cache
        asc_copy_gm2ub_align(
            load_ub_ptr, load_gm_ptr,
            1, num_stage_tokens * static_cast<uint32_t>(token_layout.get_num_bytes()), 0, 0, true,
            static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN), 0, 0
        );
        // load done: the consumer's S pipe waits on this before reading the stage
        asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(kNumMetadataStages + load_stage_idx));

        stage_num_tokens[load_stage_idx] = num_stage_tokens;
        // global recv-stream position: recv_src_metadata is indexed by the global stream
        stage_token_base_idx[load_stage_idx] = issue_iter.token_idx;

        issue_iter.advance(num_stage_tokens);

        return true;
    };

    int preload_stage_idx = 0;

    if constexpr (not kCachedMode) {
        // Clear this core's prefix row before issuing metadata loads
        fill_ubuf((__ubuf__ int*)ub_expert_token_count, kNumHistogramBins, 0);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // Clear the cumsum buffer in global memory
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2gm_align(
            workspace_layout.get_epilogue_expert_counter_ptr(vec_core_idx, 0, kNumExpertsPerRank),
            (__ubuf__ int *)ub_expert_token_count,
            kNumExpertsPerRank * sizeof(int)
        );
        asc_sync_inter_arrive(PIPE_MTE3, kExpertCountClearedBarrier);

        PSumIterator<kNumRanks> topk_iter(psum_num_recv_tokens_per_rank, token_start_idx, token_end_idx);
        #pragma unroll
        for (int histogram_stage_idx = 0; histogram_stage_idx < kNumHistogramStages; ++ histogram_stage_idx)
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(histogram_stage_idx));

        uint8_t histogram_stage_idx = 0;
        asc_set_copy_pad_val(static_cast<int32_t>(-1));
        while (not topk_iter.finished()) {
            const auto num_stage_tokens = topk_iter.get_contiguous_count(kNumMaxHistogramTokensPerStage);
            const auto ub_hist_ptr = math::advance_ptr<__ubuf__ int64_t>(ub_histogram, histogram_stage_idx * kNumHistogramStageBytes);
            const auto buffer_token = recv_buffer
                .get_rank_buffer(topk_iter.rank_idx)
                .get_token_buffer(topk_iter.get_local_idx());

            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(histogram_stage_idx));
            asc_copy_gm2ub_align(
                (__ubuf__ int32_t*)ub_hist_ptr,
                (__gm__ int32_t*)(buffer_token.template get_topk_idx_ptr<__gm__ int64_t>()),
                /* n_burst= */ num_stage_tokens, /* len_burst= */ kNumTopk * sizeof(int64_t),
                /* left_padding_num= */ 0,
                /* right_padding_num= */ (kNumHistogramTokenTopkElems - kNumTopk) * 2,
                /* enable_constant_pad= */ true,
                /* l2_cache_mode= */ static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV), // we will use the data later, so we want to cache it
                /* src_gap= */ token_layout.get_num_bytes(),
                /* dst_gap= */ kNumHistogramTokenTopkElems * sizeof(int64_t)
            );
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(histogram_stage_idx));

            asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(histogram_stage_idx));
            histogram_accumulate_u32<kNumHistogramBins>(
                ub_hist_ptr, ub_expert_token_count,
                num_stage_tokens * kNumHistogramTokenTopkElems,
                static_cast<int16_t>(rank_idx * kNumExpertsPerRank)
            );
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(histogram_stage_idx));

            histogram_stage_idx = (histogram_stage_idx + 1) % kNumHistogramStages;
            topk_iter.advance(num_stage_tokens);
        }
        asc_set_copy_pad_val(static_cast<int32_t>(0));

        #pragma unroll
        for (int histogram_stage_idx = 0; histogram_stage_idx < kNumHistogramStages; ++ histogram_stage_idx)
            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(histogram_stage_idx));

        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // reserve 1 issue slot for the later gm2ub copy
        while (preload_stage_idx < kNumMTEStages and get_iqent(PIPE_MTE2) > 1) {
            issue_load(preload_stage_idx, true);
            preload_stage_idx ++;
        }

        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_inter_wait(PIPE_MTE3, kExpertCountClearedBarrier);
        if (vec_core_idx + 1 < kNumVecCores) {
            // Atomic & Broadcast Trick
            //
            // int expert_count[kNumVecCores][kNumExperts];
            // for (int i = my_vec_core_idx + 1; i < kNumVecCores; ++i)  // n_burst = kNumVecCores - my_vec_core_idx - 1
            //    expert_count[i][:] += my_expert_count[:];              // burst_len = kNumExpertsPerRank * sizeof(int)
            asc_set_atomic_add_int();
            asc_copy_ub2gm_align(
                workspace_layout.get_epilogue_expert_counter_ptr(vec_core_idx + 1, 0, kNumExpertsPerRank),
                (__ubuf__ int*)ub_expert_token_count,
                kNumVecCores - vec_core_idx - 1, kNumExpertsPerRank * sizeof(int),
                static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                static_cast<uint64_t>(kNumExpertsPerRank) * sizeof(int), 0
            );
            asc_set_atomic_none();
        }
        asc_sync_inter_arrive(PIPE_MTE3, kExpertCountCumsumBarrier);
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        asc_sync_inter_wait(PIPE_MTE2, kExpertCountCumsumBarrier);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_copy_gm2ub_align(
            (__ubuf__ int*)ub_next_dst_slot_idx,
            workspace_layout.get_epilogue_expert_counter_ptr(vec_core_idx, 0, kNumExpertsPerRank),
            1, kNumExpertsPerRank * sizeof(int), 0, 0, true,
            static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV), 0, 0
        );
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
        initialize_dst_slot_idx<kNumHistogramBins, kExpertAlignment>(ub_next_dst_slot_idx, ub_psum_num_recv_tokens_per_expert);
        asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    }

    for (int i = preload_stage_idx; i < kNumMTEStages; ++ i) {
        issue_load(i, true);
    }

    if constexpr (kDoZeroPadding) {
        fill_ubuf((__ubuf__ int*)ub_zero_buffer, kNumZeroBytes / sizeof(int), 0);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }

    uint8_t stage_idx = 0;
    uint8_t metadata_buffer_idx = 0;
    if constexpr (not kCachedMode) {
        #pragma unroll
        for (int i = 0; i < kNumMetadataStages; ++ i)
            asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(i));
    }

    const auto num_range_tokens = token_end_idx - token_start_idx;
    int num_consumed_tokens = 0;
    while (num_consumed_tokens < num_range_tokens) {
        const auto num_stage_tokens = stage_num_tokens[stage_idx];
        const auto token_base_idx = stage_token_base_idx[stage_idx];
        // wait for this stage's load to complete
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(kNumMetadataStages + stage_idx));
        if constexpr (not kCachedMode) {
            // wait metadata buffer to be ready for writing
            asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(metadata_buffer_idx));
        }
        for (int token_in_stage = 0; token_in_stage < num_stage_tokens; ++ token_in_stage) {
            const auto token_idx = token_base_idx + token_in_stage;
            const auto token = ub_token_buffer
                .get_token_buffer(stage_idx * kNumTokensPerStage + token_in_stage);
            // Source metadata
            const auto metadata_ptr = recv_src_metadata + token_idx * kNumMetadataElemsPerToken;
            const auto ub_metadata = math::advance_ptr<__ubuf__ int>(
                ub_metadata_buffers,
                metadata_buffer_idx * kNumMetadataStageBytes + token_in_stage * kNumMetadataBytesPerToken);
            int master_topk_slot_idx = -1;

            #pragma unroll
            for (int j = 0; j < kNumTopk; ++ j) {
                // Allocate destination slots
                int dst_slot_idx;
                bool has_dst_slot;
                if constexpr (kCachedMode) {
                    dst_slot_idx = metadata_ptr[j];
                    has_dst_slot = dst_slot_idx >= 0;
                } else {
                    const auto dst_expert_idx = static_cast<int>(token.get_topk_idx_ptr<__ubuf__ int64_t>()[j]);
                    const auto local_expert_idx = dst_expert_idx - rank_idx * kNumExpertsPerRank;
                    dst_slot_idx = -1;
                    has_dst_slot = 0 <= local_expert_idx and local_expert_idx < kNumExpertsPerRank;
                    if (has_dst_slot) {
                        dst_slot_idx = static_cast<int>(ub_next_dst_slot_idx[local_expert_idx]);
                        ub_next_dst_slot_idx[local_expert_idx] = dst_slot_idx + 1;
                        master_topk_slot_idx = j;
                    }
                }

                // Issue MTE3
                if (has_dst_slot) {
                    asc_copy_ub2gm_align(
                        math::advance_ptr<__gm__ uint8_t>(recv_x, static_cast<int64_t>(dst_slot_idx) * kNumHiddenBytes),
                        token.get_hidden_ptr<__ubuf__ uint8_t>(),
                        1, kNumHiddenBytes,
                        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NOTALLOC_CI), 0, 0
                    );
                    if constexpr (kNumSFPacks > 0) {
                        asc_copy_ub2gm_align(
                            math::advance_ptr<__gm__ sf_pack_t>(
                                recv_sf, static_cast<int64_t>(dst_slot_idx) * recv_sf_token_stride * sizeof(sf_pack_t)),
                            token.get_sf_ptr<__ubuf__ sf_pack_t>(),
                            kNumSFBytes
                        );
                    }
                    if (recv_topk_weights != nullptr) {
                        AscendC::WriteGmByPassDCache(
                            reinterpret_cast<__gm__ int*>(recv_topk_weights) + dst_slot_idx,
                            token.get_topk_weights_ptr<__ubuf__ int>()[j]
                        );
                    }
                }

                if constexpr (not kCachedMode)
                    ub_metadata[j] = dst_slot_idx;
            }

            // Write metadata
            if constexpr (not kCachedMode) {
                ub_metadata[kNumTopk] = *reinterpret_cast<volatile __ubuf__ int*>(token.get_src_token_global_idx_ptr<__ubuf__ int>());
                ub_metadata[kNumTopk + 1] = master_topk_slot_idx;
            }
        }
        // stage consumed: the producer's next load for this stage waits on this signal
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kNumMetadataStages + stage_idx));

        // flush metadata buffer to global memory
        if constexpr (not kCachedMode) {
            asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(metadata_buffer_idx));
            asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(metadata_buffer_idx));
            asc_copy_ub2gm_align(
                recv_src_metadata + token_base_idx * kNumMetadataElemsPerToken,
                math::advance_ptr<__ubuf__ int>(ub_metadata_buffers, metadata_buffer_idx * kNumMetadataStageBytes),
                num_stage_tokens * kNumMetadataBytesPerToken
            );
            asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(metadata_buffer_idx));
            metadata_buffer_idx = (metadata_buffer_idx + 1) % kNumMetadataStages;
        }

        num_consumed_tokens += num_stage_tokens;
        issue_load(stage_idx, false);

        // Advance MTE stages
        stage_idx = stage_idx + 1 == kNumMTEStages ? 0 : stage_idx + 1;
    }

    if constexpr (kDoZeroPadding) {
        // Divide each expert into up to four padding parts
        static constexpr int kNumPaddingParts = math::min(4, kNumVecCores);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);

        #pragma unroll
        for (int part_idx = 0; part_idx < kNumPaddingParts; ++ part_idx) {
            const auto first_expert_idx = (vec_core_idx + part_idx * kNumVecCores / kNumPaddingParts) % kNumVecCores;
            for (int expert_idx = first_expert_idx; expert_idx < kNumExpertsPerRank; expert_idx += kNumVecCores) {
                const auto expert_token_end_idx = psum_num_recv_tokens_per_expert[expert_idx];
                const auto num_pad_tokens = math::align(expert_token_end_idx, kExpertAlignment) - expert_token_end_idx;
                const auto num_tokens_per_part = num_pad_tokens / kNumPaddingParts;
                const auto num_extra_tokens = num_pad_tokens % kNumPaddingParts;
                const auto pad_token_start_idx = part_idx * num_tokens_per_part + math::min(part_idx, num_extra_tokens);
                const auto num_part_tokens = num_tokens_per_part + (part_idx < num_extra_tokens ? 1 : 0);
                if (num_part_tokens == 0)
                    continue;

                const auto gm_recv_x = (__gm__ uint8_t*)recv_x +
                    static_cast<int64_t>(expert_token_end_idx + pad_token_start_idx) * kNumHiddenBytes;
                asc_set_ub2gm_loop_size(num_part_tokens, 1);
                asc_set_ub2gm_loop1_stride(0, kNumHiddenBytes);
                if constexpr (kNumHiddenBytes >= kNumZeroBytes) {
                    asc_copy_ub2gm_align(
                        gm_recv_x, ub_zero_buffer,
                        kNumHiddenBytes / kNumZeroBytes, kNumZeroBytes,
                        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV), kNumZeroBytes, 0
                    );
                }
                if constexpr (kNumHiddenBytes % kNumZeroBytes > 0) {
                    asc_copy_ub2gm_align(
                        gm_recv_x + kNumHiddenBytes / kNumZeroBytes * kNumZeroBytes,
                        ub_zero_buffer,
                        1, kNumHiddenBytes % kNumZeroBytes,
                        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV), kNumHiddenBytes % kNumZeroBytes, 0
                    );
                }
                asc_set_ub2gm_loop_size(1, 1);

                if (recv_topk_weights != nullptr) {
                    const auto gm_recv_topk_weights = reinterpret_cast<__gm__ int*>(recv_topk_weights) +
                        expert_token_end_idx + pad_token_start_idx;
                    for (int pad_token_idx = 0; pad_token_idx < num_part_tokens; ++ pad_token_idx) {
                        AscendC::WriteGmByPassDCache(gm_recv_topk_weights + pad_token_idx, 0);
                    }
                }

                if constexpr (kNumSFPacks > 0) {
                    const auto gm_recv_sf = recv_sf +
                        static_cast<int64_t>(expert_token_end_idx + pad_token_start_idx) * recv_sf_token_stride;
                    if constexpr (kNumSFBytes <= kNumZeroBytes) {
                        asc_copy_ub2gm_align(
                            gm_recv_sf, reinterpret_cast<__ubuf__ sf_pack_t*>(ub_zero_buffer),
                            num_part_tokens, kNumSFBytes,
                            static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                            kNumSFBytes, 0
                        );
                    } else {
                        asc_set_ub2gm_loop_size(num_part_tokens, 1);
                        asc_set_ub2gm_loop1_stride(0, static_cast<uint64_t>(recv_sf_token_stride) * sizeof(sf_pack_t));
                        asc_copy_ub2gm_align(
                            gm_recv_sf, reinterpret_cast<__ubuf__ sf_pack_t*>(ub_zero_buffer),
                            kNumSFBytes / kNumZeroBytes, kNumZeroBytes,
                            static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                            kNumZeroBytes, 0
                        );
                        asc_set_ub2gm_loop_size(1, 1);
                        if constexpr (kNumSFBytes % kNumZeroBytes > 0) {
                            asc_set_ub2gm_loop_size(num_part_tokens, 1);
                            asc_set_ub2gm_loop1_stride(0, static_cast<uint64_t>(recv_sf_token_stride) * sizeof(sf_pack_t));
                            asc_copy_ub2gm_align(
                                gm_recv_sf + kNumSFBytes / kNumZeroBytes * kNumZeroBytes / sizeof(sf_pack_t),
                                reinterpret_cast<__ubuf__ sf_pack_t*>(ub_zero_buffer),
                                1, kNumSFBytes % kNumZeroBytes,
                                static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                                kNumSFBytes % kNumZeroBytes, 0
                            );
                            asc_set_ub2gm_loop_size(1, 1);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace deep_ep
