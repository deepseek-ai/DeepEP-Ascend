#pragma once

#include <climits>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/engram/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>
#include <deep_ep/common/error.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>
#include <c_api/asc_simd.h>

namespace deep_ep {

template <int kNumScaleupRanks, int kNumHiddenBytes, int kNumSFPacks,
          int kNumEntriesPerToken, int kNumMaxTokens, int kNumVecCores,
          int kNumMTEStages, int... kNumEntriesPack>
__global__ __vector__ void engram_fetch_impl(__gm__ void* storage, __gm__ void* fetched,
                                           __gm__ int* indices, __gm__ void* workspace,
                                           __gm__ void* device_context_ptr, __gm__ sf_pack_t* sf,
                                           __gm__ sf_pack_t* fetched_sf,
                                           const int64_t sf_token_stride,
                                           const int64_t sf_hidden_stride,
                                           const int64_t sf_layer_stride,
                                           const int num_tokens, const int rank_idx) {
    AscendC::InitSocState();
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    EP_STATIC_ASSERT(kNumMaxTokens > 0 and kNumEntriesPerToken > 0);
    EP_STATIC_ASSERT(static_cast<int64_t>(kNumMaxTokens) * kNumEntriesPerToken +
                         static_cast<int64_t>(math::max(1, kNumMTEStages - 1)) * kNumVecCores <= INT_MAX,
                     "Engram prefetch and loop indices overflow int");
    EP_DEVICE_ASSERT(0 <= num_tokens and num_tokens <= kNumMaxTokens);

    using engram_layout_t = layout::EngramLayout<kNumHiddenBytes, kNumEntriesPerToken, kNumEntriesPack...>;
    constexpr int kNumLayers = engram_layout_t::kNumLayers;
    constexpr int kNumSFBytes = kNumSFPacks * sizeof(sf_pack_t);
    constexpr int kNumAlignedHiddenBytes = math::align(kNumHiddenBytes, kNumUbAlignmentBytes);
    constexpr int kNumBytesPerStage = kNumAlignedHiddenBytes + math::align(kNumSFBytes, kNumUbAlignmentBytes);
    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    auto& engram_workspace = *static_cast<__gm__ layout::EngramWorkspace*>(workspace);
    const auto ub = reinterpret_cast<__ubuf__ uint8_t*>(0);
    const auto num_entries = num_tokens * kNumEntriesPerToken;
    const auto domain_start = layout::get_scaleup_domain_start_rank<kNumScaleupRanks>(rank_idx);

    #pragma unroll
    for (int stage_idx = 0; stage_idx < kNumMTEStages; ++ stage_idx)
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));

    #pragma unroll
    for (int layer_idx = 0; layer_idx < kNumLayers; ++ layer_idx) {
        const auto num_entries_per_rank = engram_layout_t::get_num_entries(layer_idx);
        const auto num_prefix_entries = engram_layout_t::get_num_prefix_entries(layer_idx);
        const auto layer_indices = indices + static_cast<int64_t>(layer_idx) * num_entries;
        const auto layer_storage = math::advance_ptr<__gm__ uint8_t>(
            storage, engram_layout_t::get_storage_byte_offset(layer_idx, 0));
        const auto layer_fetched = math::advance_ptr<__gm__ uint8_t>(
            fetched, engram_layout_t::get_recv_byte_offset(layer_idx, 0, kNumMaxTokens));

        const auto issue_load = [=, &engram_workspace](const int& entry_idx, const uint8_t& stage_idx) __aicore__ {
            if (entry_idx >= num_entries)
                return -1;

            const auto global_idx = AscendC::ReadGmByPassDCache(layer_indices + entry_idx);
            const auto ub_entry = math::advance_ptr<__ubuf__ uint8_t>(
                ub, static_cast<int64_t>(stage_idx) * kNumBytesPerStage);

            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
            if (global_idx >= 0) {
                const auto src_scaleup_rank_idx = global_idx / num_entries_per_rank - domain_start;
                if (src_scaleup_rank_idx < 0 or src_scaleup_rank_idx >= kNumScaleupRanks)
                    record_error_and_trap(layout::DeviceContext::get_error_status(*device_context), layout::kInvalidCommMetadata);
                const auto src_entry_idx = global_idx % num_entries_per_rank;
                const auto local_entry = math::advance_ptr<__gm__ uint8_t>(
                    layer_storage, static_cast<int64_t>(src_entry_idx) * kNumHiddenBytes);
                asc_copy_gm2ub_align(
                    ub_entry,
                    layout::get_sym_ptr(engram_workspace.barrier_signals, *device_context, local_entry, src_scaleup_rank_idx),
                    kNumHiddenBytes
                );
                if constexpr (kNumSFPacks > 0) {
                    const auto local_sf = sf + (num_prefix_entries + src_entry_idx) * kNumSFPacks;
                    asc_copy_gm2ub_align(
                        math::advance_ptr<__ubuf__ sf_pack_t>(ub_entry, kNumAlignedHiddenBytes),
                        layout::get_sym_ptr(engram_workspace.barrier_signals, *device_context, local_sf, src_scaleup_rank_idx),
                        kNumSFBytes
                    );
                }
            }
            asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(stage_idx));
            return global_idx;
        };

        int stage_global_indices[kNumMTEStages];
        #pragma unroll
        for (int stage_idx = 0; stage_idx < kNumMTEStages - 1; ++ stage_idx) {
            stage_global_indices[stage_idx] = issue_load(
                vec_core_idx + stage_idx * kNumVecCores,
                static_cast<uint8_t>(stage_idx));
        }

        uint8_t stage_idx = 0;
        for (int entry_idx = vec_core_idx; entry_idx < num_entries; entry_idx += kNumVecCores) {
            const auto load_stage_idx = static_cast<uint8_t>(
                (stage_idx + kNumMTEStages - 1) % kNumMTEStages);
            stage_global_indices[load_stage_idx] = issue_load(
                entry_idx + (kNumMTEStages - 1) * kNumVecCores,
                load_stage_idx);

            const auto global_idx = stage_global_indices[stage_idx];
            const auto ub_entry = math::advance_ptr<__ubuf__ uint8_t>(
                ub, static_cast<int64_t>(stage_idx) * kNumBytesPerStage);

            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(stage_idx));
            if (global_idx >= 0) {
                asc_copy_ub2gm_align(
                    math::advance_ptr<__gm__ uint8_t>(
                        layer_fetched, static_cast<int64_t>(entry_idx) * kNumHiddenBytes),
                    ub_entry,
                    kNumHiddenBytes
                );

                if constexpr (kNumSFPacks > 0) {
                    const auto token_idx = entry_idx / kNumEntriesPerToken;
                    const auto token_entry_idx = entry_idx % kNumEntriesPerToken;
                    const auto sf_dst = fetched_sf +
                        static_cast<int64_t>(layer_idx) * sf_layer_stride +
                        token_idx * sf_token_stride +
                        token_entry_idx * kNumSFPacks * sf_hidden_stride;
                    const auto ub_sf = math::advance_ptr<__ubuf__ sf_pack_t>(ub_entry, kNumAlignedHiddenBytes);
                    asc_copy_ub2gm_align(
                        sf_dst, ub_sf,
                        kNumSFPacks, sizeof(sf_pack_t),
                        static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                        sf_hidden_stride * sizeof(sf_pack_t), sizeof(sf_pack_t)
                    );
                }
            }

            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
            stage_idx = (stage_idx + 1) % kNumMTEStages;
        }
    }
}

}  // namespace deep_ep
