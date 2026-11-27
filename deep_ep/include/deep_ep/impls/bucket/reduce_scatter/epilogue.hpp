#pragma once

#include <c_api/asc_simd.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/layout/common/context.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep {

template <int kNumIterations>
__simd_vf__ __forceinline__ void reduce_scatter_scale_vf(__ubuf__ float* data, const float scale) {
    constexpr uint32_t kNumElemsPerVector = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const auto mask = asc_create_mask_b32(PAT_ALL);
    #pragma unroll 1
    for (int i = 0; i < kNumIterations; ++ i) {
        const auto offset = i * kNumElemsPerVector;
        vector_float values;
        asc_loadalign(values, data + offset);
        asc_mul_scalar(values, values, scale, mask);
        asc_storealign(data + offset, values, mask);
    }
}

template <int kNumRanks, int kNumScaleupRanks, int kNumVecCores,
          int kNumScaleupJetties, int kNumScaleoutJetties, int64_t kNumTimeoutCycles>
__global__ __vector__ void reduce_scatter_epilogue_impl(
        __gm__ void* barrier_signals, __gm__ void* local_base, __gm__ void* buffer, __gm__ void* device_context_ptr,
        const int rank_idx, const int num_buckets, const comm::BucketList src_buckets, const float scale) {
    AscendC::InitSocState();
    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);

    // Complete local READ REDUCE operations and wait until peers have finished
    // reading our source shards before allowing the inputs to be reused.
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles, true, false, true,
                          kNumVecCores, kNumScaleupJetties, kNumScaleoutJetties>(
        *static_cast<__gm__ layout::BarrierSignals*>(barrier_signals), *device_context,
        vec_core_idx, rank_idx, local_base);

    if (scale == 1.0f)
        return;

    // Two 32-KiB buffers: MTE2 loads, V scales in place, MTE3 writes back.
    constexpr int kNumTileElems = 8192;
    constexpr int kNumIterations = kNumTileElems * sizeof(float) / AscendC::VECTOR_REG_WIDTH;
    alignas(kNumUbAlignmentBytes) __ubuf__ float ub[2][kNumTileElems];
    for (int slot = 0; slot < 2; ++ slot)
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));

    int slot = 0;
    for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
        const auto num_bytes = src_buckets[bucket_idx].num_bytes;
        const auto num_elems = num_bytes / static_cast<int64_t>(sizeof(float));
        const auto shard = math::advance_ptr<__gm__ float>(
            buffer, src_buckets[bucket_idx].offset + rank_idx * num_bytes);
        for (int64_t offset = static_cast<int64_t>(vec_core_idx) * kNumTileElems;
             offset < num_elems; offset += static_cast<int64_t>(kNumVecCores) * kNumTileElems) {
            const auto count = static_cast<uint32_t>(math::min<int64_t>(kNumTileElems, num_elems - offset));
            const auto copy_bytes = count * static_cast<uint32_t>(sizeof(float));
            const auto event = static_cast<event_t>(slot);

            // Reuse this UB slot only after its previous store has finished.
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event);
            asc_copy_gm2ub_align(ub[slot], shard + offset, 1, copy_bytes, 0, 0, true,
                                asc_load_l2_cache_mode::NOTALLOC_KEEP, 0, 0);
            asc_sync_notify(PIPE_MTE2, PIPE_V, event);
            asc_sync_wait(PIPE_MTE2, PIPE_V, event);
            reduce_scatter_scale_vf<kNumIterations>(ub[slot], scale);
            asc_sync_notify(PIPE_V, PIPE_MTE3, event);
            asc_sync_wait(PIPE_V, PIPE_MTE3, event);
            // Copy only valid bytes, including an unaligned or partial final tile.
            asc_copy_ub2gm_align(shard + offset, ub[slot], 1, copy_bytes,
                                asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event);
            slot ^= 1;
        }
    }

    for (int slot = 0; slot < 2; ++ slot)
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

}  // namespace deep_ep
