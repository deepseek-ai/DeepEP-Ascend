#pragma once

#include <deep_ep/common/utils.hpp>
#include <deep_ep/layout/bucket/chunk.hpp>

namespace deep_ep {

template <int kNumRanks, int kNumScaleupRanks, int kNumVecCores,
          int kNumScaleupJetties, int kNumScaleoutJetties, int64_t kNumTimeoutCycles>
__global__ __vector__ void all_reduce_epilogue_impl(
        __gm__ void* barrier_signals, __gm__ void* local_base, __gm__ void* buffer, __gm__ void* device_context_ptr,
        const int rank_idx, const int num_buckets, const comm::BucketList buckets, const float scale) {
    AscendC::InitSocState();
    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    auto& device_context = *static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);

    // Wait until every peer has finished reading our data before scaling or reusing it.
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles, true, false, true,
                          kNumVecCores, kNumScaleupJetties, kNumScaleoutJetties, true, true>(
        *static_cast<__gm__ layout::BarrierSignals*>(barrier_signals), device_context,
        vec_core_idx, rank_idx, local_base);

    if (scale == 1.0f)
        return;

    constexpr int kNumTileElems = 8192;
    alignas(kNumUbAlignmentBytes) __ubuf__ float ub[2 * kNumTileElems];
    Fp32Scaler<kNumTileElems> scaler(ub);
    layout::ChunkIterator<kNumTileElems * sizeof(float), kNumVecCores> tiles(num_buckets, buckets, vec_core_idx);
    while (tiles.next()) {
        auto* ptr = math::advance_ptr<__gm__ float>(buffer, tiles.bucket.offset + tiles.chunk_offset_in_bucket);
        scaler.run(ptr, tiles.num_chunk_bytes / sizeof(float), scale);
    }
    scaler.finish();
}

}  // namespace deep_ep
