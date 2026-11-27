#pragma once

#include <deep_ep/common/utils.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/layout/bucket/chunk.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

namespace deep_ep {

template <int kNumRanks, int kNumScaleupRanks, int kNumVecCores, int kNumChunkBytes, int64_t kNumTimeoutCycles>
__global__ __vector__ void urma_reduce_scatter_impl(
        __gm__ void* workspace, __gm__ void* local_base, __gm__ void* buffer,
        __gm__ void* device_context_ptr, const int rank_idx, const int num_buckets,
        comm::BucketList src_buckets) {
    constexpr bool kScaleout = kNumScaleupRanks < kNumRanks;
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks <= (kScaleout ? kNumMaxScaleoutRanks : kNumMaxScaleupRanks));
    EP_STATIC_ASSERT(kNumScaleupRanks == 1 or kNumScaleupRanks == kNumRanks);
    EP_STATIC_ASSERT(kNumChunkBytes > 0 and kNumChunkBytes % kNumUrmaAlignmentBytes == 0);
    EP_STATIC_ASSERT(kNumChunkBytes <= comm::kNumMaxReadBytes);

    AscendC::InitSocState();
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());

    auto& signals = *static_cast<__gm__ layout::BucketSignals*>(workspace);
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto error_status = layout::DeviceContext::get_error_status(*device_context);
    auto* jetties = kScaleout ? device_context->scaleout_jetties : device_context->scaleup_jetties;

    alignas(alignof(handle::HcommPeerInfo)) __ubuf__ uint8_t peer_info[sizeof(handle::HcommPeerInfo)];
    alignas(alignof(handle::HcommJettyInfo)) __ubuf__ uint8_t jetty_info[sizeof(handle::HcommJettyInfo)];
    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];

    // Every rank must finish producing its source shards before peers can read them.
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles,
                          false, false, true, kNumVecCores, 0, kScaleout ? 1 : 0>(
        signals.barrier_signals, *device_context, block_idx, rank_idx, local_base);

    // Load queue state after the barrier, which uses scale-out Jetty 0.
    handle::HcommJetty hcomm(reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(jetty_info),
                             ub_sqe, kNumRanks > 1 ? jetties : nullptr, block_idx,
                             error_status);

    // One queue owns each output chunk, seeded by this rank's original shard.
    for (int rank_offset = 1; rank_offset < kNumRanks; ++ rank_offset) {
        const auto src_rank_idx = (rank_idx + rank_offset) % kNumRanks;
        const auto src_jetty_idx = src_rank_idx < rank_idx ? src_rank_idx : src_rank_idx - 1;
        const handle::HcommPeer peer(reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(peer_info),
                                     jetties, src_jetty_idx);
        layout::ChunkIterator<kNumChunkBytes, kNumVecCores> chunks(num_buckets, src_buckets, block_idx);
        while (chunks.next()) {
            auto* ptr = math::advance_ptr<__gm__ uint8_t>(buffer,
                chunks.bucket.offset + static_cast<int64_t>(rank_idx) * chunks.bucket.num_bytes +
                chunks.chunk_offset_in_bucket);
            // Use strong placement order on this peer's first/last SQEs and relaxed
            // placement order in between to reduce ordering overhead.
            // This preserved cyclic accumulation order in testing;
            // first-only or last-only strong placement order failed bitwise checks.
            if (chunks.chunk_idx == block_idx or chunks.last())
                hcomm.template get<kNumTimeoutCycles, true, true, true>(
                    peer, local_base, ptr, ptr, chunks.num_chunk_bytes);
            else
                hcomm.template get<kNumTimeoutCycles, true, true, false>(
                    peer, local_base, ptr, ptr, chunks.num_chunk_bytes);
        }
    }
}

}  // namespace deep_ep
