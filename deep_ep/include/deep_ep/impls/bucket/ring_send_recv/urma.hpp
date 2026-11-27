#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/common/error.hpp>
#include <deep_ep/common/utils.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

namespace deep_ep {

template <int kNumRanks, int kNumJetties, int64_t kNumTimeoutNs>
EP_INLINE void ring_send_recv_impl(
        void* workspace, void* local_base, void* buffer,
        void* device_context_ptr, const int rank_idx, const int dst_rank_idx,
        const int num_buckets,
        const comm::BucketList& src_buckets,
        const comm::BucketList& dst_buckets) {
    EP_STATIC_ASSERT(kNumRanks > 1 and kNumJetties > 0);

    auto& signals = *static_cast<layout::BucketSignals*>(workspace);
    const auto& device_context = *static_cast<const layout::DeviceContext*>(device_context_ptr);
    auto* const error_status = reinterpret_cast<void*>(device_context.error_status_addr);

    // Entry barrier: ensure peers have consumed prior destinations.
    comm::aicpu::barrier<kNumRanks, kNumRanks, kNumTimeoutNs, false>(
        signals.barrier_signals, device_context, rank_idx, local_base);

    if (dst_rank_idx >= 0) {
        const auto peer_idx = dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1;
        const handle::aicpu::HcommPeer peer(device_context.scaleup_jetties, peer_idx);
        for (int jetty_idx = 0; jetty_idx < kNumJetties; ++ jetty_idx) {
            handle::aicpu::HcommJetty hcomm(device_context.scaleup_jetties, jetty_idx, error_status);
            for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
                const auto& [src_offset, num_bucket_bytes] = src_buckets[bucket_idx];
                const auto dst_offset = dst_buckets[bucket_idx].offset;
                const auto num_bytes_per_jetty = math::align<int64_t>(
                    math::ceil_div<int64_t>(num_bucket_bytes, kNumJetties), kNumRDMAAlignmentBytes);
                const auto jetty_offset = math::min<int64_t>(
                    static_cast<int64_t>(jetty_idx) * num_bytes_per_jetty, num_bucket_bytes);
                const auto num_bytes = math::min<int64_t>(num_bytes_per_jetty, num_bucket_bytes - jetty_offset);
                const auto src = math::advance_ptr<uint8_t>(buffer, src_offset + jetty_offset);
                const auto dst = math::advance_ptr<uint8_t>(buffer, dst_offset + jetty_offset);
                for (int64_t offset = 0; offset < num_bytes; offset += comm::kNumMaxWriteBytes) {
                    const auto write_bytes = math::min<int64_t>(comm::kNumMaxWriteBytes, num_bytes - offset);
                    hcomm.template put<kNumTimeoutNs, false, false, false>(
                        peer, local_base, dst + offset, src + offset, write_bytes);
                }
            }
            hcomm.ring_doorbell();
        }
    }

    // Drain issued writes and wait for peers.
    comm::aicpu::barrier<kNumRanks, kNumRanks, kNumTimeoutNs, true, kNumJetties>(
        signals.barrier_signals, device_context, rank_idx, local_base);
}

}  // namespace deep_ep
