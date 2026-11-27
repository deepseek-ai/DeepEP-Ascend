#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/common/error.hpp>
#include <deep_ep/common/utils.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

namespace deep_ep {

#ifdef EP_AICORE
// Push bytes [byte_begin, byte_end) of the selected shard to every peer in the given jetty group.
// With a marker, publish the absolute end after reaching the progress threshold or finishing the range.
template <int kNumRanks, int kNumJetties, int64_t kNumTimeoutCycles>
__aicore__ __forceinline__ void all_gather_send_bytes(
        __gm__ void* local_base, __gm__ void* buffer, __gm__ uint64_t* jetties,
        const int rank_idx, const int shard_rank_idx, const int num_buckets,
        const comm::BucketList& buckets, const int64_t byte_begin, const int64_t byte_end,
        __gm__ int64_t* landed_marker = nullptr,
        __gm__ layout::ErrorStatusSlot* error_status = nullptr) {
    EP_STATIC_ASSERT(kNumRanks == 1 or (kNumJetties > 0 and kNumJetties < kNumRanks));
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto num_blocks = static_cast<int>(AscendC::GetBlockNum());
    alignas(alignof(handle::HcommPeerInfo)) __ubuf__ uint8_t peer_info[sizeof(handle::HcommPeerInfo)];
    alignas(alignof(handle::HcommJettyInfo)) __ubuf__ uint8_t jetty_info[sizeof(handle::HcommJettyInfo)];
    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];

    for (int jetty_idx = block_idx; jetty_idx < kNumJetties; jetty_idx += num_blocks) {
        handle::HcommJetty hcomm(reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(jetty_info), ub_sqe, jetties, jetty_idx,
                                 error_status);
        for (int rank_offset = jetty_idx + 1; rank_offset < kNumRanks; rank_offset += kNumJetties) {
            const auto dst_rank_idx = (rank_idx + rank_offset) % kNumRanks;
            const auto peer_idx = dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1;
            const handle::HcommPeer peer(reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(peer_info), jetties, peer_idx);
            int64_t published_end = byte_begin;
            int64_t bucket_begin = 0;
            bool has_uncommitted = false;
            for (int bucket_idx = 0; bucket_idx < num_buckets and bucket_begin < byte_end; ++ bucket_idx) {
                const auto num_bytes = buckets[bucket_idx].num_bytes;
                const auto in_begin = math::max(byte_begin, bucket_begin) - bucket_begin;
                const auto in_end = math::min(byte_end, bucket_begin + num_bytes) - bucket_begin;
                // Peers stage shards into identical slot layouts.
                const auto dst = math::advance_ptr<__gm__ uint8_t>(
                    buffer, buckets[bucket_idx].offset + shard_rank_idx * num_bytes);
                for (int64_t offset = in_begin; offset < in_end; offset += kNumAllGatherProgressBytes) {
                    const auto write_bytes = math::min<int64_t>(kNumAllGatherProgressBytes, in_end - offset);
                    const auto shard_end = bucket_begin + offset + write_bytes;
                    // Publish after at least kNumAllGatherProgressBytes of progress, or at the end.
                    if (landed_marker != nullptr and
                        (shard_end == byte_end or shard_end - published_end >= kNumAllGatherProgressBytes)) {
                        hcomm.template post_write<kNumTimeoutCycles, false, true, true, true>(
                            peer, local_base, dst + offset, dst + offset, write_bytes, landed_marker, shard_end);
                        published_end = shard_end;
                        has_uncommitted = false;
                    } else {
                        hcomm.template put<kNumTimeoutCycles, false, false>(
                            peer, local_base, dst + offset, dst + offset, write_bytes);
                        has_uncommitted = true;
                    }
                }
                bucket_begin += num_bytes;
            }
            if (has_uncommitted)
                hcomm.ring_doorbell();
        }
    }
}

template <int kNumRanks, int kNumScaleupRanks, int kNumVecCores,
          int kNumScaleupJetties, int kNumScaleoutJetties, int64_t kNumTimeoutCycles>
__global__ __vector__ void all_gather_impl(
        __gm__ void* workspace, __gm__ void* local_base, __gm__ void* buffer,
        __gm__ void* device_context_ptr, const int rank_idx, const int num_buckets,
        const comm::BucketList buckets) {
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleoutRanks > 0 and kNumRanks % kNumScaleupRanks == 0);
    EP_STATIC_ASSERT(kNumScaleupRanks == 1 or (kNumScaleupJetties > 0 and kNumScaleupJetties < kNumScaleupRanks));
    EP_STATIC_ASSERT(kNumScaleoutRanks == 1 or (kNumScaleoutJetties > 0 and kNumScaleoutJetties < kNumScaleoutRanks));
    if constexpr (kNumRanks == 1)
        return;
    AscendC::InitSocState();

    auto& signals = *static_cast<__gm__ layout::BucketSignals*>(workspace);
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto error_status = layout::DeviceContext::get_error_status(*device_context);
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto domain_idx = rank_idx / kNumScaleupRanks;
    const auto scaleup_rank_idx = rank_idx % kNumScaleupRanks;
    int64_t total_bytes = 0;
    for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx)
        total_bytes += buckets[bucket_idx].num_bytes;

    // Clear progress markers before the entry barrier admits peer writes.
    if constexpr (kNumScaleupRanks > 1 and kNumScaleoutRanks > 1) {
        const auto num_blocks = static_cast<int>(AscendC::GetBlockNum());
        for (int i = block_idx; i < kNumScaleoutRanks; i += num_blocks)
            AscendC::WriteGmByPassDCache(&signals.all_gather_signals.landed_bytes[i], int64_t(0));
    }
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles,
                          false, false, true, kNumVecCores, 0, kNumScaleoutJetties>(
        signals.barrier_signals, *device_context, block_idx, rank_idx, local_base);

    if constexpr (kNumScaleoutRanks > 1) {
        // Scale-out all-gather
        __gm__ int64_t* landed_marker = kNumScaleupRanks > 1 ? &signals.all_gather_signals.landed_bytes[domain_idx] : nullptr;
        all_gather_send_bytes<kNumScaleoutRanks, kNumScaleoutJetties, kNumTimeoutCycles>(
            local_base, buffer, device_context->scaleout_jetties, domain_idx, rank_idx,
            num_buckets, buckets, 0, total_bytes, landed_marker, error_status);
    }

    if constexpr (kNumScaleupRanks > 1) {
        if (block_idx < kNumScaleupJetties) {
            // Scale-up all-gather
            all_gather_send_bytes<kNumScaleupRanks, kNumScaleupJetties, kNumTimeoutCycles>(
                local_base, buffer, device_context->scaleup_jetties, scaleup_rank_idx,
                rank_idx, num_buckets, buckets, 0, total_bytes, nullptr, error_status);

            if constexpr (kNumScaleoutRanks > 1) {
                // Forward remote domains' shards as chunks land
                int64_t forwarded_bytes[kNumScaleoutRanks] = {};
                comm::scalar::timeout_while<kNumTimeoutCycles>([&](const bool& timeout) __aicore__ {
                    bool pending = false;
                    for (int step = 1; step < kNumScaleoutRanks; ++ step) {
                        const auto remote = (domain_idx + step) % kNumScaleoutRanks;
                        if (forwarded_bytes[remote] >= total_bytes)
                            continue;
                        const auto landed_bytes = AscendC::ReadGmByPassDCache(&signals.all_gather_signals.landed_bytes[remote]);
                        const auto ready_end = math::min(landed_bytes, total_bytes);
                        if (ready_end > forwarded_bytes[remote]) {
                            all_gather_send_bytes<kNumScaleupRanks, kNumScaleupJetties, kNumTimeoutCycles>(
                                local_base, buffer, device_context->scaleup_jetties, scaleup_rank_idx,
                                remote * kNumScaleupRanks + scaleup_rank_idx, num_buckets, buckets,
                                forwarded_bytes[remote], ready_end, nullptr, error_status);
                            forwarded_bytes[remote] = ready_end;
                        }
                        pending = pending or forwarded_bytes[remote] < total_bytes;
                    }
                    if (timeout and pending)
                        AscendC::printf("DeepEP all-gather timeout, rank: %d, domain: %d\n", rank_idx, domain_idx);
                    return not pending;
                }, 0, error_status);
            }
        }
    }
}
#endif

#ifdef EP_AICPU
namespace aicpu {

// Push bytes [byte_begin, byte_end) of the selected shard to every peer in the given jetty group.
// With a marker, publish the absolute end after reaching the progress threshold or finishing the range.
template <int kNumRanks, int kNumJetties, int64_t kNumMaxWriteBytes, int64_t kNumTimeoutNs>
EP_INLINE void all_gather_send_bytes(
        void* local_base, void* buffer,
        aicpu::InplaceArray<handle::aicpu::HcommJetty, kNumJetties>& jetties,
        const aicpu::InplaceArray<handle::aicpu::HcommPeer, kNumRanks - 1>& peers,
        const int rank_idx, const int shard_rank_idx, const int num_buckets,
        const comm::BucketList& buckets, const int64_t byte_begin, const int64_t byte_end,
        int64_t* landed_marker = nullptr) {
    EP_STATIC_ASSERT(kNumRanks == 1 or (kNumJetties > 0 and kNumJetties < kNumRanks));
    EP_STATIC_ASSERT(kNumMaxWriteBytes > 0 and kNumMaxWriteBytes <= comm::kNumMaxWriteBytes);
    for (int jetty_idx = 0; jetty_idx < kNumJetties; ++ jetty_idx) {
        auto& hcomm = jetties[jetty_idx];
        for (int rank_offset = jetty_idx + 1; rank_offset < kNumRanks; rank_offset += kNumJetties) {
            const auto dst_rank_idx = (rank_idx + rank_offset) % kNumRanks;
            const auto peer_idx = dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1;
            const auto& peer = peers[peer_idx];
            int64_t published_end = byte_begin;
            int64_t bucket_begin = 0;
            for (int bucket_idx = 0; bucket_idx < num_buckets and bucket_begin < byte_end; ++ bucket_idx) {
                const auto num_bytes = buckets[bucket_idx].num_bytes;
                const auto in_begin = math::max(byte_begin, bucket_begin) - bucket_begin;
                const auto in_end = math::min(byte_end, bucket_begin + num_bytes) - bucket_begin;
                // Peers stage shards into identical slot layouts.
                const auto dst = math::advance_ptr<uint8_t>(
                    buffer, buckets[bucket_idx].offset + shard_rank_idx * num_bytes);
                for (int64_t offset = in_begin; offset < in_end; offset += kNumMaxWriteBytes) {
                    const auto write_bytes = math::min<int64_t>(kNumMaxWriteBytes, in_end - offset);
                    const auto shard_end = bucket_begin + offset + write_bytes;
                    if (landed_marker != nullptr and
                        (shard_end == byte_end or shard_end - published_end >= kNumAllGatherProgressBytes)) {
                        hcomm.template put_with_notify<kNumTimeoutNs, false, true>(
                            peer, local_base, dst + offset, dst + offset, write_bytes, landed_marker, shard_end);
                        published_end = shard_end;
                    } else {
                        hcomm.template put<kNumTimeoutNs, false, false, false>(
                            peer, local_base, dst + offset, dst + offset, write_bytes);
                    }
                }
                bucket_begin += num_bytes;
            }
            hcomm.ring_doorbell();
        }
    }
}

template <int kNumRanks, int kNumScaleupRanks,
          int kNumScaleupJetties, int kNumScaleoutJetties, int64_t kNumTimeoutNs>
EP_INLINE void all_gather_impl(
        void* workspace, void* local_base, void* buffer,
        void* device_context_ptr, const int rank_idx, const int num_buckets,
        const comm::BucketList& buckets) {
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    EP_STATIC_ASSERT(kNumScaleoutRanks > 0 and kNumScaleoutRanks <= kNumMaxScaleoutRanks);
    EP_STATIC_ASSERT(kNumRanks % kNumScaleupRanks == 0);
    EP_STATIC_ASSERT(kNumScaleupRanks == 1 or (kNumScaleupJetties > 0 and kNumScaleupJetties < kNumScaleupRanks));
    EP_STATIC_ASSERT(kNumScaleoutRanks == 1 or (kNumScaleoutJetties > 0 and kNumScaleoutJetties < kNumScaleoutRanks));
    if constexpr (kNumRanks == 1)
        return;

    auto& signals = *static_cast<layout::BucketSignals*>(workspace);
    const auto& device_context = *static_cast<const layout::DeviceContext*>(device_context_ptr);
    auto* const error_status = reinterpret_cast<void*>(device_context.error_status_addr);
    const auto domain_idx = rank_idx / kNumScaleupRanks;
    const auto scaleup_rank_idx = rank_idx % kNumScaleupRanks;
    int64_t total_bytes = 0;
    for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx)
        total_bytes += buckets[bucket_idx].num_bytes;

    // Clear progress markers before the entry barrier admits peer writes.
    if constexpr (kNumScaleupRanks > 1 and kNumScaleoutRanks > 1) {
        for (int i = 0; i < kNumScaleoutRanks; ++ i)
            signals.all_gather_signals.landed_bytes[i] = 0;
        aicpu::publish_gm(&signals.all_gather_signals.landed_bytes, kNumScaleoutRanks * sizeof(int64_t));
    }
    comm::aicpu::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutNs,
                         false, 0, kNumScaleoutJetties, false, true>(
        signals.barrier_signals, device_context, rank_idx, local_base);

    if constexpr (kNumScaleoutRanks > 1) {
        // Scale-out all-gather
        aicpu::InplaceArray<handle::aicpu::HcommJetty, kNumScaleoutJetties> scaleout_jetties;
        aicpu::InplaceArray<handle::aicpu::HcommPeer, kNumScaleoutRanks - 1> scaleout_peers;
        for (int i = 0; i < kNumScaleoutJetties; ++ i)
            scaleout_jetties.emplace(i, device_context.scaleout_jetties, i, error_status);
        for (int i = 0; i < kNumScaleoutRanks - 1; ++ i)
            scaleout_peers.emplace(i, device_context.scaleout_jetties, i);
        int64_t* landed_marker = kNumScaleupRanks > 1 ? &signals.all_gather_signals.landed_bytes[domain_idx] : nullptr;
        all_gather_send_bytes<kNumScaleoutRanks, kNumScaleoutJetties, kNumAllGatherProgressBytes, kNumTimeoutNs>(
            local_base, buffer, scaleout_jetties, scaleout_peers, domain_idx, rank_idx,
            num_buckets, buckets, 0, total_bytes, landed_marker);
    }

    if constexpr (kNumScaleupRanks > 1) {
        // Scale-up all-gather
        aicpu::InplaceArray<handle::aicpu::HcommJetty, kNumScaleupJetties> scaleup_jetties;
        aicpu::InplaceArray<handle::aicpu::HcommPeer, kNumScaleupRanks - 1> scaleup_peers;
        for (int i = 0; i < kNumScaleupJetties; ++ i)
            scaleup_jetties.emplace(i, device_context.scaleup_jetties, i, error_status);
        for (int i = 0; i < kNumScaleupRanks - 1; ++ i)
            scaleup_peers.emplace(i, device_context.scaleup_jetties, i);
        all_gather_send_bytes<kNumScaleupRanks, kNumScaleupJetties, comm::kNumMaxWriteBytes, kNumTimeoutNs>(
            local_base, buffer, scaleup_jetties, scaleup_peers, scaleup_rank_idx,
            rank_idx, num_buckets, buckets, 0, total_bytes);

        if constexpr (kNumScaleoutRanks > 1) {
            // Forward remote domains' shards as chunks land
            int64_t forwarded_bytes[kNumScaleoutRanks] = {};
            aicpu::timeout_while<kNumTimeoutNs>([&](const bool timeout) {
                aicpu::refresh_gm(&signals.all_gather_signals.landed_bytes, kNumScaleoutRanks * sizeof(int64_t));
                bool pending = false;
                for (int step = 1; step < kNumScaleoutRanks; ++ step) {
                    const auto remote = (domain_idx + step) % kNumScaleoutRanks;
                    if (forwarded_bytes[remote] >= total_bytes)
                        continue;
                    const auto landed = *reinterpret_cast<const volatile int64_t*>(
                        &signals.all_gather_signals.landed_bytes[remote]);
                    const auto ready_end = math::min<int64_t>(landed, total_bytes);
                    if (ready_end > forwarded_bytes[remote]) {
                        all_gather_send_bytes<kNumScaleupRanks, kNumScaleupJetties, comm::kNumMaxWriteBytes, kNumTimeoutNs>(
                            local_base, buffer, scaleup_jetties, scaleup_peers, scaleup_rank_idx,
                            remote * kNumScaleupRanks + scaleup_rank_idx, num_buckets, buckets,
                            forwarded_bytes[remote], ready_end);
                        forwarded_bytes[remote] = ready_end;
                    }
                    pending = pending or forwarded_bytes[remote] < total_bytes;
                }
                if (timeout and pending) {
                    record_error(error_status, layout::kBarrierTimeout);
                    throw EPExceptionWithLineInfo("AICPU all-gather timeout",
                        "rank=" + std::to_string(rank_idx) + ", domain=" + std::to_string(domain_idx));
                }
                return not pending;
            });
        }
    }

    // Drain issued writes and wait for peers
    comm::aicpu::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutNs,
                         true, kNumScaleupJetties, kNumScaleoutJetties, true, true>(
        signals.barrier_signals, device_context, rank_idx, local_base);
}

}  // namespace aicpu
#endif

}  // namespace deep_ep
