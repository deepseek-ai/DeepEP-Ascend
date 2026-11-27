#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/utils.hpp>
#include <deep_ep/layout/bucket/chunk.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>

namespace deep_ep {

template <int kNumRanks, int kNumScaleupRanks, int kNumJetties, int kNumStreams,
          int64_t kNumChunkBytes, int64_t kNumTimeoutNs>
EP_INLINE void urma_all_reduce_impl(
        void* workspace, void* local_base, void* buffer, void* device_context_ptr,
        const int rank_idx, const int num_buckets, const comm::BucketList& buckets) {
    constexpr bool kScaleout = kNumScaleupRanks < kNumRanks;
    constexpr int kNumBarrierJetties = kScaleout ? 1 : 0;
    using CommPlan = NhrPlan<kNumRanks, true>;
    using Jetty = handle::aicpu::HcommJetty;
    using Peer = handle::aicpu::HcommPeer;
    constexpr int kNumPhases = CommPlan::kNumPhases;

    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks <= (kScaleout ? kNumMaxScaleoutRanks : kNumMaxScaleupRanks));
    EP_STATIC_ASSERT(kNumScaleupRanks == 1 or kNumScaleupRanks == kNumRanks);
    EP_STATIC_ASSERT(kNumStreams > 0 and kNumStreams <= layout::AllReduceSignals::kNumMaxStreams);
    EP_STATIC_ASSERT(kNumChunkBytes > 0 and kNumChunkBytes <= INT32_MAX and
                     kNumChunkBytes % kNumUrmaAlignmentBytes == 0);
    // Each logical stream uses one data jetty and one notification jetty.
    // Scaleout jetty 0 is reserved for the barrier.
    EP_STATIC_ASSERT(kNumRanks == 1 or 2 * kNumStreams + kNumBarrierJetties <= kNumJetties);
    EP_STATIC_ASSERT(kNumPhases <= layout::AllReduceSignals::kNumMaxPhases);

    // The AIV epilogue drains notifications and waits for peers before scaling or source reuse.
    if constexpr (kNumRanks == 1)
        return;

    auto& signals = *static_cast<layout::BucketSignals*>(workspace);
    auto& context = *static_cast<layout::DeviceContext*>(device_context_ptr);
    auto& ready = signals.all_reduce_signals.ready;
    const auto* jetties = kScaleout ? context.scaleout_jetties : context.scaleup_jetties;
    auto* const error_status = reinterpret_cast<void*>(context.error_status_addr);
    const CommPlan plan(rank_idx);

    // Allocate cacheable stack storage.
    aicpu::InplaceArray<layout::ChunkIterator<kNumChunkBytes, kNumStreams>, kNumStreams> chunks;
    aicpu::InplaceArray<Peer, kNumPhases> read_peers[kNumStreams], notify_peers[kNumStreams];
    aicpu::InplaceArray<Jetty, kNumStreams> data_jetties, notify_jetties;
    int phase_idx[kNumStreams]{};
    bool pending[kNumStreams]{};
    uint32_t completed[kNumStreams]{}, num_chunks[kNumStreams];

    // Initialize
    for (int stream = 0; stream < kNumStreams; ++ stream) {
        const auto& chunk = chunks.emplace(stream, num_buckets, buckets, stream);
        num_chunks[stream] = math::ceil_div(math::max(0, chunk.num_total_chunks - stream), kNumStreams);
        data_jetties.emplace(stream, jetties, kNumBarrierJetties + stream, error_status);
        notify_jetties.emplace(stream, jetties, kNumBarrierJetties + kNumStreams + stream, error_status);
        for (int phase = 0; phase < kNumPhases; ++ phase) {
            ready[stream][phase] = phase == 0 ? num_chunks[stream] : 0;
            const auto& step = plan.get_phase(phase);
            const int producer = step.read_peer, consumer = step.notify_peer;
            read_peers[stream].emplace(phase, jetties, producer - (producer > rank_idx));
            notify_peers[stream].emplace(phase, jetties, consumer - (consumer > rank_idx));
        }
    }

    // Keep receive cache lines clean before any peer can update them.
    aicpu::publish_gm(&ready);
    comm::aicpu::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutNs, false, 0, kNumBarrierJetties, true, true>(
        signals.barrier_signals, context, rank_idx, local_base);

    // Each stream submits one phase, waits for completion, then publishes the next input.
    aicpu::timeout_while<kNumTimeoutNs>([&](const bool timeout) {
        bool finished = true;
        for (int stream = 0; stream < kNumStreams; ++ stream) {
            if (completed[stream] == num_chunks[stream])
                continue;
            finished = false;
            auto& phase = phase_idx[stream];
            auto& jetty = data_jetties[stream];
            auto& chunk = chunks[stream];

            // Try to schedule new chunk or advance the inflight chunk to the next phase if none is pending.
            if (not pending[stream]) {
                if (phase == 0) {
                    EP_DEVICE_ASSERT(chunk.next());
                } else if (aicpu::refresh_and_load_gm(&ready[stream][phase]) <= completed[stream]) {
                    continue;
                }
                const auto& peer = read_peers[stream][phase];
                for (int range = 0; range < plan.get_phase(phase).num_merged_slices; ++ range) {
                    const auto slice = plan.get_slice(phase, range,
                        {chunk.bucket.offset + chunk.chunk_offset_in_bucket, chunk.num_chunk_bytes});
                    auto* ptr = math::advance_ptr<uint8_t>(buffer, slice.offset);
                    for (int64_t pos = 0; pos < slice.num_bytes; pos += comm::kNumMaxReadBytes) {
                        const int bytes = math::min<int64_t>(comm::kNumMaxReadBytes, slice.num_bytes - pos);
                        // During reduce-scatter, preserve READ_REDUCE ordering at each merged range's boundaries.
                        if (phase < CommPlan::kNumLevels) {
                            if (pos == 0 or pos + bytes == slice.num_bytes)
                                jetty.template get<kNumTimeoutNs, true, false, true>(
                                    peer, local_base, ptr + pos, ptr + pos, bytes);
                            else
                                jetty.template get<kNumTimeoutNs, true, false, false>(
                                    peer, local_base, ptr + pos, ptr + pos, bytes);
                        } else {
                            jetty.template get<kNumTimeoutNs, false, false, false>(
                                peer, local_base, ptr + pos, ptr + pos, bytes);
                        }
                    }
                }
                jetty.ring_doorbell();
                pending[stream] = true;
            }

            // Wait for the pending chunk to be ready.
            const auto tail = jetty.get_expected_cq_tail();
            if (static_cast<int32_t>(jetty.try_poll_cq(tail) - tail) < 0)
                continue;
            pending[stream] = false;
            // Either the chunk is done or we should notify the next phase's peer.
            if (++ phase == kNumPhases) {
                phase = 0;
                ++ completed[stream];
            } else {
                notify_jetties[stream].template put_value<kNumTimeoutNs>(
                    notify_peers[stream][phase], local_base, &ready[stream][phase], uint64_t(completed[stream] + 1));
            }
        }
        if (timeout and not finished) {
            record_error(error_status, layout::kBarrierTimeout);
            throw EPExceptionWithLineInfo("AICPU all-reduce timeout", "rank=" + std::to_string(rank_idx) +
                ", completed=" + std::to_string(completed[0]) + ", phase=" + std::to_string(phase_idx[0]));
        }
        return finished;
    });

    // The AIV epilogue drains notifications and waits for peers before scaling or source reuse.
}

}  // namespace deep_ep
