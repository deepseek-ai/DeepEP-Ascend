#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/utils.hpp>
#include <deep_ep/layout/bucket/chunk.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>

namespace deep_ep {

template <int kNumRanks, int kNumScaleupRanks, int kNumScaleupJetties, int kNumScaleoutJetties,
          int kNumStreams, int64_t kNumChunkBytes, int kWindow, int64_t kNumTimeoutNs>
EP_INLINE void hybrid_all_reduce_impl(
        void* workspace, void* local_base, void* buffer, void* device_context_ptr,
        const int rank_idx, const int num_buckets, const comm::BucketList& buckets) {
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumRanks > 0);
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    EP_STATIC_ASSERT(kNumScaleoutRanks > 0 and kNumScaleoutRanks <= kNumMaxScaleoutRanks);
    EP_STATIC_ASSERT(kNumRanks % kNumScaleupRanks == 0);
    using SuPlan = NhrPlan<kNumScaleupRanks, true>;
    using SoPlan = NhrPlan<kNumScaleoutRanks, true>;
    using Jetty = handle::aicpu::HcommJetty;
    using Peer = handle::aicpu::HcommPeer;
    // Chunk pipeline is SU reduce-scatter -> SO reduce-scatter/all-gather -> SU all-gather.
    constexpr int kNumPhases = SuPlan::kNumPhases + SoPlan::kNumPhases;
    constexpr int kNumReducePhases = SuPlan::kNumLevels + SoPlan::kNumLevels;
    constexpr auto is_scaleout_phase = [](int phase) {
        return phase >= SuPlan::kNumLevels and phase < SuPlan::kNumLevels + SoPlan::kNumPhases;
    };
    constexpr auto get_su_or_so_phase = [](int phase) {
        if (phase < SuPlan::kNumLevels)
            return phase;
        if (phase < SuPlan::kNumLevels + SoPlan::kNumPhases)
            return phase - SuPlan::kNumLevels;
        return phase - SoPlan::kNumPhases;
    };

    EP_STATIC_ASSERT(kNumScaleupRanks > 1 and kNumScaleoutRanks > 1);
    EP_STATIC_ASSERT(kNumStreams > 0 and kNumStreams <= layout::AllReduceSignals::kNumMaxStreams and kWindow > 0);
    EP_STATIC_ASSERT(kNumPhases <= layout::AllReduceSignals::kNumMaxPhases);
    EP_STATIC_ASSERT(kNumChunkBytes > 0 and kNumChunkBytes <= INT32_MAX and
                     kNumChunkBytes % kNumUrmaAlignmentBytes == 0);

    // Reserve one data jetty per stream and phase, then divide the remaining jetties between streams.
    constexpr int kNumSuNotifyJetties = math::min(SuPlan::kNumPhases, kNumScaleupJetties / kNumStreams - SuPlan::kNumPhases);
    constexpr int kNumSoNotifyJetties = math::min(SoPlan::kNumPhases, kNumScaleoutJetties / kNumStreams - SoPlan::kNumPhases);
    EP_STATIC_ASSERT(kNumSuNotifyJetties > 0 and kNumSoNotifyJetties > 0 and
                     "Each stream needs one data queue per phase and at least one notification queue");
    const auto get_notify_jetty_index = [&](const int phase) {
        const int phase_idx = get_su_or_so_phase(phase);
        if (is_scaleout_phase(phase))
            return kNumSuNotifyJetties + phase_idx % kNumSoNotifyJetties;
        return phase_idx % kNumSuNotifyJetties;
    };

    auto& signals = *static_cast<layout::BucketSignals*>(workspace);
    auto& context = *static_cast<layout::DeviceContext*>(device_context_ptr);
    auto& ready = signals.all_reduce_signals.ready;
    const SuPlan su_plan(rank_idx % kNumScaleupRanks);
    const SoPlan so_plan(rank_idx / kNumScaleupRanks);

    // Cacheable stack storage.
    aicpu::InplaceArray<layout::ChunkIterator<kNumChunkBytes, kNumStreams>, kNumStreams> chunks;
    aicpu::InplaceArray<Peer, kNumPhases> read_peers, notify_peers;
    aicpu::InplaceArray<Jetty, kNumPhases> data_jetties[kNumStreams];
    aicpu::InplaceArray<Jetty, kNumSuNotifyJetties + kNumSoNotifyJetties> notify_jetties[kNumStreams];
    // Progress of the phase. kWindow limits the number of inflight chunks in a logical stream.
    typename SuPlan::NhrSlice slices[kNumStreams][kWindow];
    uint32_t posted[kNumStreams][kNumPhases]{}, completed[kNumStreams][kNumPhases]{};
    uint32_t cq_tails[kNumStreams][kNumPhases][kWindow];
    // Progress of the stream.
    uint32_t loaded[kNumStreams]{}, retired[kNumStreams]{}, num_chunks[kNumStreams];

    // URMA helper for posting all the slices in an NHR phase without ringing the doorbell.
    const auto post_slices = [&](const auto& plan, const int phase, Jetty& jetty, const auto& parent_slice) {
        const int phase_idx = get_su_or_so_phase(phase);
        for (int range = 0; range < plan.get_phase(phase_idx).num_merged_slices; ++ range) {
            const auto slice = plan.get_slice(phase_idx, range, {parent_slice.offset, parent_slice.num_bytes});
            auto* ptr = math::advance_ptr<uint8_t>(buffer, slice.offset);
            for (int64_t pos = 0; pos < slice.num_bytes; pos += comm::kNumMaxReadBytes) {
                const auto bytes = math::min<int64_t>(comm::kNumMaxReadBytes, slice.num_bytes - pos);
                // Preserve READ_REDUCE ordering at each merged slice's boundaries.
                if (phase < kNumReducePhases) {
                    if (pos == 0 or pos + bytes == slice.num_bytes)
                        jetty.template get<kNumTimeoutNs, true, false, true>(
                            read_peers[phase], local_base, ptr + pos, ptr + pos, bytes);
                    else
                        jetty.template get<kNumTimeoutNs, true, false, false>(
                            read_peers[phase], local_base, ptr + pos, ptr + pos, bytes);
                } else {
                    jetty.template get<kNumTimeoutNs, false, false, false>(
                        read_peers[phase], local_base, ptr + pos, ptr + pos, bytes);
                }
            }
        }
    };

    // Initialize
    for (int stream = 0; stream < kNumStreams; ++ stream) {
        const auto& chunk = chunks.emplace(stream, num_buckets, buckets, stream);
        num_chunks[stream] = math::ceil_div(math::max(0, chunk.num_total_chunks - stream), kNumStreams);
        for (int phase = 0; phase < kNumPhases; ++ phase)
            ready[stream][phase] = phase == 0 ? num_chunks[stream] : 0;
    }

    // Keep receive cache lines clean before any peer can update them.
    aicpu::publish_gm(&ready);
    comm::aicpu::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutNs, false, 0, 1, false, true>(
        signals.barrier_signals, context, rank_idx, local_base);

    // Barrier updates scaleout jetty state, so we initialize jetties after barrier.
    for (int phase = 0; phase < kNumPhases; ++ phase) {
        const bool is_so = is_scaleout_phase(phase);
        const int phase_idx = get_su_or_so_phase(phase);
        const int producer = is_so ? so_plan.get_phase(phase_idx).read_peer : su_plan.get_phase(phase_idx).read_peer;
        const int consumer = is_so ? so_plan.get_phase(phase_idx).notify_peer : su_plan.get_phase(phase_idx).notify_peer;
        const auto* jetties = is_so ? context.scaleout_jetties : context.scaleup_jetties;
        const int self = is_so ? rank_idx / kNumScaleupRanks : rank_idx % kNumScaleupRanks;
        read_peers.emplace(phase, jetties, producer - (producer > self));
        notify_peers.emplace(phase, jetties, consumer - (consumer > self));
        const int num_phases = is_so ? SoPlan::kNumPhases : SuPlan::kNumPhases;
        const int num_notify_jetties = is_so ? kNumSoNotifyJetties : kNumSuNotifyJetties;
        for (int stream = 0; stream < kNumStreams; ++ stream) {
            const int jetty_idx = stream * num_phases + phase_idx;
            data_jetties[stream].emplace(phase, jetties, jetty_idx);
            if (phase_idx < num_notify_jetties) {
                notify_jetties[stream].emplace(get_notify_jetty_index(phase), jetties,
                    kNumStreams * num_phases + stream * num_notify_jetties + phase_idx);
            }
        }
    }

    aicpu::timeout_while<kNumTimeoutNs>([&](const bool timeout) {
        aicpu::refresh_gm(&ready);
        bool finished = true;
        for (int stream = 0; stream < kNumStreams; ++ stream) {
            // Load new chunks if there is space left in the window
            while (loaded[stream] < math::min(num_chunks[stream], retired[stream] + kWindow)) {
                auto& chunk = chunks[stream];
                EP_DEVICE_ASSERT(chunk.next());
                slices[stream][loaded[stream] % kWindow] =
                    {chunk.bucket.offset + chunk.chunk_offset_in_bucket, chunk.num_chunk_bytes};
                ++ loaded[stream];
            }
            // Drain downstream phases first, allowing completed chunks to free window slots promptly.
            for (int phase = kNumPhases - 1; phase >= 0; -- phase) {
                auto& jetty = data_jetties[stream][phase];
                auto& sent = posted[stream][phase];
                auto& done = completed[stream][phase];
                // Check if any pending chunks at this phase are done.
                // If so, notify my peer of my chunk readiness.
                const auto previous_done = done;
                if (done < sent) {
                    const auto tail = jetty.try_poll_cq(cq_tails[stream][phase][(sent - 1) % kWindow]);
                    while (done < sent and static_cast<int32_t>(tail - cq_tails[stream][phase][done % kWindow]) >= 0)
                        ++ done;
                }
                if (done != previous_done and phase + 1 < kNumPhases)
                    notify_jetties[stream][get_notify_jetty_index(phase + 1)].template put_value<kNumTimeoutNs>(
                        notify_peers[phase + 1], local_base, &ready[stream][phase + 1], uint64_t(done));
                // Try to make progress at this phase if its inputs are ready.
                const auto local_ready = phase == 0 ? loaded[stream] : completed[stream][phase - 1];
                const auto remote_ready = *reinterpret_cast<const volatile uint64_t*>(&ready[stream][phase]);
                const auto limit = math::min<uint32_t>(local_ready, remote_ready);
                if (sent < limit) {
                    const bool is_so = is_scaleout_phase(phase);
                    for (; sent < limit; ++ sent) {
                        const auto slot = sent % kWindow;
                        const auto& parent_slice = slices[stream][slot];
                        if (is_so) {
                            // Scaleout subdivides the slice owned by the scaleup reduce-scatter.
                            const auto owned_slice = su_plan.get_owned_slice(parent_slice);
                            post_slices(so_plan, phase, jetty, owned_slice);
                        } else {
                            post_slices(su_plan, phase, jetty, parent_slice);
                        }
                        cq_tails[stream][phase][slot] = jetty.get_expected_cq_tail();
                    }
                    jetty.ring_doorbell();
                }
            }
            // Update stream progress.
            retired[stream] = completed[stream][kNumPhases - 1];
            finished = finished and retired[stream] == num_chunks[stream];
        }
        if (timeout)
            throw EPExceptionWithLineInfo("AICPU hybrid all-reduce timeout", "rank=" + std::to_string(rank_idx) +
                ", loaded=" + std::to_string(loaded[0]) + ", retired=" + std::to_string(retired[0]));
        return finished;
    });

    // The AIV epilogue drains notifications and waits for peers before scaling or source reuse.
}

}  // namespace deep_ep
