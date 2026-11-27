#pragma once

#include <simt_api/asc_simt.h>

#include <deep_ep/common/utils.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/layout/ep/lb.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

namespace deep_ep {

template <int kNumRanks,
          int kNumLocalExperts,
          int kNumRedundantExpertsPerRank,
          int64_t kNumTimeoutCycles,
          int kNumWarps>
__global__ __vector__ void lb_reduce_grads_issue_impl(
        __gm__ const int* redundancy_mapping, __gm__ void* workspace, __gm__ void* device_context_ptr,
        const int rank_idx, const layout::EPWeight weight) {
    AscendC::InitSocState();

    auto& signals = static_cast<__gm__ layout::EPSignals*>(workspace)->barrier_signals;
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto error_status = layout::DeviceContext::get_error_status(*device_context);
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());

    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_storage[layout::kNumLBUbBytes];
    auto& ub = *reinterpret_cast<__ubuf__ layout::EPLBUBLayout*>(ub_storage);

    asc_vf_call<layout::EPLBUBLayout::build_tasks<kNumRanks, kNumLocalExperts, kNumRedundantExpertsPerRank, kNumWarps, true>>(
        dim3(kNumWarps * warpSize, 1, 1), redundancy_mapping, rank_idx, error_status, &ub);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);

    comm::scalar::barrier<kNumRanks, kNumRanks, kNumTimeoutCycles, true, true, true>(
        signals, *device_context, block_idx, rank_idx);

    // Only use one SQ to avoid concurrent READs
    if (block_idx != 0)
        return;
    handle::HcommJetty jetty(&ub.jetty_info, ub.ub_sqe,
                             kNumRanks > 1 ? device_context->scaleup_jetties : nullptr, 0,
                             error_status);
    for (int dst_rank_idx = 0; dst_rank_idx < kNumRanks; ++ dst_rank_idx)
        if (dst_rank_idx != rank_idx)
            handle::HcommPeer(&ub.peer_infos[dst_rank_idx], device_context->scaleup_jetties, dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1);
    layout::EPWeightIterator<1, comm::kNumMaxReadBytes> iterator(ub, &weight, 0, 1);
    while (iterator.next()) {
        if (iterator.dst_rank_idx != rank_idx) {
            const handle::HcommPeer peer(&ub.peer_infos[iterator.dst_rank_idx]);
            jetty.template get<kNumTimeoutCycles, true>(peer, workspace, iterator.src_ptr, iterator.dst_ptr, iterator.num_chunk_bytes);
        } else {
            // Retain local contributions for compatibility.
            asc_set_atomic_add_float();
            block_copy(static_cast<__gm__ uint8_t*>(iterator.dst_ptr), static_cast<__gm__ uint8_t*>(iterator.src_ptr),
                       iterator.num_chunk_bytes, 0, 1);
            asc_set_atomic_none();
        }
    }
    if constexpr (kNumRanks > 1)
        jetty.template align_sq<kNumTimeoutCycles>();
}

}  // namespace deep_ep
