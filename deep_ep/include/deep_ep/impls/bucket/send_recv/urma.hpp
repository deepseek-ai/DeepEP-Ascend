#pragma once

#include <deep_ep/common/utils.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/layout/bucket/send_recv.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

namespace deep_ep {

template <int kNumRanks, int kNumScaleupRanks, int kNumVecCores, int kNumScaleoutJetties, int64_t kNumTimeoutCycles>
__global__ __vector__ void urma_send_recv_impl(__gm__ void* workspace, __gm__ void* local_base,
                                              __gm__ void* buffer, __gm__ void* device_context_ptr,
                                              const int rank_idx, const int num_entries,
                                              const layout::SendRecvList entries) {
    AscendC::InitSocState();
    constexpr bool kScaleout = kNumScaleupRanks < kNumRanks;
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks <= (kScaleout ? kNumMaxScaleoutRanks : kNumMaxScaleupRanks));
    EP_STATIC_ASSERT(kNumScaleupRanks == 1 or kNumScaleupRanks == kNumRanks);

    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto num_blocks = static_cast<int>(AscendC::GetBlockNum());
    auto& signals = *static_cast<__gm__ layout::BucketSignals*>(workspace);
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto error_status = layout::DeviceContext::get_error_status(*device_context);

    alignas(alignof(handle::HcommPeerInfo)) __ubuf__ uint8_t peer_info[sizeof(handle::HcommPeerInfo)];
    alignas(alignof(handle::HcommJettyInfo)) __ubuf__ uint8_t jetty_info[sizeof(handle::HcommJettyInfo)];
    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];

    // Ensure dsts are consumed
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles, false, false, true,
                          kNumVecCores, 0, kNumScaleoutJetties>(
        signals.barrier_signals, *device_context, block_idx, rank_idx, local_base);

    auto* jetties = kNumScaleupRanks == 1 ? device_context->scaleout_jetties : device_context->scaleup_jetties;
    handle::HcommJetty hcomm(reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(jetty_info), ub_sqe,
                            kNumRanks > 1 and block_idx < num_entries ? jetties : nullptr, block_idx,
                            error_status);
    for (int entry_idx = block_idx; entry_idx < num_entries; entry_idx += num_blocks) {
        const auto& entry = entries.entries[entry_idx];
        const auto dst_rank_idx = entry.dst_rank_idx;
        if (dst_rank_idx == rank_idx)
            continue;
        const auto dst_jetty_idx = dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1;
        const handle::HcommPeer peer(reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(peer_info), jetties, dst_jetty_idx);
        const auto src = math::advance_ptr<__gm__ uint8_t>(buffer, entry.src_offset);
        const auto dst = math::advance_ptr<__gm__ uint8_t>(buffer, entry.dst_offset);
        for (int64_t offset = 0; offset < entry.num_bytes; offset += comm::kNumMaxWriteBytes)
            hcomm.template put<kNumTimeoutCycles>(peer, local_base, dst + offset, src + offset,
                math::min<int64_t>(comm::kNumMaxWriteBytes, entry.num_bytes - offset));
    }
}

}  // namespace deep_ep
