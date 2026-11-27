#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/utils.hpp>
#include <deep_ep/layout/pp/workspace.hpp>
#include <deep_ep/layout/common/context.hpp>

#include <kernel_operator.h>

namespace deep_ep {

__aicore__ void get_buffer_offset(
        const int& src_rank_idx, const int& dst_rank_idx, const int& num_ranks,
        int& local_idx_in_dst, int& dst_idx_in_local) {
    const auto next_rank_idx = (src_rank_idx + 1) % num_ranks;
    local_idx_in_dst = dst_rank_idx == next_rank_idx ? 1 : 0;
    dst_idx_in_local = dst_rank_idx == next_rank_idx ? 0 : 1;
}

#if defined(EP_PP_SEND) or defined(__CLION_IDE__)
template <int64_t kNumTimeoutCycles>
__global__ __vector__ void pp_send_impl(__gm__ void* x, const int64_t num_x_bytes,
                                      __gm__ void* buffer, __gm__ void* workspace, __gm__ void* device_context_ptr,
                                      const int rank_idx, const int dst_rank_idx, const int num_ranks,
                                      const int64_t num_max_tensor_bytes, const int num_max_inflight_tensors) {
    AscendC::InitSocState();

    auto& pp = *static_cast<__gm__ layout::PPSignals*>(workspace);
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto error_status = layout::DeviceContext::get_error_status(*device_context);

    // Block index
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());

    // Resolve which buffer slot this (src -> dst) pair uses
    // TODO(HUAWEI): we cannot use `std::pair` and structural bindings here
    int local_idx_in_dst, dst_idx_in_local;
    get_buffer_offset(rank_idx, dst_rank_idx, num_ranks, local_idx_in_dst, dst_idx_in_local);

    // Read our running send count for this neighbor, pick the inflight ring slot
    const auto send_count_ptr = &pp.send_count[dst_idx_in_local];
    const auto send_count = AscendC::ReadGmByPassDCache(send_count_ptr);
    const auto slot_idx = send_count % num_max_inflight_tensors;

    // Compute the recv landing buffer as a local pointer (region `local_idx_in_dst`).
    const auto local_recv_ptr = math::advance_ptr<__gm__ uint8_t>(
        buffer, (local_idx_in_dst * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Flow control: block 0 waits until the receiver has freed this slot (credit signal).
    // Credit target follows NCCL: at most `num_max_inflight_tensors` outstanding sends.
    if (block_idx == 0) {
        const auto credit_ptr = &pp.release[dst_idx_in_local];
        const auto credit_target = send_count - num_max_inflight_tensors + 1;
        comm::scalar::timeout_while<kNumTimeoutCycles>([&](const bool& timeout) __aicore__ {
            const auto credit = AscendC::ReadGmByPassDCache(credit_ptr);
            if (credit >= credit_target)
                return true;

            if (timeout) {
                AscendC::printf("DeepEP PP send timeout, rank: %d, destination: %d, credit: %ld, target: %ld\n",
                                rank_idx, dst_rank_idx, credit, credit_target);
            }
            return false;
        }, 0, error_status);
    }
    AscendC::SyncAll<true>();

    // Submit the payload and absolute data-ready value as one ordered URMA write.
    if (block_idx == 0) {
        // TODO: overlap initialize and wait?
        // Initialize
        const auto dst_jetty_idx =
            dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1;
        __ubuf__ uint64_t ub_peer_info[sizeof(handle::HcommPeerInfo) / sizeof(uint64_t)];
        const auto peer_info = reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(ub_peer_info);
        const handle::HcommPeer peer(peer_info, device_context->scaleup_jetties, dst_jetty_idx);
        alignas(8) __ubuf__ uint8_t ub_jetty_info[sizeof(handle::HcommJettyInfo)];
        alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];
        handle::HcommJetty hcomm(
            reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(ub_jetty_info), ub_sqe,
            device_context->scaleup_jetties, dst_jetty_idx, error_status);

        // Issue
        hcomm.template put_with_notify<kNumTimeoutCycles>(
            peer, workspace, local_recv_ptr, x, num_x_bytes,
            &pp.arrival[local_idx_in_dst], static_cast<uint64_t>(send_count + 1));
        hcomm.template drain<kNumTimeoutCycles>();
        AscendC::WriteGmByPassDCache(send_count_ptr, send_count + 1);
    }
}
#endif

#if defined(EP_PP_RECV) or defined(__CLION_IDE__)
template <int64_t kNumTimeoutCycles>
__global__ __vector__ void pp_recv_impl(__gm__ void* x, const int64_t num_x_bytes,
                                        __gm__ void* buffer, __gm__ void* workspace,
                                        __gm__ void* device_context_ptr,
                                        const int rank_idx, const int src_rank_idx, const int num_ranks,
                                        const int64_t num_max_tensor_bytes, const int num_max_inflight_tensors) {
    AscendC::InitSocState();

    auto& pp = *static_cast<__gm__ layout::PPSignals*>(workspace);
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto error_status = layout::DeviceContext::get_error_status(*device_context);

    // Block indices
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto num_blocks = static_cast<int>(AscendC::GetBlockNum());

    // Resolve which buffer slot this (src -> dst=self) pair uses
    // TODO(HUAWEI): we cannot use `std::pair` and structural bindings here
    int src_idx_in_local, local_idx_in_src;
    get_buffer_offset(src_rank_idx, rank_idx, num_ranks, src_idx_in_local, local_idx_in_src);

    // Read our running recv count for this neighbor, pick the inflight ring slot
    const auto recv_count_ptr = &pp.recv_count[src_idx_in_local];
    const auto recv_count = AscendC::ReadGmByPassDCache(recv_count_ptr);
    const auto slot_idx = recv_count % num_max_inflight_tensors;

    // The data has already been written into our local recv buffer (region `src_idx_in_local`)
    const auto recv_buffer_ptr = math::advance_ptr<__gm__ uint8_t>(
        buffer, (src_idx_in_local * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Block 0 waits until the sender has delivered this slot (local data-arrived signal)
    if (block_idx == 0) {
        const auto data_ptr = &pp.arrival[src_idx_in_local];
        const auto data_target = recv_count + 1;
        comm::scalar::timeout_while<kNumTimeoutCycles>([&](const bool& timeout) __aicore__ {
            const auto data = AscendC::ReadGmByPassDCache(data_ptr);
            if (data >= data_target)
                return true;

            if (timeout) {
                AscendC::printf("DeepEP PP recv timeout, rank: %d, source: %d, data: %ld, target: %ld\n",
                                rank_idx, src_rank_idx, data, data_target);
            }
            return false;
        }, 0, error_status);
    }
    AscendC::SyncAll<true>();

    // Copy from the local recv buffer into the user output tensor
    block_copy(recv_buffer_ptr, static_cast<__gm__ uint8_t*>(x), num_x_bytes, block_idx, num_blocks);
    AscendC::SyncAll<true>();

    // Block 0 publishes the next recv count as the sender's remote credit
    if (block_idx == 0) {
        const auto src_jetty_idx =
            src_rank_idx < rank_idx ? src_rank_idx : src_rank_idx - 1;
        __ubuf__ uint64_t ub_peer_info[sizeof(handle::HcommPeerInfo) / sizeof(uint64_t)];
        const auto peer_info = reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(ub_peer_info);
        const handle::HcommPeer peer(peer_info, device_context->scaleup_jetties, src_jetty_idx);
        alignas(8) __ubuf__ uint8_t ub_jetty_info[sizeof(handle::HcommJettyInfo)];
        alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];
        handle::HcommJetty hcomm(
            reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(ub_jetty_info), ub_sqe,
            device_context->scaleup_jetties, src_jetty_idx, error_status);

        AscendC::WriteGmByPassDCache(recv_count_ptr, recv_count + 1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        // TODO(HUAWEI): add an inline URMA write API to avoid reading the 8-byte credit value from GM.
        hcomm.template put<kNumTimeoutCycles>(
            peer, workspace,
            &pp.release[local_idx_in_src], recv_count_ptr, sizeof(int64_t));
        hcomm.template drain<kNumTimeoutCycles>();
    }
}
#endif

}  // namespace deep_ep
