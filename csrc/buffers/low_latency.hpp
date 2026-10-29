#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_config.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/comm_args.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h>
#include "base.hpp"
#include "../comm/barrier.hpp"
#include "../utils/tensor.hpp"

namespace deep_ep {

inline uint32_t get_low_latency_combine_ub_bytes() {
    constexpr auto required = std::max(DispatchDedup::kEpServerDedupCountUbBytes,
                                      CombineDedup::kEpServerDedupCombineUbBytes);
    const auto available = runtime->get_num_ubuf_bytes_per_vec_core();
    if (available < required) {
        throw std::runtime_error("Low-latency EP requires at least " + std::to_string(required) +
                                 " bytes of usable UB per vector core; CANN reports " +
                                 std::to_string(available) + " bytes");
    }
    return static_cast<uint32_t>(std::min<int64_t>(available / 32 * 32,
                                                 CombineDedup::kEpServerDedupCombineRecvUbBytes));
}

// Each rank's 8 GiB ping-pong arena starts at a rank-specific offset. A tail
// allocation covers the bias; common HCCL barrier state precedes the arena.
class LowLatencyBuffer: public BufferBase {
    int device_idx;
    static constexpr int64_t kRankBaseBiasBytes = 1024;
    static constexpr int64_t kPeerPaddingBytes = 2 * 1024 * 1024;
    static_assert((DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE - 1) * kRankBaseBiasBytes < kPeerPaddingBytes);

public:
    std::shared_ptr<comm::Context> context;
    torch::Tensor comm_args;
    uint32_t combine_ub_bytes = 0;

    LowLatencyBuffer(int rank, int world, const std::string& hccl_group_name,
                     bool use_huge1g_page, int cpu_timeout, int gpu_timeout, bool explicitly_destroy):
        BufferBase(explicitly_destroy), device_idx(c10_npu::getCurrentNPUStream().device_index()) {
        EP_HOST_ASSERT((world == 8 or world == 16 or world == 32 or world == 64 or world == 128 or world == 256));
        EP_HOST_ASSERT(runtime->get_npu_arch() == 3510 and "Low latency requires Ascend 950/A5");
        EP_HOST_ASSERT(runtime->get_num_vec_cores() >= 64);
        combine_ub_bytes = get_low_latency_combine_ub_bytes();
        context = std::make_shared<comm::Context>(
            hccl_group_name, rank, world, kNumAllocationAlignmentBytes,
            DispatchDedup::kEpServerDedupPeerBytes + kPeerPaddingBytes, 0, use_huge1g_page, cpu_timeout, gpu_timeout);
        main_context = context;
        context->set_common_signals(context->workspace);
        DispatchDedup::CommArgs args{};
        args.rank = rank;
        args.rankSize = world;
        for (int peer = 0; peer < world; ++peer) {
            const auto base = context->hccl_context->peer_ptrs[peer];
            EP_HOST_ASSERT(base != 0 and base % 32 == 0);
            // Apply the destination rank's bias after translating its allocation,
            // so the local receiver and every remote sender use the same address.
            args.peerMems[peer] = reinterpret_cast<uint8_t*>(base + context->num_workspace_bytes +
                                                           peer * kRankBaseBiasBytes);
        }
        comm_args = torch::empty({static_cast<int64_t>(sizeof(args))},
                                 torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU));
        ACL_CHECK(aclrtMemcpy(comm_args.data_ptr(), sizeof(args), &args, sizeof(args), ACL_MEMCPY_HOST_TO_DEVICE));
    }

    ~LowLatencyBuffer() noexcept(false) override { destroy_on_destruction("low latency EP"); }

    void check_available() const {
        EP_HOST_ASSERT(not destroyed);
        context->check_available();
        EP_HOST_ASSERT(c10_npu::getCurrentNPUStream().device_index() == device_idx);
    }

    int world() const { check_available(); return context->num_ranks; }
    int64_t capacity() const { check_available(); return context->num_gpu_buffer_bytes - kPeerPaddingBytes; }

    void destroy() override {
        check_available();
        comm::barrier(*context, true);
        comm_args = torch::Tensor();
        context->finalize();
        destroyed = true;
    }
};

}  // namespace deep_ep
