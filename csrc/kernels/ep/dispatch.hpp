#pragma once

#include <format>

#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static layout::TokenLayout get_dispatch_token_layout(const int& num_hidden_bytes,
                                                     const int& num_sf_packs,
                                                     const int& num_topk) {
    return layout::TokenLayout(
        num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk);
}

static void launch_dispatch(void* x, void* sf,
                            void* topk_idx, void* topk_weights,
                            void* metadata_send_buffer,
                            void* cumulative_local_expert_recv_stats,
                            void* psum_num_recv_tokens_per_rank,
                            void* psum_num_recv_tokens_per_expert,
                            void* num_unaligned_recv_tokens_per_expert,
                            void* dst_buffer_slot_idx, void* dst_gsge_idx,
                            void* buffer, void* workspace, void* host_workspace,
                            void* jetty_ptrs,
                            const int& rank_idx, const int& num_ranks,
                            const int& num_tokens, const int& num_max_tokens_per_rank,
                            const int& num_experts, const int& num_topk,
                            const int& expert_alignment,
                            const bool& cached_mode,
                            const bool& do_cpu_sync,
                            const int& num_hidden_bytes,
                            const int& num_sf_packs,
                            const int& sf_token_stride,
                            const int& sf_hidden_stride,
                            const int& num_ub_bytes,
                            const int& num_vec_cores,
                            const int64_t& num_timeout_cycles,
                            const bool& do_barrier = false,
                            const bool& low_latency_transport = false) {
    // Calculate configs
    EP_HOST_ASSERT(num_vec_cores > 0);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_hidden_bytes > 0);
    EP_HOST_ASSERT(num_ub_bytes > 0);
    EP_HOST_ASSERT(16 <= num_vec_cores);

    // Compile
    const auto kernel = jit->compile(low_latency_transport ? "dispatch_mte" : "dispatch", std::format(R"(
#include <deep_ep/impls/ep/dispatch.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_experts, num_topk,
        expert_alignment, cached_mode, do_cpu_sync,
        num_vec_cores,
        num_hidden_bytes, num_sf_packs, num_max_tokens_per_rank, num_ub_bytes,
        num_timeout_cycles, do_barrier, low_latency_transport));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        x, sf,
        topk_idx, topk_weights,
        metadata_send_buffer,
        cumulative_local_expert_recv_stats,
        psum_num_recv_tokens_per_rank,
        psum_num_recv_tokens_per_expert,
        num_unaligned_recv_tokens_per_expert,
        dst_buffer_slot_idx, dst_gsge_idx,
        buffer, workspace, host_workspace,
        jetty_ptrs,
        rank_idx,
        num_tokens,
        sf_token_stride, sf_hidden_stride);
}

static void launch_dispatch_copy_epilogue(void* buffer,
                                          void* workspace,
                                          void* jetty_ptrs,
                                          void* psum_num_recv_tokens_per_rank,
                                          void* psum_num_recv_tokens_per_expert,
                                          void* recv_x, void* recv_sf, void* recv_topk_weights,
                                          void* recv_src_metadata,
                                          const bool& cached_mode,
                                          const bool& do_zero_padding,
                                          const int& rank_idx, const int& num_ranks,
                                          const int& num_max_tokens_per_rank,
                                          const int& num_experts, const int& num_topk,
                                          const int& expert_alignment,
                                          const int& num_hidden_bytes,
                                          const int& num_sf_packs,
                                          const int& recv_sf_token_stride,
                                          const int& recv_sf_hidden_stride,
                                          const int& num_ub_bytes,
                                          const int& num_vec_cores,
                                          const int64_t& num_timeout_cycles,
                                          const bool& do_barrier = true) {
    // Ascend dispatch epilogue is expand-only and non-hybrid.
    EP_HOST_ASSERT(num_vec_cores > 0);
    EP_HOST_ASSERT(num_ub_bytes > 0);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_sf_packs == 0 or
                   (recv_sf_token_stride == num_sf_packs and recv_sf_hidden_stride == 1));

    // Compile
    const auto kernel = jit->compile("dispatch_copy_epilogue", std::format(R"(
#include <deep_ep/impls/ep/dispatch_copy_epilogue.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_experts, num_topk,
        expert_alignment, cached_mode, do_zero_padding,
        num_vec_cores,
        num_hidden_bytes, num_sf_packs,
        num_max_tokens_per_rank, num_ub_bytes, num_timeout_cycles, do_barrier));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        buffer,
        workspace,
        jetty_ptrs,
        psum_num_recv_tokens_per_rank,
        psum_num_recv_tokens_per_expert,
        recv_x, recv_sf, recv_topk_weights,
        recv_src_metadata,
        rank_idx,
        recv_sf_token_stride, recv_sf_hidden_stride);
}

}  // namespace deep_ep
