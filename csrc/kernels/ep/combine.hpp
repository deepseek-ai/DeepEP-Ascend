#pragma once

#include <cstdint>
#include <format>

#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static layout::TokenLayout get_combine_token_layout(const int& num_hidden_bytes, const int& num_topk) {
    return layout::TokenLayout(num_hidden_bytes, 0, num_topk, false);
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            void* src_metadata,
                            void* psum_num_recv_tokens_per_rank,
                            void* buffer,
                            void* send_buffer,
                            void* workspace,
                            void* jetty_ptrs,
                            const int& rank_idx, const int& num_ranks,
                            const int& num_max_tokens_per_rank,
                            const int& num_topk,
                            const int& hidden,
                            const int& num_ub_bytes,
                            const int& num_vec_cores,
                            const int64_t& num_timeout_cycles,
                            const bool& do_barrier = false,
                            const bool& low_latency_transport = false) {
    EP_HOST_ASSERT(num_vec_cores > 0);
    EP_HOST_ASSERT(num_ub_bytes > 0);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);

    // Compile
    const auto kernel = jit->compile(low_latency_transport ? "combine_mte" : "combine", std::format(R"(
#include <deep_ep/impls/ep/combine.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_topk, hidden, num_max_tokens_per_rank, num_vec_cores,
        num_timeout_cycles, num_ub_bytes, do_barrier, topk_weights != nullptr, low_latency_transport));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        x,
        topk_weights,
        src_metadata,
        psum_num_recv_tokens_per_rank,
        buffer,
        send_buffer,
        workspace,
        jetty_ptrs,
        rank_idx);
    return buffer;
}

static void launch_combine_reduce_epilogue(void* combined_x,
                                           void* combined_topk_weights,
                                           void* combined_topk_idx,
                                           void* reduce_buffer,
                                           void* bias_0,
                                           void* bias_1,
                                           void* workspace,
                                           void* jetty_ptrs,
                                           const int& num_ranks,
                                           const int& num_combined_tokens,
                                           const int& num_max_tokens_per_rank,
                                           const int& num_experts, const int& num_topk,
                                           const int& hidden,
                                           const int& num_ub_bytes,
                                           const int& num_vec_cores,
                                           const int64_t& num_timeout_cycles,
                                           const bool& do_barrier = true) {
    EP_HOST_ASSERT(num_vec_cores > 0);
    EP_HOST_ASSERT(num_ub_bytes > 0);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
    EP_HOST_ASSERT(hidden % (kNumVectorRegisterBytes / static_cast<int>(sizeof(uint16_t))) == 0);

    // Compile
    const auto kernel = jit->compile("combine_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/ep/combine_reduce_epilogue.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_experts, num_topk,
        hidden, num_max_tokens_per_rank, num_ub_bytes, num_vec_cores, num_timeout_cycles, do_barrier));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        combined_x,
        combined_topk_weights,
        combined_topk_idx,
        reduce_buffer,
        bias_0,
        bias_1,
        workspace,
        jetty_ptrs,
        num_combined_tokens);
}

}  // namespace deep_ep
