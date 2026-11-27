#pragma once

#include <format>

#include <deep_ep/layout/ep/lb.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_lb_reduce_grads(void* workspace, void* device_context,
                                   const int& rank_idx, const int& num_ranks,
                                   float* redundant_expert_grads, float* expert_grads,
                                   const int* redundancy_mapping,
                                   const int& num_redundant_experts_per_rank, const int& num_local_experts,
                                   const int64_t& hidden, const int& num_vec_cores, const int& num_ub_bytes,
                                   const int64_t& num_timeout_cycles) {
    const int num_warps = math::max(1, math::min(16, math::ceil_div(num_ranks * num_redundant_experts_per_rank, 32)));
    EP_HOST_ASSERT(num_vec_cores >= 2 and num_vec_cores % 2 == 0);
    EP_HOST_ASSERT(num_vec_cores <= runtime->get_num_vec_cores());
    EP_HOST_ASSERT(layout::kNumLBUbBytes + kNumBlockCopyUbBytes <= num_ub_bytes);
    const layout::EPWeight weight = {redundant_expert_grads, expert_grads, hidden * int64_t(sizeof(float))};
    EP_HOST_ASSERT(weight.num_bytes_per_expert % kNumUbAlignmentBytes == 0);

    // Compile
    const auto kernel = jit->compile("lb_reduce_grads_issue", std::format(R"(
#include <deep_ep/impls/ep/reduce_grads.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&lb_reduce_grads_issue_impl<{}, {}, {}, {}, {}>);
}}
)", num_ranks, num_local_experts, num_redundant_experts_per_rank, num_timeout_cycles, num_warps));

    // Launch
    jit->launch(kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = 0},
                redundancy_mapping, workspace, device_context, rank_idx, weight);
}

}  // namespace deep_ep
