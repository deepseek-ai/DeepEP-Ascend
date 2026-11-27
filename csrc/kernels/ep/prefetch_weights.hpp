#pragma once

#include <format>

#include <deep_ep/layout/ep/lb.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_lb_prefetch_weights(void* workspace, void* device_context,
                                       const int& rank_idx, const int& num_ranks,
                                       const int& num_weights,
                                       const layout::EPWeightList& weights,
                                       const int* redundancy_mapping,
                                       const int& num_redundant_experts_per_rank,
                                       const int& num_local_experts,
                                       const int& num_vec_cores,
                                       const int& num_ub_bytes,
                                       const int64_t& num_timeout_cycles) {
    constexpr int kNumWriteBytes = 1024 * 1024;    // slightly better than 64 MB
    const int num_warps = math::max(1, math::min(16, math::ceil_div(num_ranks * num_redundant_experts_per_rank, 32)));
    EP_HOST_ASSERT(num_vec_cores >= 2 and num_vec_cores % 2 == 0);
    EP_HOST_ASSERT(num_vec_cores <= runtime->get_num_vec_cores());
    EP_HOST_ASSERT(layout::kNumLBUbBytes + kNumBlockCopyUbBytes <= num_ub_bytes);

    // Compile
    const auto kernel = jit->compile("lb_prefetch_weights_issue", std::format(R"(
#include <deep_ep/impls/ep/prefetch_weights.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&lb_prefetch_weights_issue_impl<{}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_local_experts, num_redundant_experts_per_rank,
    num_weights, kNumWriteBytes, num_timeout_cycles, num_warps));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = 0},
        redundancy_mapping, workspace, device_context, rank_idx, weights
    );
}

}  // namespace deep_ep
