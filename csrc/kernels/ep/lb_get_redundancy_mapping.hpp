#pragma once

#include <cstdint>
#include <format>

#include <deep_ep/layout/ep/lb.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"
#include "../comm/api.hpp"

namespace deep_ep {

static void launch_lb_get_redundancy_mapping(const comm::Context& context, const torch::Tensor& topk_idx, const torch::Tensor& redundancy_mapping,
                                             const torch::Tensor& remapped_topk_idx, const int& num_experts, const int& num_routes,
                                             const int& token_alignment, const int& num_vec_cores, const int& num_redundant_experts_per_rank,
                                             const c10_npu::NPUStream& stream) {
    constexpr int kNumWarps = 32;
    constexpr int kNumThreadsPerWarp = 32;
    constexpr int kNumThreads = kNumWarps * kNumThreadsPerWarp;
    const int num_main_experts_per_rank = num_experts / context.num_ranks;
    const int num_plan_ints = 2 * kNumThreads + 4 * num_experts + 2 * context.num_ranks;
    const int num_ub_bytes = math::align(
        math::max(kNumWarps * num_experts, num_plan_ints) * static_cast<int>(sizeof(int)), kNumUbAlignmentBytes);
    EP_HOST_ASSERT(num_ub_bytes <= runtime->get_num_ubuf_bytes_per_vec_core());
    EP_HOST_ASSERT(math::ceil_div<int64_t>(num_routes, kNumThreads) * num_experts <=
                   layout::LBPrepareSignals::kNumMaxPrepareRoutes);

    // Compile
    const auto kernel = jit->compile("lb_get_redundancy_mapping", std::format(R"(
#include <deep_ep/impls/ep/lb_get_redundancy_mapping.hpp>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::lb_get_redundancy_mapping_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", context.num_ranks, context.hcomm_context->num_scaleup_ranks, num_main_experts_per_rank, num_redundant_experts_per_rank,
        token_alignment, num_vec_cores, kNumThreads, context.num_gpu_timeout_cycles));

    // Launch
    jit->launch(
        kernel, {.stream = stream.stream(), .num_blocks = num_vec_cores},
        topk_idx.data_ptr(), num_routes, context.rank_idx,
        redundancy_mapping.data_ptr<int>(), remapped_topk_idx.data_ptr(),
        context.workspace, context.hcomm_context->device_context);
}

}  // namespace deep_ep
