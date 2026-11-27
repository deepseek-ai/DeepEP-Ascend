#pragma once

#include <format>

#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_barrier(void* barrier_signals, void* local_base, void* device_context,
                           const int& rank_idx, const int& num_ranks,
                           const int& num_scaleup_ranks, const int& num_scaleup_jetties,
                           const int& num_scaleout_jetties, const bool& sequential,
                           const int64_t& num_timeout_cycles, const bool& wait_completion,
                           const c10_npu::NPUStream& stream) {
    const auto num_scaleout_ranks = num_ranks / num_scaleup_ranks;
    const bool parallel = not sequential and num_scaleup_ranks > 1 and num_scaleout_ranks > 1;
    auto num_blocks = math::max(num_scaleup_jetties, num_scaleout_jetties + (parallel ? 1 : 0));
    num_blocks = math::max(1, math::min(num_blocks, runtime->get_num_vec_cores()));
    // Compile
    const auto kernel = jit->compile("barrier", std::format(R"(
#include <deep_ep/impls/comm/barrier.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_scaleup_ranks, num_blocks, num_scaleup_jetties, num_scaleout_jetties,
    num_timeout_cycles, sequential, wait_completion));

    // Launch
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = num_blocks},
                barrier_signals, local_base, device_context, rank_idx);
}

}  // namespace deep_ep
