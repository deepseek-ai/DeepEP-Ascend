#pragma once

#include <format>

#include <deep_ep/comm/bucket.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_all_gather_aiv(void* workspace, void* local_base, void* buffer, void* device_context,
                                  const int& rank_idx, const int& num_ranks, const int& num_scaleup_ranks,
                                  const int& num_vec_cores,
                                  const int& num_scaleup_jetties, const int& num_scaleout_jetties,
                                  const int& num_buckets, const comm::BucketList& buckets,
                                  const int64_t& num_timeout_cycles, const c10_npu::NPUStream& stream) {
    const auto kernel = jit->compile("all_gather_impl", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/urma.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&all_gather_impl<{}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_scaleup_ranks, num_vec_cores,
        num_scaleup_jetties, num_scaleout_jetties, num_timeout_cycles));
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = num_vec_cores},
                workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
}

static void launch_all_gather_aicpu(void* workspace, void* local_base, void* buffer, void* device_context,
                                    const int& rank_idx, const int& num_ranks, const int& num_scaleup_ranks,
                                    const int& num_scaleup_jetties, const int& num_scaleout_jetties,
                                    const int& num_buckets, const comm::BucketList& buckets,
                                    const int64_t& num_timeout_ns, const c10_npu::NPUStream& stream) {
    const auto kernel = jit->compile("all_gather", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/urma.hpp>

using namespace deep_ep;

extern "C" __global__ __aicpu__ uint32_t all_gather(
        void* workspace, void* local_base, void* buffer, void* device_context,
        int rank_idx, int num_buckets, comm::BucketList buckets) {{
    aicpu::all_gather_impl<{}, {}, {}, {}, {}>(
        workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
    return 0;
}}
)", num_ranks, num_scaleup_ranks,
        num_scaleup_jetties, num_scaleout_jetties, num_timeout_ns), {.aicpu = true});
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = 1},
                workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
}

}  // namespace deep_ep
