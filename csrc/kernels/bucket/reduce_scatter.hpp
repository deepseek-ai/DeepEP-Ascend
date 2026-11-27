#pragma once

#include <algorithm>
#include <format>

#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static constexpr int kNumReduceScatterVecCores = 2;

static void launch_reduce_scatter(void* workspace, void* local_base, void* buffer,
                                  void* device_context,
                                  const int& num_buckets, const comm::BucketList& src_buckets,
                                  const int& rank_idx, const int& num_ranks, const int& num_scaleup_ranks,
                                  const int64_t& num_timeout_cycles, const c10_npu::NPUStream& stream) {
    const int64_t max_chunk_bytes = (num_scaleup_ranks == num_ranks ? 128 : 16) * 1024 * 1024;
    const int64_t min_chunk_bytes = (num_scaleup_ranks == num_ranks ? 4 : 8) * 1024 * 1024;
    int64_t total_bytes = 0;
    for (int i = 0; i < num_buckets; ++ i)
        total_bytes += src_buckets[i].num_bytes;
    auto chunk_bytes = std::clamp<int64_t>(
        math::align<int64_t>(math::ceil_div<int64_t>(total_bytes, kNumReduceScatterVecCores), kNumUrmaAlignmentBytes),
        min_chunk_bytes, max_chunk_bytes);
    // Reducing chunk size if the busiest producer's load is imbalanced (33/32 = 1.03x)
    while (chunk_bytes > min_chunk_bytes) {
        int64_t core_bytes[kNumReduceScatterVecCores] = {};
        int64_t chunk_idx = 0;
        for (int i = 0; i < num_buckets; ++ i) {
            const auto bytes = src_buckets[i].num_bytes;
            const auto num_full_chunks = bytes / chunk_bytes;
            for (int c = 0; c < kNumReduceScatterVecCores; ++ c) {
                const auto count = num_full_chunks / kNumReduceScatterVecCores +
                    ((c - chunk_idx % kNumReduceScatterVecCores + kNumReduceScatterVecCores) %
                     kNumReduceScatterVecCores < num_full_chunks % kNumReduceScatterVecCores);
                core_bytes[c] += count * chunk_bytes;
            }
            chunk_idx += num_full_chunks;
            if (const auto rem = bytes % chunk_bytes; rem > 0)
                core_bytes[(chunk_idx ++) % kNumReduceScatterVecCores] += rem;
        }
        int64_t max_bytes = 0;
        for (int c = 0; c < kNumReduceScatterVecCores; ++ c)
            max_bytes = math::max(max_bytes, core_bytes[c]);
        if (max_bytes * kNumReduceScatterVecCores * 32 <= total_bytes * 33)
            break;
        chunk_bytes = math::max(min_chunk_bytes, chunk_bytes / 2 / kNumUrmaAlignmentBytes * kNumUrmaAlignmentBytes);
    }

    // Compile
    const auto kernel = jit->compile("reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/urma.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&urma_reduce_scatter_impl<{}, {}, {}, {}, {}>);
}}
)", num_ranks, num_scaleup_ranks, kNumReduceScatterVecCores, chunk_bytes, num_timeout_cycles));

    // Launch
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = kNumReduceScatterVecCores},
                workspace, local_base, buffer, device_context, rank_idx, num_buckets, src_buckets);
}

static void launch_reduce_scatter_epilogue(void* barrier_signals, void* local_base, void* buffer,
                                          void* device_context,
                                          const int& rank_idx, const int& num_ranks, const int& num_scaleup_ranks,
                                          const int& num_scaleup_jetties, const int& num_scaleout_jetties,
                                          const int& num_buckets, const comm::BucketList& src_buckets, const float& scale,
                                          const int64_t& num_timeout_cycles, const c10_npu::NPUStream& stream) {
    const auto num_active_scaleup_jetties = std::min(kNumReduceScatterVecCores, num_scaleup_jetties);
    const auto num_active_scaleout_jetties = std::min(kNumReduceScatterVecCores, num_scaleout_jetties);
    const auto num_vec_cores = scale == 1.0f ?
        kNumReduceScatterVecCores : runtime->get_num_vec_cores();

    // Compile
    const auto kernel = jit->compile("reduce_scatter_epilogue", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/epilogue.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&reduce_scatter_epilogue_impl<{}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_scaleup_ranks, num_vec_cores, num_active_scaleup_jetties, num_active_scaleout_jetties, num_timeout_cycles));

    // Launch
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = num_vec_cores},
                barrier_signals, local_base, buffer, device_context, rank_idx, num_buckets, src_buckets, scale);
}

}  // namespace deep_ep
