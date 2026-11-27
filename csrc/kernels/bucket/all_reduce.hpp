#pragma once

#include <format>

#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_all_reduce(void* workspace, void* local_base, void* buffer, void* device_context,
                              const int& rank_idx, const int& num_ranks, const int& num_scaleup_ranks,
                              const int& num_scaleup_jetties, const int& num_scaleout_jetties,
                              const int& num_buckets, const comm::BucketList& buckets,
                              const int64_t& num_timeout_ns, const c10_npu::NPUStream& stream) {
    constexpr int kNumStreams = 2;
    // Hybrid all-reduce
    if (num_scaleup_ranks > 1 and num_scaleup_ranks < num_ranks) {
        constexpr int64_t kMiB = 1024 * 1024;
        constexpr int kWindow = 8;
        const int num_scaleout_ranks = num_ranks / num_scaleup_ranks;
        const double so_to_su_traffic = static_cast<double>(num_scaleout_ranks - 1) /
                                        num_scaleout_ranks / (num_scaleup_ranks - 1);
        // Use finer chunks as relative SO traffic increases. The thresholds are empirical.
        int64_t num_chunk_bytes = (so_to_su_traffic <= 0.08 ? 512 :
                                  so_to_su_traffic <= 0.10 ? 256 :
                                  so_to_su_traffic <  0.12 ? 64 : 32) * kMiB;
        int64_t total_bytes = 0;
        for (int i = 0; i < num_buckets; ++ i)
            total_bytes += buckets[i].num_bytes;
        // Shrink chunks for smaller buffers, where we target half of each stream's window capacity, rounded up.
        constexpr int kNumTargetChunksPerStream = math::ceil_div(kWindow, 2);
        while (num_chunk_bytes > 32 * kMiB and num_chunk_bytes * kNumStreams * kNumTargetChunksPerStream > total_bytes)
            num_chunk_bytes >>= 1;

        const auto kernel = jit->compile("hybrid_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/hybrid.hpp>

using namespace deep_ep;

extern "C" __global__ __aicpu__ uint32_t hybrid_all_reduce(
        void* workspace, void* local_base, void* buffer, void* device_context,
        int rank_idx, int num_buckets, comm::BucketList buckets) {{
    hybrid_all_reduce_impl<{}, {}, {}, {}, {}, {}, {}, {}>(
        workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
    return 0;
}}
)", num_ranks, num_scaleup_ranks, num_scaleup_jetties, num_scaleout_jetties,
            kNumStreams, num_chunk_bytes, kWindow, num_timeout_ns), {.aicpu = true});
        jit->launch(kernel, {.stream = stream.stream(), .num_blocks = 1},
                    workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
    } else {
        // Pure scaleup/scaleout all-reduce
        const auto num_jetties = num_scaleup_ranks == num_ranks ? num_scaleup_jetties : num_scaleout_jetties;
        const auto num_chunk_bytes = [&]() {
            constexpr int64_t kNumMaxChunkBytes = 1024 * 1024 * 1024;
            constexpr int64_t kNumMinChunkBytes = 128 * 1024 * 1024;
            // Halve chunks until the two logical NHR streams are balanced (33/32 = 1.03x).
            for (auto chunk_bytes = kNumMaxChunkBytes; chunk_bytes > kNumMinChunkBytes; chunk_bytes /= 2) {
                int64_t stream_bytes[kNumStreams]{};
                int next_stream = 0;
                const auto round_bytes = kNumStreams * chunk_bytes;
                // Continue chunk numbering across bucket boundaries, including short tails.
                for (int i = 0; i < num_buckets; ++ i) {
                    const auto bytes = buckets[i].num_bytes;
                    const auto first_bytes = bytes / round_bytes * chunk_bytes + math::min(chunk_bytes, bytes % round_bytes);
                    stream_bytes[next_stream] += first_bytes;
                    stream_bytes[next_stream ^ 1] += bytes - first_bytes;
                    next_stream ^= math::ceil_div(bytes, chunk_bytes) & 1;
                }
                if (math::max(stream_bytes[0], stream_bytes[1]) * 64 <= (stream_bytes[0] + stream_bytes[1]) * 33)
                    return chunk_bytes;
            }
            return kNumMinChunkBytes;
        }();

        const auto kernel = jit->compile("all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/urma.hpp>

using namespace deep_ep;

extern "C" __global__ __aicpu__ uint32_t all_reduce(
        void* workspace, void* local_base, void* buffer, void* device_context,
        int rank_idx, int num_buckets, comm::BucketList buckets) {{
    urma_all_reduce_impl<{}, {}, {}, {}, {}, {}>(
        workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
    return 0;
}}
)", num_ranks, num_scaleup_ranks, num_jetties, kNumStreams, num_chunk_bytes, num_timeout_ns), {.aicpu = true});
        jit->launch(kernel, {.stream = stream.stream(), .num_blocks = 1},
                    workspace, local_base, buffer, device_context, rank_idx, num_buckets, buckets);
    }
}

static void launch_all_reduce_epilogue(void* barrier_signals, void* local_base, void* buffer, void* device_context,
                                       const int& rank_idx, const int& num_ranks, const int& num_scaleup_ranks,
                                       const int& num_scaleup_jetties, const int& num_scaleout_jetties,
                                       const int& num_buckets, const comm::BucketList& buckets, const float& scale,
                                       const int64_t& num_timeout_cycles, const c10_npu::NPUStream& stream) {
    const auto num_vec_cores = scale == 1.0f ? 2 : runtime->get_num_vec_cores();

    const auto kernel = jit->compile("all_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/epilogue.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&all_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_scaleup_ranks, num_vec_cores, num_scaleup_jetties, num_scaleout_jetties, num_timeout_cycles));
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = num_vec_cores},
                barrier_signals, local_base, buffer, device_context, rank_idx, num_buckets, buckets, scale);
}

}  // namespace deep_ep
