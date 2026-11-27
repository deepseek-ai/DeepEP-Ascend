#pragma once

#include <format>

#include <deep_ep/comm/bucket.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_ring_send_recv(void* workspace, void* local_base, void* buffer, void* device_context,
                                  const int& rank_idx, const int& dst_rank_idx,
                                  const int& num_ranks, const int& num_jetties, const int& num_buckets,
                                  const comm::BucketList& src_buckets, const comm::BucketList& dst_buckets,
                                  const int64_t& num_timeout_ns, const c10_npu::NPUStream& stream) {
    const auto kernel = jit->compile("ring_send_recv", std::format(R"(
#include <deep_ep/impls/bucket/ring_send_recv/urma.hpp>

using namespace deep_ep;

extern "C" __global__ __aicpu__ uint32_t ring_send_recv(
        void* workspace, void* local_base, void* buffer, void* device_context,
        int rank_idx, int dst_rank_idx, int num_buckets,
        comm::BucketList src_buckets, comm::BucketList dst_buckets) {{
    ring_send_recv_impl<{}, {}, {}>(
        workspace, local_base, buffer, device_context,
        rank_idx, dst_rank_idx, num_buckets, src_buckets, dst_buckets);
    return 0;
}}
)", num_ranks, num_jetties, num_timeout_ns), {.aicpu = true});
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = 1},
                workspace, local_base, buffer, device_context,
                rank_idx, dst_rank_idx, num_buckets, src_buckets, dst_buckets);
}

}  // namespace deep_ep
