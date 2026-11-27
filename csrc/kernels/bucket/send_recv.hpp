#pragma once

#include <format>

#include <deep_ep/layout/bucket/send_recv.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_send_recv(void* workspace, void* local_base, void* buffer,
                             void* device_context,
                             const int& rank_idx, const int& num_ranks,
                             const int& num_scaleup_ranks, const int& num_scaleout_jetties,
                             const int& num_entries, const layout::SendRecvList& entries,
                             const int64_t& num_timeout_cycles,
                             const c10_npu::NPUStream& stream) {
    EP_HOST_ASSERT(num_scaleup_ranks == 1 or num_scaleup_ranks == num_ranks);
    const auto num_blocks = runtime->get_num_vec_cores();
    const auto kernel = jit->compile("send_recv", std::format(R"(
#include <deep_ep/impls/bucket/send_recv/urma.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&urma_send_recv_impl<{}, {}, {}, {}, {}>);
}}
)", num_ranks, num_scaleup_ranks, num_blocks, num_scaleout_jetties, num_timeout_cycles));
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = num_blocks},
                workspace, local_base, buffer, device_context, rank_idx, num_entries, entries);

    // Handle self copy separately
    for (int entry_idx = 0; entry_idx < num_entries; ++ entry_idx) {
        const auto& entry = entries.entries[entry_idx];
        if (entry.dst_rank_idx == rank_idx and entry.src_offset != entry.dst_offset) {
            const auto src = math::advance_ptr<void>(buffer, entry.src_offset);
            const auto dst = math::advance_ptr<void>(buffer, entry.dst_offset);
            ACL_CHECK(aclrtMemcpyAsync(dst, entry.num_bytes, src, entry.num_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        }
    }
}

}  // namespace deep_ep
