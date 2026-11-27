#pragma once

#include <format>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_pp_send(void* x, const int64_t& num_x_bytes,
                           void* buffer, void* workspace, void* device_context,
                           const int& rank_idx, const int& dst_rank_idx, const int& num_ranks,
                           const int64_t& num_max_tensor_bytes,
                           const int& num_max_inflight_tensors,
                           const int& num_blocks,
                           const int64_t& num_timeout_cycles) {
    // Compile
    const auto kernel = jit->compile("pp_send", std::format(R"(
#define EP_PP_SEND
#include <deep_ep/impls/pp/pp_send_recv.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pp_send_impl<{}>);
}}
)", num_timeout_cycles));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_blocks},
        x, num_x_bytes,
        buffer, workspace, device_context,
        rank_idx, dst_rank_idx, num_ranks,
        num_max_tensor_bytes, num_max_inflight_tensors);
}

static void launch_pp_recv(void* x, const int64_t& num_x_bytes,
                           void* buffer, void* workspace, void* device_context,
                           const int& rank_idx, const int& src_rank_idx, const int& num_ranks,
                           const int64_t& num_max_tensor_bytes,
                           const int& num_max_inflight_tensors,
                           const int& num_blocks,
                           const int64_t& num_timeout_cycles) {
    // Compile
    const auto kernel = jit->compile("pp_recv", std::format(R"(
#define EP_PP_RECV
#include <deep_ep/impls/pp/pp_send_recv.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pp_recv_impl<{}>);
}}
)", num_timeout_cycles));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_blocks},
        x, num_x_bytes,
        buffer, workspace, device_context,
        rank_idx, src_rank_idx, num_ranks,
        num_max_tensor_bytes, num_max_inflight_tensors);
}

}  // namespace deep_ep
