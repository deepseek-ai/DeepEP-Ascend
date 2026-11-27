#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include <deep_jit/utils/str.hpp>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_engram_fetch(void* storage, void* fetched,
                                int* indices, void* workspace,
                                void* device_context, void* sf, void* fetched_sf,
                                const int64_t& sf_token_stride, const int64_t& sf_hidden_stride,
                                const int64_t& sf_layer_stride,
                                const std::vector<int>& num_entries_per_layer,
                                const int& rank_idx, const int& num_scaleup_ranks, const int& num_hidden_bytes,
                                const int& num_sf_packs, const int& num_entries_per_token,
                                const int& num_tokens, const int& num_max_tokens,
                                const int& num_ub_bytes, const int& num_vec_cores) {
    EP_HOST_ASSERT(num_ub_bytes > 0 and num_vec_cores > 0);

    const auto num_stage_bytes = math::align(num_hidden_bytes, kNumUbAlignmentBytes) +
                                 math::align<int>(num_sf_packs * sizeof(sf_pack_t), kNumUbAlignmentBytes);
    const auto num_mte_stages = std::min(8, num_ub_bytes / num_stage_bytes);
    EP_HOST_ASSERT(num_mte_stages > 0);

    std::vector<std::string> num_entries_per_layer_strings;
    num_entries_per_layer_strings.reserve(num_entries_per_layer.size());
    for (const auto& num_entries: num_entries_per_layer)
        num_entries_per_layer_strings.emplace_back(std::to_string(num_entries));

    // Compile
    const auto kernel = jit->compile("engram_fetch", std::format(R"(
#include <deep_ep/impls/engram/engram_fetch.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&engram_fetch_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_scaleup_ranks, num_hidden_bytes, num_sf_packs,
        num_entries_per_token, num_max_tokens, num_vec_cores,
        num_mte_stages, deep_jit::str::join(num_entries_per_layer_strings, ", ")));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        storage, fetched, indices, workspace,
        device_context,
        static_cast<sf_pack_t*>(sf), static_cast<sf_pack_t*>(fetched_sf),
        sf_token_stride, sf_hidden_stride, sf_layer_stride,
        num_tokens, rank_idx);
}

}  // namespace deep_ep
