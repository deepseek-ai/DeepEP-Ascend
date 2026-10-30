#pragma once

#include <cstdlib>
#include <cstring>

namespace deep_ep {

// Retain the native small-batch MTE policy only for EP128 after transport
// comparisons. Other EP sizes use the original URMA transport. EP128 keeps
// its existing model, hardware and capacity guards and large-batch fallback.
// The wire layout, allocation, routing metadata and collective sequence remain
// those of EPBuffer. Peers may consequently use different transports when their
// actual token counts differ, without introducing another agreement collective.
constexpr bool prefer_low_latency_transport(int arch, int num_ranks, int num_tokens,
                                            int num_max_tokens_per_rank, int hidden,
                                            int num_topk, int num_experts, int num_vec_cores,
                                            bool use_fp8, int num_ub_bytes = 190 * 1024) {
    if (arch != 3510 or use_fp8 or hidden != 7168 or num_topk != 8 or num_experts != 128 or
        num_vec_cores != 64 or num_ub_bytes < 190 * 1024 or num_tokens <= 0 or
        num_tokens > num_max_tokens_per_rank or num_max_tokens_per_rank > 256)
        return false;
    return num_ranks == 128 and num_tokens <= 32;
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
