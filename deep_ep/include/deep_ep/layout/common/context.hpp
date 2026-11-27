#pragma once

#include <cstdint>

#include <deep_ep/layout/barrier.hpp>
#include <deep_ep/layout/common/error.hpp>

namespace deep_ep::layout {

// Jetty tables store non-self peers first, in rank order, and are also indexed by AIV to select a send queue.
struct DeviceContext {
    uint64_t scaleup_jetties[kNumMaxScaleupRanks];
    uint64_t scaleup_peer_ptrs[kNumMaxScaleupRanks];
    uint64_t scaleout_jetties[kNumMaxScaleoutRanks];
    uint64_t scaleout_peer_ptrs[kNumMaxScaleoutRanks];
    uint64_t error_status_addr;

    static EP_INLINE __gm__ ErrorStatusSlot* get_error_status(const __gm__ DeviceContext& device_context) {
        return reinterpret_cast<__gm__ ErrorStatusSlot*>(device_context.error_status_addr);
    }
};

template <int kNumScaleupRanks>
constexpr EP_INLINE int get_scaleup_domain_start_rank(const int rank_idx) {
    static_assert(kNumScaleupRanks > 0);
    return rank_idx / kNumScaleupRanks * kNumScaleupRanks;
}

template <typename dtype_t>
constexpr __aicore__ __forceinline__ __gm__ dtype_t* get_sym_ptr(
    const __gm__ BarrierSignals& signals, const __gm__ DeviceContext& device_context,
    __gm__ dtype_t* local_ptr, const int dst_scaleup_rank_idx) {
    const auto offset = reinterpret_cast<int64_t>(local_ptr) - reinterpret_cast<int64_t>(&signals);
    return reinterpret_cast<__gm__ dtype_t*>(device_context.scaleup_peer_ptrs[dst_scaleup_rank_idx] + offset);
}

}  // namespace deep_ep::layout
