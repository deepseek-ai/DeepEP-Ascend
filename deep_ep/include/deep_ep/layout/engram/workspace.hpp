#pragma once

#include <deep_ep/layout/barrier.hpp>

namespace deep_ep::layout {

struct alignas(kNumAllocationAlignmentBytes) EngramWorkspace {
    BarrierSignals barrier_signals;
};

// Each layer has a packed storage shard and a fixed-size receive region.
template <int kHiddenBytes, int kEntriesPerToken, int... kEntriesPack>
struct EngramLayout {
    static constexpr int kNumLayers = sizeof...(kEntriesPack);

    static constexpr __aicore__ __forceinline__ int get_num_entries(const int& layer_idx) {
        constexpr int kNumEntriesPerLayer[kNumLayers] = {kEntriesPack...};
        return kNumEntriesPerLayer[layer_idx];
    }

    static constexpr __aicore__ __forceinline__ int64_t get_num_prefix_entries(const int& layer_idx) {
        int64_t num_prefix_entries = 0;
        for (int i = 0; i < layer_idx; ++ i)
            num_prefix_entries += get_num_entries(i);
        return num_prefix_entries;
    }

    static constexpr __aicore__ __forceinline__ int64_t get_storage_byte_offset(
        const int& layer_idx, const int& entry_idx) {
        return (get_num_prefix_entries(layer_idx) + entry_idx) * kHiddenBytes;
    }

    static constexpr __aicore__ __forceinline__ int64_t get_recv_byte_offset(
        const int& layer_idx, const int& entry_idx, const int& num_max_tokens) {
        return (static_cast<int64_t>(layer_idx) * kEntriesPerToken * num_max_tokens + entry_idx) * kHiddenBytes;
    }
};

}  // namespace deep_ep::layout
