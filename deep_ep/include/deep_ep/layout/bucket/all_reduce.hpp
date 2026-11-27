#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>

namespace deep_ep::layout {

struct alignas(kNumAicpuCacheLineBytes) AllReduceSignals {
    static constexpr int kNumMaxStreams = 2;
    // RS and AG each need ceil(log2(SU)) + ceil(log2(SO)) phases.
    static constexpr int kNumMaxPhases = 2 * (64 - __builtin_clz(unsigned(kNumMaxScaleupRanks - 1)) -
                                              __builtin_clz(unsigned(kNumMaxScaleoutRanks - 1)));
    // Input-ready chunk count, published by its read peer.
    uint64_t ready[kNumMaxStreams][kNumMaxPhases];
};

}  // namespace deep_ep::layout
