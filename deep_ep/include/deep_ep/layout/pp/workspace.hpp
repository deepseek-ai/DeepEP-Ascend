#pragma once

#include <deep_ep/layout/barrier.hpp>

namespace deep_ep::layout {

// Pipeline-parallel send/recv state, indexed by the ring direction.
struct alignas(kNumAllocationAlignmentBytes) PPSignals {
    BarrierSignals barrier_signals;
    int64_t send_count[2];
    int64_t recv_count[2];
    int64_t arrival[2];
    int64_t release[2];
};

}  // namespace deep_ep::layout
