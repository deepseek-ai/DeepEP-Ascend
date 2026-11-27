#pragma once

#include <deep_ep/layout/barrier.hpp>
#include <deep_ep/layout/bucket/all_gather.hpp>
#include <deep_ep/layout/bucket/all_reduce.hpp>

namespace deep_ep::layout {

struct BucketSignals {
    BarrierSignals barrier_signals;
    AllReduceSignals all_reduce_signals;
    AllGatherSignals all_gather_signals;
};

struct alignas(kNumAllocationAlignmentBytes) BucketWorkspace {
    BucketSignals signals[kNumMaxContexts];
};

}  // namespace deep_ep::layout
