#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>

#ifdef __HOST_ONLY__
#include <acl/acl.h>
#include <deep_ep/common/exception.hpp>
#endif

namespace deep_ep::layout {

// Align slots for better scale-up barrier performance.
struct alignas(kNumAicpuCacheLineBytes) ScaleupBarrierSlot {
    int64_t value;
};

// Align slots for better scale-out barrier performance.
struct alignas(kNumUrmaAlignmentBytes) ScaleoutBarrierSlot {
    int64_t value;
};

struct BarrierSignals {
    // Barriers
    alignas(kNumSectorBytes) uint32_t grid_sync_count;
    alignas(kNumSectorBytes) int64_t scaleup_barrier_counter;
    // For scale-up performance, AICore uses SIMT atomic add and AICPU uses non-atomic write.
    alignas(kNumAicpuCacheLineBytes) int scaleup_barrier_signals[2];
    ScaleupBarrierSlot scaleup_barrier_slots[2][kNumMaxScaleupRanks];
    alignas(kNumSectorBytes) int64_t scaleout_barrier_counter;
    ScaleoutBarrierSlot scaleout_barrier_slots[2][kNumMaxScaleoutRanks];

#ifdef __HOST_ONLY__
    void reset() {
        ACL_CHECK(aclrtMemset(&grid_sync_count, sizeof(grid_sync_count), 0, sizeof(grid_sync_count)));
        ACL_CHECK(aclrtMemset(&scaleup_barrier_counter, sizeof(scaleup_barrier_counter), 0, sizeof(scaleup_barrier_counter)));
        ACL_CHECK(aclrtMemset(scaleup_barrier_signals, sizeof(scaleup_barrier_signals), 0, sizeof(scaleup_barrier_signals)));
        ACL_CHECK(aclrtMemset(scaleup_barrier_slots, sizeof(scaleup_barrier_slots), 0, sizeof(scaleup_barrier_slots)));
        ACL_CHECK(aclrtMemset(&scaleout_barrier_counter, sizeof(scaleout_barrier_counter), 0, sizeof(scaleout_barrier_counter)));
        ACL_CHECK(aclrtMemset(scaleout_barrier_slots, sizeof(scaleout_barrier_slots), 0, sizeof(scaleout_barrier_slots)));
    }
#endif
};

}  // namespace deep_ep::layout
