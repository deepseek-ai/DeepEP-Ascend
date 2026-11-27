#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>

namespace deep_ep::layout {

struct alignas(kNumAicpuCacheLineBytes) AllGatherSignals {
    // Per source domain: how many bytes of that domain's shard have landed in this rank's output buffer.
    int64_t landed_bytes[kNumMaxScaleoutRanks];
};

}  // namespace deep_ep::layout

namespace deep_ep {

// Progress exposed by one exchange write and one publication. It has to stay well below a shard, otherwise a
// large shard is a single write and its forward cannot start before the whole shard has landed.
static constexpr int64_t kNumAllGatherProgressBytes = 32 << 20;

}  // namespace deep_ep
