// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#pragma once
#include <cstdint>
#include <type_traits>
namespace deepep {
struct EpServerDedupCombineKernelArgs {
    uint8_t *commArgs, *expertOut, *destinationIndex, *sendCounts;
    uint8_t *tokenType, *relayReadIndex, *sourceMask, *yOut, *tilingData;
};
static_assert(std::is_standard_layout<EpServerDedupCombineKernelArgs>::value, "combine launch ABI");
static_assert(sizeof(EpServerDedupCombineKernelArgs) == 72, "combine launch ABI");
}  // namespace deepep
