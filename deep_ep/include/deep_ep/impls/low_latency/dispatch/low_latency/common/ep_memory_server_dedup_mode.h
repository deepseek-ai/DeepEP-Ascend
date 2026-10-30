// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_MODE_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_MODE_H

#include <cstdint>

namespace DispatchDedup {
// The dedup resource layout is fixed for the lifetime of the binary. The
// execution path is chosen at runtime from the per-rank capacity instead.
constexpr bool kEpServerDedupEnable = true;
constexpr uint32_t kEpServerDedupDedupThreshold = 32U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#endif
    constexpr bool
    EpServerDedupEnabled(uint32_t numMaxTokensPerRank)
{
    return numMaxTokensPerRank > kEpServerDedupDedupThreshold;
}
}  // namespace DispatchDedup
#endif
