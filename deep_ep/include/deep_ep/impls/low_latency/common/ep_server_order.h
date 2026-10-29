// Copyright (c) 2026, Lu Lu
// Modified by liuboxi, 2026

#ifndef DEEPEP_EP_SERVER_ORDER_H
#define DEEPEP_EP_SERVER_ORDER_H

#include <cstdint>

namespace EpServerOrder {
// Preconditions: servers > 0, ownServer < servers, ordinal < servers.
// Rotate each half from the next relative position. At 16+ servers, both
// dedup and direct paths alternate own/other half; smaller worlds retain
// their original ordering.
// This mapping is independent of ownership and pipeline cursors.
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__
#endif
inline uint32_t ServerDedupHalfOrder(uint32_t servers, uint32_t ownServer,
                                                uint32_t ordinal)
{
    const uint32_t split = servers / 2U;
    const bool lower = ownServer < split;
    const uint32_t ownBase = lower ? 0U : split;
    const uint32_t ownLength = lower ? split : servers - split;
    const uint32_t relative = ownServer - ownBase;
    if (servers >= 16U) {  // Eight ranks per server: 128P and above.
        // Map alternating visits back to the contiguous half-order index.
        // For odd worlds, visit the larger half's unmatched last entry at the end.
        if (ordinal < 2U * split)
            ordinal = ordinal / 2U + (ordinal % 2U == 0U ? 0U : ownLength);
        else
            ordinal = ownLength > split ? split : servers - 1U;
    }
    if (ordinal < ownLength) return ownBase + (relative + ordinal + 1U) % ownLength;
    const uint32_t otherBase = lower ? split : 0U;
    const uint32_t otherLength = servers - ownLength;
    return otherBase + (relative + ordinal - ownLength + 1U) % otherLength;
}

// Dispatch direct packets interleave halves at 128 ranks or more.
// Smaller Dispatch worlds retain their own-server-first full-ring traversal.
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__
#endif
inline uint32_t ServerDedupDispatchDirectOrder(uint32_t servers, uint32_t ownServer, uint32_t ordinal)
{
    return servers >= 16U ? ServerDedupHalfOrder(servers, ownServer, ordinal) :
                           (ownServer + ordinal) % servers;
}
}  // namespace EpServerOrder
#endif
