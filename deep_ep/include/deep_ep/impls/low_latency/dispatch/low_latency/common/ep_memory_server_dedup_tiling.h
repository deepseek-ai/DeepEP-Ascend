// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_TILING_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_TILING_H

#include <cstddef>
#include <cstdint>

namespace DispatchDedup {
// Shared Host/AICore wire layout. Host validates before narrowing.
// Magic is persistent device state, not tiling or a Host launch value.
struct EpServerDedupTilingData {
    uint32_t bs;
    uint32_t h;
    uint32_t numMaxTokensPerRank;
    uint16_t moeExpertNum;
    uint8_t topK;
    uint8_t reservedSendCoreNum;
    uint8_t expertTokenNumsType;
    uint8_t quantMode;
    uint8_t dtype;
    uint8_t expandXOutDtype;
    uint8_t inputMode;  // EpServerDedupInputMode; P is derived from H.
    uint8_t reserved[11];
};
static_assert(sizeof(EpServerDedupTilingData) == 32U, "tiling occupies one 32B block");
// Keep the wire structure naturally aligned. Kernel reads use scalar GM
// accesses, not an implicit vector/MTE load of the entire block.
static_assert(alignof(EpServerDedupTilingData) == 4U, "tiling uses natural scalar alignment");
static_assert(offsetof(EpServerDedupTilingData, numMaxTokensPerRank) == 8U, "per-rank capacity offset");
static_assert(offsetof(EpServerDedupTilingData, moeExpertNum) == 12U, "expert count offset");
static_assert(offsetof(EpServerDedupTilingData, topK) == 14U, "topK offset");
static_assert(offsetof(EpServerDedupTilingData, reservedSendCoreNum) == 15U, "reserved offset");
static_assert(offsetof(EpServerDedupTilingData, inputMode) == 20U, "input mode offset");

}  // namespace DispatchDedup

#endif
