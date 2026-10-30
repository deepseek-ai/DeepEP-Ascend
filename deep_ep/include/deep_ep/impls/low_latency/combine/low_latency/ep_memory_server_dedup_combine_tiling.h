// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_TILING_H
#define COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_TILING_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace CombineDedup {
constexpr uint32_t kEpServerDedupCombineCoreNum = 64;

// Independent Host/AICore wire contract. Host validates before narrowing.
struct EpServerDedupCombineTilingData {
    uint32_t bs;
    uint32_t h;
    uint16_t moeExpertNum;
    uint8_t topK;
    uint8_t dtype;
    uint16_t numMaxTokensPerRank;
    uint8_t quantMode;
    uint8_t reserved;
    uint32_t recvUbBytes;
};
static_assert(std::is_standard_layout<EpServerDedupCombineTilingData>::value, "standard layout tiling");
static_assert(sizeof(EpServerDedupCombineTilingData) == 20, "Combine tiling is 20 bytes");
static_assert(alignof(EpServerDedupCombineTilingData) == 4, "scalar GM alignment");
static_assert(offsetof(EpServerDedupCombineTilingData, bs) == 0, "BS offset");
static_assert(offsetof(EpServerDedupCombineTilingData, h) == 4, "H offset");
static_assert(offsetof(EpServerDedupCombineTilingData, moeExpertNum) == 8, "E offset");
static_assert(offsetof(EpServerDedupCombineTilingData, topK) == 10, "K offset");
static_assert(offsetof(EpServerDedupCombineTilingData, dtype) == 11, "dtype offset");
static_assert(offsetof(EpServerDedupCombineTilingData, numMaxTokensPerRank) == 12, "per-rank capacity offset");
static_assert(offsetof(EpServerDedupCombineTilingData, quantMode) == 14, "quantMode offset");
static_assert(offsetof(EpServerDedupCombineTilingData, reserved) == 15, "reserved zero offset");
static_assert(offsetof(EpServerDedupCombineTilingData, recvUbBytes) == 16, "receive UB budget offset");
}  // namespace CombineDedup
#endif
