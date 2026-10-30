// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_PROTOCOL_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_PROTOCOL_H

#include <cstddef>
#include <cstdint>

namespace DispatchDedup {
constexpr uint32_t kEpServerDedupRanksPerServer = 8U;
constexpr uint32_t kEpServerDedupAlignmentBytes = 32U;
constexpr uint32_t kEpServerDedupMaxTopK = 32U;

// The wire header and TokenType output use the same values.
enum class EpServerDedupTokenType : uint32_t { Direct = 1U, FirstHop = 2U, Terminal = 3U };

struct alignas(32) EpServerDedupPacketHeader {
    uint32_t srcRank;
    uint32_t srcTokenId;
    uint32_t topK;  // Original route index / returnK, not the shape K.
    uint32_t type;
    uint32_t padding[4];
};

// Block 0: index=N, weightBits=selfWeight. Member: index=expert|(q<<16).
// A RelayReadIndex row contains exactly K blocks; only blocks 0..N are valid.
struct EpServerDedupReadIndexBlock {
    uint32_t index;
    uint32_t weightBits;
};

struct EpServerDedupDirectTask {
    uint32_t tokenOffset;
    uint32_t originalK;
    uint32_t expertId;
    uint32_t q;
};
static_assert(sizeof(EpServerDedupTokenType) == 4U, "token type ABI");
static_assert(sizeof(EpServerDedupPacketHeader) == 32U, "packet header ABI");
static_assert(alignof(EpServerDedupPacketHeader) == 32U, "packet header alignment");
static_assert(offsetof(EpServerDedupPacketHeader, srcTokenId) == 4U, "source token offset");
static_assert(offsetof(EpServerDedupPacketHeader, topK) == 8U, "return k offset");
static_assert(offsetof(EpServerDedupPacketHeader, type) == 12U, "packet role offset");
static_assert(sizeof(EpServerDedupReadIndexBlock) == 8U, "read index block ABI");
static_assert(offsetof(EpServerDedupReadIndexBlock, weightBits) == 4U, "weight bits offset");
static_assert(sizeof(EpServerDedupDirectTask) == 16U, "source direct task ABI");
}  // namespace DispatchDedup
#endif
