// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef COMBINE_DEDUP_EP_SERVER_DEDUP_SELECT_H
#define COMBINE_DEDUP_EP_SERVER_DEDUP_SELECT_H

#include <kernel_operator.h>
#include <deep_ep/impls/low_latency/common/ep_server_order.h>

namespace CombineDedup {
// Dispatch and Combine share the same large-world half ordering.
using EpServerOrder::ServerDedupHalfOrder;

// Table2 carries source ranks. Two comparisons select one eight-rank server
// without scalar division or a per-candidate GetValue loop.
__aicore__ inline void ServerDedupRankMask(AscendC::LocalTensor<uint8_t> mask, AscendC::LocalTensor<uint8_t> upper,
                                           AscendC::LocalTensor<int32_t> ranks, int32_t first, int32_t end,
                                           uint32_t count)
{
    using namespace AscendC;
    // CANN's public debug contract requires 256B of comparison input. All
    // rank arrays reserve that padding; GatherMask still uses the true count.
    const uint32_t compareCount = (count + 63U) / 64U * 64U;
    Compares(mask, ranks, first, CMPMODE::GE, compareCount);
    Compares(upper, ranks, end, CMPMODE::LT, compareCount);
    PipeBarrier<PIPE_V>();
    And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), upper.ReinterpretCast<uint16_t>(),
        int32_t((count + 15U) / 16U));
    PipeBarrier<PIPE_V>();
}
// The compact stream is striped across cores, continuing from prior servers
// (and prior vector tiles). No scalar traversal of rejected candidates.
__aicore__ inline uint32_t ServerDedupFirstOwned(uint32_t core, uint32_t owner, uint32_t cores)
{
    return (core + cores - owner) % cores;
}
}  // namespace CombineDedup
#endif
