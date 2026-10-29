// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_SERVER_DEDUP_MAGIC_H
#define DISPATCH_SERVER_DEDUP_MAGIC_H

#include <kernel_operator.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h>

namespace DispatchDedup {
// All 64 AIVs call exactly once per launch, including cores with no count work.
// The initial Host clear seeds 0; launches use 1, 0, 1, 0, ... without overflow.
// Each core owns a 512-byte slot in the LOCAL symmetric allocation's fixed
// prefix. No other core/rank writes it. Both accesses use DataCopy.
__aicore__ inline uint32_t RotateDispatchServerDedupMagic(const __gm__ CommArgs* args, uint32_t core,
                                                          AscendC::TPipe* pipe)
{
    using namespace AscendC;
    TBuf<> scratch;
    pipe->InitBuffer(scratch, 32U);
    auto state = scratch.Get<uint32_t>();
    auto* slot =
        reinterpret_cast<__gm__ uint32_t*>(args->peerMems[args->rank] + uint64_t(core) * kEpServerDedupMagicStride);
    GlobalTensor<uint32_t> slotGm;
    slotGm.SetGlobalBuffer(slot);
    DataCopy(state, slotGm, 8U);
    const auto readEvent = pipe->FetchEventID(HardEvent::MTE2_S);
    SetFlag<HardEvent::MTE2_S>(readEvent);
    WaitFlag<HardEvent::MTE2_S>(readEvent);
    const uint32_t magic = state.GetValue(0) == 0U ? 1U : 0U;
    state.SetValue(0, magic);
    const auto scalarEvent = pipe->FetchEventID(HardEvent::S_MTE3);
    SetFlag<HardEvent::S_MTE3>(scalarEvent);
    WaitFlag<HardEvent::S_MTE3>(scalarEvent);
    DataCopy(slotGm, state, 8U);
    const auto writeEvent = pipe->FetchEventID(HardEvent::MTE3_S);
    SetFlag<HardEvent::MTE3_S>(writeEvent);
    WaitFlag<HardEvent::MTE3_S>(writeEvent);
    // Retire the write before releasing scratch. Counts gets its full UB budget.
    pipe->Reset();
    return magic;
}
}  // namespace DispatchDedup
#endif
