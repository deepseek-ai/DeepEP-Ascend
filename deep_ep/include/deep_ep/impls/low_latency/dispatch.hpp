// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#pragma once
#include <cstdint>
#include <kernel_operator.h>

#include <deep_ep/impls/low_latency/dispatch/low_latency/dispatch_server_dedup_kernel.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/dispatch_dedup_types.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h>

// One production entry and one 64-core process with a compile-time send/recv split.
template <int = 0>
__global__ __vector__ void dispatch_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
    GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, GM_ADDR tilingData,
    int64_t reservedMagic, GM_ADDR inputScales)
{
    AscendC::InitSocState();
    // Host owns validation; each core reads its configuration once before work.
    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tilingData);
    const uint32_t bs = tiling->bs;
    const uint32_t h = tiling->h;
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(tiling->numMaxTokensPerRank);
    const uint32_t topK = tiling->topK;
    const uint32_t moeExpertNum = tiling->moeExpertNum;
    const uint32_t expertTokenNumsType = tiling->expertTokenNumsType;
    const uint8_t quantMode = tiling->quantMode;
    const uint8_t dtype = tiling->dtype;
    const uint8_t expandXOutDtype = tiling->expandXOutDtype;
    const auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(tiling->inputMode);
    AscendC::TPipe pipe;
#define DISPATCH_SERVER_DEDUP_RUN(XType, WireFp8, E4M3, Prequantized)                                                  \
    DispatchDedup::DispatchMemoryServerDedupKernel<XType, WireFp8, E4M3, Prequantized> kernel;                         \
    kernel.Init(commArgs, x, expertIds, expertScales, expandXOut, dynamicScalesOut, expertTokenNumsOut, sendCountsOut, \
                tokenTypeOut, destinationIndexOut, relayReadIndexOut, sourceMaskOut, expertScalesOut, bs, h, topK,     \
                moeExpertNum, expertTokenNumsType, &pipe, inputScales, enableDedup);                                   \
    kernel.Process()
    if (inputMode == DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs && quantMode == 4 &&
        dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 &&
        expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3) {
        DISPATCH_SERVER_DEDUP_RUN(uint8_t, true, true, true);
    } else if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16) {
        if (quantMode == 0) {
            DISPATCH_SERVER_DEDUP_RUN(half, false, false, false);
        } else if (expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3) {
            DISPATCH_SERVER_DEDUP_RUN(half, true, true, false);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(half, true, false, false);
        }
    } else if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16) {
        if (quantMode == 0) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, false, false, false);
        } else if (expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, true, true, false);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, true, false, false);
        }
    }
#undef DISPATCH_SERVER_DEDUP_RUN
}
