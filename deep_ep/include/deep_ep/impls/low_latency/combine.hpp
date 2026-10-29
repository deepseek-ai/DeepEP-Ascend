// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#pragma once
#include <cstdint>
#include <kernel_operator.h>

#include <deep_ep/impls/low_latency/combine/low_latency/combine_server_dedup_kernel.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_tiling.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_magic.h>
#include <deep_ep/impls/low_latency/combine/low_latency/launch.hpp>

template <int = 0>
__global__ __vector__ void combine_server_dedup_kernel(
    GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR destinationIndex, GM_ADDR sendCounts,
    GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask, GM_ADDR yOut, GM_ADDR tilingData)
{
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
    AscendC::InitSocState();
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, AscendC::GetBlockIdx(), &pipe);
    }
    const auto* td = reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(tilingData);
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(td->numMaxTokensPerRank);
    CombineDedup::CombineMemoryServerDedupKernel kernel;
    kernel.Init(commArgs, expertOut, destinationIndex, sendCounts, tokenType, relayReadIndex, sourceMask,
                yOut, td->bs, td->h, td->topK, td->moeExpertNum, td->dtype, 0, magic, td->recvUbBytes, enableDedup);
    kernel.Process();
}
