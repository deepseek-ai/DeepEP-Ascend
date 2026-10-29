// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace deepep {
struct EpServerDedupKernelArgs {
    uint8_t* commArgs;  // Persistent device configuration owned by BufferRuntime.
    uint8_t* x;
    uint8_t* expertIds;
    uint8_t* expertScales;
    uint8_t* expandXOut;
    uint8_t* dynamicScalesOut;
    uint8_t* expertTokenNumsOut;
    uint8_t* sendCountsOut;
    uint8_t* tokenTypeOut;
    uint8_t* destinationIndexOut;
    uint8_t* relayReadIndexOut;
    uint8_t* sourceMaskOut;
    uint8_t* expertScalesOut;
    uint8_t* tilingData;    // Per-launch device tiling, read from GM by the kernel.
    int64_t reservedMagic;  // Must be zero; device owns the persistent bank selector.
    uint8_t* inputScales;   // ABI: complete INT32 packed E8M0 input rows.
};
static_assert(std::is_standard_layout<EpServerDedupKernelArgs>::value, "launch arguments must be standard layout");
static_assert(sizeof(EpServerDedupKernelArgs) == 128, "launch ABI has fifteen addresses and a reserved slot");
static_assert(offsetof(EpServerDedupKernelArgs, dynamicScalesOut) == 40, "quant scale address");
static_assert(offsetof(EpServerDedupKernelArgs, tokenTypeOut) == 64, "token type address");
static_assert(offsetof(EpServerDedupKernelArgs, destinationIndexOut) == 72, "destination index address");
static_assert(offsetof(EpServerDedupKernelArgs, relayReadIndexOut) == 80, "relay read index address");
static_assert(offsetof(EpServerDedupKernelArgs, sourceMaskOut) == 88, "source mask address");
static_assert(offsetof(EpServerDedupKernelArgs, expertScalesOut) == 96, "source weight address");
static_assert(offsetof(EpServerDedupKernelArgs, tilingData) == 104, "tiling GM address");
static_assert(offsetof(EpServerDedupKernelArgs, reservedMagic) == 112, "reserved former magic offset");

static_assert(offsetof(EpServerDedupKernelArgs, inputScales) == 120, "input packed scale address");

}  // namespace deepep
