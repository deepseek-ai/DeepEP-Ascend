// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef ASCEND_EP_SERVER_DEDUP_INPUT_H
#define ASCEND_EP_SERVER_DEDUP_INPUT_H

#include <cstdint>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/dispatch_dedup_types.h>

namespace DispatchDedup {
// Wire FP8 does not imply internal quantization. Values also form the tiling ABI.
enum class EpServerDedupInputMode : uint8_t {
    Plain16 = 0,
    Quantize16 = 1,
    PrequantizedFp8Packs = 2,
};

// Host-only classification; no implicit conversion or dtype-based quantization.
inline bool DispatchDedupResolveInputMode(int64_t quantMode, int64_t dtype, bool hasInputScales, int64_t outputDtype,
                                          EpServerDedupInputMode* mode)
{
    if (mode == nullptr) return false;
    const bool input16 =
        dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16 || dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (quantMode == 0 && input16 && !hasInputScales &&
        (outputDtype == dtype || outputDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_RESERVED)) {
        *mode = EpServerDedupInputMode::Plain16;
        return true;
    }
    if (quantMode != 4) return false;
    if (input16 && !hasInputScales &&
        (outputDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 ||
         outputDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E5M2)) {
        *mode = EpServerDedupInputMode::Quantize16;
        return true;
    }
    if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 && hasInputScales && outputDtype == dtype) {
        *mode = EpServerDedupInputMode::PrequantizedFp8Packs;
        return true;
    }
    return false;
}
}  // namespace DispatchDedup
#endif
