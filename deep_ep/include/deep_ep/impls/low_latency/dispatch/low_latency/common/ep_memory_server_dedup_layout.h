// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_LAYOUT_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h>

namespace DispatchDedup {
struct EpMemoryServerDedupLayout {
    uint64_t globalBs = 0;
    uint64_t localExpertNum = 0;
    uint64_t capacityRows = 0;
    EpServerDedupInputMode inputMode = EpServerDedupInputMode::Plain16;
    uint64_t xBytes = 0;
    uint64_t inputScalesBytes = 0;
    uint64_t expertIdsBytes = 0;
    uint64_t weightBytes = 0;
    uint64_t expandXBytes = 0;
    uint64_t dynamicScalesBytes = 0;
    uint64_t expertTokenNumsBytes = 0;
    uint64_t sendCountsBytes = 0;
    uint64_t countStateBytes = 0;
    uint64_t tokenTypeBytes = 0;
    uint64_t destinationIndexBytes = 0;
    uint64_t relayReadIndexRowStrideBytes = 0;
    uint64_t relayReadIndexBytes = 0;
    uint64_t sourceMaskBytes = 0;
    EpServerDedupDataLayout data{};
};

// Caller-side allocation records only. The public bare-pointer API cannot infer these sizes.
struct EpServerDedupAllocation {
    const void* data = nullptr;
    uint64_t bytes = 0;
    uint64_t requiredBytes = 0;
    uint32_t alignment = 1;
    int32_t deviceId = -1;
};

// Host-only layout validation; the device entry uses the compile-time core split.
int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                   EpMemoryServerDedupLayout* out, bool quantized = false);
int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                   EpMemoryServerDedupLayout* out, EpServerDedupInputMode mode);
int ValidateDispatchServerDedupAllocations(const EpServerDedupAllocation* allocations, std::size_t count,
                                           int32_t deviceId);

}  // namespace DispatchDedup

#include <algorithm>
#include <limits>

#include <deep_ep/impls/low_latency/dispatch/low_latency/common/comm_args.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/dispatch_dedup_types.h>

namespace DispatchDedup {
namespace {
bool MultiplyBytes(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    const uint64_t limit = std::numeric_limits<std::size_t>::max();
    if (lhs != 0 && rhs > limit / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

}  // namespace

inline int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                          EpMemoryServerDedupLayout* out, EpServerDedupInputMode mode)
{
    if (out == nullptr) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    *out = EpMemoryServerDedupLayout{};
    if ((mode != EpServerDedupInputMode::Plain16 && mode != EpServerDedupInputMode::Quantize16 &&
         mode != EpServerDedupInputMode::PrequantizedFp8Packs) ||
        rankSize < 8 || rankSize > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE ||
        rankSize % kEpServerDedupRanksPerServer != 0 || bs <= 0 || h <= 0 || h > UINT32_MAX || topK <= 0 ||
        topK > kEpServerDedupMaxTopK || moeExpertNum <= 0 || moeExpertNum > kEpServerDedupMaxExperts ||
        topK > moeExpertNum || moeExpertNum % rankSize != 0 || bs > UINT32_MAX / topK) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }

    const uint32_t countCores =
        bs < kEpServerDedupAllCoreCountBsThreshold ? kEpServerDedupCoreNum : kEpServerDedupRecvCoreNum;
    if (DispatchServerDedupCountSendUbBytes(uint64_t(bs) * topK, uint32_t(moeExpertNum), countCores) >
        kEpServerDedupCountUbBytes) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }

    const bool wireFp8 = mode != EpServerDedupInputMode::Plain16;
    const bool prequantized = mode == EpServerDedupInputMode::PrequantizedFp8Packs;
    EpMemoryServerDedupLayout next{};
    next.inputMode = mode;
    next.globalBs = static_cast<uint64_t>(rankSize * bs);
    next.localExpertNum = static_cast<uint64_t>(moeExpertNum / rankSize);
    next.countStateBytes = DispatchServerDedupCountLayout(uint32_t(moeExpertNum)).stateBytes;
    next.data = DispatchServerDedupDataLayout(uint32_t(bs), uint32_t(h), uint32_t(topK), uint32_t(moeExpertNum), mode);
    if (next.countStateBytes > kEpServerDedupControlBytes ||
        next.data.requiredBytes > kEpServerDedupWindowUsableBytes) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    next.relayReadIndexRowStrideBytes = static_cast<uint64_t>(topK) * 8U;
    uint64_t rowBytes = 0;
    uint64_t routeCount = 0;
    if (!MultiplyBytes(next.globalBs, std::min<uint64_t>(topK, next.localExpertNum), &next.capacityRows) ||
        !MultiplyBytes(h, prequantized ? 1U : 2U, &rowBytes) || !MultiplyBytes(bs, rowBytes, &next.xBytes) ||
        !MultiplyBytes(bs, prequantized ? next.data.scaleBytes : 0U, &next.inputScalesBytes) ||
        !MultiplyBytes(bs, topK, &routeCount) || !MultiplyBytes(routeCount, sizeof(int32_t), &next.expertIdsBytes) ||
        !MultiplyBytes(routeCount, sizeof(float), &next.weightBytes) ||
        !MultiplyBytes(next.capacityRows, wireFp8 ? uint64_t(h) : rowBytes, &next.expandXBytes) ||
        !MultiplyBytes(next.capacityRows, next.data.scaleBytes, &next.dynamicScalesBytes) ||
        !MultiplyBytes(next.localExpertNum, sizeof(int64_t), &next.expertTokenNumsBytes) ||
        !MultiplyBytes(moeExpertNum, sizeof(int32_t), &next.sendCountsBytes) ||
        !MultiplyBytes(next.capacityRows, sizeof(uint32_t), &next.tokenTypeBytes) ||
        !MultiplyBytes(next.capacityRows, 3U * sizeof(uint32_t), &next.destinationIndexBytes) ||
        !MultiplyBytes(bs, sizeof(uint32_t), &next.sourceMaskBytes) ||
        // Sparse first-hop indices use the full Dispatch output-row capacity.
        !MultiplyBytes(next.capacityRows, next.relayReadIndexRowStrideBytes, &next.relayReadIndexBytes)) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    *out = next;
    return DispatchDedup::DISPATCH_DEDUP_SUCCESS;
}

inline int BuildDispatchServerDedupLayout(int64_t rankSize, int64_t bs, int64_t h, int64_t topK, int64_t moeExpertNum,
                                          EpMemoryServerDedupLayout* out, bool quantized)
{
    return BuildDispatchServerDedupLayout(
        rankSize, bs, h, topK, moeExpertNum, out,
        quantized ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);
}

inline int ValidateDispatchServerDedupAllocations(const EpServerDedupAllocation* allocations, std::size_t count,
                                                  int32_t deviceId)
{
    if (allocations == nullptr || count == 0 || deviceId < 0) {
        return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& allocation = allocations[i];
        const uintptr_t begin = reinterpret_cast<uintptr_t>(allocation.data);
        if (allocation.data == nullptr || allocation.requiredBytes == 0 ||
            allocation.bytes < allocation.requiredBytes || allocation.deviceId != deviceId ||
            allocation.alignment == 0 || (allocation.alignment & (allocation.alignment - 1U)) != 0 ||
            begin % allocation.alignment != 0 || allocation.bytes > std::numeric_limits<uintptr_t>::max() - begin) {
            return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
        }
        const uintptr_t end = begin + allocation.bytes;
        for (std::size_t j = 0; j < i; ++j) {
            const uintptr_t otherBegin = reinterpret_cast<uintptr_t>(allocations[j].data);
            const uintptr_t otherEnd = otherBegin + allocations[j].bytes;
            if (begin < otherEnd && otherBegin < end) {
                return DispatchDedup::DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL;
            }
        }
    }
    return DispatchDedup::DISPATCH_DEDUP_SUCCESS;
}

}  // namespace DispatchDedup

namespace DispatchDedup {
// Matches each input mode, including quantization scratch or packed scale staging.
inline bool DispatchDedupPacketUbFits(const EpMemoryServerDedupLayout& layout, int64_t b, int64_t k,
                                      int64_t num_experts, int64_t world_size)
{
    const auto align = [](uint64_t n, uint64_t a) { return (n + a - 1) / a * a; };
    const uint64_t cores = b < kEpServerDedupAllCoreCountBsThreshold ? 64 : 32;
    const uint64_t reserved =
        b < kEpServerDedupAllCoreCountBsThreshold ? align((num_experts + cores - 1) / cores, 8) * 32 : 0;
    const uint64_t tokens = (b + 31) / 32;
    const uint64_t groups = std::min<uint64_t>(world_size / 8 - 1, k / 2);
    const bool needQuantize = layout.inputMode == EpServerDedupInputMode::Quantize16;
    const uint64_t inputBytes = needQuantize ? layout.data.quantCount * 2U : layout.data.packedPayloadBytes;
    // Prequantized scales occupy the two full payload input buffers (no alias to quantBuf).
    const uint64_t qBytes = align(b * k * 4, 32);
    const uint64_t scaleCount = layout.data.quantCount / 32U;
    const uint64_t quantWork = align(align(scaleCount, 32) * 4U + scaleCount * 2U, 32);
    const uint64_t scratchBytes = needQuantize ? std::max(qBytes, quantWork) : qBytes;
    const uint64_t sender = reserved + 2 * inputBytes + (needQuantize ? layout.data.packedPayloadBytes : 0U) +
                            align(b * k * 4, 256) + 2 * scratchBytes + qBytes + 2 * align(tokens * k * 4, 32) +
                            align(tokens * 4, 32) + align(tokens * k * 16, 32) + 64 + align(groups * 4, 32) +
                            layout.data.tokenStride;
    const uint64_t serverStateExtra = std::max<uint64_t>(64U, align(world_size / 8U * 4U, 32U)) - 64U;
    const uint64_t receiver = align(num_experts * 4, 32) + 2 * align(((layout.capacityRows + 31) / 32) * 4, 32) +
                              16 * layout.data.blockCount * 32 + 256 +
                              16 * align((layout.data.blockCount + 7) / 8, 32) * 4 + 256 + 16 * 36 +
                              layout.data.tokenStride + 64;
    return layout.data.blockCount > 0 && layout.data.blockCount <= UINT8_MAX && sender + serverStateExtra <= kEpServerDedupCountUbBytes &&
           receiver <= kEpServerDedupCountUbBytes;
}
}  // namespace DispatchDedup

#endif  // DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_LAYOUT_H
