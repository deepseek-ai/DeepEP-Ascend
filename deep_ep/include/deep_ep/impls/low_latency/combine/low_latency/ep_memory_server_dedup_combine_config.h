// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_CONFIG_H
#define COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_CONFIG_H

#include <cstdint>

// Independent of Dispatch. Rebuild Host and kernel together on all ranks.
#ifndef COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SEND_CORE_NUM
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SEND_CORE_NUM 32U
#endif

namespace CombineDedup {
constexpr uint32_t kEpServerDedupCombineSendCoreNum = COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SEND_CORE_NUM;
static_assert(kEpServerDedupCombineSendCoreNum >= 16U && kEpServerDedupCombineSendCoreNum <= 56U &&
                  kEpServerDedupCombineSendCoreNum % 4U == 0U,
              "COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SEND_CORE_NUM must be 16..56 in steps of 4");
constexpr uint32_t kEpServerDedupCombineRecvCoreNum = 64U - kEpServerDedupCombineSendCoreNum;
constexpr uint32_t kEpServerDedupCombineMaxTokenNum = 10000U;
// M1/M2 keep all resources distinct: three row arrays, compare mask, one
// 512-row table-2 cache, per-level detector/work/result/zero buffers and flags.
constexpr uint32_t kEpServerDedupCombineUbBytes = 190U * 1024U;
// Upper bound inherited from the source kernel. The host selects the actual
// M4/M5 arena from the runtime-reported usable UB and passes it in tiling.
// M1/M3 keep their independent 190 KiB budget.
constexpr uint32_t kEpServerDedupCombineRecvUbBytes = 240U * 1024U;
constexpr uint32_t kEpServerDedupCombineM4MaxInputLanes = 3U;
constexpr uint32_t kEpServerDedupCombineM5MaxInputLanes = 7U;  // MTE2_V(7) belongs to probe
constexpr uint32_t kEpServerDedupCombineM5MaxOutputLanes = 2U;
constexpr uint32_t kEpServerDedupCombineTableRows = 512U;
// Even before metadata is reserved, one admitted input/output pair has fewer
// than 256 blocks, so packing never narrows an oversized vector repeat count.
static_assert(kEpServerDedupCombineUbBytes / (480U + 512U) <= 255U, "M3 vector repeat budget");
constexpr uint32_t kEpServerDedupCombineFixedUbBytes = kEpServerDedupCombineTableRows * 12U + 256U + 2304U + 512U +
                                                       32U + 2048U + 512U + 256U + 32U + 512U +
                                                       kEpServerDedupCombineRecvCoreNum * 32U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define COMBINE_DEDUP_EP_COMBINE_CONSTEXPR __aicore__ inline constexpr
#else
#define COMBINE_DEDUP_EP_COMBINE_CONSTEXPR constexpr
#endif
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineAlign(uint32_t n, uint32_t a)
{
    return (n + a - 1U) / a * a;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineRowUbBytes(uint32_t tokens)
{
    return 3U * EpCombineAlign(tokens == 0U ? 4U : tokens * 4U, 256U) + EpCombineAlign((tokens + 7U) / 8U + 32U, 256U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombinePayloadLaneCount(uint32_t tokens, uint32_t h)
{
    if (h == 0U || h > kEpServerDedupCombineUbBytes / 2U || tokens > kEpServerDedupCombineMaxTokenNum) {
        return 0U;
    }
    const uint32_t available =
        kEpServerDedupCombineUbBytes - kEpServerDedupCombineFixedUbBytes - EpCombineRowUbBytes(tokens);
    // M3 vector packing reads every 480B payload block, including its tail.
    // M1 uses the same input lanes and drains them before M3. Output lanes
    // remain distinct until their MTE3 consumers have completed.
    const uint32_t blocks = (h * 2U + 479U) / 480U;
    const uint32_t lanes = available / (blocks * (480U + 512U));
    return lanes > 8U ? 8U : lanes;
}
// Control buffers retain independent storage throughout per-core stage reuse.
constexpr uint32_t kEpServerDedupCombineResidentUbBytes =
    kEpServerDedupCombineFixedUbBytes - kEpServerDedupCombineTableRows * 12U;
constexpr uint32_t kEpServerDedupCombineM4TableRows = 32U;
// During server selection M4's table caches and low-precision packing buffer
// are idle. Alias only that phase range, never inputs or in-flight outputs.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM4ScanBytes(uint32_t rows)
{
    return rows * 12U + 2U * EpCombineAlign(rows * 4U, 256U) + 3U * EpCombineAlign((rows + 7U) / 8U, 32U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM4ScanRows(uint32_t h, uint32_t k)
{
    const uint32_t available =
        384U + EpCombineAlign(kEpServerDedupCombineM4TableRows * k * 8U, 32U) + ((h * 2U + 479U) / 480U) * 480U;
    uint32_t rows = 32U;
    while (rows < 512U && EpCombineM4ScanBytes(rows * 2U) <= available) rows *= 2U;
    return rows;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM4TaskBytes(uint32_t tokens)
{
    return EpCombineAlign(((tokens + kEpServerDedupCombineRecvCoreNum - 1U) / kEpServerDedupCombineRecvCoreNum) * 4U,
                          32U);
}
// M5 input events use IDs 0..6; MTE2_V(7) is reserved for probe.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM5TokenCount(uint32_t bs)
{
    return bs / kEpServerDedupCombineRecvCoreNum + (bs % kEpServerDedupCombineRecvCoreNum != 0U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM5FlagBytes(uint32_t h, uint32_t k)
{
    // CANN 9.1 dav_3510 ReduceSum(mask=1, stride=1) loads a full 256B
    // vector at every 32B step. Reserve its final vector beyond the flags.
    return k * ((h * 2U + 479U) / 480U) * 32U + 256U;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM5WorkBytes(uint32_t h, uint32_t k)
{
    // The intermediate B floats are reduced with full 256B vector loads.
    return EpCombineAlign(k * ((h * 2U + 479U) / 480U) * 4U, 256U);
}
// Offsets are relative to the scratch after resident control buffers. Only
// phase metadata changes meaning; output slots stay live across M4 -> M5.
struct EpCombineM45WorkspaceLayout {
    uint32_t arenaBytes = 0U;
    uint32_t computeBytes = 0U, phaseOffset = 0U, phaseBytes = 0U;
    uint32_t inputOffset = 0U, outputOffset = 0U, finalInputOffset = 0U;
    uint32_t inputBytes = 0U, outputBytes = 0U;
    uint32_t inputLanes = 0U, outputLanes = 0U, finalOutputLanes = 0U;
    uint32_t totalBytes = 0U;
};
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR EpCombineM45WorkspaceLayout EpCombineM45Layout(uint32_t tokens, uint32_t bs,
                                                                                  uint32_t h, uint32_t k,
                                                                                  uint32_t ubBytes = kEpServerDedupCombineRecvUbBytes)
{
    EpCombineM45WorkspaceLayout result{};
    if (ubBytes < kEpServerDedupCombineUbBytes || ubBytes > kEpServerDedupCombineRecvUbBytes || ubBytes % 32U ||
        tokens > kEpServerDedupCombineMaxTokenNum || bs == 0U || h == 0U ||
        h > ubBytes / 12U || k == 0U || k > 32U)
        return result;
    const uint64_t owned = EpCombineM5TokenCount(bs);
    const uint64_t partial = EpCombineM4TaskBytes(tokens) + 384U +
                             EpCombineAlign(kEpServerDedupCombineM4TableRows * k * 8U, 32U) +
                             ((h * 2U + 479U) / 480U) * 480U;
    const uint64_t final = 2U * ((owned * 4U + 31U) / 32U * 32U) + EpCombineM5FlagBytes(h, k) +
                           EpCombineM5WorkBytes(h, k) + 32U;
    const uint64_t phase = partial > final ? partial : final;
    const uint32_t compute = EpCombineAlign(h * 4U, 256U);
    const uint64_t fixed = kEpServerDedupCombineResidentUbBytes + 2U * compute + phase;
    const uint32_t input = ((h * 2U + 479U) / 480U) * 480U;
    const uint32_t output = ((h * 2U + 479U) / 480U) * 512U;
    if (fixed + input + output > ubBytes) return result;
    result.arenaBytes = ubBytes;
    const uint32_t reads = uint32_t((ubBytes - fixed - output) / input);
    result.inputLanes = reads > kEpServerDedupCombineM4MaxInputLanes ? kEpServerDedupCombineM4MaxInputLanes : reads;
    const uint32_t writes = uint32_t((ubBytes - fixed - result.inputLanes * input) / output);
    result.outputLanes = writes > 8U ? 8U : writes;
    result.finalOutputLanes = result.outputLanes > kEpServerDedupCombineM5MaxOutputLanes
                                  ? kEpServerDedupCombineM5MaxOutputLanes
                                  : result.outputLanes;
    result.computeBytes = compute;
    result.phaseOffset = 2U * compute;
    result.phaseBytes = uint32_t(phase);
    result.inputOffset = result.phaseOffset + result.phaseBytes;
    result.finalInputOffset = result.phaseOffset + uint32_t(final);
    result.outputOffset = result.inputOffset + result.inputLanes * input;
    result.inputBytes = input;
    result.outputBytes = output;
    result.totalBytes = kEpServerDedupCombineResidentUbBytes + result.outputOffset + result.outputLanes * output;
    return result;
}
// M5 reuses all space after its own metadata except the retained physical
// output slots. Each input remembers which M4 sends must retire before overwrite.
struct EpCombineM5WorkspaceLayout {
    uint32_t inputLanes = 0U;
    uint32_t inputOffsets[kEpServerDedupCombineM5MaxInputLanes] = {};
    uint32_t inputRetireMasks[kEpServerDedupCombineM5MaxInputLanes] = {};
    uint32_t outputSlots[kEpServerDedupCombineM5MaxOutputLanes] = {};
};
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR EpCombineM5WorkspaceLayout
EpCombineM5Layout(const EpCombineM45WorkspaceLayout& shared, uint32_t partialTasks)
{
    EpCombineM5WorkspaceLayout result{};
    if (shared.outputLanes == 0U || shared.inputBytes == 0U) return result;
    // Keep the two latest M4 sends; with fewer sends include unused slots.
    const uint32_t retained = partialTasks < shared.finalOutputLanes ? partialTasks : shared.finalOutputLanes;
    const uint32_t first = (partialTasks - retained) % shared.outputLanes;
    for (uint32_t lane = 0; lane < shared.finalOutputLanes; ++lane)
        result.outputSlots[lane] = (first + lane) % shared.outputLanes;
    const uint32_t limit = shared.arenaBytes - kEpServerDedupCombineResidentUbBytes;
    uint32_t cursor = shared.finalInputOffset;
    while (result.inputLanes < kEpServerDedupCombineM5MaxInputLanes && cursor + shared.inputBytes <= limit) {
        bool overlapsOutput = false;
        for (uint32_t lane = 0; lane < shared.finalOutputLanes; ++lane) {
            const uint32_t begin = shared.outputOffset + result.outputSlots[lane] * shared.outputBytes;
            if (cursor < begin + shared.outputBytes && cursor + shared.inputBytes > begin) {
                cursor = begin + shared.outputBytes;
                overlapsOutput = true;
                break;
            }
        }
        if (overlapsOutput) continue;
        const uint32_t inputLane = result.inputLanes++;
        result.inputOffsets[inputLane] = cursor;
        for (uint32_t slot = 0; slot < shared.outputLanes; ++slot) {
            const uint32_t begin = shared.outputOffset + slot * shared.outputBytes;
            if (cursor < begin + shared.outputBytes && cursor + shared.inputBytes > begin)
                result.inputRetireMasks[inputLane] |= 1U << slot;
        }
        cursor += shared.inputBytes;
    }
    return result;
}
// Keep the existing H admission bound despite freeing the product buffer.
// It also bounds K*N so each token's flag clear fits CANN's 12-bit blockCount.
static_assert(32U * ((2U * (kEpServerDedupCombineRecvUbBytes / 12U) + 479U) / 480U) <= 4095U,
              "M5 flag clear blockCount budget");
#undef COMBINE_DEDUP_EP_COMBINE_CONSTEXPR
}  // namespace CombineDedup
#endif
