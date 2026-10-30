// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_SERVER_DEDUP_KERNEL_H
#define DISPATCH_SERVER_DEDUP_KERNEL_H

#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_magic.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_counts.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_send.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_recv.h>

namespace DispatchDedup {
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class DispatchMemoryServerDedupKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
                                GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, uint32_t bs,
                                uint32_t h, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType, TPipe* pipe,
                                GM_ADDR inputScales = nullptr, bool enableDedup = true)
    {
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        pipe_ = pipe;
        const uint32_t magic = RotateDispatchServerDedupMagic(args_, core_, pipe);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType,
                     magic, pipe);
        if (core_ < kEpServerDedupSendCoreNum)
            sender_.Init(args_, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe,
                         magic, inputScales, enableDedup);
        else
            receiver_.Init(args_, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut,
                           bs, h, topK, experts, pipe, magic, enableDedup);
    }

    __aicore__ inline void Process()
    {
        counts_.SendCnt();

        if (core_ < kEpServerDedupSendCoreNum) {
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes());

        } else {
            counts_.RecvCnt();
            counts_.CalcCumsum();
            counts_.CompleteCumsum();
            // CompleteCumsum may still be writing expert counts from its UB.
            // Retire that stage before receiver resets the pipe and reloads GM.
            PipeBarrier<PIPE_ALL>();
            receiver_.Process();
        }

        counts_.Finish();

        // No shared work follows local completion. Same-stream kernel completion
        // orders the next launch after all cores finish their writes and cleanup.
    }

private:
    ServerDedupCounts counts_;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t core_ = 0;
};
}  // namespace DispatchDedup
#endif
