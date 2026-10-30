// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H
#define COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H

#include <kernel_operator.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/comm_args.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_buffer_layout.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_config.h>
#include <deep_ep/impls/low_latency/combine/low_latency/combine_server_dedup_partial.h>
#include <deep_ep/impls/low_latency/combine/low_latency/combine_server_dedup_source_aggregate.h>
#include <deep_ep/impls/low_latency/combine/low_latency/combine_server_dedup_select.h>

namespace CombineDedup {
using namespace AscendC;

// Local stage, readiness, DIRECT/partial return and source-token aggregation.
class CombineMemoryServerDedupKernel {
    friend struct CombineModuleTestAccess;

public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR sendCounts,
                                GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask,
                                GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t topK, uint32_t experts, uint8_t dtype,
                                uint8_t quantMode, int64_t magic, uint32_t recvUbBytes, bool enableDedup = true)
    {
        (void)quantMode;
        core_ = GetBlockIdx();
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        enableDedup_ = enableDedup;
        h_ = h;
        topK_ = topK;
        bs_ = bs;
        experts_ = experts;
        dtype_ = dtype;
        expertAddr_ = expertOut;
        tableAddr_ = assistInfoForCombine;
        relayAddr_ = relayReadIndex;
        typesAddr_ = tokenType;
        masksAddr_ = sourceMask;
        outputAddr_ = yOut;
        if (recvUbBytes < kEpServerDedupCombineUbBytes || recvUbBytes > kEpServerDedupCombineRecvUbBytes ||
            recvUbBytes % 32U) {
            Trap();
            return;
        }
        // Load the device-produced prefix through MTE2 rather than caching a
        // mutable GM scalar across launches. This 32B temporary becomes the
        // independent M2 prepare sum only after the scalar count is consumed.
        pipe_.InitBuffer(arena_, core_ >= kEpServerDedupCombineSendCoreNum ? recvUbBytes
                                                                           : kEpServerDedupCombineUbBytes);
        prepareSum_ = Allocate<float>(32U);
        GlobalTensor<int32_t> prefix;
        prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCounts));
        auto tokenCount = prepareSum_.ReinterpretCast<int32_t>();
        DataCopyPad(tokenCount, prefix[experts - 1U], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        Sync<HardEvent::MTE2_S>();
        const int32_t tokens = tokenCount.GetValue(0U);
        const uint32_t localExperts = experts / uint32_t(args_->rankSize);
        const uint32_t copies = topK < localExperts ? topK : localExperts;
        const uint64_t capacity = uint64_t(args_->rankSize) * bs * copies;
        if (tokens < 0 || uint32_t(tokens) > kEpServerDedupCombineMaxTokenNum || uint64_t(tokens) > capacity) {
            Trap();  // Invalid device data must not become a truncated successful launch.
            return;
        }
        tokens_ = uint32_t(tokens);
        lanes_ = EpCombinePayloadLaneCount(tokens_, h_);
        workspace_ = EpCombineM45Layout(tokens_, bs_, h_, topK_, recvUbBytes);
        if (lanes_ == 0U || workspace_.inputLanes == 0U) {
            Trap();
            return;
        }

        blocks_ = (h_ * 2U + 479U) / 480U;
        laneBytes_ = blocks_ * 480U;
        resultStride_ = blocks_ * 512U;
        window_ = (uint64_t(magic) % 2U) * DispatchDedup::kEpServerDedupWindowBytes;
        control_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET;
        stageStride_ = (uint64_t(h_) * 2U + 511U) / 512U * 512U;
        stageBase_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + uint64_t(bs) * topK * resultStride_;
        expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(expertOut));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(assistInfoForCombine));
        types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenType));
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(args_->peerMems[args_->rank] + stageBase_));
        preparePublish_ = Allocate<float>(256U);
        prepareInput_ = Allocate<float>(2304U);
        prepareWork_ = Allocate<float>(512U);
        prepareZero_ = Allocate<float>(2048U);
        readyInput_ = Allocate<float>(512U);
        readyWork_ = Allocate<float>(256U);
        readySum_ = Allocate<float>(32U);
        readyZero_ = Allocate<float>(512U);
        notifyUb_ = Allocate<float>(kEpServerDedupCombineRecvCoreNum * 32U);
        const uint32_t rowBytes = EpCombineAlign(tokens_ == 0U ? 4U : tokens_ * 4U, 256U);
        typeUb_ = Allocate<int32_t>(rowBytes);
        indexUb_ = Allocate<int32_t>(rowBytes);
        packedUb_ = Allocate<int32_t>(rowBytes);
        maskUb_ = Allocate<uint8_t>(EpCombineAlign((tokens_ + 7U) / 8U + 32U, 256U));
        tableUb_ = Allocate<int32_t>(kEpServerDedupCombineTableRows * 12U);
        payloadUb_ = Allocate<uint16_t>(lanes_ * laneBytes_);
        resultUb_ = Allocate<uint8_t>(lanes_ * resultStride_);
    }

    __aicore__ inline void Process()
    {
        if (enableDedup_) CopyLocalPendingTokens();
        if (core_ >= kEpServerDedupCombineSendCoreNum) {
            if (enableDedup_) {
                if (core_ >= 56U) {
                    NotifyServerCard();
                    // Notification has been issued; do not add an MTE3 drain.
                }
                WaitServerReady();
                ReturnPartialTokens();
                const uint32_t retiredOutputs = AggregateSourceTokens();
                DrainSharedOutputs(retiredOutputs);
            } else {
                AggregateSourceTokensM6();
            }
        } else {
            ReturnDirectTokens();
            // Retain the full drain after direct-return DMA completion.
            PipeBarrier<PIPE_ALL>();
        }
        // Retires prepare publication, both zero sources and server notification;
        // no arena/pipe reset or exit may precede completion of their DMA reads.
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void AggregateSourceTokensM6()
    {
        ServerDedupSourceAggregate aggregate;
        // Count DMA has retired. Auto uses the same final-stage scratch view as
        // M5, but owns its output events because M4 did not run on this launch.
        aggregate.InitDirectShared(
            arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes],
            args_->peerMems[args_->rank] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET,
            outputAddr_, bs_, h_, topK_, dtype_, core_ - kEpServerDedupCombineSendCoreNum,
            kEpServerDedupCombineRecvCoreNum, workspace_);
        aggregate.Process();
        aggregate.DrainM6();
    }
    __aicore__ inline uint32_t AggregateSourceTokens()
    {
        ServerDedupSourceAggregate aggregate;
        aggregate.Init(arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes],
                       args_->peerMems[args_->rank] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET,
                       masksAddr_, outputAddr_, bs_, h_, topK_, dtype_,
                       core_ - kEpServerDedupCombineSendCoreNum, kEpServerDedupCombineRecvCoreNum, workspace_,
                       partialTasks_, enableDedup_);
        return aggregate.Process();
    }
    __aicore__ inline void DrainSharedOutputs(uint32_t retiredOutputs = 0U)
    {
        // M5 already consumed slots repurposed as inputs. Retire the remaining
        // physical slots exactly once, including unused seeds and M5 outputs.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane)
            if (((retiredOutputs >> lane) & 1U) == 0U) WaitFlag<HardEvent::MTE3_V>(lane);
    }
    template <class T>
    __aicore__ inline LocalTensor<T> Allocate(uint32_t bytes)
    {
        auto result = arena_.Get<uint8_t>()[usedBytes_].template ReinterpretCast<T>();
        usedBytes_ += bytes;
        return result;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const TEventID id = pipe_.FetchEventID(E);
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }
    __aicore__ inline GlobalTensor<float> Control(uint32_t rank, uint64_t offset)
    {
        GlobalTensor<float> result;
        result.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args_->peerMems[rank] + control_ + offset));
        return result;
    }
    __aicore__ inline void SelectRows(int32_t type)
    {
        selected_ = 0U;
        if (tokens_ != 0U) {
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();
            Compares(maskUb_, typeUb_, type, CMPMODE::EQ, tokens_);
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();
        }
    }
    __aicore__ inline void CopyLocalPendingTokens()
    {
        SelectRows(3);
        const uint32_t base = uint32_t(selected_) / 64U, extra = uint32_t(selected_) % 64U;
        const uint32_t count = base + (core_ < extra ? 1U : 0U);
        const uint32_t start = core_ * base + (core_ < extra ? core_ : extra);
        for (uint32_t lane = 0; lane < lanes_; ++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
        uint32_t cacheBegin = 0U, cacheEnd = 0U;
        for (uint32_t task = 0; task < count; ++task) {
            const uint32_t row = uint32_t(packedUb_.GetValue(start + task));
            CacheTableRows(row, start, task, count, cacheBegin, cacheEnd);
            const uint32_t slot = uint32_t(tableUb_.GetValue((row - cacheBegin) * 3U));
            const uint32_t lane = task % lanes_;
            WaitFlag<HardEvent::MTE3_MTE2>(lane);
            auto payload = payloadUb_[lane * laneBytes_ / 2U];
            DataCopyPad(payload, expert_[uint64_t(row) * h_], {1U, h_ * 2U, 0U, 0U, 0U}, {false, 0U, 0U, uint16_t(0)});
            SetFlag<HardEvent::MTE2_MTE3>(lane);
            WaitFlag<HardEvent::MTE2_MTE3>(lane);
            DataCopyPad(stage_[uint64_t(slot) * stageStride_ / 2U], payload, {1U, h_ * 2U, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE3_MTE2>(lane);
        }
        // Also consumes seeded, unused lane events. All stage stores complete
        // before publication; individual tokens never wait immediately after store.
        for (uint32_t lane = 0; lane < lanes_; ++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
        Duplicate<float>(preparePublish_, 0.0f, 64U);
        Sync<HardEvent::V_S>();
        for (uint32_t j = 0; j < 8U; ++j) preparePublish_.SetValue(j * 8U, 1.0f);
        Sync<HardEvent::S_MTE3>();
        auto done = Control(args_->rank, COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET);
        DataCopyPad(done[core_ * 8U], preparePublish_, {8U, 32U, 0U, 2016U, 0U});
    }
    __aicore__ inline void CacheTableRows(uint32_t row, uint32_t start, uint32_t task, uint32_t count,
                                          uint32_t& cacheBegin, uint32_t& cacheEnd)
    {
        if (row >= cacheEnd) {
            cacheBegin = row;
            cacheEnd = row + 1U;
            for (uint32_t next = task + 1U; next < count; ++next) {
                const uint32_t nextRow = uint32_t(packedUb_.GetValue(start + next));
                if (nextRow - cacheBegin >= kEpServerDedupCombineTableRows) break;
                cacheEnd = nextRow + 1U;
            }
            // The old metadata has no users after its scalar offset is read.
            // Payload DMA owns separate lanes, so changing this cache needs
            // no payload drain and preserves the lane sequence across batches.
            DataCopyPad(tableUb_, table_[uint64_t(cacheBegin) * 3U], {1U, (cacheEnd - cacheBegin) * 12U, 0U, 0U, 0U},
                        {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void PrepareDirectRows()
    {
        selected_ = 0U;
        if (tokens_ == 0U) return;
        if (enableDedup_) {
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();
            Compares(maskUb_, typeUb_, int32_t(1), CMPMODE::EQ, (tokens_ + 63U) / 64U * 64U);
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();
        } else {
            // Every row is a direct row without the dedup token type table.
            CreateVecIndex(packedUb_, int32_t(0), tokens_);
            selected_ = tokens_;
            Sync<HardEvent::V_S>();
        }
        if (selected_ == 0U) return;  // no routing reads for a type2/type3-only rank
        const uint32_t chunk = tokens_ < kEpServerDedupCombineTableRows ? tokens_ : kEpServerDedupCombineTableRows;
        CreateVecIndex(typeUb_, int32_t(0), chunk);
        PipeBarrier<PIPE_V>();
        Muls(typeUb_, typeUb_, int32_t(12), chunk);  // Table2 byte offsets
        for (uint32_t begin = 0U; begin < tokens_; begin += chunk) {
            const uint32_t span = tokens_ - begin < chunk ? tokens_ - begin : chunk;
            Sync<HardEvent::V_MTE2>();  // table cache is no longer a Gather source
            DataCopyPad(tableUb_, table_[uint64_t(begin) * 3U], {1U, span * 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();
            Gather(indexUb_[begin], tableUb_, typeUb_.ReinterpretCast<uint32_t>(), 0U, span);
        }
        PipeBarrier<PIPE_V>();
        // Keep two persistent compact arrays: ranks in typeUb_, original rows
        // in packedUb_. indexUb_ becomes the per-server compact output list.
        if (enableDedup_) {
            GatherMask(typeUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
        } else {
            Muls(typeUb_, indexUb_, int32_t(1), tokens_);
        }
        Sync<HardEvent::V_S>();
        ServerDedupRankMask(maskUb_, tableUb_.ReinterpretCast<uint8_t>(), typeUb_, 0, int32_t(args_->rankSize),
                            uint32_t(selected_));
        uint64_t valid = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, valid);
        Sync<HardEvent::V_S>();
        if (valid != selected_) Trap();
    }
    __aicore__ inline uint32_t SelectDirectServer(uint32_t server)
    {
        ServerDedupRankMask(maskUb_, tableUb_.ReinterpretCast<uint8_t>(), typeUb_, int32_t(server * 8U),
                            int32_t((server + 1U) * 8U), uint32_t(selected_));
        uint64_t count = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, count);
        Sync<HardEvent::V_S>();
        return uint32_t(count);
    }
    __aicore__ inline void ReturnDirectTokens()
    {
        PrepareDirectRows();
        if (selected_ == 0U) return;
        const uint32_t servers = uint32_t(args_->rankSize) / 8U;
        const uint32_t ownServer = uint32_t(args_->rank) / 8U;
        uint32_t owner = 0U, sendSequence = 0U;
        uint64_t flagMask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < lanes_; ++lane) {
            // The entire 32B Flag is initialized once and is not touched by Copy.
            Duplicate<float>(resultUb_[lane * resultStride_].ReinterpretCast<float>()[120U], 1.0f, flagMask,
                             uint8_t(blocks_), 1U, 16U);
            SetFlag<HardEvent::V_MTE2>(lane);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t step = 1U; step <= servers; ++step) {
            const uint32_t count = SelectDirectServer(ServerDedupHalfOrder(servers, ownServer, step - 1U));
            const uint32_t first = ServerDedupFirstOwned(core_, owner, kEpServerDedupCombineSendCoreNum);
            owner = (owner + count) % kEpServerDedupCombineSendCoreNum;
            for (uint32_t task = first; task < count; task += kEpServerDedupCombineSendCoreNum) {
                const uint32_t row = uint32_t(indexUb_.GetValue(task));
                // Only owned rows reload a routing triple. The persistent rank
                // and original-row arrays are untouched between server rounds.
                DataCopyPad(tableUb_, table_[uint64_t(row) * 3U], {1U, 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
                Sync<HardEvent::MTE2_S>();
                const uint32_t source = uint32_t(tableUb_.GetValue(0U));
                const uint32_t token = uint32_t(tableUb_.GetValue(1U));
                const uint32_t topk = uint32_t(tableUb_.GetValue(2U));
                const uint32_t lane = sendSequence++ % lanes_;
                // Input lifetime ends after Vector has consumed its last block.
                WaitFlag<HardEvent::V_MTE2>(lane);
                auto input = payloadUb_[lane * laneBytes_ / 2U];
                DataCopyPad(input, expert_[uint64_t(row) * h_], {1U, h_ * 2U, 0U, 0U, 0U},
                            {false, 0U, 0U, uint16_t(0)});
                SetFlag<HardEvent::MTE2_V>(lane);
                WaitFlag<HardEvent::MTE2_V>(lane);
                // Output lifetime ends only after the previous send has read it.
                WaitFlag<HardEvent::MTE3_V>(lane);
                auto output = resultUb_[lane * resultStride_];
                auto dstWords = output.ReinterpretCast<int32_t>();
                auto srcWords = input.ReinterpretCast<int32_t>();
                // Same nonquantized vector packing as Dispatch, without its tail.
                // Input UB covers blocks_*480B even though GM reads exactly 2H.
                Copy(dstWords, srcWords, uint64_t(64U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
                Copy(dstWords[64U], srcWords[64U], uint64_t(56U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
                SetFlag<HardEvent::V_MTE2>(lane);
                SetFlag<HardEvent::V_MTE3>(lane);
                WaitFlag<HardEvent::V_MTE3>(lane);
                GlobalTensor<uint8_t> destination;
                destination.SetGlobalBuffer(args_->peerMems[source] + window_ +
                                            COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET +
                                            (uint64_t(token) * topK_ + topk) * resultStride_);
                DataCopy(destination, output, resultStride_);
                SetFlag<HardEvent::MTE3_V>(lane);
            }
            // No server fence: the next scan can overlap pending output DMA.
        }
        // Final retirement includes unused seeded lanes, never a per-token drain.
        for (uint32_t lane = 0; lane < lanes_; ++lane) {
            WaitFlag<HardEvent::V_MTE2>(lane);
            WaitFlag<HardEvent::MTE3_V>(lane);
        }
    }
    __aicore__ inline void ReturnPartialTokens()
    {
        // M1 payload lanes are retired; complete this core's M2 publications
        // before reusing scratch. Independent control buffers remain resident.
        PipeBarrier<PIPE_ALL>();
        // Seed exactly once, including empty M4. M5 inherits these events.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane) SetFlag<HardEvent::MTE3_V>(lane);
        ServerDedupPartial partial;
        partial.Init(args_, &pipe_, arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes], expertAddr_,
                     tableAddr_, relayAddr_, typesAddr_, tokens_, bs_, h_, topK_, experts_, dtype_, window_,
                     workspace_);
        partialTasks_ = partial.Process();
        // Retire only input and Vector users before scalar metadata reuse.
        // M4 output slots and MTE3_V events stay live for M5's per-slot waits.
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::V_S>();
    }
    __aicore__ inline void WaitFlags(GlobalTensor<float> gm, LocalTensor<float> input, LocalTensor<float> work,
                                     LocalTensor<float> sum, uint32_t flags)
    {
        while (true) {
            // The previous reduction has completed before MTE2 overwrites input.
            Sync<HardEvent::V_MTE2>();
            DataCopyPad(input, gm, {1U, flags * 32U, 0U, 0U, 0U}, {false, 0U, 0U, 0.0f});
            Sync<HardEvent::MTE2_V>();
            ReduceSum<float>(sum, input, work, 1U, flags, 1U);
            Sync<HardEvent::V_S>();
            if (sum.GetValue(0U) == float(flags)) return;
        }
    }
    __aicore__ inline void NotifyServerCard()
    {
        auto done = Control(args_->rank,
                            COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET + uint64_t(core_ - 56U) * 2048U);
        WaitFlags(done, prepareInput_, prepareWork_, prepareSum_, 64U);
        Duplicate<float>(prepareZero_, 0.0f, 512U);
        Duplicate<float>(notifyUb_, 0.0f, kEpServerDedupCombineRecvCoreNum * 8U);
        Sync<HardEvent::V_S>();
        for (uint32_t r = 0; r < kEpServerDedupCombineRecvCoreNum; ++r) notifyUb_.SetValue(r * 8U, 1.0f);
        Sync<HardEvent::V_MTE3>();
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(done, prepareZero_, {1U, 2048U, 0U, 0U, 0U});
        const uint32_t target = uint32_t(args_->rank) / 8U * 8U + core_ - 56U;
        auto ready = Control(target, COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET);
        DataCopyPad(ready[(uint32_t(args_->rank) % 8U) * 8U], notifyUb_,
                    {uint16_t(kEpServerDedupCombineRecvCoreNum), 32U, 0U, 480U, 0U});
    }
    __aicore__ inline void WaitServerReady()
    {
        auto ready = Control(args_->rank, COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET +
                                              uint64_t(core_ - kEpServerDedupCombineSendCoreNum) * 512U);
        WaitFlags(ready, readyInput_, readyWork_, readySum_, 8U);
        Duplicate<float>(readyZero_, 0.0f, 128U);
        Sync<HardEvent::V_MTE3>();
        DataCopyPad(ready, readyZero_, {1U, 512U, 0U, 0U, 0U});
    }

    EpCombineM45WorkspaceLayout workspace_{};
    uint32_t partialTasks_ = 0U;
    TPipe pipe_;
    TBuf<> arena_;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    GM_ADDR expertAddr_ = nullptr;
    GM_ADDR typesAddr_ = nullptr;
    GM_ADDR tableAddr_ = nullptr;
    GM_ADDR relayAddr_ = nullptr;
    GM_ADDR masksAddr_ = nullptr;
    GM_ADDR outputAddr_ = nullptr;
    bool enableDedup_ = false;
    uint32_t bs_ = 0, experts_ = 0;
    uint8_t dtype_ = 0;
    uint32_t core_ = 0, h_ = 0, topK_ = 0, blocks_ = 0, resultStride_ = 0, tokens_ = 0, lanes_ = 0, laneBytes_ = 0,
             usedBytes_ = 0;
    uint64_t selected_ = 0, window_ = 0, control_ = 0, stageBase_ = 0, stageStride_ = 0;
    GlobalTensor<uint16_t> expert_, stage_;
    GlobalTensor<int32_t> table_, types_;
    LocalTensor<int32_t> typeUb_, indexUb_, packedUb_, tableUb_;
    LocalTensor<uint8_t> maskUb_, resultUb_;
    LocalTensor<uint16_t> payloadUb_;
    LocalTensor<float> preparePublish_, prepareInput_, prepareWork_, prepareSum_, prepareZero_;
    LocalTensor<float> readyInput_, readyWork_, readySum_, readyZero_, notifyUb_;
};
}  // namespace CombineDedup
#endif
