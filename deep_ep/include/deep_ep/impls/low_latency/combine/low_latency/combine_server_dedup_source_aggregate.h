// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_SERVER_DEDUP_SOURCE_AGGREGATE_H
#define COMBINE_DEDUP_EP_SERVER_DEDUP_SOURCE_AGGREGATE_H

#include <kernel_operator.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_config.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/dispatch_dedup_types.h>

namespace CombineDedup {
using namespace AscendC;

// M5/M6 sum expert contributions without applying routing weights.
// M5 inherits M4 output slots and their completion events. Metadata/compute/input
// are handed off after MTE2/Vector completion, while M4 MTE3 may still run.
class ServerDedupSourceAggregate {
public:
    __aicore__ inline void Init(LocalTensor<uint8_t> scratch, GM_ADDR result, GM_ADDR masks,
                                GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t k, uint8_t dtype, uint32_t participant,
                                uint32_t participants, const EpCombineM45WorkspaceLayout& layout, uint32_t partialTasks,
                                bool enableDedup)
    {
        outputBytes_ = layout.outputBytes;
        finalLayout_ = EpCombineM5Layout(layout, partialTasks);
        inputLanes_ = finalLayout_.inputLanes;
        sharedOutputLanes_ = layout.outputLanes;
        outputLanes_ = layout.finalOutputLanes;
        if (inputLanes_ == 0U || outputLanes_ == 0U) {
            Trap();
            return;
        }
        InitCommon(scratch, result, masks, yOut, bs, h, k, dtype, participant, participants,
                   layout.phaseOffset, layout.computeBytes, layout.outputOffset, enableDedup);
    }

    // Auto direct aggregation shares the M5 resource layout without inheriting
    // any M4 sends or events. Only the slots actually used for outputs are seeded.
    __aicore__ inline void InitDirectShared(LocalTensor<uint8_t> scratch, GM_ADDR result, GM_ADDR yOut,
                                            uint32_t bs, uint32_t h, uint32_t k, uint8_t dtype, uint32_t participant,
                                            uint32_t participants, const EpCombineM45WorkspaceLayout& layout)
    {
        finalLayout_ = EpCombineM5Layout(layout, 0U);
        inputLanes_ = finalLayout_.inputLanes;
        outputLanes_ = layout.finalOutputLanes;
        outputBytes_ = layout.outputBytes;
        if (inputLanes_ == 0U || outputLanes_ == 0U) {
            Trap();
            return;
        }
        for (uint32_t lane = 0; lane < outputLanes_; ++lane) SetFlag<HardEvent::MTE3_V>(finalLayout_.outputSlots[lane]);
        InitCommon(scratch, result, nullptr, yOut, bs, h, k, dtype, participant, participants,
                   layout.phaseOffset, layout.computeBytes, layout.outputOffset, false);
    }

    __aicore__ inline void DrainM6()
    {
        for (uint32_t lane = 0; lane < outputLanes_; ++lane)
            WaitFlag<HardEvent::MTE3_V>(finalLayout_.outputSlots[lane]);
    }

    __aicore__ inline uint32_t Process()
    {
        if (count_ == 0U) return 0U;
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) SetFlag<HardEvent::V_MTE2>(lane);
        uint32_t remaining = count_, scan = 0U, completed = 0U;
        while (remaining != 0U) {
            const uint32_t token = pending_.GetValue(scan);
            const uint32_t mask = enableDedup_ ? masks_.GetValue(token - begin_) : 0U;
            if (enableDedup_ && (mask == 0U || (k_ < 32U && (mask >> k_) != 0U))) {
                Trap();
                return retiredOutputs_;
            }
            if (!Probe(token, mask)) {
                if (++scan == remaining) scan = 0U;
                continue;
            }
            // No further probes after this point: clear every original token's
            // complete K*N flags, including mask-inactive slots. This source is
            // independent of the final token's in-flight payload and yOut.
            if (remaining == 1U) ClearFlags();
            if (dtype_ == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16)
                Aggregate<half>(token, mask, completed);
            else
                Aggregate<bfloat16_t>(token, mask, completed);
            pending_.SetValue(scan, pending_.GetValue(remaining - 1U));
            --remaining;
            ++completed;
            if (scan == remaining) scan = 0U;  // re-probe the swapped real TokenId
        }
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) WaitFlag<HardEvent::V_MTE2>(lane);
        return retiredOutputs_;
    }

private:
    __aicore__ inline void InitCommon(LocalTensor<uint8_t> scratch, GM_ADDR result, GM_ADDR masks,
                                      GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t k, uint8_t dtype,
                                      uint32_t participant, uint32_t participants, uint32_t phaseOffset,
                                      uint32_t computeBytes, uint32_t outputOffset, bool enableDedup)
    {
        enableDedup_ = enableDedup;
        scratch_ = scratch[phaseOffset];
        h_ = h;
        k_ = k;
        dtype_ = dtype;
        const uint32_t base = bs / participants, extra = bs % participants;
        count_ = base + (participant < extra ? 1U : 0U);
        begin_ = participant * base + (participant < extra ? participant : extra);
        blocks_ = (h * 2U + 479U) / 480U;
        packetBytes_ = blocks_ * 512U;
        // The shared Host budget uses the largest contiguous Recv-core segment.
        const uint32_t owned = EpCombineM5TokenCount(bs);
        masks_ = Allocate<uint32_t>(EpCombineAlign(owned * 4U, 32U));
        pending_ = Allocate<uint32_t>(EpCombineAlign(owned * 4U, 32U));
        flags_ = Allocate<float>(EpCombineM5FlagBytes(h, k));
        work_ = Allocate<float>(EpCombineM5WorkBytes(h, k));
        probeSum_ = Allocate<float>(32U);
        cast_ = scratch.ReinterpretCast<float>();
        sum_ = scratch[computeBytes].ReinterpretCast<float>();
        inputs_ = scratch.ReinterpretCast<uint16_t>();
        outputs_ = scratch[outputOffset].ReinterpretCast<uint16_t>();
        result_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(result));
        resultFlags_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(yOut));
        if (count_ == 0U) return;
        GlobalTensor<uint32_t> maskGm;
        if (enableDedup_) maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(masks));
        if (enableDedup_) {
            DataCopyPad(masks_, maskGm[begin_], {1U, count_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            Sync<HardEvent::MTE2_S>();
        }
        for (uint32_t p = 0; p < count_; ++p) pending_.SetValue(p, begin_ + p);
    }
    template <class T>
    __aicore__ inline LocalTensor<T> Allocate(uint32_t bytes)
    {
        auto tensor = scratch_[used_].template ReinterpretCast<T>();
        used_ += bytes;
        return tensor;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        SetFlag<E>(7U);
        WaitFlag<E>(7U);
    }
    __aicore__ inline bool Probe(uint32_t token, uint32_t mask)
    {
        // The previous probe's V_S retired its flag-buffer reader. This UB is
        // independent of payload compute/output, so do not wait for that Vector
        // work before issuing the next flag read. Inactive slot flags stay zero.
        const uint32_t flags = k_ * blocks_;
        const uint32_t expected = (enableDedup_ ? uint32_t(GetBitCount<1>(uint64_t(mask))) : k_) * blocks_;
        auto src = resultFlags_[uint64_t(token) * k_ * packetBytes_ / 4U + 120U];
        DataCopy(flags_, src, {uint16_t(flags), 1U, 15U, 0U});
        Sync<HardEvent::MTE2_V>();
        ReduceSum(probeSum_, flags_, work_, 1U, flags, 1U);
        Sync<HardEvent::V_S>();
        return probeSum_.GetValue(0U) == float(expected);
    }
    __aicore__ inline void ClearFlags()
    {
        const uint32_t flags = k_ * blocks_;
        Duplicate<float>(flags_, 0.0f, flags * 8U);
        // M4 may still own an unconsumed V_MTE3(7) publication for slot 7.
        // Publish zero via Scalar instead: neither wait drains MTE3, and
        // S_MTE3(7) is independent of the inherited payload event directions.
        Sync<HardEvent::V_S>();
        Sync<HardEvent::S_MTE3>();
        for (uint32_t p = begin_; p < begin_ + count_; ++p) {
            auto target = resultFlags_[uint64_t(p) * k_ * packetBytes_ / 4U + 120U];
            DataCopyPad(target, flags_, {uint16_t(flags), 32U, 0U, 480U, 0U});
        }
    }
    __aicore__ inline void Load(uint32_t token, uint32_t k, uint32_t member)
    {
        const uint32_t lane = (inputSequence_ + member) % inputLanes_;
        WaitFlag<HardEvent::V_MTE2>(lane);
        const uint32_t retire = enableDedup_ ? finalLayout_.inputRetireMasks[lane] & ~retiredOutputs_ : 0U;
        if (retire != 0U) {
            for (uint32_t slot = 0; slot < sharedOutputLanes_; ++slot)
                if ((retire >> slot) & 1U) WaitFlag<HardEvent::MTE3_V>(slot);
            // MTE3_V protects Vector, not MTE2. Bridge only first-use reclamation
            // through Vector before this DMA overwrites any inherited output.
            SetFlag<HardEvent::V_MTE2>(lane);
            WaitFlag<HardEvent::V_MTE2>(lane);
            retiredOutputs_ |= retire;
        }
        DataCopy(inputs_[finalLayout_.inputOffsets[lane] / 2U], result_[(uint64_t(token) * k_ + k) * packetBytes_ / 2U],
                 {uint16_t(blocks_), 15U, 1U, 0U});
        SetFlag<HardEvent::MTE2_V>(lane);
    }
    template <class T>
    __aicore__ inline void Aggregate(uint32_t token, uint32_t mask, uint32_t completed)
    {
        uint32_t nextK = 0U, loaded = 0U;
        for (; nextK < k_ && loaded < inputLanes_; ++nextK)
            if (!enableDedup_ || ((mask >> nextK) & 1U)) Load(token, nextK, loaded++);
        uint32_t member = 0U;
        for (uint32_t k = 0U; k < k_; ++k)
            if (!enableDedup_ || ((mask >> k) & 1U)) {
                const uint32_t lane = (inputSequence_ + member) % inputLanes_;
                WaitFlag<HardEvent::MTE2_V>(lane);
                Cast(member == 0U ? sum_ : cast_, inputs_[finalLayout_.inputOffsets[lane] / 2U].template ReinterpretCast<T>(),
                     RoundMode::CAST_NONE, h_);
                SetFlag<HardEvent::V_MTE2>(lane);
                while (enableDedup_ && nextK < k_ && ((mask >> nextK) & 1U) == 0U) ++nextK;
                if (nextK < k_) Load(token, nextK++, loaded++);
                PipeBarrier<PIPE_V>();
                if (member != 0U) Add(sum_, sum_, cast_, h_);
                PipeBarrier<PIPE_V>();
                ++member;
            }
        inputSequence_ += member;
        const uint32_t outputLane = finalLayout_.outputSlots[completed % outputLanes_];
        WaitFlag<HardEvent::MTE3_V>(outputLane);
        auto output = outputs_[outputLane * outputBytes_ / 2U];
        Cast(output.template ReinterpretCast<T>(), sum_, RoundMode::CAST_RINT, h_);
        SetFlag<HardEvent::V_MTE3>(outputLane);
        WaitFlag<HardEvent::V_MTE3>(outputLane);
        DataCopyPad(output_[uint64_t(token) * h_], output, {1U, h_ * 2U, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_V>(outputLane);
    }

    bool enableDedup_ = false;
    LocalTensor<uint8_t> scratch_;
    LocalTensor<uint32_t> masks_, pending_;
    LocalTensor<float> flags_, work_, probeSum_, cast_, sum_;
    LocalTensor<uint16_t> inputs_, outputs_;
    GlobalTensor<uint16_t> result_, output_;
    GlobalTensor<float> resultFlags_;
    uint32_t used_ = 0U, h_ = 0U, k_ = 0U, dtype_ = 0U, begin_ = 0U, count_ = 0U;
    uint32_t blocks_ = 0U, packetBytes_ = 0U, outputBytes_ = 0U;
    EpCombineM5WorkspaceLayout finalLayout_{};
    uint32_t inputLanes_ = 0U, outputLanes_ = 0U, inputSequence_ = 0U;
    uint32_t sharedOutputLanes_ = 0U, retiredOutputs_ = 0U;
};
}  // namespace CombineDedup
#endif
