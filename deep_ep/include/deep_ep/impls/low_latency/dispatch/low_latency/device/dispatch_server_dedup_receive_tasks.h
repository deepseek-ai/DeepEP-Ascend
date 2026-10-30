// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_SERVER_DEDUP_RECEIVE_TASKS_H
#define DISPATCH_SERVER_DEDUP_RECEIVE_TASKS_H

#include <kernel_operator.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h>

namespace DispatchDedup {
using namespace AscendC;

// R05-01/R05-02: stable physical slot positions and a compact pending list.
// The caller supplies the complete, ready prefix in localExpert/source order.
// Both tables remain in UB until this core has finished clearing its slots.
class ServerDedupReceiveTasks {
public:
    __aicore__ inline void Build(LocalTensor<int32_t> prefix, uint32_t rank, uint32_t ranks, uint32_t experts,
                                 uint32_t bs, uint32_t receiver, TPipe* pipe)
    {
        const uint32_t total = uint32_t(prefix.GetValue(experts - 1U));
        begin_ = uint32_t(uint64_t(receiver) * total / kEpServerDedupRecvCoreNum);
        const uint32_t end = uint32_t(uint64_t(receiver + 1U) * total / kEpServerDedupRecvCoreNum);
        count_ = end - begin_;
        pendingCount_ = count_;
        if (count_ == 0U) return;

        const uint32_t bytes = uint32_t(DispatchDedupAlign(uint64_t(count_) * sizeof(uint32_t), 32U));
        pipe->InitBuffer(tokenPosBuf_, bytes);
        pipe->InitBuffer(pendingIdxBuf_, bytes);
        tokenPos_ = tokenPosBuf_.Get<uint32_t>();
        pendingIdx_ = pendingIdxBuf_.Get<uint32_t>();

        const uint32_t localExperts = experts / ranks;
        // Upper bound skips records ending at begin_, including zero-count runs.
        // A nonempty partition has begin_ < total, so the result is < experts.
        uint32_t record = 0U, limit = experts;
        while (record < limit) {
            const uint32_t middle = record + (limit - record) / 2U;
            if (uint32_t(prefix.GetValue(middle)) <= begin_)
                record = middle + 1U;
            else
                limit = middle;
        }
        // Keep the original record start: begin_ may split its q range.
        uint32_t previous = record == 0U ? 0U : uint32_t(prefix.GetValue(record - 1U));
        for (; record < experts && previous < end; ++record) {
            const uint32_t cumulative = uint32_t(prefix.GetValue(record));
            const uint32_t first = previous > begin_ ? previous : begin_;
            const uint32_t last = cumulative < end ? cumulative : end;
            const uint32_t source = record % ranks;
            const uint32_t localExpert = record / ranks;
            const uint32_t region = ((source + rank) % ranks) * localExperts + localExpert;
            for (uint32_t row = first; row < last; ++row) {
                const uint32_t j = row - begin_;
                tokenPos_.SetValue(j, region * bs + row - previous);
                pendingIdx_.SetValue(j, j);
            }
            previous = cumulative;
        }
    }

    __aicore__ inline uint32_t Begin() const
    {
        return begin_;
    }
    __aicore__ inline uint32_t Count() const
    {
        return count_;
    }
    __aicore__ inline uint32_t PendingCount() const
    {
        return pendingCount_;
    }
    __aicore__ inline uint32_t Position(uint32_t j)
    {
        return tokenPos_.GetValue(j);
    }
    __aicore__ inline uint32_t PendingIndex(uint32_t p)
    {
        return pendingIdx_.GetValue(p);
    }

    // Call only after every required write for PendingIndex(p) has been issued.
    // Recheck p after removal; output row and cleanup ownership still use j.
    __aicore__ inline void RemovePending(uint32_t p)
    {
        --pendingCount_;
        pendingIdx_.SetValue(p, pendingIdx_.GetValue(pendingCount_));
    }

private:
    uint32_t begin_ = 0U, count_ = 0U, pendingCount_ = 0U;
    TBuf<> tokenPosBuf_, pendingIdxBuf_;
    LocalTensor<uint32_t> tokenPos_, pendingIdx_;
};
}  // namespace DispatchDedup

#endif
