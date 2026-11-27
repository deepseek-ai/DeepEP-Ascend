#pragma once

#include <deep_ep/common/math.hpp>
#include <deep_ep/layout/barrier.hpp>
#include <deep_ep/layout/ep/lb.hpp>

namespace deep_ep::layout {

struct alignas(kNumAllocationAlignmentBytes) EPSignals {
    BarrierSignals barrier_signals;
    // Local histogram counters occupy separate sectors.
    int local_expert_histogram[kNumMaxExperts][kNumSectorInts];
    int remote_expert_histogram[kNumMaxExperts];
    int local_rank_histogram[kNumMaxVecCores][math::align(kNumMaxEPRanks, kNumSectorInts)];
    int remote_rank_histogram[kNumMaxEPRanks];
    int host_rank_expert_count[kNumMaxEPRanks + kNumMaxExperts];
    // Rows are packed with the active number of experts per rank.
    int epilogue_expert_counter[kNumMaxVecCores * kNumMaxExperts];
    LBPrepareSignals lb_prepare_signals;

#ifdef __HOST_ONLY__
    void reset() {
        barrier_signals.reset();
        lb_prepare_signals.barrier_signals.reset();
        // Dispatch clears these after consumption, which an abort can skip.
        // Host notifications and epilogue counters are initialized before use.
        ACL_CHECK(aclrtMemset(local_expert_histogram, sizeof(local_expert_histogram), 0, sizeof(local_expert_histogram)));
        ACL_CHECK(aclrtMemset(remote_expert_histogram, sizeof(remote_expert_histogram), 0, sizeof(remote_expert_histogram)));
        ACL_CHECK(aclrtMemset(local_rank_histogram, sizeof(local_rank_histogram), 0, sizeof(local_rank_histogram)));
        ACL_CHECK(aclrtMemset(remote_rank_histogram, sizeof(remote_rank_histogram), 0, sizeof(remote_rank_histogram)));
    }
#endif
};

struct EPWorkspaceLayout {
    __gm__ EPSignals& signals;

    constexpr __aicore__ __forceinline__ explicit EPWorkspaceLayout(__gm__ void* base):
        signals(*static_cast<__gm__ EPSignals*>(base)) {}

    static constexpr __aicore__ __forceinline__ int64_t get_num_bytes() {
        return sizeof(EPSignals);
    }

    constexpr __aicore__ __forceinline__ __gm__ BarrierSignals& get_barrier_signals() const {
        return signals.barrier_signals;
    }

    constexpr __aicore__ __forceinline__ __gm__ int* get_local_expert_histogram_ptr(const int& expert_idx = 0) const {
        return signals.local_expert_histogram[expert_idx];
    }

    constexpr __aicore__ __forceinline__ __gm__ int* get_remote_expert_histogram_ptr(const int& expert_idx = 0) const {
        return signals.remote_expert_histogram + expert_idx;
    }

    constexpr __aicore__ __forceinline__ __gm__ int* get_local_rank_histogram_ptr(
        const int& vec_core_idx = 0, const int& rank_idx = 0) const {
        return signals.local_rank_histogram[vec_core_idx] + rank_idx;
    }

    constexpr __aicore__ __forceinline__ __gm__ int* get_remote_rank_histogram_ptr(const int& rank_idx = 0) const {
        return signals.remote_rank_histogram + rank_idx;
    }

    constexpr __aicore__ __forceinline__ __gm__ int* get_host_rank_expert_count_ptr() const {
        return signals.host_rank_expert_count;
    }

    constexpr __aicore__ __forceinline__ volatile __gm__ int* get_host_rank_count_ptr(const int& rank_idx = 0) const {
        return signals.host_rank_expert_count + rank_idx;
    }

    constexpr __aicore__ __forceinline__ volatile __gm__ int* get_host_expert_count_ptr(const int& expert_idx = 0) const {
        return signals.host_rank_expert_count + kNumMaxEPRanks + expert_idx;
    }

    constexpr __aicore__ __forceinline__ __gm__ int* get_epilogue_expert_counter_ptr(
        const int& vec_core_idx = 0, const int& expert_idx = 0,
        const int& num_experts_per_rank = kNumMaxExperts) const {
        return signals.epilogue_expert_counter + vec_core_idx * num_experts_per_rank + expert_idx;
    }
};

}  // namespace deep_ep::layout
