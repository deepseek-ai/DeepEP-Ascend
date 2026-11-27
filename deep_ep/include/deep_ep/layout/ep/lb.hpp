#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/barrier.hpp>

#ifdef __CCE__
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/common/simt.hpp>
#include <simt_api/device_sync_functions.h>
#endif
#include <deep_ep/common/math.hpp>

namespace deep_ep::layout {

// One exchange pair between an original expert and a redundant expert slot
struct EPLBTask {
    int local_expert_idx;
    int dst_rank_idx;
    int redundant_expert_idx;
};

static constexpr int kNumLBUbBytes = 64 * 1024;

#ifdef __CCE__
struct alignas(kNumUbAlignmentBytes) EPLBUBLayout {
    static constexpr int kNumBytes = kNumLBUbBytes;
    static constexpr int kNumMaxWarps = 32;
    static constexpr int kNumMaxTasks =
        (kNumBytes - math::align<int>(kNumMaxEPRanks * sizeof(handle::HcommPeerInfo) + sizeof(handle::HcommJettyInfo), kNumUbAlignmentBytes) -
         handle::HcommJetty::kNumMaxSQEBytes - (kNumMaxWarps + 1) * sizeof(int)) / sizeof(EPLBTask);

    handle::HcommPeerInfo peer_infos[kNumMaxEPRanks];
    handle::HcommJettyInfo jetty_info;
    alignas(kNumUbAlignmentBytes) uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];
    int warp_counts[kNumMaxWarps];
    int num_tasks;
    EPLBTask tasks[kNumMaxTasks];

    template <int kNumRanks, int kNumLocalExperts, int kNumRedundantExpertsPerRank, int kNumWarps, bool kSlotMajor = false>
    __simt_vf__ __aicore__ __launch_bounds__(kNumWarps * warpSize) static void build_tasks(
        __gm__ const int* redundancy_mapping, const int rank_idx, __gm__ ErrorStatusSlot* error_status,
        __ubuf__ EPLBUBLayout* ub) {
        EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks <= kNumMaxEPRanks);
        EP_STATIC_ASSERT(kNumLocalExperts > 0);
        EP_STATIC_ASSERT(kNumWarps > 0 and kNumWarps <= kNumMaxWarps);
        constexpr int kNumThreads = kNumWarps * warpSize;
        const int thread_idx = threadIdx.x;
        const int warp_idx = thread_idx / warpSize;
        const int lane_idx = thread_idx % warpSize;

        // Prefetch uses rank/slot order; reduction interleaves rotated ranks within each slot.
        int num_found_tasks = 0;
        for (int base = 0; base < kNumRanks * kNumRedundantExpertsPerRank; base += kNumThreads) {
            int i = base + thread_idx;
            if constexpr (kSlotMajor) {
                const int peer = i % kNumRanks;
                const int slot = i / kNumRanks;
                i = i < kNumRanks * kNumRedundantExpertsPerRank ?
                    (peer + rank_idx + 1) % kNumRanks * kNumRedundantExpertsPerRank + slot : i;
            }
            const int expert_idx = i < kNumRanks * kNumRedundantExpertsPerRank ? redundancy_mapping[i] : -1;
            if (expert_idx < -1 or expert_idx >= kNumRanks * kNumLocalExperts)
                record_error_and_trap_simt(error_status, kInvalidCommMetadata);
            const int is_local = expert_idx >= 0 and expert_idx / kNumLocalExperts == rank_idx;
            const int inclusive_sum = simt::warp_inclusive_sum(is_local, lane_idx);
            if (lane_idx == warpSize - 1)
                ub->warp_counts[warp_idx] = inclusive_sum;
            asc_threadfence_block();
            asc_syncthreads();

            int task_idx = num_found_tasks + inclusive_sum - is_local;
            int num_iter_found_tasks = 0;
            for (int w = 0; w < kNumWarps; ++ w) {
                const int count = ub->warp_counts[w];
                if (w < warp_idx)
                    task_idx += count;
                num_iter_found_tasks += count;
            }
            if (is_local) {
                if (task_idx >= kNumMaxTasks)
                    record_error_and_trap_simt(error_status, kSQCapacityExceeded);
                ub->tasks[task_idx].local_expert_idx = expert_idx % kNumLocalExperts;
                ub->tasks[task_idx].dst_rank_idx = i / kNumRedundantExpertsPerRank;
                ub->tasks[task_idx].redundant_expert_idx = i % kNumRedundantExpertsPerRank;
            }
            num_found_tasks += num_iter_found_tasks;
            asc_syncthreads();
        }
        if (thread_idx == 0)
            ub->num_tasks = num_found_tasks;
    }
};
EP_STATIC_ASSERT(sizeof(EPLBUBLayout) <= EPLBUBLayout::kNumBytes, "LB metadata exceeds UB capacity");
#endif

struct LBPrepareSignals {
    static constexpr int kNumMaxPrepareRoutes = 262144;

    BarrierSignals barrier_signals;
    // Alternate between two halves across calls so a fast rank never overwrites a peer still in `lb_plan`
    int num_tokens_per_rank_expert[2][kNumMaxPrepareRoutes];
    int expert_token_offset[kNumMaxExperts];
    int physical_expert_idx[kNumMaxPrepareRoutes];
    int psum_num_tokens_per_expert_rank[kNumMaxPrepareRoutes];
    int block_expert_histogram[kNumMaxPrepareRoutes];
    int expert_token_ordinal_in_block[kNumMaxPrepareRoutes];
};

static constexpr int kNumMaxWeightEntries = 16;

struct EPWeight {
    void* redundant_expert_weights;
    void* expert_weights;
    int64_t num_bytes_per_expert;
};

struct EPWeightList {
    EPWeight weights[kNumMaxWeightEntries] = {};

    __forceinline__ __aicore__ EPWeight& operator[](const int& weight_idx) {
        return weights[weight_idx];
    }

    __forceinline__ __aicore__ const EPWeight& operator[](const int& weight_idx) const {
        return weights[weight_idx];
    }
};

#ifdef __CCE__
// Stripe chunks across AIVs in weight, chunk, task order
template <int kNumWeights, int kNumBytesPerChunk>
struct EPWeightIterator {
    const __ubuf__ EPLBUBLayout& ubuf_layout;
    const EPWeight* weights;
    const int num_tasks;
    const int num_blocks;

    // States, valid after a successful `next()`
    int num_chunk_bytes;
    __gm__ void* src_ptr;
    __gm__ void* dst_ptr;
    int dst_rank_idx;

    __forceinline__ __aicore__ EPWeightIterator(const __ubuf__ EPLBUBLayout& ubuf_layout,
                                                const EPWeight* weights,
                                                const int block_idx, const int num_blocks):
        ubuf_layout(ubuf_layout), weights(weights), num_tasks(ubuf_layout.num_tasks),
        num_blocks(num_blocks), next_chunk_idx(block_idx) {
    }

    __forceinline__ __aicore__ bool next() {
        // Advance weight monotonically
        while (weight_idx < kNumWeights) {
            const int64_t num_chunks = math::ceil_div<int64_t>(weights[weight_idx].num_bytes_per_expert, kNumBytesPerChunk) * num_tasks;
            if (next_chunk_idx < num_chunks)
                break;

            next_chunk_idx -= num_chunks;
            weight_idx ++;
        }
        if (weight_idx == kNumWeights)
            return false;

        // Locate task and resolve pointers
        const int64_t chunk_idx = next_chunk_idx;
        next_chunk_idx += num_blocks;
        const auto& [redundant_expert_weights, expert_weights, num_bytes_per_expert] = weights[weight_idx];
        const auto& [local_expert_idx, dst_rank_idx, redundant_expert_idx] = ubuf_layout.tasks[chunk_idx % num_tasks];
        const int64_t chunk_offset = static_cast<int64_t>(chunk_idx / num_tasks) * kNumBytesPerChunk;
        num_chunk_bytes = static_cast<int>(math::min(static_cast<int64_t>(kNumBytesPerChunk), num_bytes_per_expert - chunk_offset));
        src_ptr = math::advance_ptr<__gm__ void>(expert_weights, local_expert_idx * num_bytes_per_expert + chunk_offset);
        dst_ptr = math::advance_ptr<__gm__ void>(redundant_expert_weights, redundant_expert_idx * num_bytes_per_expert + chunk_offset);
        this->dst_rank_idx = dst_rank_idx;

        return true;
    }

private:
    int weight_idx = 0;
    int64_t next_chunk_idx;
};
#endif

}  // namespace deep_ep::layout
