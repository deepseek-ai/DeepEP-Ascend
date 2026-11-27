#pragma once

#include <climits>

#include <kernel_operator.h>
#include <simt_api/asc_simt.h>

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/layout/ep/lb.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/simt.hpp>

namespace deep_ep {

// One domain's aggregation and planner state, kept in UB on AIV 0.
template <int kNumExperts, int kNumRanks, int kNumThreads>
struct LBPlanUB {
    int num_total_tokens[kNumThreads];  // Partial totals per rank group
    int num_tokens_before_rank[kNumThreads];  // Partial pre-rank prefixes per rank group
    int num_tokens_per_expert[kNumExperts];  // Domain-wide token count per expert
    int num_main_loads[kNumExperts];  // Aligned loads this rank keeps before donating
    int num_redundant_loads[kNumExperts];  // Aligned loads each receiver pulls from its donor
    int redundant_slot_idx[kNumExperts];  // Redundant slot on the receiver for each donated expert
    int donor_rank_idx[kNumRanks];  // Every receiver's donor, or -1
    int num_chain_loads[kNumRanks];  // Aligned loads a receiver pulls from its donor
};

// Per-AIV UB scratch shared across the sequential count/publish, plan, and remap stages.
template <int kNumExperts, int kNumRanks, int kNumThreads>
union alignas(kNumUbAlignmentBytes) LBPrepareUB {
    int scratch[(kNumThreads / warpSize) * kNumExperts];
    LBPlanUB<kNumExperts, kNumRanks, kNumThreads> plan;
};

// Count routes per (block, expert), turn the block counts into offsets, and publish each expert's total to every peer.
template <int kNumRanks, int kNumExperts, int kNumVecCores, int kNumThreads>
__simt_vf__ __aicore__ __launch_bounds__(kNumThreads) void lb_count_and_publish(
    __gm__ const int64_t* topk_idx, __gm__ const layout::DeviceContext* device_context, const int num_routes,
    const int num_blocks, const int rank_idx, const int vec_core_idx, __gm__ uint32_t* grid_sync_count,
    __gm__ int* block_expert_histogram, __gm__ int* expert_token_ordinal_in_block,
    __gm__ int* local_num_tokens_per_expert, __ubuf__ int* ub_scratch) {
    constexpr int kNumWarps = kNumThreads / warpSize;
    const int thread_idx = static_cast<int>(threadIdx.x);
    const int warp_idx = thread_idx / warpSize;
    const int lane_idx = thread_idx % warpSize;
    for (int block_idx = vec_core_idx; block_idx < num_blocks; block_idx += kNumVecCores) {
        for (int cell_idx = thread_idx; cell_idx < kNumWarps * kNumExperts; cell_idx += kNumThreads)
            ub_scratch[cell_idx] = 0;
        asc_threadfence_block();
        asc_syncthreads();

        const int route_idx = block_idx * kNumThreads + thread_idx;
        const auto value = route_idx < num_routes ? topk_idx[route_idx] : int64_t(-1);
        if (value >= kNumExperts)
            record_error_and_trap_simt(
                layout::DeviceContext::get_error_status(*device_context), layout::kInvalidCommMetadata);
        const int route_expert_idx = value >= 0 ? static_cast<int>(value) : -1;
        auto matching_lane_mask = asc_ballot(route_expert_idx >= 0);
        #pragma unroll 1
        for (int bit = 0; (1u << bit) < kNumExperts; ++ bit) {
            const auto value_bit = (static_cast<uint32_t>(route_expert_idx) >> bit) & 1u;
            matching_lane_mask &= asc_ballot(value_bit) ^ (value_bit - 1u);
        }
        const int warp_expert_ordinal = __popc(matching_lane_mask & ((1u << lane_idx) - 1));
        if (route_expert_idx >= 0 and lane_idx == __ffs(static_cast<int>(matching_lane_mask)) - 1)
            ub_scratch[warp_idx * kNumExperts + route_expert_idx] = __popc(matching_lane_mask);
        asc_threadfence_block();
        asc_syncthreads();

        // Turn the per-warp counts into this block's offsets, and store the block count in `[expert][block]` order
        for (int expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumThreads) {
            int num_total_tokens = 0;
            #pragma unroll
            for (int source_warp_idx = 0; source_warp_idx < kNumWarps; ++ source_warp_idx) {
                const int num_tokens = ub_scratch[source_warp_idx * kNumExperts + expert_idx];
                ub_scratch[source_warp_idx * kNumExperts + expert_idx] = num_total_tokens;
                num_total_tokens += num_tokens;
            }
            asc_stcg(block_expert_histogram + expert_idx * num_blocks + block_idx, num_total_tokens);
        }
        asc_threadfence_block();
        asc_syncthreads();
        if (route_expert_idx >= 0)
            expert_token_ordinal_in_block[route_idx] =
                warp_expert_ordinal + ub_scratch[warp_idx * kNumExperts + route_expert_idx];
        asc_threadfence_block();
        asc_syncthreads();
    }
    asc_threadfence();
    comm::simt::grid_sync<kNumVecCores>(grid_sync_count, vec_core_idx);

    for (int expert_begin = vec_core_idx * kNumWarps; expert_begin < kNumExperts;
         expert_begin += kNumVecCores * kNumWarps) {
        const int expert_idx = expert_begin + warp_idx;
        int num_total_tokens = 0;
        if (expert_idx < kNumExperts) {
            for (int block_begin = 0; block_begin < num_blocks; block_begin += warpSize) {
                const int block_idx = block_begin + lane_idx;
                const int num_tokens = block_idx < num_blocks ?
                    asc_ldcg(block_expert_histogram + expert_idx * num_blocks + block_idx) : 0;
                const int num_inclusive_tokens = simt::warp_inclusive_sum(num_tokens, lane_idx);
                if (block_idx < num_blocks)
                    asc_stcg(block_expert_histogram + expert_idx * num_blocks + block_idx,
                             num_total_tokens + num_inclusive_tokens - num_tokens);
                num_total_tokens += simt::exchange(num_inclusive_tokens, warpSize - 1);
            }
            if (lane_idx == 0)
                ub_scratch[warp_idx] = num_total_tokens;
        }
        asc_threadfence_block();
        asc_syncthreads();

        // Transpose through UB so each warp writes 32 contiguous experts to one peer rank
        const int dst_expert_idx = expert_begin + lane_idx;
        if (dst_expert_idx < kNumExperts) {
            const int num_dst_tokens = ub_scratch[lane_idx];
            if (warp_idx == 0)
                asc_stcg(local_num_tokens_per_expert + dst_expert_idx, num_dst_tokens);
            const auto& base_signals = *reinterpret_cast<__gm__ const layout::BarrierSignals*>(
                device_context->scaleup_peer_ptrs[rank_idx]);
            for (int peer_rank_idx = warp_idx; peer_rank_idx < kNumRanks; peer_rank_idx += kNumWarps) {
                if (peer_rank_idx != rank_idx) {
                    auto* peer_num_tokens_per_expert = layout::get_sym_ptr(
                        base_signals, *device_context, local_num_tokens_per_expert, peer_rank_idx);
                    asc_stcg(peer_num_tokens_per_expert + dst_expert_idx, num_dst_tokens);
                }
            }
        }
        asc_threadfence_block();
        asc_syncthreads();
    }
    asc_threadfence();
}

// Sum gathered counts, balance the aligned loads, and compact each expert's active destination segments.
template <int kNumRanks, int kNumMainExpertsPerRank, int kNumRedundantExpertsPerRank, int kTokenAlignment, int kNumThreads>
__simt_vf__ __aicore__ __launch_bounds__(kNumThreads) void lb_plan(
    __gm__ const int* num_tokens_per_rank_expert, const int local_rank_idx, __gm__ int* expert_token_offset,
    __gm__ int* physical_expert_idx, __gm__ int* psum_num_tokens_per_expert_rank, __gm__ int* redundancy_mapping,
    __ubuf__ LBPlanUB<kNumRanks * kNumMainExpertsPerRank, kNumRanks, kNumThreads>* plan) {
    constexpr int kNumExperts = kNumRanks * kNumMainExpertsPerRank;
    constexpr int kNumPhysicalExpertsPerRank = kNumMainExpertsPerRank + kNumRedundantExpertsPerRank;
    constexpr int kNumWarps = kNumThreads / warpSize;
    constexpr int kNumRankSlots = math::ceil_div(kNumRanks, warpSize);
    constexpr int kNumGroups = 4;
    constexpr int kNumExpertsPerGroup = kNumThreads / kNumGroups;
    const int thread_idx = static_cast<int>(threadIdx.x);
    const int warp_idx = thread_idx / warpSize;
    const int lane_idx = thread_idx % warpSize;
    const int group_idx = thread_idx / kNumExpertsPerGroup;
    const int expert_idx_in_group = thread_idx % kNumExpertsPerGroup;

    // Pick the largest value across the warp and break ties by smallest key
    auto warp_max_with_key = [](int& value, int& key) __simt_callee__ {
        const int best_value = asc_reduce_max(value);
        key = asc_reduce_min(value == best_value ? key : INT_MAX);
        value = best_value;
    };

    // Sum gathered counts across source ranks, record this rank's prefix, and initialize home aligned loads
    for (int expert_begin_idx = 0; expert_begin_idx < kNumExperts; expert_begin_idx += kNumExpertsPerGroup) {
        const int expert_idx = expert_begin_idx + expert_idx_in_group;
        int num_total_tokens = 0;
        int num_tokens_before_rank = 0;
        if (expert_idx < kNumExperts) {
            for (int source_rank_idx = group_idx; source_rank_idx < kNumRanks; source_rank_idx += kNumGroups) {
                const int num_tokens = asc_ldcg(const_cast<__gm__ int*>(num_tokens_per_rank_expert) +
                                                source_rank_idx * kNumExperts + expert_idx);
                num_total_tokens += num_tokens;
                if (source_rank_idx < local_rank_idx)
                    num_tokens_before_rank += num_tokens;
            }
        }
        plan->num_total_tokens[thread_idx] = num_total_tokens;
        plan->num_tokens_before_rank[thread_idx] = num_tokens_before_rank;
        asc_threadfence_block();
        asc_syncthreads();
        if (thread_idx < kNumExpertsPerGroup and expert_idx < kNumExperts) {
            #pragma unroll
            for (int source_group_idx = 1; source_group_idx < kNumGroups; ++ source_group_idx) {
                num_total_tokens += plan->num_total_tokens[source_group_idx * kNumExpertsPerGroup + expert_idx_in_group];
                num_tokens_before_rank += plan->num_tokens_before_rank[source_group_idx * kNumExpertsPerGroup + expert_idx_in_group];
            }
            plan->num_tokens_per_expert[expert_idx] = num_total_tokens;
            plan->num_main_loads[expert_idx] = static_cast<int>(
                math::ceil_div<int64_t>(num_total_tokens, kTokenAlignment));
            plan->num_redundant_loads[expert_idx] = 0;
            asc_stcg(expert_token_offset + expert_idx, num_tokens_before_rank);
        }
        asc_threadfence_block();
        asc_syncthreads();
    }
    for (int cell_idx = thread_idx; cell_idx < kNumRanks * kNumRedundantExpertsPerRank; cell_idx += kNumThreads)
        asc_stcg(redundancy_mapping + cell_idx, -1);

    // Pair each deficit receiver with one surplus donor and record its requested chain loads
    if (warp_idx == 0) {
        int load_imbalance[kNumRankSlots];
        int local_donor_rank_idx[kNumRankSlots];
        int local_num_chain_loads[kNumRankSlots];
        int num_total_loads = 0;
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
            const int rank_idx = slot_idx * warpSize + lane_idx;
            int num_rank_loads = 0;
            if (rank_idx < kNumRanks) {
                for (int local_expert_idx = 0; local_expert_idx < kNumMainExpertsPerRank; ++ local_expert_idx)
                    num_rank_loads += plan->num_main_loads[rank_idx * kNumMainExpertsPerRank + local_expert_idx];
            }
            load_imbalance[slot_idx] = num_rank_loads;
            local_donor_rank_idx[slot_idx] = -1;
            local_num_chain_loads[slot_idx] = 0;
            num_total_loads += num_rank_loads;
        }
        num_total_loads = asc_reduce_add(num_total_loads);
        const int num_avg_loads = num_total_loads / kNumRanks;
        const int num_remainder = num_total_loads % kNumRanks;
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
            const int rank_idx = slot_idx * warpSize + lane_idx;
            if (rank_idx < kNumRanks)
                load_imbalance[slot_idx] -= num_avg_loads + (rank_idx < num_remainder);
        }

        // One donor per receiver; a donor above the fair share keeps draining while it can cover the next deficit
        int sticky_donor_idx = -1;
        int num_surplus_loads = 0;
        #pragma unroll 1
        for (int step_idx = 0; step_idx < kNumRanks; ++ step_idx) {
            // Both picks read this step's snapshot (`imbalance < min_imbalance` avoids `bisheng`'s broken `ISETP -R`)
            int min_imbalance = 0;
            int receiver_idx = kNumRanks;
            int num_best_surplus_loads = 0;
            int best_donor_idx = kNumRanks;
            #pragma unroll
            for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                const int rank_idx = slot_idx * warpSize + lane_idx;
                const int imbalance = rank_idx < kNumRanks ? load_imbalance[slot_idx] : 0;
                if (imbalance < min_imbalance) {
                    min_imbalance = imbalance;
                    receiver_idx = rank_idx;
                }
                if (imbalance > num_best_surplus_loads) {
                    num_best_surplus_loads = imbalance;
                    best_donor_idx = rank_idx;
                }
            }
            const int best_min_imbalance = asc_reduce_min(min_imbalance);
            receiver_idx = asc_reduce_min(min_imbalance == best_min_imbalance ? receiver_idx : INT_MAX);
            warp_max_with_key(num_best_surplus_loads, best_donor_idx);
            const bool needs_donor =
                num_surplus_loads + num_surplus_loads / kNumMainExpertsPerRank + best_min_imbalance < 0;
            if (best_min_imbalance >= 0 or (needs_donor and num_best_surplus_loads <= 0))
                break;
            if (needs_donor) {
                sticky_donor_idx = best_donor_idx;
                num_surplus_loads = num_best_surplus_loads > num_avg_loads ? num_best_surplus_loads : 0;
            }
            const int num_deficit_loads = -best_min_imbalance;
            num_surplus_loads -= num_deficit_loads;
            #pragma unroll
            for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                const int rank_idx = slot_idx * warpSize + lane_idx;
                if (rank_idx == sticky_donor_idx)
                    load_imbalance[slot_idx] -= num_deficit_loads;
                if (rank_idx == receiver_idx) {
                    load_imbalance[slot_idx] += num_deficit_loads;
                    local_donor_rank_idx[slot_idx] = sticky_donor_idx;
                    local_num_chain_loads[slot_idx] = num_deficit_loads;
                }
            }
        }
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
            const int rank_idx = slot_idx * warpSize + lane_idx;
            if (rank_idx < kNumRanks) {
                plan->donor_rank_idx[rank_idx] = local_donor_rank_idx[slot_idx];
                plan->num_chain_loads[rank_idx] = local_num_chain_loads[slot_idx];
            }
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Each donor warp walks its own receivers, largest request first, and hands over whole experts
    if constexpr (kNumRedundantExpertsPerRank > 0) {
        for (int donor_idx = warp_idx; donor_idx < kNumRanks; donor_idx += kNumWarps) {
            while (true) {
                int num_remaining_loads = 0;
                int receiver_idx = kNumRanks;
                #pragma unroll
                for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                    const int rank_idx = slot_idx * warpSize + lane_idx;
                    const int num_loads = rank_idx < kNumRanks and plan->donor_rank_idx[rank_idx] == donor_idx ?
                        plan->num_chain_loads[rank_idx] : 0;
                    if (num_loads > num_remaining_loads) {
                        num_remaining_loads = num_loads;
                        receiver_idx = rank_idx;
                    }
                }
                warp_max_with_key(num_remaining_loads, receiver_idx);
                if (num_remaining_loads <= 0)
                    break;
                if (lane_idx == 0)
                    plan->num_chain_loads[receiver_idx] = 0;
                int redundant_expert_idx = 0;
                while (num_remaining_loads > 0 and redundant_expert_idx < kNumRedundantExpertsPerRank) {
                    int num_max_available_loads = 0;
                    int best_local_expert_idx = kNumMainExpertsPerRank;
                    for (int local_expert_idx = lane_idx; local_expert_idx < kNumMainExpertsPerRank; local_expert_idx += warpSize) {
                        const int num_available_loads = plan->num_main_loads[donor_idx * kNumMainExpertsPerRank + local_expert_idx];
                        if (num_available_loads > num_max_available_loads) {
                            num_max_available_loads = num_available_loads;
                            best_local_expert_idx = local_expert_idx;
                        }
                    }
                    warp_max_with_key(num_max_available_loads, best_local_expert_idx);
                    const int num_taken_loads = math::min(num_remaining_loads, num_max_available_loads);
                    const int expert_idx = donor_idx * kNumMainExpertsPerRank + best_local_expert_idx;
                    if (lane_idx == 0) {
                        plan->num_main_loads[expert_idx] = num_max_available_loads - num_taken_loads;
                        plan->num_redundant_loads[receiver_idx * kNumMainExpertsPerRank + best_local_expert_idx] = num_taken_loads;
                        plan->redundant_slot_idx[receiver_idx * kNumMainExpertsPerRank + best_local_expert_idx] = redundant_expert_idx;
                        asc_stcg(redundancy_mapping + receiver_idx * kNumRedundantExpertsPerRank + redundant_expert_idx, expert_idx);
                    }
                    asc_threadfence_block();
                    num_remaining_loads -= num_taken_loads;
                    ++ redundant_expert_idx;
                }
            }
        }
        asc_threadfence_block();
        asc_syncthreads();
    }

    // Compact each expert's active destination segments and cumulative token thresholds, one warp per expert
    for (int expert_idx = warp_idx; expert_idx < kNumExperts; expert_idx += kNumWarps) {
        const int owner_rank_idx = expert_idx / kNumMainExpertsPerRank;
        const int local_expert_idx = expert_idx % kNumMainExpertsPerRank;
        const int home_physical_expert_idx = owner_rank_idx * kNumPhysicalExpertsPerRank + local_expert_idx;
        int tail_rank_idx = owner_rank_idx;
        if (plan->num_main_loads[expert_idx] == 0) {
            int num_max_loads = 0;
            int best_rank_idx = kNumRanks;
            #pragma unroll
            for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                const int rank_idx = slot_idx * warpSize + lane_idx;
                const int num_rank_loads = rank_idx < kNumRanks and plan->donor_rank_idx[rank_idx] == owner_rank_idx ?
                    plan->num_redundant_loads[rank_idx * kNumMainExpertsPerRank + local_expert_idx] : 0;
                if (num_rank_loads > num_max_loads) {
                    num_max_loads = num_rank_loads;
                    best_rank_idx = rank_idx;
                }
            }
            warp_max_with_key(num_max_loads, best_rank_idx);
            if (num_max_loads > 0)
                tail_rank_idx = best_rank_idx;
        }
        const int num_tokens = plan->num_tokens_per_expert[expert_idx];
        const int num_padding_tokens = num_tokens % kTokenAlignment == 0 ? 0 : kTokenAlignment - num_tokens % kTokenAlignment;
        int num_previous_rank_tokens = 0;
        int num_active_segments = 0;
        for (int rank_begin = 0; rank_begin < kNumRanks; rank_begin += warpSize) {
            const int rank_idx = rank_begin + lane_idx;
            const int num_copied_loads = rank_idx < kNumRanks and plan->donor_rank_idx[rank_idx] == owner_rank_idx ?
                plan->num_redundant_loads[rank_idx * kNumMainExpertsPerRank + local_expert_idx] : 0;
            const int dst_physical_expert_idx = num_copied_loads > 0 ?
                rank_idx * kNumPhysicalExpertsPerRank + kNumMainExpertsPerRank +
                    plan->redundant_slot_idx[rank_idx * kNumMainExpertsPerRank + local_expert_idx] :
                home_physical_expert_idx;
            const int num_rank_loads = (rank_idx == owner_rank_idx ? plan->num_main_loads[expert_idx] : 0) + num_copied_loads;
            int num_rank_tokens = static_cast<int>(
                static_cast<int64_t>(num_rank_loads) * kTokenAlignment - (rank_idx == tail_rank_idx ? num_padding_tokens : 0));
            const uint32_t active_mask = asc_ballot(num_rank_tokens > 0);
            const int segment_idx = num_active_segments + __popc(active_mask & ((1u << lane_idx) - 1));
            num_rank_tokens = simt::warp_inclusive_sum(num_rank_tokens, lane_idx);
            if (active_mask & (1u << lane_idx)) {
                asc_stcg(physical_expert_idx + expert_idx * kNumRanks + segment_idx, dst_physical_expert_idx);
                asc_stcg(psum_num_tokens_per_expert_rank + expert_idx * kNumRanks + segment_idx,
                         num_previous_rank_tokens + num_rank_tokens);
            }
            num_previous_rank_tokens += simt::exchange(num_rank_tokens, warpSize - 1);
            num_active_segments += __popc(active_mask);
        }
    }
    asc_threadfence();
}

// Point every route at the physical expert that owns its stable index in that expert's sequence.
template <int kNumExperts, int kNumRanks, int kNumVecCores, int kNumThreads>
__simt_vf__ __aicore__ __launch_bounds__(kNumThreads) void lb_remap(
    __gm__ const int64_t* topk_idx, __gm__ const int* block_expert_histogram, __gm__ const int* expert_token_ordinal_in_block,
    __gm__ const int* local_num_tokens_per_expert, __gm__ const int* expert_token_offset, __gm__ const int* physical_expert_idx,
    __gm__ const int* psum_num_tokens_per_expert_rank, const int num_routes, const int num_blocks,
    const int vec_core_idx, __gm__ int64_t* remapped_topk_idx, __ubuf__ int* ub_expert_scratch) {
    __ubuf__ int* ub_expert_token_offset = ub_expert_scratch;
    __ubuf__ int* ub_single_physical_expert_idx = ub_expert_token_offset + kNumExperts;
    __ubuf__ int* ub_start_segment_idx = ub_single_physical_expert_idx + kNumExperts;

    // Pre-advance each expert's segment pointer to this rank's starting offset
    const int thread_idx = static_cast<int>(threadIdx.x);
    for (int expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumThreads) {
        const int num_local_tokens = asc_ldcg(const_cast<__gm__ int*>(local_num_tokens_per_expert) + expert_idx);
        int first_ordinal = 0;
        int start_segment_idx = 0;
        int single_physical_expert_idx = 0;
        if (num_local_tokens > 0) {
            first_ordinal = asc_ldcg(const_cast<__gm__ int*>(expert_token_offset) + expert_idx);
            const int last_ordinal = first_ordinal + num_local_tokens - 1;
            while (asc_ldcg(const_cast<__gm__ int*>(psum_num_tokens_per_expert_rank) + expert_idx * kNumRanks + start_segment_idx) <= first_ordinal)
                ++ start_segment_idx;
            single_physical_expert_idx =
                last_ordinal < asc_ldcg(const_cast<__gm__ int*>(psum_num_tokens_per_expert_rank) + expert_idx * kNumRanks + start_segment_idx) ?
                asc_ldcg(const_cast<__gm__ int*>(physical_expert_idx) + expert_idx * kNumRanks + start_segment_idx) : -1;
        }
        ub_expert_token_offset[expert_idx] = first_ordinal;
        ub_single_physical_expert_idx[expert_idx] = single_physical_expert_idx;
        ub_start_segment_idx[expert_idx] = start_segment_idx;
    }
    asc_threadfence_block();
    asc_syncthreads();

    for (int route_idx = vec_core_idx * kNumThreads + thread_idx; route_idx < num_routes;
         route_idx += kNumVecCores * kNumThreads) {
        const auto value = topk_idx[route_idx];
        if (value < 0) {
            remapped_topk_idx[route_idx] = value;
            continue;
        }
        const int expert_idx = static_cast<int>(value);
        int dst_physical_expert_idx = ub_single_physical_expert_idx[expert_idx];
        if (dst_physical_expert_idx < 0) {
            const int block_expert_offset = asc_ldcg(const_cast<__gm__ int*>(block_expert_histogram) +
                                                     expert_idx * num_blocks + route_idx / kNumThreads);
            const int ordinal = ub_expert_token_offset[expert_idx] + block_expert_offset +
                                expert_token_ordinal_in_block[route_idx];
            int segment_idx = ub_start_segment_idx[expert_idx];
            while (asc_ldcg(const_cast<__gm__ int*>(psum_num_tokens_per_expert_rank) + expert_idx * kNumRanks + segment_idx) <= ordinal)
                ++ segment_idx;
            dst_physical_expert_idx = asc_ldcg(const_cast<__gm__ int*>(physical_expert_idx) + expert_idx * kNumRanks + segment_idx);
        }
        remapped_topk_idx[route_idx] = static_cast<int64_t>(dst_physical_expert_idx);
    }
    asc_threadfence();
}

template <int kNumRanks, int kNumScaleupRanks, int kNumMainExpertsPerRank, int kNumRedundantExpertsPerRank,
          int kTokenAlignment, int kNumVecCores, int kNumThreads, int64_t kNumTimeoutCycles>
__global__ __vector__ void lb_get_redundancy_mapping_impl(
    __gm__ const int64_t* topk_idx, const int num_routes, const int rank_idx,
    __gm__ int* redundancy_mapping, __gm__ int64_t* remapped_topk_idx,
    __gm__ void* workspace, __gm__ void* device_context_ptr) {
    constexpr int kNumExperts = kNumRanks * kNumMainExpertsPerRank;
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks <= kNumMaxEPRanks);
    EP_STATIC_ASSERT(kNumRanks == kNumScaleupRanks, "LB prepare requires a single supernode");

    AscendC::InitSocState();
    const int num_blocks = math::ceil_div(num_routes, kNumThreads);
    EP_DEVICE_ASSERT(num_blocks * kNumExperts <= layout::LBPrepareSignals::kNumMaxPrepareRoutes);
    const int vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    auto& signals = static_cast<__gm__ layout::EPSignals*>(workspace)->lb_prepare_signals;
    auto& device_context = *static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const int epoch = static_cast<int>(AscendC::ReadGmByPassDCache(&signals.barrier_signals.scaleup_barrier_counter));
    __gm__ int* num_tokens_per_rank_expert = signals.num_tokens_per_rank_expert[epoch & 1];
    __gm__ int* local_num_tokens_per_expert = num_tokens_per_rank_expert + rank_idx * kNumExperts;
    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_storage[sizeof(LBPrepareUB<kNumExperts, kNumRanks, kNumThreads>)];
    auto& ub = *reinterpret_cast<__ubuf__ LBPrepareUB<kNumExperts, kNumRanks, kNumThreads>*>(ub_storage);

    // Count local routes and publish per-expert totals to all peers before every rank plans
    asc_vf_call<lb_count_and_publish<kNumRanks, kNumExperts, kNumVecCores, kNumThreads>>(
        dim3(kNumThreads), topk_idx, &device_context, num_routes, num_blocks, rank_idx, vec_core_idx,
        &signals.barrier_signals.grid_sync_count, signals.block_expert_histogram,
        signals.expert_token_ordinal_in_block, local_num_tokens_per_expert, ub.scratch);
    asc_sync_pipe(PIPE_ALL);
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles, false, true, false, kNumVecCores, 0>(
        signals.barrier_signals, device_context, vec_core_idx, rank_idx);

    // Aggregate and plan on AIV 0, then remap in parallel across all AIVs
    if (vec_core_idx == 0) {
        asc_vf_call<lb_plan<kNumRanks, kNumMainExpertsPerRank, kNumRedundantExpertsPerRank, kTokenAlignment, kNumThreads>>(
            dim3(kNumThreads), num_tokens_per_rank_expert, rank_idx, signals.expert_token_offset,
            signals.physical_expert_idx, signals.psum_num_tokens_per_expert_rank, redundancy_mapping, &ub.plan);
        asc_sync_pipe(PIPE_ALL);
    }
    AscendC::SyncAll<true>();
    asc_vf_call<lb_remap<kNumExperts, kNumRanks, kNumVecCores, kNumThreads>>(
        dim3(kNumThreads), topk_idx, signals.block_expert_histogram, signals.expert_token_ordinal_in_block,
        local_num_tokens_per_expert, signals.expert_token_offset, signals.physical_expert_idx,
        signals.psum_num_tokens_per_expert_rank, num_routes, num_blocks, vec_core_idx, remapped_topk_idx,
        ub.scratch);
    asc_sync_pipe(PIPE_ALL);
}

}  // namespace deep_ep
