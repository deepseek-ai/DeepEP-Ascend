#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>

#ifdef EP_AICPU
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace deep_ep::aicpu {

// Fixed-capacity storage with explicit in-place construction and no heap allocation.
// Construct each element once with emplace() before accessing it through operator[].
// Elements must be trivially destructible.
template <typename T, int kSize>
class InplaceArray {
    EP_STATIC_ASSERT(kSize >= 0 and std::is_trivially_destructible_v<T>);
    alignas(T) std::byte bytes[kSize > 0 ? kSize : 1][sizeof(T)];

public:
    EP_INLINE InplaceArray() = default;
    InplaceArray(const InplaceArray&) = delete;
    InplaceArray& operator=(const InplaceArray&) = delete;

    template <typename... Args>
    EP_INLINE T& emplace(const int index, Args&&... args) {
        return *::new (static_cast<void*>(bytes[index])) T(std::forward<Args>(args)...);
    }

    EP_INLINE T& operator[](const int index) {
        return *std::launder(reinterpret_cast<T*>(bytes[index]));
    }

    EP_INLINE const T& operator[](const int index) const {
        return *std::launder(reinterpret_cast<const T*>(bytes[index]));
    }
};

}  // namespace deep_ep::aicpu
#endif

namespace deep_ep {

// TODO: how about byte alignment?
// TODO: copy pipeline optimization
__aicore__ void block_copy(__gm__ uint8_t* src, __gm__ uint8_t* dst, const int64_t& num_bytes,
                          const int& block_idx, const int& num_blocks) {
    // UB staging buffer for this core
    __ubuf__ uint8_t ub[kNumBlockCopyUbBytes];

    // Stride the byte range in chunks of `kNumBlockCopyUbBytes`, round-robin across cores
    const auto stride = static_cast<int64_t>(num_blocks) * kNumBlockCopyUbBytes;
    for (int64_t offset = static_cast<int64_t>(block_idx) * kNumBlockCopyUbBytes; offset < num_bytes; offset += stride) {
        // Align with the end
        const auto num_copy_bytes = static_cast<uint32_t>(
            math::min<int64_t>(kNumBlockCopyUbBytes, num_bytes - offset));

        // GM -> UB (PIPE_MTE2)
        // TODO: what if the address is not aligned?
        asc_copy_gm2ub_align(ub, src + offset,
                             /* n_burst= */ 1, /* len_burst= */ num_copy_bytes,
                             /* left_padding_num= */ 0, /* right_padding_num= */ 0,
                             /* enable_constant_pad= */ true, /* l2_cache_mode= */ 0,
                             /* src_gap= */ 0, /* dst_gap= */ 0);

        // Wait for UB to be filled before issuing the UB -> GM store
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

        // UB -> GM (PIPE_MTE3)
        asc_copy_ub2gm_align(dst + offset, ub,
                             /* n_burst= */ 1, /* len_burst= */ num_copy_bytes,
                             /* l2_cache_mode= */ 0,
                             /* dst_gap= */ 0, /* src_gap= */ 0);

        // Wait for UB -> GM to drain before reusing the UB buffer next iteration
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

// Scales FP32 GM ranges in-place using MTE2 -> SIMD -> MTE3 pipeline and double buffering.
// The caller enqueues independent ranges with `run()`, then call `finish()` once after the
// last run before reading GM results or reusing the UB storage/events.
template <int kNumTileElems>
struct Fp32Scaler {
    EP_STATIC_ASSERT(kNumTileElems > 0, "Invalid tile size");
    EP_STATIC_ASSERT(kNumTileElems * sizeof(float) % AscendC::VECTOR_REG_WIDTH == 0,
                     "Tile size must be a multiple of the SIMD vector width");

    __ubuf__ float* ub;
    int next_tile = 0;

    // `storage` holds 2 * kNumTileElems floats, aligned to kNumUbAlignmentBytes (32 bytes).
    __aicore__ __forceinline__ explicit Fp32Scaler(__ubuf__ float* storage): ub(storage) {
        #pragma unroll
        for (int i = 0; i < 2; ++ i)
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(i));
    }

    // Returns before stores complete; GM ranges must not overlap while in flight. `ptr` needs float alignment.
    __aicore__ __forceinline__ void run(__gm__ float* ptr, const int64_t num_elems, const float scale) {
        for (int64_t offset = 0; offset < num_elems; offset += kNumTileElems) {
            const auto bytes = static_cast<uint32_t>(math::min<int64_t>(kNumTileElems, num_elems - offset) * sizeof(float));
            const auto event = static_cast<event_t>(next_tile);
            auto* data = ub + next_tile * kNumTileElems;
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event);
            asc_copy_gm2ub_align(data, ptr + offset, 1, bytes, 0, 0, true,
                                asc_load_l2_cache_mode::NOTALLOC_KEEP, 0, 0);
            asc_sync_notify(PIPE_MTE2, PIPE_V, event);
            asc_sync_wait(PIPE_MTE2, PIPE_V, event);
            // Unused lanes of a partial tile stay in UB and are never written back to GM.
            scale_vf(data, scale);
            asc_sync_notify(PIPE_V, PIPE_MTE3, event);
            asc_sync_wait(PIPE_V, PIPE_MTE3, event);
            asc_copy_ub2gm_align(ptr + offset, data, 1, bytes,
                                asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event);
            next_tile ^= 1;
        }
    }

    // Drain both UB tiles and wait on PIPE_S for the stores.
    __aicore__ __forceinline__ void finish() {
        #pragma unroll
        for (int i = 0; i < 2; ++ i)
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(i));
        asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

private:
    __simd_vf__ __forceinline__ static void scale_vf(__ubuf__ float* data, const float scale) {
        constexpr int kNumElemsPerVector = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        constexpr int kNumIterations = kNumTileElems / kNumElemsPerVector;
        const auto mask = asc_create_mask_b32(PAT_ALL);
        #pragma unroll 1
        for (int i = 0; i < kNumIterations; ++ i) {
            vector_float values;
            asc_loadalign(values, data + i * kNumElemsPerVector);
            asc_mul_scalar(values, values, scale, mask);
            asc_storealign(data + i * kNumElemsPerVector, values, mask);
        }
    }
};

// Nonuniform hierarchical ring (NHR) communication plan for N=`kNumRanks` ranks.
// Each input chunk is divided into N slices. All-reduce takes ceil(log2(N))
// reduce-scatter phases followed by the same number of all-gather phases.
//
// By default, rank i owns slice i after reduce-scatter, and a phase may transfer
// several disjoint slices. `kSwizzleOwners` permutes slice ownership to merge
// transfers. It is intended for all-reduce, since it changes reduce-scatter's
// output layout. For power-of-two groups, this permutation is bit reversal:
// with N=8, slices 0,1,2,3,4,5,6,7 are owned by ranks 0,4,2,6,1,5,3,7.
//
// Use READ_REDUCE for reduce-scatter and READ for all-gather. Each phase gives
// the peer to read from, the peer to notify when my input is ready, and merged
// slices in increasing offset order. The caller handles communication and
// synchronization; the plan only describes peer and slice selection.
//
// Slice offsets within the parent are rounded up to `kNumSliceAlignmentBytes`
// and clamped to the parent's size. This preserves the alignment of an aligned
// parent address; it does not align the parent itself. Tail slices can be empty,
// and the caller must skip their transfers. All peers must use the same parent.
template <int kNumRanks, bool kSwizzleOwners = false,
          int kNumSliceAlignmentBytes = kNumUrmaAlignmentBytes>
struct NhrPlan {
    EP_STATIC_ASSERT(kNumRanks > 0 and "NHR requires a positive group size");
    EP_STATIC_ASSERT(kNumSliceAlignmentBytes > 0 and "NHR requires a positive slice alignment");
    static constexpr int kNumLevels = kNumRanks == 1 ? 0 : 32 - __builtin_clz(unsigned(kNumRanks - 1));
    static constexpr int kNumPhases = 2 * kNumLevels;
    static constexpr int kNumMaxSlices = kSwizzleOwners and math::is_power_of_2(kNumRanks)
        ? kNumPhases : 2 * (kNumRanks - 1);

    // A byte range within a chunk.
    struct NhrSlice {
        int64_t offset;
        int64_t num_bytes;
    };

    // Work at each phase.
    struct NhrPhase {
        int read_peer;             // Read this peer's input for this phase
        int notify_peer;           // Notify this peer when my input for this phase is ready
        int num_merged_slices = 0; // Number of merged slices, including possibly empty tails
    };

    EP_INLINE explicit NhrPlan(const int rank_idx): owned_slice_idx(slice_index(rank_idx)) {
        int num_slice_ranges = 0;
        for (int phase_idx = 0; phase_idx < kNumPhases; ++ phase_idx) {
            auto& phase = phases[phase_idx];
            const bool is_reduce_scatter = phase_idx < kNumLevels;
            const int level = is_reduce_scatter ? phase_idx : kNumPhases - 1 - phase_idx;
            const int distance = 1 << level;
            // Power-of-two AG can read in either direction; reuse the RS peer.
            const bool forward = is_reduce_scatter or math::is_power_of_2(kNumRanks);
            const int delta = forward ? distance : -distance;
            phase.read_peer = (rank_idx + kNumRanks + delta) % kNumRanks;
            phase.notify_peer = (rank_idx + kNumRanks - delta) % kNumRanks;
            // RS selects owners r, r-2d, ...; AG selects r-d, r-3d, ... (mod N).
            bool selected[kNumRanks]{};
            const int num_owners = (kNumRanks - 1 + distance) / (2 * distance);
            for (int i = 0; i < num_owners; ++ i) {
                const int owner = (rank_idx + 2 * kNumRanks -
                    (is_reduce_scatter ? 0 : distance) - i * 2 * distance) % kNumRanks;
                selected[slice_index(owner)] = true;
            }
            // Merge contiguous slices.
            for (int slice_idx = 0; slice_idx < kNumRanks;) {
                if (not selected[slice_idx]) {
                    ++ slice_idx;
                    continue;
                }
                auto& range = slice_ranges[num_slice_ranges ++];
                EP_DEVICE_ASSERT(num_slice_ranges <= kNumMaxSlices);
                range.begin = slice_idx;
                while (slice_idx < kNumRanks and selected[slice_idx])
                    ++ slice_idx;
                range.end = slice_idx;
                ++ phase.num_merged_slices;
            }
            cumsum_slice_ranges[phase_idx] = num_slice_ranges;
        }
    }

    // Reduce-scatter first, then all-gather.
    EP_INLINE const NhrPhase& get_phase(const int phase_idx) const {
        return phases[phase_idx];
    }

    // The returned offset includes parent.offset.
    EP_INLINE NhrSlice get_slice(const int phase_idx, const int slice_idx, const NhrSlice parent) const {
        const int range_idx = (phase_idx ? cumsum_slice_ranges[phase_idx - 1] : 0) + slice_idx;
        const auto& range = slice_ranges[range_idx];
        const auto begin = slice_offset(parent.num_bytes, range.begin);
        const auto end = slice_offset(parent.num_bytes, range.end);
        return {parent.offset + begin, end - begin};
    }

    // This rank's owned slice. The returned offset includes parent.offset.
    EP_INLINE NhrSlice get_owned_slice(const NhrSlice parent) const {
        const auto begin = slice_offset(parent.num_bytes, owned_slice_idx);
        const auto end = slice_offset(parent.num_bytes, owned_slice_idx + 1);
        return {parent.offset + begin, end - begin};
    }

private:
    // Half-open ranges of permuted owner-slice indices.
    struct SliceRange {
        int begin;
        int end;
    };

    const int owned_slice_idx;
    NhrPhase phases[math::max(1, kNumPhases)];
    // Cumulative number of slice ranges through each phase, including that phase.
    int cumsum_slice_ranges[math::max(1, kNumPhases)];
    SliceRange slice_ranges[math::max(1, kNumMaxSlices)];

    // Find the slice index owned by an owner.
    EP_INLINE static int slice_index(const int owner) {
        if constexpr (not kSwizzleOwners)
            return owner;
        // Put even positions before odd positions within each group. No need to swizzle
        // when each peer reads at most one slice in the group, i.e., group_size / 2 <= 1.
        int index = owner;
        for (int group_size = kNumRanks; group_size >= 4; group_size /= 2) {
            const int group_begin = index / group_size * group_size;
            const int position = index % group_size;
            index = group_begin + position / 2;
            if (position & 1)
                index += (group_size + 1) / 2;
            // Odd-sized groups cannot be divided into equal halves. Stop swizzling.
            if (group_size & 1)
                break;
        }
        return index;
    }

    // Find the slice offset when a parent chunk is divided into N slices.
    EP_INLINE static int64_t slice_offset(const int64_t num_parent_bytes, const int index) {
        return math::min(num_parent_bytes, math::ceil_div(num_parent_bytes * index,
            int64_t(kNumRanks) * kNumSliceAlignmentBytes) * kNumSliceAlignmentBytes);
    }
};

}  // namespace deep_ep
