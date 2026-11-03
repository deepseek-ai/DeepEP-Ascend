#pragma once

#include <cstdint>

#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <c_api/vector_datamove/vector_datamove.h>

namespace deep_ep {

static constexpr uint32_t kL2CacheCleanTileBytes = 64 * 1024;

__simd_vf__ inline void fill_l2_cache_clean_ub(__ubuf__ uint32_t* buffer) {
    vector_uint32_t zeros;
    const auto mask = asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(zeros, static_cast<uint32_t>(0), mask);
    for (uint32_t offset = 0; offset < kL2CacheCleanTileBytes / sizeof(uint32_t); offset += 64)
        asc_storealign(buffer + offset, zeros, mask);
}

// This kernel is the second stage of a clean L2 flush. The first stage writes
// an oversized GM buffer with the normal allocation policy, evicting the old
// working set but leaving the flush buffer's resident lines Dirty. Revisiting
// the same addresses with NOTALLOC_CLEAN does not allocate on a miss; on a hit
// it keeps the resident line and marks it Clean.
__global__ __vector__ void l2_cache_clean_impl(__gm__ uint8_t* buffer, const int64_t num_bytes) {
    AscendC::InitSocState();

    auto ub_buffer = reinterpret_cast<__ubuf__ uint32_t*>(0);
    fill_l2_cache_clean_ub(ub_buffer);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    const auto block_idx = static_cast<uint64_t>(AscendC::GetBlockIdx());
    const auto num_blocks = static_cast<uint64_t>(AscendC::GetBlockNum());
    for (uint64_t offset = block_idx * kL2CacheCleanTileBytes;
         offset < static_cast<uint64_t>(num_bytes);
         offset += num_blocks * kL2CacheCleanTileBytes) {
        const auto remaining = static_cast<uint64_t>(num_bytes) - offset;
        const auto copy_bytes = static_cast<uint32_t>(
            remaining < kL2CacheCleanTileBytes ? remaining : kL2CacheCleanTileBytes);
        asc_copy_ub2gm_align(
            buffer + offset, reinterpret_cast<__ubuf__ uint8_t*>(ub_buffer),
            /* n_burst= */ 1, /* len_burst= */ copy_bytes,
            asc_store_l2_cache_mode::NOTALLOC_CLEAN,
            /* dst_gap= */ 0, /* src_gap= */ 0);
    }
    pipe_barrier(PIPE_ALL);
}

}  // namespace deep_ep
