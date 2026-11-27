#pragma once

#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep::layout {

// Iterates over all buckets as one flat sequence of `kNumBytesPerChunk`-sized chunks, e.g.
//     [bucket 0: chunk 0, 1, ..., n0 - 1][bucket 1: chunk n0, n0 + 1, ..., n0 + n1 - 1]...
// Each iterator visits chunks `start_chunk_idx, start_chunk_idx + kStride, ...`, so `kStride`
// iterators with distinct starts in `[0, kStride)` cover every chunk exactly once
template <int kNumBytesPerChunk, int kStride>
struct ChunkIterator {
    EP_STATIC_ASSERT(kNumBytesPerChunk > 0, "Invalid chunk size");
    EP_STATIC_ASSERT(kStride > 0, "Invalid chunk stride");

    const int num_buckets;
    const comm::BucketList& buckets;
    int num_total_chunks = 0; // Total across all buckets, including each bucket's partial tail

    // States, valid after a successful `next()`
    comm::Bucket bucket;
    int chunk_idx;                      // Global chunk index across all buckets
    int num_chunk_bytes;                // Only the last chunk of a bucket may be smaller than `kNumBytesPerChunk`
    int64_t chunk_offset_in_bucket;     // Byte offset of this chunk within `bucket`

    // NOTES: `int` is enough for chunk indices, as a chunk is normally at least several KB, which covers several TB in total
    EP_INLINE ChunkIterator(const int num_buckets, const comm::BucketList& buckets, const int start_chunk_idx):
        num_buckets(num_buckets), buckets(buckets), chunk_idx(start_chunk_idx - kStride) {
        for (int i = 0; i < num_buckets; ++ i)
            num_total_chunks += static_cast<int>(math::ceil_div<int64_t>(buckets[i].num_bytes, kNumBytesPerChunk));
    }

    EP_INLINE bool next() {
        // Move chunk
        chunk_idx += kStride;

        // Locate bucket
        while (bucket_idx < num_buckets) {
            bucket = buckets[bucket_idx];
            const int bucket_end_chunk_idx = bucket_start_chunk_idx +
                static_cast<int>(math::ceil_div<int64_t>(bucket.num_bytes, kNumBytesPerChunk));
            if (chunk_idx < bucket_end_chunk_idx) {
                chunk_offset_in_bucket = static_cast<int64_t>(chunk_idx - bucket_start_chunk_idx) * kNumBytesPerChunk;
                // NOTES: only the last chunk may be partial, so the remainder fits in `int` without a 64-bit `min`
                num_chunk_bytes = chunk_idx + 1 == bucket_end_chunk_idx ?
                    static_cast<int>(bucket.num_bytes - chunk_offset_in_bucket) : kNumBytesPerChunk;
                return true;
            }

            bucket_start_chunk_idx = bucket_end_chunk_idx;
            ++ bucket_idx;
        }
        return false;
    }

    EP_INLINE bool last() const {
        return chunk_idx + kStride >= num_total_chunks;
    }

private:
    int bucket_idx = 0, bucket_start_chunk_idx = 0;
};

}  // namespace deep_ep::layout
