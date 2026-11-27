#pragma once

#include <cstdint>

namespace deep_ep::layout {

static constexpr int kNumMaxSendRecvEntries = 500;

struct SendRecvEntry {
    int64_t src_offset;
    int64_t dst_offset;
    int64_t num_bytes;
    int dst_rank_idx;
};

struct SendRecvList {
    SendRecvEntry entries[kNumMaxSendRecvEntries] = {};
};

}  // namespace deep_ep::layout
