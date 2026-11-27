#pragma once

#include <string>
#include <tuple>

#include "process_group.hpp"

namespace deep_ep::comm {

struct Topology {
    int num_scaleup_ranks;
    std::string scaleup_eid;
};

// Collective on the first call for `group` and cached locally for subsequent calls.
// Requires the current NPU device to be set and the caller to hold the GIL.
Topology discover_topology(const ProcessGroup& group, int num_timeout_secs = 300);

// Shares the collective first discovery and local cache with Buffer initialization.
std::tuple<int, int> get_physical_domain_size(const ProcessGroup& group);

}  // namespace deep_ep::comm
