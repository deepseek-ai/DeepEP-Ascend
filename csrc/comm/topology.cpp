#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include <acl/acl_rt.h>

#include "topology.hpp"

// CANN ships libtopoaddrinfo without its header. These declarations match
// HCOMM's topo_addr_info.h; keep this compatibility dependency in this file.
extern "C" {
int TopoAddrInfoGetSize(int phyId, size_t* size);
int TopoAddrInfoGet(int phyId, char* rankInfo, size_t* bufSize);
}

namespace deep_ep::comm {
namespace {

namespace py = pybind11;

struct LocalTopology {
    std::string instance, plane, eid;
};

struct TopologyReport {
    // IDs are typically much shorter. Both fields use HCOMM's 1024-byte capacity in topo_addr_info.
    char instance[1024];
    char plane[1024];
    // Truncated to 255 bytes.
    char error[256];
};

using WeakGroup = c10::weak_intrusive_ptr<c10d::ProcessGroup>;
std::mutex topology_cache_mutex;
std::unordered_map<WeakGroup, Topology> topology_cache;

std::optional<Topology> get_cached_topology(const ProcessGroup& group) {
    std::lock_guard<std::mutex> lock(topology_cache_mutex);
    const auto it = topology_cache.find(WeakGroup(group));
    if (it != topology_cache.end())
        return it->second;
    return std::nullopt;
}

Topology cache_topology(const ProcessGroup& group, Topology topology) {
    std::lock_guard<std::mutex> lock(topology_cache_mutex);
    std::erase_if(topology_cache, [](const auto& entry) { return entry.first.expired(); });
    topology_cache.insert_or_assign(WeakGroup(group), topology);
    return topology;
}

LocalTopology read_local_topology() {
    // aclrtGetDevice returns the user device ID (remapped by ASCEND_RT_VISIBLE_DEVICES),
    // while TopoAddrInfoGet expects the hardware physical device ID.
    int32_t user_device_id = -1, physical_device_id = -1;
    TORCH_CHECK(aclrtGetDevice(&user_device_id) == 0, "aclrtGetDevice failed; select the NPU before topology discovery");
    TORCH_CHECK(aclrtGetPhyDevIdByUserDevId(user_device_id, &physical_device_id) == 0,
                "cannot resolve physical device for ", user_device_id);
    size_t topo_json_size = 0;
    TORCH_CHECK(TopoAddrInfoGetSize(physical_device_id, &topo_json_size) == 0 and topo_json_size > 0,
                "TopoAddrInfoGetSize failed for device ", physical_device_id);
    std::string topo_json(topo_json_size, '\0');
    TORCH_CHECK(TopoAddrInfoGet(physical_device_id, topo_json.data(), &topo_json_size) == 0,
                "TopoAddrInfoGet failed for device ", physical_device_id);

    const auto parsed_topology = py::module_::import("json").attr("loads")(topo_json.c_str());
    const auto rank_list = parsed_topology["rank_list"].cast<py::list>();
    py::object local_rank_info;
    if (rank_list.size() == 1) {
        local_rank_info = rank_list[0];
    } else {
        for (size_t i = 0; i < rank_list.size(); ++ i) {
            const auto rank_info = rank_list[i];
            // CANN topology JSON may encode device_id and net_layer as integers or decimal strings(e.g. 0 or "0").
            if (py::int_(rank_info.attr("get")("device_id", -1)).cast<int>() == physical_device_id)
                local_rank_info = rank_info;
        }
    }
    TORCH_CHECK(static_cast<bool>(local_rank_info), "cannot identify physical device ", physical_device_id);

    for (const auto layer_info: local_rank_info["level_list"]) {
        // Layer 1 carries one address per rank, unlike mesh.
        if (py::int_(layer_info["net_layer"]).cast<int>() != 1)
            continue;
        TORCH_CHECK(layer_info["net_type"].cast<std::string>() == "CLOS",
                    "automatic scale-up discovery requires a CLOS layer 1");
        const auto addr_info = layer_info["rank_addr_list"][py::int_(0)];
        LocalTopology local_topology{layer_info["net_instance_id"].cast<std::string>(),
                                     addr_info["plane_id"].cast<std::string>(),
                                     addr_info["addr"].cast<std::string>()};
        TORCH_CHECK(not local_topology.instance.empty() and local_topology.instance.size() < sizeof(TopologyReport{}.instance) and
                    not local_topology.plane.empty() and local_topology.plane.size() < sizeof(TopologyReport{}.plane),
                    "invalid scale-up instance or plane ID in topology report");
        TORCH_CHECK(addr_info["addr_type"].cast<std::string>() == "EID" and local_topology.eid.size() == 32 and
                    local_topology.eid.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos,
                    "scale-up layer has no valid EID endpoint");
        return local_topology;
    }
    TORCH_CHECK(false, "the device reports no scale-up network layer 1");
}

int infer_scaleup_ranks(const std::vector<TopologyReport>& peers) {
    std::map<std::string, int> counts;
    for (const auto& peer: peers)
        ++ counts[peer.instance];
    int num_scaleup_ranks = 0;
    for (const auto& [instance, count]: counts)
        num_scaleup_ranks = std::gcd(num_scaleup_ranks, count);
    for (size_t begin = 0; begin < peers.size(); begin += num_scaleup_ranks)
        for (size_t rank = begin; rank < begin + num_scaleup_ranks; ++ rank)
            TORCH_CHECK(std::strcmp(peers[rank].instance, peers[begin].instance) == 0 and
                        std::strcmp(peers[rank].plane, peers[begin].plane) == 0,
                        "group ranks [", begin, ", ", begin + num_scaleup_ranks, ") do not share a scale-up instance and endpoint plane");
    return num_scaleup_ranks;
}

}  // namespace

Topology discover_topology(const ProcessGroup& group, const int num_timeout_secs) {
    TORCH_CHECK(group != nullptr and group->getSize() > 0, "a live ProcessGroup is required");
    if (const auto cached = get_cached_topology(group))
        return *cached;
    if (group->getSize() == 1)
        return cache_topology(group, {1, {}});

    // TODO: Replace the topoaddrinfo JSON probe when HCOMM exposes the local scale-up domain, plane and EID.
    // Include local errors in the exchange so a failed probe cannot leave peers waiting.
    LocalTopology local;
    TopologyReport report{};
    try {
        local = read_local_topology();
        std::memcpy(report.instance, local.instance.c_str(), local.instance.size() + 1);
        std::memcpy(report.plane, local.plane.c_str(), local.plane.size() + 1);
    } catch (const std::exception& error) {
        std::snprintf(report.error, sizeof(report.error), "%s", error.what());
    }
    const auto peers = all_gather_fixed(group, report, num_timeout_secs);
    std::string errors;
    for (size_t rank = 0; rank < peers.size(); ++ rank)
        if (peers[rank].error[0] != '\0')
            errors += "rank " + std::to_string(rank) + ": " + peers[rank].error + "; ";
    TORCH_CHECK(errors.empty(), "scale-up topology discovery failed: ", errors);
    const auto num_scaleup_ranks = infer_scaleup_ranks(peers);
    return cache_topology(group, {num_scaleup_ranks, local.eid});
}

std::tuple<int, int> get_physical_domain_size(const ProcessGroup& group) {
    const auto topology = discover_topology(group);
    return {group->getSize() / topology.num_scaleup_ranks, topology.num_scaleup_ranks};
}

}  // namespace deep_ep::comm
