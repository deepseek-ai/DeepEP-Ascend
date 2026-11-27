#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdint>
#include <optional>

#include <acl/acl.h>
#include <hcomm/hcomm_res.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_ep/common/math.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/layout/common/context.hpp>

#include "symmetric.hpp"
#include "../../comm/topology.hpp"

namespace deep_ep {

// Peers share the c10d group name and create buffers in the same order within that group.
static std::string make_buffer_uid(const comm::ProcessGroup& group) {
    const auto name = pybind11::cast(group).attr("group_name").cast<std::string>();
    using WeakGroup = c10::weak_intrusive_ptr<c10d::ProcessGroup>;
    static std::unordered_map<WeakGroup, int> context_idx_per_group;
    std::erase_if(context_idx_per_group, [](const auto& entry) { return entry.first.expired(); });
    const auto buffer_idx = context_idx_per_group[WeakGroup(group)] ++;
    return "deep_ep/" + name + "/" + std::to_string(buffer_idx);
}

// One local endpoint and the port it listens on.
struct EndpointEntry {
    EndpointDesc desc{};
    uint16_t port = 0;
};

// One fixed-size record per rank. A zero port denotes an unused endpoint.
struct PeerEndpoints {
    EndpointEntry scaleup, scaleup_mem, scaleout;
};

static std::string from_hex(const std::string& hex) {
    EP_HOST_ASSERT(hex.size() % 2 == 0 and "the scale-up EID must be hex encoded");
    const auto digit = [](const char c) {
        if (c >= '0' and c <= '9')
            return c - '0';
        if (c >= 'a' and c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' and c <= 'F')
            return c - 'A' + 10;
        EP_HOST_UNREACHABLE("invalid hex digit in the scale-up EID");
    };
    std::string raw;
    raw.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2)
        raw.push_back(static_cast<char>((digit(hex[i]) << 4) | digit(hex[i + 1])));
    return raw;
}

struct HCommContext {
    std::shared_ptr<symmetric::SymmetricMemory> memory;
    void* buffer = nullptr;
    int64_t num_bytes = 0;

    int num_ranks = 0;
    int rank_idx = 0;
    int num_scaleout_ranks = 0;
    int num_scaleup_ranks = 0;
    std::string buffer_uid;
    std::vector<uint64_t> scaleup_peer_ptrs;
    std::vector<uint64_t> scaleout_peer_ptrs;

    int num_scaleup_jetties = 0;
    int num_scaleout_jetties = 0;
    EndpointHandle scaleup_mem_endpoint = nullptr;
    std::vector<ChannelHandle> scaleup_mem_channel_handles;
    std::vector<EndpointHandle> scaleup_jetty_endpoints;
    std::vector<ChannelHandle> scaleup_channel_handles;
    std::vector<EndpointHandle> scaleout_jetty_endpoints;
    std::vector<ChannelHandle> scaleout_channel_handles;
    // Device copy of the channel tables.
    layout::DeviceContext* device_context = nullptr;

    HCommContext(const comm::ProcessGroup& group, const int& rank_idx,
                const int& num_ranks, const int64_t& num_bytes,
                const int& num_scaleup_jetties, const int& num_scaleout_jetties,
                const int& num_cpu_timeout_secs, const bool& use_huge1g_page,
                const std::shared_ptr<symmetric::SymmetricMemory>& shared_memory = nullptr):
        num_bytes(num_bytes),
        num_ranks(num_ranks), rank_idx(rank_idx) {
        EP_HOST_ASSERT(num_bytes > 0);
        EP_HOST_ASSERT(num_ranks > 0 and rank_idx >= 0 and rank_idx < num_ranks);
        EP_HOST_ASSERT(group != nullptr and group->getRank() == rank_idx and group->getSize() == num_ranks);

        buffer_uid = make_buffer_uid(group);

        const auto topology = comm::discover_topology(group, num_cpu_timeout_secs);
        num_scaleup_ranks = topology.num_scaleup_ranks;
        EP_HOST_ASSERT(num_scaleup_ranks > 0 and num_scaleup_ranks <= kNumMaxScaleupRanks);
        EP_HOST_ASSERT(num_scaleup_ranks == 1 or num_scaleup_ranks % 2 == 0);
        EP_HOST_ASSERT(num_ranks % num_scaleup_ranks == 0);
        num_scaleout_ranks = num_ranks / num_scaleup_ranks;
        EP_HOST_ASSERT(num_scaleout_ranks <= kNumMaxScaleoutRanks);
        EP_HOST_ASSERT(num_scaleup_ranks == 1 or num_scaleup_jetties > 0);
        EP_HOST_ASSERT(num_scaleout_ranks == 1 or num_scaleout_jetties > 0);

        this->num_scaleup_jetties = num_scaleup_ranks > 1 ? num_scaleup_jetties : 0;
        // TODO(Huawei): HCOMM supports UB_RTP loopback.
        this->num_scaleout_jetties = num_scaleout_ranks > 1 ? num_scaleout_jetties : 0;

        // Share the allocation when registering the same storage with another group.
        memory = shared_memory == nullptr ? std::make_shared<symmetric::SymmetricMemory>(num_bytes, use_huge1g_page) : shared_memory;
        EP_HOST_ASSERT(memory->num_bytes == num_bytes);
        buffer = memory->ptr;

        // Discover peer buffer pointers
        scaleup_peer_ptrs.resize(num_scaleup_ranks, 0);
        scaleup_peer_ptrs[rank_idx % num_scaleup_ranks] = reinterpret_cast<uint64_t>(buffer);

        layout::DeviceContext host_device_context{};

        if (num_ranks > 1) {
            PeerEndpoints local{};
            if (num_scaleout_ranks > 1)
                local.scaleout = create_scaleout_endpoints();
            if (num_scaleup_ranks > 1)
                create_scaleup_endpoints(topology.scaleup_eid, local);
            const auto peer_entries = comm::all_gather_fixed(group, local, num_cpu_timeout_secs);
            if (num_scaleup_ranks > 1) {
                acquire_scaleup_mem_channels(peer_entries, num_cpu_timeout_secs);
                acquire_scaleup_jetties(peer_entries, num_cpu_timeout_secs, host_device_context);
            }
            if (num_scaleout_ranks > 1)
                acquire_scaleout_jetties(peer_entries, num_cpu_timeout_secs, host_device_context);
        }

        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device_context), sizeof(layout::DeviceContext),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(device_context, sizeof(layout::DeviceContext), &host_device_context,
                              sizeof(layout::DeviceContext), ACL_MEMCPY_HOST_TO_DEVICE));
    }

    ~HCommContext() noexcept(false) {
        if (device_context != nullptr) {
            ACL_CHECK(aclrtFree(device_context));
            device_context = nullptr;
        }
        if (not scaleup_channel_handles.empty())
            HCOMM_CHECK(HcommChannelDestroy(scaleup_channel_handles.data(), static_cast<uint32_t>(scaleup_channel_handles.size())));
        for (size_t i = 0; i < scaleup_jetty_endpoints.size(); ++ i) {
            HCOMM_CHECK(HcommMemUnreg(scaleup_jetty_endpoints[i], scaleup_jetty_mem_handles[i]));
            HCOMM_CHECK(HcommEndpointDestroy(scaleup_jetty_endpoints[i]));
        }
        if (not scaleout_channel_handles.empty())
            HCOMM_CHECK(HcommChannelDestroy(scaleout_channel_handles.data(), static_cast<uint32_t>(scaleout_channel_handles.size())));
        for (size_t i = 0; i < scaleout_jetty_endpoints.size(); ++ i) {
            HCOMM_CHECK(HcommMemUnreg(scaleout_jetty_endpoints[i], scaleout_jetty_mem_handles[i]));
            HCOMM_CHECK(HcommEndpointDestroy(scaleout_jetty_endpoints[i]));
        }
        // Release all UB_MEM imports and registrations before the allocation.
        if (not scaleup_mem_channel_handles.empty()) {
            HCOMM_CHECK(HcommChannelDestroy(scaleup_mem_channel_handles.data(), static_cast<uint32_t>(scaleup_mem_channel_handles.size())));
            HCOMM_CHECK(HcommMemUnreg(scaleup_mem_endpoint, scaleup_mem_handle));
            HCOMM_CHECK(HcommEndpointDestroy(scaleup_mem_endpoint));
        }
        memory.reset();
        buffer = nullptr;
    }

private:
    HcommMemHandle scaleup_mem_handle = nullptr;
    // HCOMM borrows channelName until channel destruction.
    std::vector<std::string> scaleup_mem_channel_names;
    std::vector<HcommMemHandle> scaleup_jetty_mem_handles;
    // HCOMM shallow-copies channelName pointers; keep scaleup_channel_names until channel destruction.
    std::vector<std::string> scaleup_channel_names;
    std::vector<HcommMemHandle> scaleout_jetty_mem_handles;
    std::vector<std::string> scaleout_channel_names;

    // Scale-up peers are the rest of this rank's block, in increasing rank order.
    int scaleup_peer_rank(const int& slot) const {
        const auto block_start_rank_idx = rank_idx - rank_idx % num_scaleup_ranks;
        const auto slot_in_block = rank_idx - block_start_rank_idx;
        EP_HOST_ASSERT(slot >= 0 and slot < num_scaleup_ranks - 1);
        return block_start_rank_idx + (slot < slot_in_block ? slot : slot + 1);
    }

    static std::vector<EndpointDesc> enumerate_local_endpoints() {
        // Although HcommEndpointGetDescNum/GetDescs name their parameter `deviceLogicId`,
        // they actually expect the user device ID returned by `aclrtGetDevice`.
        int32_t device = 0;
        ACL_CHECK(aclrtGetDevice(&device));
        uint32_t num = 0;
        HCOMM_CHECK(HcommEndpointGetDescNum(device, &num));
        EP_HOST_ASSERT(num > 0 and "the device reports no HCOMM endpoints");
        std::vector<EndpointDesc> descs(num);
        uint32_t capacity = num;
        HCOMM_CHECK(HcommEndpointGetDescs(device, &capacity, descs.data()));
        descs.resize(capacity);
        return descs;
    }

    static std::optional<EndpointDesc> pick_endpoint(const std::vector<EndpointDesc>& descs,
                                                     const CommProtocol& protocol) {
        for (const auto& desc: descs)
            if (desc.protocol == protocol)
                return desc;
        return std::nullopt;
    }

    CommMem registration() const {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = static_cast<uint64_t>(num_bytes);
        return mem;
    }

    static uint16_t get_listen_port(const EndpointHandle& endpoint) {
        uint32_t port = 0;
        HCOMM_CHECK(HcommEndpointGetListenPort(endpoint, &port));
        EP_HOST_ASSERT(port > 0 and port <= std::numeric_limits<uint16_t>::max());
        return static_cast<uint16_t>(port);
    }

    EndpointEntry create_scaleout_endpoints() {
        const auto rtp = pick_endpoint(enumerate_local_endpoints(), COMM_PROTOCOL_UB_RTP);
        EP_HOST_ASSERT(rtp.has_value() and "the device reports no UB_RTP endpoint");
        const auto mem = registration();
        scaleout_jetty_endpoints.resize(num_scaleout_jetties);
        scaleout_jetty_mem_handles.resize(num_scaleout_jetties);
        for (int i = 0; i < num_scaleout_jetties; ++ i) {
            HCOMM_CHECK(HcommEndpointCreate(&*rtp, &scaleout_jetty_endpoints[i]));
            HCOMM_CHECK(HcommMemReg(scaleout_jetty_endpoints[i], buffer_uid.c_str(), &mem, &scaleout_jetty_mem_handles[i]));
        }
        return EndpointEntry{*rtp, get_listen_port(scaleout_jetty_endpoints.front())};
    }

    // Every scale-up peer shares one local endpoint, so the whole block needs a single port. The
    // endpoints themselves stay one per channel: a shared endpoint would share one Jetty, which
    // the concurrent AIVs cannot use.
    void create_scaleup_endpoints(const std::string& scaleup_eid, PeerEndpoints& local) {
        // The scale-up layer gives every rank one address shared by all of its peers, so the local
        // endpoint is that address rather than one of the node-local mesh ports.
        EP_HOST_ASSERT(not scaleup_eid.empty() and "scale-up topology has no endpoint");
        const auto eid = from_hex(scaleup_eid);
        EP_HOST_ASSERT(eid.size() == sizeof(CommAddr{}.eid) and "the scale-up endpoint id is malformed");
        std::optional<EndpointDesc> desc;
        for (const auto& candidate: enumerate_local_endpoints())
            if (candidate.protocol == COMM_PROTOCOL_UBC_CTP and
                std::memcmp(candidate.commAddr.raws, eid.data(), eid.size()) == 0)
                desc = candidate;
        EP_HOST_ASSERT(desc.has_value() and "the device reports no UBC_CTP endpoint");
        EP_HOST_ASSERT(math::max(num_scaleup_jetties, num_scaleup_ranks - 1) <= kNumMaxScaleupRanks);

        const auto mem = registration();
        scaleup_jetty_endpoints.resize(num_scaleup_jetties);
        scaleup_jetty_mem_handles.resize(num_scaleup_jetties);
        for (int i = 0; i < num_scaleup_jetties; ++ i) {
            HCOMM_CHECK(HcommEndpointCreate(&*desc, &scaleup_jetty_endpoints[i]));
            HCOMM_CHECK(HcommMemReg(scaleup_jetty_endpoints[i], buffer_uid.c_str(), &mem,
                                    &scaleup_jetty_mem_handles[i]));
        }

        // The block-wide read/write mapping uses its own UB_MEM endpoint, like HCCL.
        auto mem_desc = *desc;
        mem_desc.protocol = COMM_PROTOCOL_UB_MEM;
        HCOMM_CHECK(HcommEndpointCreate(&mem_desc, &scaleup_mem_endpoint));
        HCOMM_CHECK(HcommMemReg(scaleup_mem_endpoint, buffer_uid.c_str(), &mem, &scaleup_mem_handle));
        local.scaleup_mem = {mem_desc, get_listen_port(scaleup_mem_endpoint)};
        local.scaleup = {*desc, get_listen_port(scaleup_jetty_endpoints.front())};
    }

    HcommChannelDesc make_peer_desc(const EndpointEntry& remote) const {
        HcommChannelDesc desc;
        HCOMM_CHECK(HcommChannelDescInit(&desc, 1));
        desc.remoteEndpoint = remote.desc;
        desc.notifyNum = 3;  // same as HCCL
        return desc;
    }

    // Cover every scale-out peer, then alternate ring neighbors to fill the Jetty pool.
    std::vector<std::pair<int, EndpointEntry>> build_scaleout_peers(const std::vector<PeerEndpoints>& peer_entries,
                                                                   const int& num_channels,
                                                                   const CommProtocol& protocol,
                                                                   const char* err) {
        const int num_peers = num_scaleout_ranks - 1;
        EP_HOST_ASSERT(num_channels >= num_peers and num_channels <= kNumMaxScaleoutRanks);
        const int scaleout_rank_idx = rank_idx / num_scaleup_ranks;
        const int scaleup_rank_idx = rank_idx % num_scaleup_ranks;
        const int num_extra = num_channels - num_peers;
        if (num_scaleout_ranks % 2 != 0)
            EP_HOST_ASSERT(num_extra % 2 == 0);

        std::vector<std::pair<int, EndpointEntry>> peers;
        peers.reserve(num_channels);
        for (int channel_idx = 0; channel_idx < num_channels; ++ channel_idx) {
            int peer_idx;
            if (channel_idx < num_peers) {
                peer_idx = channel_idx < scaleout_rank_idx ? channel_idx : channel_idx + 1;
            } else {
                // Alternate ring neighbors; adjacent ranks start toward each other.
                const bool next_peer = (channel_idx - num_peers + scaleout_rank_idx) % 2 == 0;
                peer_idx = (scaleout_rank_idx + (next_peer ? 1 : num_scaleout_ranks - 1)) % num_scaleout_ranks;
            }
            const int peer_rank_idx = peer_idx * num_scaleup_ranks + scaleup_rank_idx;
            const auto& remote = peer_entries[peer_rank_idx].scaleout;
            EP_HOST_ASSERT(remote.port != 0 and err);
            EP_HOST_ASSERT(remote.desc.protocol == protocol);
            peers.emplace_back(peer_rank_idx, remote);
        }
        return peers;
    }

    // Match registration by tag and size; UB_MEM may leave the memory type unset.
    uint64_t find_remote_registration(ChannelHandle channel) const {
        uint32_t num_mems = 0;
        CommMem* remote_mems = nullptr;
        char** mem_tags = nullptr;
        HCOMM_CHECK(HcommChannelGetRemoteMems(channel, &num_mems, &remote_mems, &mem_tags));

        uint64_t peer_ptr = 0;
        int num_matches = 0;
        for (uint32_t mem_idx = 0; mem_idx < num_mems; ++ mem_idx) {
            if (mem_tags[mem_idx] == nullptr or std::strcmp(mem_tags[mem_idx], buffer_uid.c_str()) != 0 or
                remote_mems[mem_idx].size != static_cast<uint64_t>(num_bytes))
                continue;
            peer_ptr = reinterpret_cast<uint64_t>(remote_mems[mem_idx].addr);
            ++ num_matches;
        }
        EP_HOST_ASSERT(num_matches == 1 and "the peer registration is not unique");
        EP_HOST_ASSERT(peer_ptr != 0);
        return peer_ptr;
    }

    // One UBMEM channel per non-self peer of this rank's scale-up block; resolves
    // scaleup_peer_ptrs to each remote's registered workspace address.
    void acquire_scaleup_mem_channels(const std::vector<PeerEndpoints>& peer_entries, const int& num_cpu_timeout_secs) {
        const auto num_peers = num_scaleup_ranks - 1;
        // A single-rank block has no peer to wire up.
        if (num_peers == 0)
            return;

        std::vector<HcommChannelDesc> descs(num_peers);
        scaleup_mem_channel_names.resize(descs.size());
        for (size_t i = 0; i < descs.size(); ++ i) {
            const auto peer_rank_idx = scaleup_peer_rank(i);
            const auto& remote = peer_entries[peer_rank_idx].scaleup_mem;
            EP_HOST_ASSERT(remote.port != 0 and "the peer did not publish a UB_MEM endpoint");
            scaleup_mem_channel_names[i] = buffer_uid + "/ubmem/" + std::to_string(std::min(rank_idx, peer_rank_idx)) + "-" +
                std::to_string(std::max(rank_idx, peer_rank_idx));
            EP_HOST_ASSERT(scaleup_mem_channel_names[i].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
            auto& desc = descs[i];
            HCOMM_CHECK(HcommChannelDescInit(&desc, 1));
            desc.remoteEndpoint = remote.desc;
            desc.memHandles = &scaleup_mem_handle;
            desc.memHandleNum = 1;
            desc.role = rank_idx < peer_rank_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            desc.port = peer_entries[std::min(rank_idx, peer_rank_idx)].scaleup_mem.port;
            desc.channelName = scaleup_mem_channel_names[i].c_str();
        }
        scaleup_mem_channel_handles.resize(descs.size());
        HCOMM_CHECK(HcommChannelCreate(scaleup_mem_endpoint, CommEngine::COMM_ENGINE_AIV,
                                      descs.data(), static_cast<uint32_t>(descs.size()), scaleup_mem_channel_handles.data()));
        wait_for_channels(scaleup_mem_channel_handles, num_cpu_timeout_secs);

        for (size_t i = 0; i < descs.size(); ++ i) {
            const auto peer_rank_idx = scaleup_peer_rank(i);
            scaleup_peer_ptrs[peer_rank_idx % num_scaleup_ranks] = find_remote_registration(scaleup_mem_channel_handles[i]);
        }
    }

    // Acquire exactly one shared Jetty per AIV, with separate channels for peer metadata.
    void acquire_scaleout_jetties(const std::vector<PeerEndpoints>& peer_entries, const int& num_cpu_timeout_secs,
                                  layout::DeviceContext& host_device_context) {
        const int num_peers = num_scaleout_ranks - 1;
        const auto num_channels = math::max(num_scaleout_jetties, num_peers);
        const auto peers = build_scaleout_peers(peer_entries, num_channels, COMM_PROTOCOL_UB_RTP,
                                              "no URMA link to the peer scale-out group");

        // Number channels independently for each peer to match both ends of the connection.
        HcommChannelConfig config = nullptr;
        HCOMM_CHECK(HcommChannelConfigCreate(&config));
        HCOMM_CHECK(HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1));
        scaleout_channel_handles.resize(num_channels);
        scaleout_channel_names.resize(num_channels);
        std::vector<int> peer_channel_counts(num_scaleout_ranks, 0);
        for (int channel_idx = 0; channel_idx < num_channels; ++ channel_idx) {
            const auto peer_rank_idx = peers[channel_idx].first;
            const auto peer_idx = peer_rank_idx / num_scaleup_ranks;
            const auto pair_idx = peer_channel_counts[peer_idx] ++;
            scaleout_channel_names[channel_idx] = buffer_uid + "/scaleout/" +
                std::to_string(std::min(rank_idx, peer_rank_idx)) + "-" +
                std::to_string(std::max(rank_idx, peer_rank_idx)) + "/" + std::to_string(pair_idx);
            EP_HOST_ASSERT(scaleout_channel_names[channel_idx].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
            HcommChannelDesc desc = make_peer_desc(peers[channel_idx].second);
            desc.memHandles = &scaleout_jetty_mem_handles[channel_idx % num_scaleout_jetties];
            desc.memHandleNum = 1;
            // The lower rank listens; the higher rank connects socket.
            desc.role = rank_idx < peer_rank_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            desc.port = peer_entries[std::min(rank_idx, peer_rank_idx)].scaleout.port;
            desc.channelName = scaleout_channel_names[channel_idx].c_str();
            HCOMM_CHECK(HcommChannelCreateWithConfig(scaleout_jetty_endpoints[channel_idx % num_scaleout_jetties], CommEngine::COMM_ENGINE_AIV,
                                                     &desc, 1, config, scaleout_channel_handles.data() + channel_idx));
        }
        HCOMM_CHECK(HcommChannelConfigDestroy(config));
        wait_for_channels(scaleout_channel_handles, num_cpu_timeout_secs);

        // Device code indexes this table by both AIV and non-self peer.
        for (size_t channel_idx = 0; channel_idx < scaleout_channel_handles.size(); ++ channel_idx)
            host_device_context.scaleout_jetties[channel_idx] = scaleout_channel_handles[channel_idx];

        scaleout_peer_ptrs.assign(num_scaleout_ranks, 0);
        for (int channel_idx = 0; channel_idx < num_peers; ++ channel_idx) {
            char** mem_tags = nullptr;
            CommMem* remote_mems = nullptr;
            uint32_t num_mems = 0;
            HCOMM_CHECK(HcommChannelGetRemoteMems(scaleout_channel_handles[channel_idx], &num_mems, &remote_mems, &mem_tags));
            EP_HOST_ASSERT(num_mems == 1 and remote_mems != nullptr and mem_tags != nullptr);
            EP_HOST_ASSERT(mem_tags[0] != nullptr and buffer_uid == mem_tags[0]);
            EP_HOST_ASSERT(remote_mems[0].addr != nullptr and remote_mems[0].size == static_cast<uint64_t>(num_bytes));
            const int peer_idx = peers[channel_idx].first / num_scaleup_ranks;
            scaleout_peer_ptrs[peer_idx] = reinterpret_cast<uint64_t>(remote_mems[0].addr);
        }
    }

    // Acquire exactly one shared Jetty per AIV, with separate channels for peer metadata.
    void acquire_scaleup_jetties(const std::vector<PeerEndpoints>& peer_entries, const int& num_cpu_timeout_secs,
                                 layout::DeviceContext& host_device_context) {
        const auto block_start_rank_idx = rank_idx - rank_idx % num_scaleup_ranks;
        const auto scaleup_rank_idx = rank_idx % num_scaleup_ranks;
        const auto num_peers = num_scaleup_ranks - 1;
        const auto num_channels = math::max(num_scaleup_jetties, num_peers);

        // The first entries describe every peer; pair the remaining Jetty owners directly.
        HcommChannelConfig config = nullptr;
        HCOMM_CHECK(HcommChannelConfigCreate(&config));
        HCOMM_CHECK(HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1));
        scaleup_channel_handles.resize(num_channels);
        scaleup_channel_names.resize(num_channels);
        for (int channel_idx = 0; channel_idx < num_channels; ++ channel_idx) {
            int peer_idx = channel_idx, pair_idx = 0;
            if (channel_idx >= num_peers) {
                const auto peer_rank_idx = block_start_rank_idx + (scaleup_rank_idx ^ 1);
                peer_idx = peer_rank_idx - block_start_rank_idx - (peer_rank_idx > rank_idx ? 1 : 0);
                pair_idx = channel_idx - num_peers + 1;
            }
            const auto peer_rank_idx = scaleup_peer_rank(peer_idx);
            const auto& remote = peer_entries[peer_rank_idx].scaleup;
            EP_HOST_ASSERT(remote.port != 0 and "the peer did not publish a scale-up endpoint");
            scaleup_channel_names[channel_idx] = buffer_uid + "/" +
                std::to_string(std::min(rank_idx, peer_rank_idx)) + "-" +
                std::to_string(std::max(rank_idx, peer_rank_idx)) + "/" + std::to_string(pair_idx);
            EP_HOST_ASSERT(scaleup_channel_names[channel_idx].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
            HcommChannelDesc desc = make_peer_desc(remote);
            desc.memHandles = &scaleup_jetty_mem_handles[channel_idx % num_scaleup_jetties];
            desc.memHandleNum = 1;
            // The lower rank listens; the higher rank connects socket.
            desc.role = rank_idx < peer_rank_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            desc.port = peer_entries[std::min(rank_idx, peer_rank_idx)].scaleup.port;
            desc.channelName = scaleup_channel_names[channel_idx].c_str();
            // Leave desc.qos at 0xFFFFFFFF to use HCOMM's default UB QoS (4).
            // Leave desc.ubAttr.sqDepth at 0xFFFFFFFF (UB_SQ_DEPTH_NOT_SET) to use HCOMM's default depth.
            HCOMM_CHECK(HcommChannelCreateWithConfig(scaleup_jetty_endpoints[channel_idx % num_scaleup_jetties], CommEngine::COMM_ENGINE_AIV,
                                                     &desc, 1, config, scaleup_channel_handles.data() + channel_idx));
        }
        HCOMM_CHECK(HcommChannelConfigDestroy(config));
        wait_for_channels(scaleup_channel_handles, num_cpu_timeout_secs);

        // Device code indexes this table by both AIV and non-self peer.
        for (size_t channel_idx = 0; channel_idx < scaleup_channel_handles.size(); ++ channel_idx)
            host_device_context.scaleup_jetties[channel_idx] = scaleup_channel_handles[channel_idx];
    }

    void wait_for_channels(const std::vector<ChannelHandle>& channels, const int& num_cpu_timeout_secs) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(num_cpu_timeout_secs);
        std::vector<int32_t> statuses(channels.size(), HCOMM_CHANNEL_STATUS_CONNECTING);
        while (true) {
            const auto result = HcommChannelGetStatus(channels.data(), static_cast<uint32_t>(channels.size()), statuses.data());
            if (result != HCCL_E_AGAIN)
                HCOMM_CHECK(result);

            if (result == HCCL_SUCCESS) {
                bool all_ready = true;
                for (const int& status: statuses) {
                    if (status == HCOMM_CHANNEL_STATUS_READY)
                        continue;

                    all_ready = false;
                    EP_HOST_ASSERT(status < HCOMM_CHANNEL_STATUS_FAILED_INTERNAL);
                }
                if (all_ready)
                    return;
            }
            EP_HOST_ASSERT(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
};

}  // namespace deep_ep
