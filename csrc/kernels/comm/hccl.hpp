#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <optional>

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hcomm/hcomm_res.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_ep/common/math.hpp>
#include <deep_ep/common/exception.hpp>

#include "symmetric.hpp"

namespace deep_ep {

struct HCCLContext {
    std::shared_ptr<symmetric::SymmetricMemory> memory;
    HcclComm comm = nullptr;
    void* buffer = nullptr;
    int64_t num_bytes = 0;

    int num_ranks = 0;
    int rank_idx = 0;
    std::vector<int64_t> peer_ptrs;

    // HCCL registration for UB_MEM address discovery
    HcclMemHandle mem_handle = nullptr;

    // Jetty endpoints, peer channels and their device lookup table
    std::vector<EndpointHandle> jetty_endpoints;
    std::vector<ChannelHandle> channel_handles;
    void* jetty_device_ptrs = nullptr;  // Device copy of `channel_handles`

    HCCLContext(const std::string& hccl_group_name, const int& rank_idx,
                const int& num_ranks, const int64_t& num_bytes,
                const int& num_jetties, const int& num_cpu_timeout_secs, const bool& use_huge1g_page,
                const std::shared_ptr<symmetric::SymmetricMemory>& shared_memory = nullptr):
        num_bytes(num_bytes),
        num_ranks(num_ranks), rank_idx(rank_idx) {
        EP_HOST_ASSERT(num_bytes > 0);
        EP_HOST_ASSERT(num_ranks > 0 and rank_idx >= 0 and rank_idx < num_ranks);

        // TODO: relationships between layer and Jetties
        EP_HOST_ASSERT(num_jetties > 0);

        // All ranks must create buffers in the same order within each group.
        static std::unordered_map<std::string, int> context_idx_per_group;
        const auto buffer_idx = context_idx_per_group[hccl_group_name] ++;
        const auto buffer_uid = "deep_ep/" + hccl_group_name + "/" + std::to_string(buffer_idx);

        // Share the allocation when registering the same storage with another group.
        memory = shared_memory == nullptr ? std::make_shared<symmetric::SymmetricMemory>(num_bytes, use_huge1g_page) : shared_memory;
        EP_HOST_ASSERT(memory->num_bytes == num_bytes);
        buffer = memory->ptr;

        // Register buffer, make it visible to peers
        // TODO(HUAWEI): the class naming is so normal without namespaces
        CommMem mem;
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = static_cast<uint64_t>(this->num_bytes);
        HCCL_CHECK(HcclCommGetHandleWithName(hccl_group_name.c_str(), &comm));
        HCCL_CHECK(HcclCommMemReg(comm, buffer_uid.c_str(), &mem, &mem_handle));
        // TODO: do we need a barrier after member reg?

        // Discover peer buffer pointers
        peer_ptrs.resize(num_ranks, 0);
        peer_ptrs[rank_idx] = reinterpret_cast<int64_t>(buffer);

        if (num_ranks > 1) {
            EP_HOST_ASSERT(num_ranks % 2 == 0);
            // Prefer the second network layer, or the only layer on single-layer fabrics.
            uint32_t* layers = nullptr;
            uint32_t num_layers = 0;
            HCCL_CHECK(HcclRankGraphGetLayers(comm, &layers, &num_layers));
            EP_HOST_ASSERT(layers != nullptr and num_layers > 0);
            // HCCL owns `layers` and subsequent calls may invalidate it, so copy the layer value immediately.
            const auto layer = layers[num_layers > 1 ? 1 : 0];

            // Acquire UBMEM channels and Jetties
            acquire_ubmem_channels(layer);
            acquire_jetties(layer, num_jetties, buffer_uid, num_cpu_timeout_secs);
        }
    }

    ~HCCLContext() noexcept(false) {
        // TODO: do we have to destroy the connections as well?
        if (jetty_device_ptrs != nullptr) {
            ACL_CHECK(aclrtFree(jetty_device_ptrs));
            jetty_device_ptrs = nullptr;
        }
        if (not channel_handles.empty())
            HCOMM_CHECK(HcommChannelDestroy(channel_handles.data(), static_cast<uint32_t>(channel_handles.size())));
        for (size_t i = 0; i < jetty_endpoints.size(); ++ i) {
            HCOMM_CHECK(HcommMemUnreg(jetty_endpoints[i], jetty_mem_handles[i]));
            HCOMM_CHECK(HcommEndpointDestroy(jetty_endpoints[i]));
        }
        memory.reset();
        buffer = nullptr;
    }

private:
    std::vector<HcommMemHandle> jetty_mem_handles;
    // HCOMM shallow-copies channelName pointers; keep channel_names until channel destruction.
    std::vector<std::string> channel_names;

    // Build a channel desc for the first link to `peer_rank_idx` on `layer` whose
    // protocol satisfies `accept`. Returns nullptr if no such link exists.
    template <typename Accept>
    std::optional<HcclChannelDesc> make_peer_desc(uint32_t layer, int peer_rank_idx, Accept accept) {
        CommLink* links = nullptr;
        uint32_t num_links = 0;
        HCCL_CHECK(HcclRankGraphGetLinks(comm, layer, rank_idx, peer_rank_idx, &links, &num_links));
        for (uint32_t i = 0; i < num_links; ++ i) {
            const auto& link = links[i];
            const auto proto = link.linkAttr.linkProtocol;
            if (not accept(proto))
                continue;

            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank              = peer_rank_idx;
            desc.channelProtocol         = proto;
            desc.localEndpoint.protocol  = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr  = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc       = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc      = link.dstEndpointDesc.loc;
            // TODO: what do the magic numbers (3 and 1) mean?
            desc.notifyNum               = 3;
            // Attach our registered workspace so peers' writes land in our region.
            desc.memHandles              = &mem_handle;
            desc.memHandleNum            = 1;
            return desc;
        }
        return std::nullopt;
    }

    // Build descs for every non-self peer on `layer`, one per peer via the first
    // link matching `accept`. Asserts `err` if some peer has no matching link.
    template <typename Accept>
    std::vector<HcclChannelDesc> build_peer_descs(uint32_t layer, Accept accept, const char* err) {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(num_ranks - 1);
        for (int peer_rank_idx = 0; peer_rank_idx < num_ranks; ++ peer_rank_idx) {
            if (peer_rank_idx == rank_idx)
                continue;

            auto desc = make_peer_desc(layer, peer_rank_idx, accept);
            EP_HOST_ASSERT(desc.has_value() and err);
            descs.push_back(*desc);
        }
        return descs;
    }

    // One UBMEM channel per non-self peer on `layer`; resolves peer_ptrs to each
    // remote's registered workspace address.
    void acquire_ubmem_channels(uint32_t layer) {
        auto descs = build_peer_descs(layer,
            [](auto proto) { return proto == COMM_PROTOCOL_UB_MEM; },
            "no UB_MEM link to peer on the chosen layer");

        std::vector<ChannelHandle> channels(descs.size(), 0);
        HCCL_CHECK(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV,
                                      descs.data(), static_cast<uint32_t>(descs.size()),
                                      channels.data()));

        // The last remote mem is the registered userbuffer.
        for (size_t i = 0; i < descs.size(); ++ i) {
            const auto peer_rank_idx = static_cast<int>(descs[i].remoteRank);
            uint32_t num_mems = 0;
            CommMem* remote_mems = nullptr;
            char** mem_tags = nullptr;
            HCCL_CHECK(HcclChannelGetRemoteMems(comm, channels[i], &num_mems, &remote_mems, &mem_tags));
            EP_HOST_ASSERT(num_mems >= 1);
            EP_HOST_ASSERT(remote_mems[num_mems - 1].size == this->num_bytes);
            peer_ptrs[peer_rank_idx] = reinterpret_cast<int64_t>(remote_mems[num_mems - 1].addr);
        }
    }

    // Acquire exactly one shared Jetty per AIV, with separate channels for peer metadata.
    void acquire_jetties(uint32_t layer, const int& num_jetties,
                         const std::string& buffer_uid, const int& num_cpu_timeout_secs) {
        const auto peer_descs = build_peer_descs(layer,
            [](auto proto) { return proto == COMM_PROTOCOL_UBC_CTP; },
            "no UBC_CTP link to peer on the chosen layer");

        const auto num_peers = num_ranks - 1;
        const auto num_channels = math::max(num_jetties, num_peers);

        // Register the buffer on the HCOMM endpoint for UB_CTP; HCCL registration for UB_MEM.
        const auto& jetty_endpoint_desc = peer_descs.front().localEndpoint;

        // Each AIV uses one Jetty for all peers, so require same local endpoint.
        for (const auto& peer_desc: peer_descs) {
            const auto& endpoint = peer_desc.localEndpoint;
            EP_HOST_ASSERT(endpoint.protocol == COMM_PROTOCOL_UBC_CTP and peer_desc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP);
            EP_HOST_ASSERT(endpoint.protocol == jetty_endpoint_desc.protocol and
                           endpoint.commAddr.type == jetty_endpoint_desc.commAddr.type and
                           std::memcmp(endpoint.commAddr.raws, jetty_endpoint_desc.commAddr.raws, sizeof(endpoint.commAddr.raws)) == 0 and
                           endpoint.loc.locType == jetty_endpoint_desc.loc.locType and
                           std::memcmp(endpoint.loc.raws, jetty_endpoint_desc.loc.raws, sizeof(endpoint.loc.raws)) == 0 and
                           "multiple local UBC_CTP endpoints are not supported");
        }
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = buffer;
        mem.size = static_cast<uint64_t>(num_bytes);
        jetty_endpoints.resize(num_jetties);
        jetty_mem_handles.resize(num_jetties);
        for (int i = 0; i < num_jetties; ++ i) {
            HCOMM_CHECK(HcommEndpointCreate(&jetty_endpoint_desc, &jetty_endpoints[i]));
            HCOMM_CHECK(HcommMemReg(jetty_endpoints[i], buffer_uid.c_str(), &mem, &jetty_mem_handles[i]));
        }

        // Allocate a private listen port; port 0 in a channel desc otherwise selects the fixed port 60001.
        uint32_t listen_port = 0;
        HCOMM_CHECK(HcommEndpointGetListenPort(jetty_endpoints.front(), &listen_port));
        EP_HOST_ASSERT(listen_port > 0 and listen_port <= std::numeric_limits<uint16_t>::max());
        const auto listen_ports = gather_listen_ports(static_cast<uint16_t>(listen_port), num_cpu_timeout_secs);

        // The first entries describe every peer; pair the remaining Jetty owners directly.
        HcommChannelConfig config = nullptr;
        HCOMM_CHECK(HcommChannelConfigCreate(&config));
        HCOMM_CHECK(HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1));
        channel_handles.resize(num_channels);
        channel_names.resize(num_channels);
        for (int channel_idx = 0; channel_idx < num_channels; ++ channel_idx) {
            int peer_idx = channel_idx, pair_idx = 0;
            if (channel_idx >= num_peers) {
                const auto peer_rank_idx = rank_idx ^ 1;
                peer_idx = peer_rank_idx < rank_idx ? peer_rank_idx : peer_rank_idx - 1;
                pair_idx = channel_idx - num_peers + 1;
            }
            const auto& peer_desc = peer_descs[peer_idx];
            const auto peer_rank_idx = static_cast<int>(peer_desc.remoteRank);
            channel_names[channel_idx] = buffer_uid + "/" +
                std::to_string(std::min(rank_idx, peer_rank_idx)) + "-" +
                std::to_string(std::max(rank_idx, peer_rank_idx)) + "/" + std::to_string(pair_idx);
            EP_HOST_ASSERT(channel_names[channel_idx].size() <= HCOMM_CHANNEL_NAME_MAX_LEN);
            HcommChannelDesc desc;
            HCOMM_CHECK(HcommChannelDescInit(&desc, 1));
            desc.remoteEndpoint = peer_desc.remoteEndpoint;
            desc.notifyNum = 3;  // same as HCCL
            desc.memHandles = &jetty_mem_handles[channel_idx % num_jetties];
            desc.memHandleNum = 1;
            // The lower rank listens; the higher rank connects socket.
            desc.role = rank_idx < peer_rank_idx ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
            desc.port = listen_ports[std::min(rank_idx, peer_rank_idx)];
            desc.channelName = channel_names[channel_idx].c_str();
            // Leave desc.qos at 0xFFFFFFFF to use HCOMM's default UB QoS (4).
            // Leave desc.ubAttr.sqDepth at 0xFFFFFFFF (UB_SQ_DEPTH_NOT_SET) to use HCOMM's default depth.
            HCOMM_CHECK(HcommChannelCreateWithConfig(jetty_endpoints[channel_idx % num_jetties], CommEngine::COMM_ENGINE_AIV,
                                                     &desc, 1, config, channel_handles.data() + channel_idx));
        }
        HCOMM_CHECK(HcommChannelConfigDestroy(config));
        wait_for_channels(num_cpu_timeout_secs);

        // Device code indexes this table by both AIV and non-self peer.
        // Keep all peer entries visible even when peers outnumber AIV-owned queues.
        const auto num_table_bytes = sizeof(ChannelHandle) * channel_handles.size();
        ACL_CHECK(aclrtMalloc(&jetty_device_ptrs, num_table_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(jetty_device_ptrs, num_table_bytes,
                              channel_handles.data(), num_table_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::vector<uint16_t> gather_listen_ports(const uint16_t& listen_port, const int& num_cpu_timeout_secs) const {
        // Exchange once per buffer, regardless of the number of peers or Jetties.
        std::vector<uint16_t> listen_ports(num_ranks);
        const auto num_port_bytes = sizeof(uint16_t) * num_ranks;
        void* device_ports = nullptr;
        ACL_CHECK(aclrtMalloc(&device_ports, num_port_bytes + sizeof(uint16_t), ACL_MEM_MALLOC_NORMAL_ONLY));
        const auto stream = c10_npu::getCurrentNPUStream();
        const auto local_port = math::advance_ptr<void>(device_ports, num_port_bytes);
        ACL_CHECK(aclrtMemcpy(local_port, sizeof(listen_port), &listen_port, sizeof(listen_port), ACL_MEMCPY_HOST_TO_DEVICE));
        HCCL_CHECK(HcclAllGather(local_port, device_ports, 1, HCCL_DATA_TYPE_UINT16, comm, stream));
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, num_cpu_timeout_secs * 1000));
        ACL_CHECK(aclrtMemcpy(listen_ports.data(), num_port_bytes, device_ports, num_port_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtFree(device_ports));
        for (const auto& port: listen_ports)
            EP_HOST_ASSERT(port > 0);
        return listen_ports;
    }

    void wait_for_channels(const int& num_cpu_timeout_secs) const {
        // State values from the `HcommChannelGetStatus` contract.
        // TODO(HUAWEI): do not hardcode
        static constexpr int kChannelReady = 0;
        static constexpr int kChannelConnecting = 1;
        static constexpr int kChannelFailed = 2;
        static constexpr int kChannelTimeout = 3;

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(num_cpu_timeout_secs);
        std::vector<int32_t> statuses(channel_handles.size(), kChannelConnecting);
        while (true) {
            const auto result = HcommChannelGetStatus(channel_handles.data(), static_cast<uint32_t>(channel_handles.size()), statuses.data());
            if (result != HCCL_E_AGAIN)
                HCOMM_CHECK(result);

            if (result == HCCL_SUCCESS) {
                bool all_ready = true;
                for (const int& status: statuses) {
                    if (status == kChannelReady)
                        continue;

                    all_ready = false;
                    EP_HOST_ASSERT(status != kChannelFailed and status != kChannelTimeout);
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
