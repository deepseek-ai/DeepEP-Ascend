#pragma once

#include <cstring>
#include <string>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "barrier.hpp"

namespace deep_ep::comm {

static pybind11::bytearray get_local_unique_id() {
    HcclRootInfo root_info;
    HCCL_CHECK(HcclGetRootInfo(&root_info));
    return pybind11::bytearray(reinterpret_cast<const char*>(&root_info), sizeof(root_info));
}

static std::string create_hccl_comm(const pybind11::bytearray& root_unique_id,
                                   const int& num_ranks, const int& rank_idx) {
    EP_HOST_ASSERT(num_ranks > 0 and rank_idx >= 0 and rank_idx < num_ranks);
    const auto bytes = root_unique_id.cast<std::string>();
    EP_HOST_ASSERT(bytes.size() == sizeof(HcclRootInfo));
    HcclRootInfo root_info;
    std::memcpy(&root_info, bytes.data(), sizeof(root_info));
    HcclComm comm = nullptr;
    HCCL_CHECK(HcclCommInitRootInfo(num_ranks, &root_info, rank_idx, &comm));
    char name[COMM_NAME_MAX_LENGTH] = {};
    HCCL_CHECK(HcclGetCommName(comm, name));
    return name;
}

static void destroy_hccl_comm(const std::string& name) {
    HcclComm comm = nullptr;
    HCCL_CHECK(HcclCommGetHandleWithName(name.c_str(), &comm));
    HCCL_CHECK(HcclCommDestroy(comm));
}

static std::tuple<int, int> get_physical_domain_size(const std::string& hccl_group_name) {
    HcclComm comm = nullptr;
    HCCL_CHECK(HcclCommGetHandleWithName(hccl_group_name.c_str(), &comm));
    uint32_t rank_idx = 0, num_ranks = 0;
    HCCL_CHECK(HcclGetRankId(comm, &rank_idx));
    HCCL_CHECK(HcclGetRankSize(comm, &num_ranks));
    EP_HOST_ASSERT(num_ranks > 0 and rank_idx < num_ranks);
    if (num_ranks == 1)
        return {1, 1};
    uint32_t* layers = nullptr;
    uint32_t num_layers = 0;
    HCCL_CHECK(HcclRankGraphGetLayers(comm, &layers, &num_layers));
    EP_HOST_ASSERT(layers != nullptr and num_layers > 0);
    // Copy the selected layer before subsequent HCCL calls invalidate the list.
    const auto layer = layers[num_layers > 1 ? 1 : 0];
    for (int peer_rank_idx = 0; peer_rank_idx < num_ranks; ++ peer_rank_idx) {
        if (peer_rank_idx == rank_idx)
            continue;
        CommLink* links = nullptr;
        uint32_t num_links = 0;
        HCCL_CHECK(HcclRankGraphGetLinks(comm, layer, rank_idx, peer_rank_idx, &links, &num_links));
        bool has_ubmem = false;
        for (uint32_t link_idx = 0; link_idx < num_links; ++ link_idx)
            has_ubmem = has_ubmem or links[link_idx].linkAttr.linkProtocol == COMM_PROTOCOL_UB_MEM;
        EP_HOST_ASSERT(has_ubmem and "Ascend currently requires direct UBMEM connectivity to every peer");
    }
    return {1, num_ranks};
}

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Context, std::shared_ptr<Context>>(m, "Context")
        .def("get_physical_domain_size", &Context::get_physical_domain_size)
        .def("get_logical_domain_size", &Context::get_logical_domain_size)
        .def_readonly("num_workspace_bytes", &Context::num_workspace_bytes)
        .def_readonly("num_gpu_buffer_bytes", &Context::num_gpu_buffer_bytes)
        .def_readonly("num_rdma_storage_bytes", &Context::num_rdma_storage_bytes);
    m.def("barrier", [](const std::shared_ptr<Context>& context, const bool& with_cpu_sync,
                        const bool& sequential, const bool& wait_comm_stream) {
        EP_HOST_ASSERT(context != nullptr);
        barrier(*context, with_cpu_sync, sequential, wait_comm_stream);
    });
    m.def("get_physical_domain_size", &get_physical_domain_size);
    m.def("get_logical_domain_size", [](const std::string& hccl_group_name, const bool& allow_hybrid_mode) {
        return get_physical_domain_size(hccl_group_name);
    });
    m.def("get_local_hccl_unique_id", &get_local_unique_id);
    m.def("create_hccl_comm", &create_hccl_comm);
    m.def("destroy_hccl_comm", &destroy_hccl_comm);
    m.def("get_comm_stream", []() -> torch::Stream { return get_comm_stream(); });
}

}  // namespace deep_ep::comm
