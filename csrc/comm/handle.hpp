#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "barrier.hpp"

namespace deep_ep::comm {

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Context, std::shared_ptr<Context>>(m, "Context")
        .def("get_physical_domain_size", &Context::get_physical_domain_size)
        .def("get_logical_domain_size", &Context::get_logical_domain_size)
        .def_readonly("num_workspace_bytes", &Context::num_workspace_bytes)
        .def_readonly("num_gpu_buffer_bytes", &Context::num_gpu_buffer_bytes)
        .def_readonly("num_rdma_storage_bytes", &Context::num_rdma_storage_bytes)
        .def("get_error_status", [](const std::shared_ptr<Context>& context) {
            EP_HOST_ASSERT(context != nullptr);
            return context->get_error_status();
        });

    m.def("barrier", [](const std::shared_ptr<Context>& context, const bool& with_cpu_sync,
                        const bool& sequential, const bool& wait_comm_stream) {
        EP_HOST_ASSERT(context != nullptr);
        barrier(*context, with_cpu_sync, sequential, wait_comm_stream);
    });
    m.def("get_physical_domain_size", [](const ProcessGroup& group) {
        return get_physical_domain_size(group);
    });
    m.def("get_logical_domain_size", [](const ProcessGroup& group, const bool& allow_hybrid_mode) {
        return get_physical_domain_size(group);
    });
    m.def("get_comm_stream", []() -> torch::Stream { return get_comm_stream(); });
}

}  // namespace deep_ep::comm
