#pragma once

#include <cstdio>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>

#include "../kernels/comm/api.hpp"

namespace deep_ep {

class BufferBase {
protected:
    bool explicitly_destroy;
    bool destroyed = false;

    explicit BufferBase(const bool& explicitly_destroy): explicitly_destroy(explicitly_destroy) {}

    // Called from the derived destructor, while its members and destroy() implementation are still available.
    void destroy_on_destruction(const char* name) {
        if (not explicitly_destroy and not destroyed) {
            destroy();
            main_context = nullptr;
        }

        if (not destroyed) {
            std::printf("`destroy()` is not called before DeepEP %s buffer destruction, which can leak resources.\n", name);
            std::fflush(stdout);
        }
    }

public:
    std::shared_ptr<comm::Context> main_context;

    virtual ~BufferBase() noexcept(false) = default;
    virtual void destroy() = 0;

    // BucketBuffer overrides this pair to cover every context, not only the main one.
    virtual std::vector<layout::ErrorCode> get_error_status() const {
        EP_HOST_ASSERT(not destroyed and main_context != nullptr and "error record unavailable after destroy");
        return {main_context->get_error_status()};
    }

    virtual void clear_error_status() const {
        EP_HOST_ASSERT(not destroyed and main_context != nullptr and "error record unavailable after destroy");
        main_context->clear_error_status();
    }

    void print_memory_usage() const {
        EP_HOST_ASSERT(not destroyed and main_context != nullptr);
        const auto num_workspace_gibs = main_context->num_workspace_bytes / static_cast<float>(1 << 30);
        const auto num_buffer_gibs = main_context->num_gpu_buffer_bytes / static_cast<float>(1 << 30);
        const auto num_gpu_storage_gibs = main_context->use_cpu_rdma_storage ? 0.0 : main_context->num_rdma_storage_bytes / static_cast<float>(1 << 30);
        const auto num_cpu_storage_gibs = main_context->use_cpu_rdma_storage ? main_context->num_rdma_storage_bytes / static_cast<float>(1 << 30) : 0.0;
        std::printf("[DeepEP memory usage] rank_idx=%d, num_ranks=%d, GPU total %.3f GiB "
                    "(workspace %.3f GiB, buffer %.3f GiB, RDMA storage %.3f GiB), CPU total %.3f GiB\n",
                    main_context->rank_idx, main_context->num_ranks,
                    num_workspace_gibs + num_buffer_gibs + num_gpu_storage_gibs,
                    num_workspace_gibs, num_buffer_gibs, num_gpu_storage_gibs, num_cpu_storage_gibs);
        std::fflush(stdout);
    }
};

}  // namespace deep_ep

namespace deep_ep::base {

static void register_apis(pybind11::module_& m) {
    pybind11::class_<BufferBase>(m, "BufferBase")
        .def_readonly("main_context", &BufferBase::main_context)
        .def("print_memory_usage", &BufferBase::print_memory_usage)
        .def("get_error_status", &BufferBase::get_error_status)
        .def("clear_error_status", &BufferBase::clear_error_status)
        .def("destroy", &BufferBase::destroy);
}

}  // namespace deep_ep::base
