#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "buffers/base.hpp"
#include "buffers/bucket.hpp"
#include "buffers/engram.hpp"
#include "buffers/ep.hpp"
#include "buffers/pp.hpp"
#include "comm/api.hpp"
#include "runtime/jit.hpp"
#include "utils/event.hpp"
#include "utils/l2_cache.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    using namespace deep_ep;

    m.doc() = "DeepEP communication buffers (Ascend)";
    config::register_apis(m);
    l2_cache::register_apis(m);
    m.attr("topk_idx_t") = py::cast(torch::kInt64);

    py::class_<EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    comm::register_apis(m);
    base::register_apis(m);

    py::class_<EPBuffer, BufferBase>(m, "EPBuffer")
        .def(py::init<int, int, std::string, int64_t, int64_t, bool, int, int, bool>())
        .def_readonly("context", &EPBuffer::context)
        .def_readonly("lb_storage", &EPBuffer::lb_storage)
        .def("lb_prefetch_weights", &EPBuffer::lb_prefetch_weights)
        .def("lb_reduce_grads", &EPBuffer::lb_reduce_grads)
        .def("dispatch", &EPBuffer::dispatch)
        .def("combine", &EPBuffer::combine);

    py::class_<PPBuffer, BufferBase>(m, "PPBuffer")
        .def(py::init<int, int, std::string, int64_t, int, bool, int, bool>())
        .def_readonly("context", &PPBuffer::context)
        .def_readonly("storage", &PPBuffer::storage)
        .def("send", &PPBuffer::send)
        .def("recv", &PPBuffer::recv);

    py::class_<EngramBuffer, BufferBase>(m, "EngramBuffer")
        .def(py::init<int, int, std::string, int64_t, int64_t, bool, int, bool>())
        .def_readonly("context", &EngramBuffer::context)
        .def("set_config", &EngramBuffer::set_config)
        .def("write", &EngramBuffer::write)
        .def("fetch", &EngramBuffer::fetch);

    py::class_<BucketBuffer, BufferBase>(m, "BucketBuffer")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<std::string>, int64_t, bool, int, bool>())
        .def_readonly("contexts", &BucketBuffer::contexts)
        .def_readonly("storage", &BucketBuffer::storage)
        .def("reduce_scatter", &BucketBuffer::reduce_scatter)
        .def("all_reduce", &BucketBuffer::all_reduce)
        .def("all_gather", &BucketBuffer::all_gather);

    m.def("calculate_ep_buffer_size", &EPBuffer::calculate_buffer_size);
    m.def("get_num_allocation_alignment", []() {
        return kNumAllocationAlignmentBytes;
    });
    m.def("get_num_tma_alignment", []() {
        return kNumTMAAlignmentBytes;
    });
    m.def("get_num_rdma_alignment", []() {
        return kNumRDMAAlignmentBytes;
    });
}
