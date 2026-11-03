#pragma once

#include <pybind11/pybind11.h>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>

#include "tensor.hpp"
#include "../kernels/utils/l2_cache.hpp"

namespace deep_ep::l2_cache {

static void zero_l2_cache_clean(const torch::Tensor& buffer) {
    EP_HOST_ASSERT(buffer.device().type() == TORCH_DEVICE_NPU);
    EP_HOST_ASSERT(buffer.is_contiguous() and buffer.nbytes() > 0);
    EP_HOST_ASSERT(buffer.nbytes() % kNumUbAlignmentBytes == 0);

    launch_l2_cache_clean(
        buffer.data_ptr(), buffer.nbytes(), c10_npu::getCurrentNPUStream());
}

static void register_apis(pybind11::module_& m) {
    m.def(
        "zero_l2_cache_clean", &zero_l2_cache_clean,
        pybind11::arg("buffer"),
        "Overwrite an already-zeroed NPU buffer with NOTALLOC_CLEAN zero stores.");
}

}  // namespace deep_ep::l2_cache
