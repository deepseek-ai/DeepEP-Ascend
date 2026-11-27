#pragma once

#include <chrono>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace deep_ep::comm {

using ProcessGroup = c10::intrusive_ptr<c10d::ProcessGroup>;

// Exchange a fixed-size POD record across a CPU or device (NPU) ProcessGroup.
// The caller holds the GIL. Release it so Python backends can make progress while we wait.
template <typename T>
inline std::vector<T> all_gather_fixed(const ProcessGroup& group, const T& payload, const int num_timeout_secs) {
    static_assert(std::is_trivially_copyable_v<T>);
    pybind11::gil_scoped_release release;
    TORCH_CHECK(group != nullptr and group->getSize() > 0, "a live ProcessGroup is required");
    TORCH_CHECK(num_timeout_secs > 0, "the ProcessGroup exchange timeout must be positive");
    const int size = group->getSize();
    if (size == 1)
        return {payload};

    // Single-backend groups use their registered device (CPU or NPU); multi-device groups prefer CPU.
    const auto devices = group->getDeviceTypes();
    const auto device = devices.size() == 1 ? devices.front() : c10::Device(c10::kCPU);
    auto bytes = at::empty({static_cast<int64_t>(sizeof(T))}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(bytes.data_ptr(), &payload, sizeof(T));
    std::vector<at::Tensor> inputs{bytes.to(device)};
    auto output = at::empty({size, bytes.numel()}, inputs[0].options());
    std::vector<std::vector<at::Tensor>> outputs(1);
    outputs[0].reserve(size);
    for (int rank = 0; rank < size; ++ rank)
        outputs[0].push_back(output[rank]);
    c10d::AllgatherOptions options;
    const auto timeout = std::chrono::milliseconds(static_cast<int64_t>(num_timeout_secs) * 1000);
    options.timeout = timeout;
    auto work = group->allgather(outputs, inputs, options);
    TORCH_CHECK(work != nullptr and work->wait(timeout), "ProcessGroup allgather did not complete");
    if (output.device().type() == c10::DeviceType::PrivateUse1) {
        // HCCL wait() may only enqueue a stream dependency; bound the device wait before copying to CPU.
        TORCH_CHECK(timeout.count() <= std::numeric_limits<int32_t>::max(), "ProcessGroup exchange timeout is too large");
        const auto stream = c10_npu::getCurrentNPUStream(output.get_device());
        const auto status = aclrtSynchronizeStreamWithTimeout(stream, static_cast<int32_t>(timeout.count()));
        TORCH_CHECK(status == ACL_SUCCESS, "ProcessGroup allgather stream synchronization failed, ACL error code: ", status);
    }

    std::vector<T> result(size);
    std::memcpy(result.data(), output.cpu().const_data_ptr(), result.size() * sizeof(T));
    return result;
}

}  // namespace deep_ep::comm
