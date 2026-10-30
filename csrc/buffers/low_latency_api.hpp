#pragma once
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "low_latency.hpp"
#include "../kernels/ep/low_latency.hpp"
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h>
#include <deep_ep/impls/low_latency/combine/low_latency/combine_layout.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_tiling.h>

namespace py = pybind11;
namespace deep_ep::low_latency {
namespace {
using Shape = std::vector<int64_t>;
uint8_t* tensor(py::handle value, const Shape& shape, const char* dtype)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(value).cast<bool>()) {
        throw std::invalid_argument("LL requires tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(value);
    if (!obj.attr("is_contiguous")().cast<bool>() || obj.attr("shape").cast<Shape>() != shape ||
        !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("LL tensor shape, dtype or contiguity mismatch");
    }
    const auto device = obj.attr("device");
    const auto type = device.attr("type").cast<std::string>();
    if ((type != "npu" && type != "privateuseone") ||
        device.attr("index").cast<int>() != c10_npu::getCurrentNPUStream().device_index()) {
        throw std::invalid_argument("LL tensors must be on the current NPU");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
DispatchDedup::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    DispatchDedup::EpMemoryServerDedupLayout result{};
    auto mode = fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                    : DispatchDedup::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 256 ||
        DispatchDedup::BuildDispatchServerDedupLayout(p, b, h, k, e, &result, mode) !=
            DispatchDedup::DISPATCH_DEDUP_SUCCESS ||
        !DispatchDedup::DispatchDedupPacketUbFits(result, b, k, e, p) || result.capacityRows > CombineDedup::kEpServerDedupCombineMaxTokenNum) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    return result;
}
template <class T>
py::bytes bytes(const T& data)
{
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}
uint8_t dtype_code(const std::string& dtype)
{
    if (dtype == "bfloat16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (dtype == "float16") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (dtype == "float8_e4m3fn") return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    throw std::invalid_argument("LL dispatch supports BF16/FP16 or packed FP8 E4M3FN");
}
void validate_capacity(int b, int capacity)
{
    if (b <= 0 || b > capacity || capacity > 256)
        throw std::invalid_argument("LL requires 0 < B <= num_max_tokens_per_rank <= 256");
}
}  // namespace

inline void register_apis(py::module_& module)
{
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        dtype_code(dtype);
        try {
            auto plan = layout(p, b, h, k, e, dtype == "float8_e4m3fn");
            py::dict result;
            result["capacity_rows"] = plan.capacityRows;
            return py::object(result);
        } catch (const std::invalid_argument&) {
            return py::object(py::none());
        }
    });
    module.def("low_latency_dispatch_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype,
                                                 int num_max_tokens_per_rank) {
        validate_capacity(b, num_max_tokens_per_rank);
        const bool fp8 = dtype == "float8_e4m3fn";
        layout(p, b, h, k, e, fp8);
        DispatchDedup::EpServerDedupTilingData data{};
        data.bs = b;
        data.h = h;
        data.numMaxTokensPerRank = num_max_tokens_per_rank;
        data.topK = k;
        data.moeExpertNum = e;
        data.expertTokenNumsType = 1;
        data.dtype = dtype_code(dtype);
        data.quantMode = fp8 ? 4 : 0;
        data.inputMode = static_cast<uint8_t>(fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs
                                                  : DispatchDedup::EpServerDedupInputMode::Plain16);
        data.expandXOutDtype = data.dtype;
        return bytes(data);
    });
    module.def("low_latency_combine_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype,
                                                int num_max_tokens_per_rank) {
        validate_capacity(b, num_max_tokens_per_rank);
        CombineDedup::Layout plan{};
        const auto ub_bytes = get_low_latency_combine_ub_bytes();
        if ((dtype != "bfloat16" && dtype != "float16") || !CombineDedup::BuildLayout(p, b, h, k, e, plan, ub_bytes)) {
            throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
        }
        CombineDedup::EpServerDedupCombineTilingData data{};
        data.bs = b;
        data.h = h;
        data.numMaxTokensPerRank = num_max_tokens_per_rank;
        data.topK = k;
        data.moeExpertNum = e;
        data.dtype = dtype_code(dtype);
        data.recvUbBytes = ub_bytes;
        return bytes(data);
    });
    module.def("dispatch_low_latency", [](LowLatencyBuffer& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          const std::string& dtype) {
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto plan = layout(session.world(), b, h, k, e, fp8);
        const int64_t c = plan.capacityRows;
        if (rows < c || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupKernelArgs args{};
        args.commArgs = static_cast<uint8_t*>(session.comm_args.data_ptr());
        args.x = get("input", {b, h}, dtype.c_str());
        args.expertIds = get("indices", {b, k}, "int32");
        args.expertScales = get("weights", {b, k}, "float32");
        args.expandXOut = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        args.dynamicScalesOut = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        args.expertTokenNumsOut = get("expert_recv_counts", {e / session.world()}, "int64");
        args.sendCountsOut = get("send_counts", {e}, "int32");
        args.tokenTypeOut = get("token_type", {c}, "int32");
        args.destinationIndexOut = get("destination_index", {c, 3}, "int32");
        args.relayReadIndexOut = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMaskOut = get("source_mask", {b}, "int32");
        args.expertScalesOut = get("source_weights", {b, k}, "float32");
        args.tilingData = get("tiling", {sizeof(DispatchDedup::EpServerDedupTilingData)}, "uint8");
        args.inputScales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        session.check_available();
        py::gil_scoped_release release;
        launch_dispatch_low_latency(args);
    });
    module.def("combine_low_latency", [](LowLatencyBuffer& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype) {
        CombineDedup::Layout plan{};
        if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16") ||
            !CombineDedup::BuildLayout(session.world(), b, h, k, e, plan, session.combine_ub_bytes)) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        const int64_t c = plan.capacityRows;
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupCombineKernelArgs args{};
        args.commArgs = static_cast<uint8_t*>(session.comm_args.data_ptr());
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {b}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.check_available();
        py::gil_scoped_release release;
        launch_combine_low_latency(args);
    });
}
}  // namespace deep_ep::low_latency
