#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <deep_ep/layout/engram/workspace.hpp>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../utils/tensor.hpp"
#include "../kernels/engram/engram.hpp"

namespace deep_ep {

class EngramBuffer: public BufferBase {
    std::vector<int> num_engram_entries_per_layer;
    int engram_hidden = 0;
    int engram_elem_size = 0;
    int num_max_engram_tokens = 0;
    int num_engram_entries_per_token = 0;
    int num_engram_sf_packs = 0;
    int64_t num_engram_recv_bytes = 0;

public:
    std::shared_ptr<comm::Context> context;

    EngramBuffer(const int& rank_idx, const int& num_ranks, const std::string& hccl_group_name,
                 const int64_t& num_gpu_bytes, const int64_t& num_rdma_storage_bytes,
                 const bool& use_huge1g_page, const int& num_gpu_timeout_secs, const bool& explicitly_destroy):
        BufferBase(explicitly_destroy) {
        EP_HOST_ASSERT(num_gpu_bytes > 0 and num_gpu_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_rdma_storage_bytes > 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0);
        context = std::make_shared<comm::Context>(
            hccl_group_name, rank_idx, num_ranks, sizeof(layout::EngramWorkspace), num_gpu_bytes,
            num_rdma_storage_bytes, use_huge1g_page, 300, num_gpu_timeout_secs);
        main_context = context;
        auto& workspace = *static_cast<layout::EngramWorkspace*>(context->workspace);
        context->set_common_signals(&workspace.common_signals);
    }

    ~EngramBuffer() noexcept(false) override {
        destroy_on_destruction("Engram");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        comm::barrier(*context, true);
        context->finalize();
        context = nullptr;
        destroyed = true;
    }

    void set_config(const std::vector<int>& num_entries_per_layer,
                    const int& hidden, const int& elem_size,
                    const int& num_max_tokens, const int& num_entries_per_token,
                    const int& num_sf_packs) {
        EP_HOST_ASSERT(not destroyed);
        comm::barrier(*context);

        EP_HOST_ASSERT(num_max_tokens > 0 and num_entries_per_token > 0 and hidden > 0 and num_sf_packs >= 0);
        EP_HOST_ASSERT(not num_entries_per_layer.empty());
        EP_HOST_ASSERT((elem_size == 1 and num_sf_packs > 0) or (elem_size == 2 and num_sf_packs == 0));
        int64_t num_total_entries = 0;
        for (const auto& num_entries: num_entries_per_layer) {
            EP_HOST_ASSERT(num_entries > 0);
            num_total_entries += num_entries;
        }

        const auto num_storage_bytes = num_total_entries * hidden * elem_size;
        num_engram_recv_bytes = static_cast<int64_t>(num_entries_per_layer.size()) *
                                num_max_tokens * num_entries_per_token * hidden * elem_size;
        const auto num_sf_bytes = static_cast<int64_t>(num_total_entries) *
                                  num_sf_packs * sizeof(sf_pack_t);
        EP_HOST_ASSERT(num_engram_recv_bytes + num_sf_bytes <= context->num_gpu_buffer_bytes);
        EP_HOST_ASSERT(num_storage_bytes <= context->num_rdma_storage_bytes);

        num_engram_entries_per_layer = num_entries_per_layer;
        engram_hidden = hidden;
        engram_elem_size = elem_size;
        num_max_engram_tokens = num_max_tokens;
        num_engram_entries_per_token = num_entries_per_token;
        num_engram_sf_packs = num_sf_packs;
    }

    void write(const std::vector<torch::Tensor>& storages,
               const std::optional<std::vector<torch::Tensor>>& sfs) {
        EP_HOST_ASSERT(not destroyed);
        comm::barrier(*context);
        // ensure data is ready
        ACL_CHECK(aclrtSynchronizeDevice());

        const auto num_layers = static_cast<int>(num_engram_entries_per_layer.size());
        EP_HOST_ASSERT(num_layers > 0 and storages.size() == num_layers);
        EP_HOST_ASSERT(sfs.has_value() == (num_engram_sf_packs > 0));

        // Write local storage
        auto* storage_ptr = context->local_rdma_storage_ptr;
        for (int layer_idx = 0; layer_idx < num_layers; ++ layer_idx) {
            const auto& storage = storages[layer_idx];
            EP_HOST_ASSERT(storage.dim() == 2 and storage.size(0) == num_engram_entries_per_layer[layer_idx] and
                           storage.size(1) == engram_hidden and storage.element_size() == engram_elem_size);
            EP_HOST_ASSERT(storage.scalar_type() == (num_engram_sf_packs > 0 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
            EP_HOST_ASSERT(storage.device().type() == TORCH_DEVICE_NPU and storage.is_contiguous());

            ACL_CHECK(aclrtMemcpy(
                storage_ptr, storage.nbytes(),
                storage.data_ptr(), storage.nbytes(), ACL_MEMCPY_DEVICE_TO_DEVICE
            ));
            storage_ptr = math::advance_ptr<void>(storage_ptr, storage.nbytes());
        }
        // Write the local SF shard
        if (sfs.has_value()) {
            EP_HOST_ASSERT(static_cast<int>(sfs->size()) == num_layers);

            auto* sf_ptr = math::advance_ptr<void>(context->buffer, num_engram_recv_bytes);
            for (int layer_idx = 0; layer_idx < num_layers; ++ layer_idx) {
                const auto& sf = sfs->at(layer_idx);
                EP_HOST_ASSERT(sf.dim() == 2 and sf.size(0) == context->num_ranks * num_engram_entries_per_layer[layer_idx] and
                               sf.size(1) == num_engram_sf_packs);
                EP_HOST_ASSERT(sf.device().type() == TORCH_DEVICE_NPU and sf.is_contiguous() and sf.element_size() == sizeof(sf_pack_t));

                const auto num_sf_shard_bytes = static_cast<int64_t>(sf.nbytes()) / context->num_ranks;
                ACL_CHECK(aclrtMemcpy(
                    sf_ptr, num_sf_shard_bytes,
                    math::advance_ptr<void>(sf.data_ptr(), context->rank_idx * num_sf_shard_bytes),
                    num_sf_shard_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE
                ));
                sf_ptr = math::advance_ptr<void>(sf_ptr, num_sf_shard_bytes);
            }
        }

        comm::barrier(*context);
    }

    std::vector<std::function<pybind11::object()>>
    fetch(const torch::Tensor& indices, const int& num_qps,
          const bool& use_tma_aligned_col_major_sf) const {
        EP_HOST_ASSERT(not destroyed and num_qps == 0);
        EP_HOST_ASSERT(indices.dim() == 3);
        const auto use_fp8 = num_engram_sf_packs > 0;
        const auto fetched_dtype = use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(indices.is_contiguous() and indices.device().type() == TORCH_DEVICE_NPU);
        const auto num_layers = static_cast<int>(indices.size(0));
        const auto num_tokens = static_cast<int>(indices.size(1));
        const auto num_entries_per_token = static_cast<int>(indices.size(2));
        EP_HOST_ASSERT(num_layers == num_engram_entries_per_layer.size());
        EP_HOST_ASSERT(num_tokens <= num_max_engram_tokens);
        EP_HOST_ASSERT(num_entries_per_token == num_engram_entries_per_token);

        const auto num_layer_bytes = static_cast<int64_t>(num_max_engram_tokens) *
                                     num_entries_per_token * engram_hidden * engram_elem_size;
        std::vector<torch::Tensor> fetched;
        for (int layer_idx = 0; layer_idx < num_layers; ++ layer_idx)
            fetched.push_back(at_npu::native::from_blob(
                math::advance_ptr<void>(context->buffer, layer_idx * num_layer_bytes),
                {num_tokens, num_entries_per_token * engram_hidden},
                [context = context](void*) {},
                torch::TensorOptions().dtype(fetched_dtype).device(TORCH_DEVICE_NPU)));

        // Create SF views
        std::vector<torch::Tensor> fetched_sf;
        void* sf_ptr = nullptr;
        void* fetched_sf_ptr = nullptr;
        int64_t sf_token_stride = 0, sf_hidden_stride = 0, sf_layer_stride = 0;
        if (use_fp8) {
            sf_token_stride = use_tma_aligned_col_major_sf ? 1 : num_entries_per_token * num_engram_sf_packs;
            sf_hidden_stride = use_tma_aligned_col_major_sf ? num_tokens : 1;
            sf_layer_stride = static_cast<int64_t>(num_tokens) * num_entries_per_token * num_engram_sf_packs;

            const auto sf_backing = torch::empty(
                {num_layers, sf_layer_stride},
                torch::TensorOptions().dtype(torch::kInt16).device(TORCH_DEVICE_NPU));
            for (int layer_idx = 0; layer_idx < num_layers; ++ layer_idx)
                fetched_sf.push_back(sf_backing[layer_idx].as_strided(
                    {num_tokens, num_entries_per_token * num_engram_sf_packs},
                    {sf_token_stride, sf_hidden_stride}));

            sf_ptr = math::advance_ptr<void>(context->buffer, num_engram_recv_bytes);
            fetched_sf_ptr = sf_backing.data_ptr();
        }

        // Launch fetch
        launch_engram_fetch(
            context->local_rdma_storage_ptr,
            context->buffer,
            indices.data_ptr<int>(), context->workspace,
            sf_ptr, fetched_sf_ptr,
            sf_token_stride, sf_hidden_stride, sf_layer_stride,
            num_engram_entries_per_layer,
            context->num_ranks, engram_hidden * engram_elem_size,
            num_engram_sf_packs, num_entries_per_token,
            num_tokens, num_max_engram_tokens,
            static_cast<int>(runtime->get_num_ubuf_bytes_per_vec_core()),
            runtime->get_num_vec_cores()
        );

        // Create hooks
        std::vector<std::function<pybind11::object()>> hooks;
        for (int layer_idx = 0; layer_idx < num_layers; ++ layer_idx)
            hooks.emplace_back([=]() -> pybind11::object {
                return use_fp8 ? pybind11::make_tuple(fetched[layer_idx], fetched_sf[layer_idx])
                               : pybind11::cast(fetched[layer_idx]);
            });
        return hooks;
    }

};

}  // namespace deep_ep
