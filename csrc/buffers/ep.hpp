#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/common/reduction.hpp>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../utils/tensor.hpp"
#include "../kernels/ep/api.hpp"
#include "../kernels/ep/low_latency_policy.hpp"
#include "../utils/event.hpp"

namespace deep_ep {

class EPBuffer: public BufferBase {
    int64_t num_buffer_bytes;
    void* host_workspace = nullptr;
    void* mapped_host_workspace = nullptr;

    static int get_num_used_vec_cores(const int num_ai_cores) {
        // Persistent kernels require all participating AIVs to be resident.
        // Query the device directly: runtime vector-core overrides are not physical limits.
        const auto physical_ai_cores = jit->device.get_num_aicore_cores();
        const auto physical_vec_cores = jit->device.get_num_vec_cores();
        EP_HOST_ASSERT(physical_ai_cores > 0 and physical_vec_cores > 0);
        if (num_ai_cores <= 0 or num_ai_cores > physical_ai_cores)
            throw EPExceptionWithLineInfo("EPBuffer", "num_sms must be between 1 and " +
                std::to_string(physical_ai_cores) + " physical AI cores; requested " +
                std::to_string(num_ai_cores));

        const auto configured_vec_cores = runtime->get_num_vec_cores();
        if (configured_vec_cores <= 0 or configured_vec_cores > physical_vec_cores)
            throw EPExceptionWithLineInfo("EPBuffer", "Configured AIV count " +
                std::to_string(configured_vec_cores) + " exceeds the available range 1.." +
                std::to_string(physical_vec_cores));
        const auto num_used_vec_cores = static_cast<int64_t>(num_ai_cores) *
            (configured_vec_cores / physical_ai_cores);
        if (num_used_vec_cores <= 0 or num_used_vec_cores > physical_vec_cores)
            throw EPExceptionWithLineInfo("EPBuffer", "Requested AI cores map to " +
                std::to_string(num_used_vec_cores) + " AIVs; the physical device provides " +
                std::to_string(physical_vec_cores));
        return static_cast<int>(num_used_vec_cores);
    }

public:
    std::shared_ptr<comm::Context> context;
    torch::Tensor lb_storage;

    EPBuffer(const int& rank_idx, const int& num_ranks, const std::string& hccl_group_name,
             const int64_t& num_buffer_bytes, const int64_t& num_lb_buffer_bytes,
             const bool& use_huge1g_page,
             const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
             const bool& explicitly_destroy):
        BufferBase(explicitly_destroy), num_buffer_bytes(num_buffer_bytes) {
        EP_HOST_ASSERT(num_buffer_bytes >= 0 and num_buffer_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_lb_buffer_bytes >= 0 and num_lb_buffer_bytes % kNumAllocationAlignmentBytes == 0);
        context = std::make_shared<comm::Context>(
            hccl_group_name, rank_idx, num_ranks, sizeof(layout::EPSignals), num_buffer_bytes + num_lb_buffer_bytes, 0,
            use_huge1g_page, num_cpu_timeout_secs, num_gpu_timeout_secs);
        main_context = context;
        auto& workspace = *static_cast<layout::EPSignals*>(context->workspace);
        context->set_common_signals(&workspace.common_signals);
        lb_storage = torch::from_blob(
            context->buffer, {num_buffer_bytes + num_lb_buffer_bytes}, [context = context](void*) {},
            torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU)
        ).narrow(0, num_buffer_bytes, num_lb_buffer_bytes);
        ACL_CHECK(aclrtMallocHost(&host_workspace, sizeof(layout::EPSignals)));
        ACL_CHECK(aclrtHostRegister(host_workspace, sizeof(layout::EPSignals), ACL_HOST_REGISTER_MAPPED, &mapped_host_workspace));
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(host_workspace) % alignof(layout::EPSignals) == 0 and
                       reinterpret_cast<uintptr_t>(mapped_host_workspace) % alignof(layout::EPSignals) == 0);
        std::memset(host_workspace, 0, sizeof(layout::EPSignals));
    }

    ~EPBuffer() noexcept(false) override {
        destroy_on_destruction("EP");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        comm::barrier(*context, true);
        ACL_CHECK(aclrtHostUnregister(host_workspace));
        ACL_CHECK(aclrtFreeHost(host_workspace));
        context->finalize();
        lb_storage = torch::Tensor();
        destroyed = true;
    }

    EventHandle lb_prefetch_weights(const std::vector<torch::Tensor>& redundant_expert_weights,
                                    const std::vector<torch::Tensor>& expert_weights,
                                    const torch::Tensor& redundancy_mapping, const int& num_sms,
                                    const std::optional<EventHandle>& previous_event) const {
        EP_HOST_UNREACHABLE("EPBuffer::lb_prefetch_weights is not implemented on Ascend");
    }

    EventHandle lb_reduce_grads(const torch::Tensor& redundant_expert_grads,
                                const torch::Tensor& expert_grads, const torch::Tensor& redundancy_mapping,
                                const int& num_sms, const std::optional<EventHandle>& previous_event) const {
        EP_HOST_UNREACHABLE("EPBuffer::lb_reduce_grads is not implemented on Ascend");
    }

    pybind11::tuple
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_dst_gsge_idx,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_ai_cores,
             const std::optional<EventHandle>& previous_event,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_cube_aligned_col_major_sf,
             const bool& defer_epilogue) const {
        EP_HOST_ASSERT(not destroyed);

        // Ascend EP has some limitations over NVIDIA version
        EP_HOST_ASSERT(do_expand);
        EP_HOST_ASSERT(not do_handle_copy);

        // Cached dispatch is only supported for the expand-only, non-hybrid Ascend path
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
            EP_HOST_ASSERT(not cumulative_local_expert_recv_stats.has_value());
            EP_HOST_ASSERT(cached_num_expanded_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_num_unaligned_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());
            EP_HOST_ASSERT(cached_dst_gsge_idx.has_value());
            EP_HOST_ASSERT(cached_recv_src_metadata.has_value());
        }

        // Check data tensor
        const auto num_tokens = static_cast<int>(x.size(0));
        const auto hidden = static_cast<int>(x.size(1));
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        const auto num_experts_per_rank = num_experts / context->num_ranks;
        EP_HOST_ASSERT(num_experts % context->num_ranks == 0);
        EP_HOST_ASSERT(x.dim() == 2);
        EP_HOST_ASSERT(x.is_contiguous());
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        int num_sf_packs = 0;
        int sf_token_stride = 0;
        int sf_hidden_stride = 0;
        void* sf_ptr = nullptr;
        if (sf.has_value()) {
            EP_HOST_ASSERT(x.scalar_type() == torch::kFloat8_e4m3fn);
            EP_HOST_ASSERT(sf->dim() == 2 and sf->size(0) == num_tokens);
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            EP_HOST_ASSERT(sf->device() == x.device());
            EP_HOST_ASSERT(sf->stride(0) == 1 or sf->stride(1) == 1);
            EP_HOST_ASSERT(sf->stride(0) <= std::numeric_limits<int>::max() and
                           sf->stride(1) <= std::numeric_limits<int>::max());
            num_sf_packs = static_cast<int>(sf->size(1));
            sf_ptr = sf->data_ptr();
            sf_token_stride = static_cast<int>(sf->stride(0));
            sf_hidden_stride = static_cast<int>(sf->stride(1));
        } else {
            EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        }

        // Check top-k stuffs
        const auto num_topk = static_cast<int>(topk_idx.size(1));
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens);
        EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(topk_idx.is_contiguous());

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->size(0) == num_tokens and topk_weights->size(1) == num_topk);
            EP_HOST_ASSERT(topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and
                           cumulative_local_expert_recv_stats->size(0) == num_experts_per_rank);
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == x.device());
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Resource configs
        const auto num_used_vec_cores = get_num_used_vec_cores(num_ai_cores);
        const auto num_ub_bytes = static_cast<int>(runtime->get_num_ubuf_bytes_per_vec_core());
        EP_HOST_ASSERT(num_ub_bytes > 0);

        // Check buffer size
        EP_HOST_ASSERT(get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk,
            static_cast<int>(x.element_size()), context->num_ranks) <= num_buffer_bytes);

        // Handle tensors written by the dispatch notify path
        const auto int_options = at::TensorOptions(x.device()).dtype(torch::kInt);
        const auto metadata_send_buffer = torch::empty(
            {num_tokens, get_dispatch_token_layout(num_hidden_bytes, num_sf_packs, num_topk).get_num_bytes(true)},
            x.options().dtype(torch::kByte));
        const auto psum_num_recv_tokens_per_rank = cached_mode ?
            cached_psum_num_recv_tokens_per_rank.value() :
            torch::empty({context->num_ranks}, int_options);
        const auto psum_num_recv_tokens_per_expert = cached_mode ?
            cached_psum_num_recv_tokens_per_expert.value() :
            torch::empty({num_experts_per_rank}, int_options);
        const auto num_unaligned_recv_tokens_per_expert = cached_mode ?
            cached_num_unaligned_recv_tokens_per_expert.value() :
            torch::empty({num_experts_per_rank}, int_options);
        const auto dst_buffer_slot_idx = cached_mode ?
            cached_dst_buffer_slot_idx.value() :
            torch::empty({num_tokens, num_topk}, int_options);
        const auto dst_gsge_idx = cached_mode ?
            cached_dst_gsge_idx.value() :
            torch::empty({num_tokens, num_topk}, int_options);
        if (cached_mode) {
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.dim() == 1 and
                           psum_num_recv_tokens_per_rank.size(0) == context->num_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.dim() == 1 and
                           psum_num_recv_tokens_per_expert.size(0) == num_experts_per_rank);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.dim() == 1 and
                           num_unaligned_recv_tokens_per_expert.size(0) == num_experts_per_rank);
            EP_HOST_ASSERT(dst_buffer_slot_idx.dim() == 2 and
                           dst_buffer_slot_idx.size(0) == num_tokens and
                           dst_buffer_slot_idx.size(1) == num_topk);
            EP_HOST_ASSERT(dst_gsge_idx.sizes() == dst_buffer_slot_idx.sizes());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(dst_buffer_slot_idx.is_contiguous());
            EP_HOST_ASSERT(dst_gsge_idx.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(dst_gsge_idx.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.device() == x.device());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.device() == x.device());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.device() == x.device());
            EP_HOST_ASSERT(dst_buffer_slot_idx.device() == x.device());
            EP_HOST_ASSERT(dst_gsge_idx.device() == x.device());
        }

        // Clean host workspace
        const auto host_workspace_layout = layout::EPWorkspaceLayout(host_workspace);
        if (not cached_mode) {
            std::fill_n(host_workspace_layout.get_host_rank_count_ptr(), context->num_ranks, 0);
            std::fill_n(host_workspace_layout.get_host_expert_count_ptr(), num_experts_per_rank, 0);
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        // Launch dispatch
        const auto barrier_in_prologue = runtime->get_barrier_in_prologue();
        const auto low_latency_transport = low_latency_transport_enabled() &&
            prefer_low_latency_transport(runtime->get_npu_arch(), context->num_ranks, num_tokens,
                                         num_max_tokens_per_rank, hidden, num_topk, num_experts,
                                         num_used_vec_cores, sf.has_value(), num_ub_bytes);
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr(), topk_weights_ptr,
                        metadata_send_buffer.data_ptr(),
                        cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        num_unaligned_recv_tokens_per_expert.data_ptr<int>(),
                        dst_buffer_slot_idx.data_ptr<int>(),
                        dst_gsge_idx.data_ptr<int>(),
                        context->buffer, context->workspace, mapped_host_workspace,
                        context->hccl_context->jetty_device_ptrs,
                        context->rank_idx, context->num_ranks,
                        num_tokens, num_max_tokens_per_rank,
                        num_experts, num_topk,
                        expert_alignment,
                        cached_mode,
                        do_cpu_sync,
                        num_hidden_bytes,
                        num_sf_packs,
                        sf_token_stride,
                        sf_hidden_stride,
                        num_ub_bytes,
                        num_used_vec_cores,
                        context->num_gpu_timeout_cycles,
                        barrier_in_prologue, low_latency_transport);

        // URMA still reads x and metadata_send_buffer after the issue kernel exits.
        // No NPU stream tracks that traffic; capture its sources until the epilogue queues the drain.
        auto epilogue = [=, x = x, metadata_send_buffer = metadata_send_buffer, context = context]() {
            // Wait GPU notify or reuse cached allocation sizes
            int num_recv_tokens = context->num_ranks * num_max_tokens_per_rank;
            int num_expanded_tokens = 0;
            std::vector<int> num_recv_tokens_per_expert_list;
            if (cached_mode) {
                num_recv_tokens = cached_num_recv_tokens.value();
                num_expanded_tokens = cached_num_expanded_tokens.value();
                num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
            } else if (do_cpu_sync) {
                const auto cpu_sync_deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(context->num_cpu_timeout_secs);

                // Wait rank status arrival
                num_recv_tokens = 0;
                for (int i = 0; i < context->num_ranks; ++ i) {
                    const auto status = reduction::host::wait_ready(
                        host_workspace_layout.get_host_rank_count_ptr(i), context->num_ranks, cpu_sync_deadline);
                    num_recv_tokens += reduction::get_reduction_value(status);
                }

                // Wait expert status arrival
                num_recv_tokens_per_expert_list.reserve(num_experts_per_rank);
                for (int i = 0; i < num_experts_per_rank; ++ i) {
                    const auto status = reduction::host::wait_ready(
                        host_workspace_layout.get_host_expert_count_ptr(i), context->num_ranks, cpu_sync_deadline);
                    const auto count = reduction::get_reduction_value(status);
                    num_expanded_tokens += count;
                    num_recv_tokens_per_expert_list.push_back(count);
                }
            } else {
                num_expanded_tokens = context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_experts_per_rank);
                num_expanded_tokens += (expert_alignment - 1) * num_experts_per_rank;
                num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
            }

            // Allocate output tensors
            auto recv_x = torch::empty({num_expanded_tokens, hidden}, x.options());
            auto recv_sf = std::optional<torch::Tensor>();
            int recv_sf_token_stride = 0;
            int recv_sf_hidden_stride = 0;
            void* recv_sf_ptr = nullptr;
            if (sf.has_value()) {
                recv_sf_token_stride = num_sf_packs;
                recv_sf_hidden_stride = 1;
                recv_sf = torch::empty_strided(
                    {num_expanded_tokens, num_sf_packs},
                    {recv_sf_token_stride, recv_sf_hidden_stride},
                    sf->options());
                recv_sf_ptr = recv_sf->data_ptr();
            }
            auto recv_topk_weights = std::optional<torch::Tensor>();
            if (topk_weights.has_value())
                recv_topk_weights = torch::empty({num_expanded_tokens}, topk_weights->options());
            auto recv_src_metadata = cached_mode ?
                cached_recv_src_metadata.value() :
                torch::empty({num_recv_tokens, num_topk + 2}, int_options);
            if (cached_mode) {
                EP_HOST_ASSERT(num_recv_tokens >= 0 and num_expanded_tokens >= 0);
                EP_HOST_ASSERT(recv_src_metadata.dim() == 2 and
                               recv_src_metadata.size(0) == num_recv_tokens and
                               recv_src_metadata.size(1) == num_topk + 2);
                EP_HOST_ASSERT(recv_src_metadata.is_contiguous());
                EP_HOST_ASSERT(recv_src_metadata.scalar_type() == torch::kInt);
                EP_HOST_ASSERT(recv_src_metadata.device() == x.device());
            }

            launch_dispatch_copy_epilogue(
                context->buffer,
                context->workspace,
                context->hccl_context->jetty_device_ptrs,
                psum_num_recv_tokens_per_rank.data_ptr<int>(),
                psum_num_recv_tokens_per_expert.data_ptr<int>(),
                recv_x.data_ptr(),
                recv_sf_ptr,
                recv_topk_weights.has_value() ? recv_topk_weights->data_ptr() : nullptr,
                recv_src_metadata.data_ptr<int>(),
                cached_mode,
                do_zero_padding,
                context->rank_idx, context->num_ranks,
                num_max_tokens_per_rank,
                num_experts, num_topk,
                expert_alignment,
                num_hidden_bytes,
                num_sf_packs,
                recv_sf_token_stride,
                recv_sf_hidden_stride,
                num_ub_bytes,
                num_used_vec_cores,
                context->num_gpu_timeout_cycles,
                not barrier_in_prologue
            );

            // Convert row-major epilogue output to column-major
            if (recv_sf.has_value() and use_cube_aligned_col_major_sf)
                recv_sf = recv_sf->transpose(0, 1).contiguous().transpose(0, 1);

            return pybind11::make_tuple(
                recv_x, recv_sf, recv_topk_weights,
                num_recv_tokens,
                num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                dst_gsge_idx);
        };

        // Defer epilogue and host sync
        if (defer_epilogue) {
            EP_HOST_ASSERT(async_with_compute_stream);
            std::function<pybind11::object()> epilogue_hook = std::move(epilogue);
            return pybind11::make_tuple(
                pybind11::none(), std::optional<EventHandle>(EventHandle()), epilogue_hook);
        }

        // Do epilogue immediately
        const auto result = epilogue();
        const auto event = async_with_compute_stream ?
            std::optional<EventHandle>(EventHandle()) : std::nullopt;
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    pybind11::tuple
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_rank,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms,
            const std::optional<EventHandle>& previous_event,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& defer_epilogue) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(num_experts % context->num_ranks == 0);

        // Check data tensor
        const auto num_tokens = static_cast<int>(x.size(0));
        const auto hidden = static_cast<int>(x.size(1));
        EP_HOST_ASSERT(x.dim() == 2);
        EP_HOST_ASSERT(x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);

        // Check metadata from dispatch
        const auto num_combined_tokens = static_cast<int>(combined_topk_idx.size(0));
        const auto num_topk = static_cast<int>(combined_topk_idx.size(1));
        const auto num_reduced_tokens = static_cast<int>(src_metadata.size(0));
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);
        EP_HOST_ASSERT(combined_topk_idx.dim() == 2);
        EP_HOST_ASSERT(combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(src_metadata.dim() == 2 and src_metadata.size(1) == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.size(0) <= context->num_ranks * num_max_tokens_per_rank);
        EP_HOST_ASSERT(src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.dim() == 1);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.size(0) == context->num_ranks);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.scalar_type() == torch::kInt);

        // Check optional tensors
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->dim() == 1 and topk_weights->size(0) == num_tokens);
            EP_HOST_ASSERT(topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        const std::optional<torch::Tensor> bias_opts[2] = {bias_0, bias_1};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                const auto& bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2);
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                EP_HOST_ASSERT(bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            }
        }

        // Resource configs
        const auto num_used_vec_cores = get_num_used_vec_cores(num_sms);
        const auto num_ub_bytes = static_cast<int>(runtime->get_num_ubuf_bytes_per_vec_core());
        EP_HOST_ASSERT(num_ub_bytes > 0);

        // Check buffer size
        EP_HOST_ASSERT(get_combine_buffer_size(
            context->num_ranks, num_max_tokens_per_rank, hidden, num_topk) <= num_buffer_bytes);

        const auto token_layout = get_combine_token_layout(
            hidden * static_cast<int>(sizeof(uint16_t)), num_topk);
        const auto send_buffer = torch::empty(
            {num_reduced_tokens, token_layout.get_num_bytes()}, x.options().dtype(torch::kByte));

        // Push data into remote reduce buffers
        const auto barrier_in_prologue = runtime->get_barrier_in_prologue();
        const auto low_latency_transport = low_latency_transport_enabled() &&
            prefer_low_latency_transport(runtime->get_npu_arch(), context->num_ranks, num_combined_tokens,
                                         num_max_tokens_per_rank, hidden, num_topk, num_experts,
                                         num_used_vec_cores, false, num_ub_bytes);
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),
            topk_weights_ptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_rank.data_ptr<int>(),
            context->buffer,
            send_buffer.data_ptr(),
            context->workspace,
            context->hccl_context->jetty_device_ptrs,
            context->rank_idx, context->num_ranks,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            num_ub_bytes,
            num_used_vec_cores,
            context->num_gpu_timeout_cycles,
            barrier_in_prologue, low_latency_transport
        );

        // URMA still reads x (direct sends) and send_buffer after the issue kernel exits.
        // No NPU stream tracks that traffic; capture its sources until the epilogue queues the drain.
        auto epilogue = [=, x = x, send_buffer = send_buffer, context = context]() {
            // Allocate output tensors
            auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
            auto combined_topk_weights = std::optional<torch::Tensor>();
            float* combined_topk_weights_ptr = nullptr;
            if (topk_weights.has_value()) {
                combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
                combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
            }

            // Reduce pushed data
            launch_combine_reduce_epilogue(
                combined_x.data_ptr(),
                combined_topk_weights_ptr,
                combined_topk_idx.data_ptr<int64_t>(),
                reduce_buffer,
                bias_0.has_value() ? bias_0->data_ptr() : nullptr,
                bias_1.has_value() ? bias_1->data_ptr() : nullptr,
                context->workspace,
                context->hccl_context->jetty_device_ptrs,
                context->num_ranks,
                num_combined_tokens,
                num_max_tokens_per_rank,
                num_experts, num_topk,
                hidden,
                num_ub_bytes,
                num_used_vec_cores,
                context->num_gpu_timeout_cycles,
                not barrier_in_prologue
            );

            return pybind11::make_tuple(combined_x, combined_topk_weights);
        };

        // Defer epilogue
        if (defer_epilogue) {
            EP_HOST_ASSERT(async_with_compute_stream);
            std::function<pybind11::object()> epilogue_hook = std::move(epilogue);
            return pybind11::make_tuple(
                pybind11::none(), std::optional<EventHandle>(EventHandle()), epilogue_hook);
        }

        // Do epilogue immediately
        const auto result = epilogue();
        const auto event = async_with_compute_stream ?
            std::optional<EventHandle>(EventHandle()) : std::nullopt;
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs,
                                            const int& num_topk, const int& element_size,
                                            const int& num_ranks) {
        const auto token_layout = get_dispatch_token_layout(
            hidden * element_size, num_sf_packs, num_topk);
        const auto recv_buffer_layout = layout::BufferLayout(
            token_layout, num_ranks, num_max_tokens_per_rank);
        return recv_buffer_layout.get_num_bytes();
    }

    static int64_t get_combine_buffer_size(const int& num_ranks,
                                           const int& num_max_tokens_per_rank,
                                           const int& hidden, const int& num_topk) {
        const auto token_layout = get_combine_token_layout(
            hidden * static_cast<int>(sizeof(uint16_t)), num_topk);
        const auto reduce_buffer_layout = layout::BufferLayout(
            token_layout, std::min(num_topk, num_ranks), num_max_tokens_per_rank);
        return reduce_buffer_layout.get_num_bytes();
    }

    static int64_t calculate_buffer_size(const int& num_ranks,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch) {
        EP_HOST_ASSERT(num_ranks > 0);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
        EP_HOST_ASSERT(hidden > 0);
        EP_HOST_ASSERT(num_topk >= 0);

        // NOTES: dispatch requires `kNumTopk <= 32`, so reserve the maximum when top-k is unspecified
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Dispatch layout
        // NOTES: approximate the SF pack count because the size-hint API does not receive an SF tensor
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;
        const auto elem_size = use_fp8_dispatch ? sizeof(c10::Float8_e4m3fn) : sizeof(c10::BFloat16);
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size, num_ranks);

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_ranks, num_max_tokens_per_rank, hidden, num_topk);

        // Return the maximum of those layouts, aligned to 2 MB
        return math::align(std::max(num_dispatch_bytes, num_combine_bytes), kNumAllocationAlignmentBytes);
    }
};

}  // namespace deep_ep
