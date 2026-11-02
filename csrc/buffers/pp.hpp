#pragma once

#include <limits>

#include <deep_ep/layout/pp/workspace.hpp>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../utils/tensor.hpp"
#include "../kernels/pp/pp_send_recv.hpp"

namespace deep_ep {

class PPBuffer: public BufferBase {
    int prev_rank_idx, next_rank_idx;
    int64_t num_max_pp_tensor_bytes;
    int num_max_pp_inflight_tensors;

public:
    std::shared_ptr<comm::Context> context;
    torch::Tensor storage;

    PPBuffer(const int& rank_idx, const int& num_ranks, const std::string& hccl_group_name,
             const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors,
             const bool& use_huge1g_page, const int& num_gpu_timeout_secs, const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        prev_rank_idx((rank_idx + num_ranks - 1) % num_ranks),
        next_rank_idx((rank_idx + 1) % num_ranks),
        num_max_pp_tensor_bytes(math::align<int64_t>(num_max_tensor_bytes, kNumRDMAAlignmentBytes)),
        num_max_pp_inflight_tensors(num_max_inflight_tensors) {
        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);
        EP_HOST_ASSERT(num_max_pp_tensor_bytes <= std::numeric_limits<uint32_t>::max());
        const auto num_storage_bytes = math::align<int64_t>(
            4 * num_max_pp_tensor_bytes * num_max_inflight_tensors, kNumAllocationAlignmentBytes);
        context = std::make_shared<comm::Context>(
            hccl_group_name, rank_idx, num_ranks, sizeof(layout::PPSignals), num_storage_bytes, 0,
            use_huge1g_page, 300, num_gpu_timeout_secs);
        main_context = context;
        auto& pp = *static_cast<layout::PPSignals*>(context->workspace);
        context->set_common_signals(&pp.common_signals);
        storage = at_npu::native::from_blob(context->buffer, {num_storage_bytes}, [context = context](void*) {},
                                   torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU));
    }

    ~PPBuffer() noexcept(false) override {
        destroy_on_destruction("PP");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        comm::barrier(*context, true);
        context->finalize();
        storage = torch::Tensor();
        context = nullptr;
        destroyed = true;
    }

    void send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.device().type() == TORCH_DEVICE_NPU and x.is_contiguous() and x.nbytes() > 0 and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);

        launch_pp_send(
            x.data_ptr(), x.nbytes(),
            context->buffer, context->workspace, context->hccl_context->jetty_device_ptrs,
            context->rank_idx, dst_rank_idx, context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? runtime->get_num_vec_cores() : num_sms,
            context->num_gpu_timeout_cycles
        );
    }

    void recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.device().type() == TORCH_DEVICE_NPU and x.is_contiguous() and x.nbytes() > 0 and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(src_rank_idx == prev_rank_idx or src_rank_idx == next_rank_idx);

        launch_pp_recv(
            x.data_ptr(), x.nbytes(),
            context->buffer, context->workspace, context->hccl_context->jetty_device_ptrs,
            context->rank_idx, src_rank_idx, context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? runtime->get_num_vec_cores() : num_sms,
            context->num_gpu_timeout_cycles
        );
    }

};

}  // namespace deep_ep
