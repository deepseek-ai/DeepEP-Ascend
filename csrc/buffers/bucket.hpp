#pragma once

#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../kernels/bucket/all_gather.hpp"
#include "../utils/event.hpp"
#include "../utils/tensor.hpp"

namespace deep_ep {

class BucketBuffer: public BufferBase {
    int64_t num_storage_bytes;

public:
    std::vector<std::shared_ptr<comm::Context>> contexts;
    torch::Tensor storage;

    BucketBuffer(const std::vector<int>& rank_indices, const std::vector<int>& num_ranks,
                 const std::vector<std::string>& hccl_group_names, const int64_t& num_storage_bytes,
                 const bool& use_huge1g_page, const int& num_gpu_timeout_secs, const bool& explicitly_destroy):
        BufferBase(explicitly_destroy), num_storage_bytes(num_storage_bytes) {
        const auto num_groups = static_cast<int>(hccl_group_names.size());
        EP_HOST_ASSERT(num_groups > 0 and num_groups <= kNumMaxContexts);
        EP_HOST_ASSERT(rank_indices.size() == num_groups and num_ranks.size() == num_groups);
        EP_HOST_ASSERT(num_storage_bytes > 0 and num_storage_bytes % kNumAllocationAlignmentBytes == 0);
        for (int context_idx = 0; context_idx < num_groups; ++ context_idx) {
            contexts.emplace_back(std::make_shared<comm::Context>(
                hccl_group_names[context_idx], rank_indices[context_idx], num_ranks[context_idx],
                sizeof(layout::BucketWorkspace), num_storage_bytes, 0, use_huge1g_page, 300, num_gpu_timeout_secs,
                contexts.empty() ? nullptr : contexts.front()));
            auto& workspace = *static_cast<layout::BucketWorkspace*>(contexts.back()->workspace);
            contexts.back()->set_common_signals(&workspace.signals[context_idx].common_signals);
        }
        main_context = contexts.front();
        storage = at_npu::native::from_blob(main_context->buffer, {num_storage_bytes}, [context = main_context](void*) {},
                                   torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU));
    }

    ~BucketBuffer() noexcept(false) override {
        destroy_on_destruction("bucket");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        for (const auto& context: contexts)
            comm::barrier(*context, true);
        for (const auto& context: contexts)
            context->finalize();
        storage = torch::Tensor();
        contexts.clear();
        destroyed = true;
    }

    std::tuple<std::vector<torch::Tensor>, EventHandle, int> reduce_scatter(
        const std::vector<torch::Tensor>& srcs, const int& context_idx, const int& num_sms,
        const std::string& comm_precision, const float& scale) const {
        EP_HOST_UNREACHABLE("BucketBuffer::reduce_scatter is not implemented on Ascend");
    }

    std::tuple<std::vector<torch::Tensor>, EventHandle, int> all_reduce(
        const std::vector<torch::Tensor>& srcs, const int& context_idx, const int& num_sms, const float& scale) const {
        EP_HOST_UNREACHABLE("BucketBuffer::all_reduce is not implemented on Ascend");
    }

    pybind11::tuple all_gather(const std::vector<torch::Tensor>& srcs,
                              const std::vector<torch::Tensor>& dsts,
                              const int& context_idx, const int& num_sms) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < contexts.size());
        EP_HOST_ASSERT(num_sms == 0);
        EP_HOST_ASSERT(not srcs.empty() and srcs.size() <= comm::kNumMaxBuckets);
        EP_HOST_ASSERT(dsts.empty() or dsts.size() == srcs.size());
        const auto& context = contexts[context_idx];
        const auto rank_idx = context->rank_idx, num_ranks = context->num_ranks;

        // Describe each gathered tensor and its optional external source by byte offsets.
        const auto num_buckets = static_cast<int>(srcs.size());
        comm::BucketList buckets;
        auto src_buckets = dsts.empty() ? std::nullopt : std::make_optional<comm::BucketList>();
        std::vector<torch::Tensor> gathered;
        const auto buffer_ptr = reinterpret_cast<int64_t>(storage.data_ptr());
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& src = srcs[bucket_idx];
            EP_HOST_ASSERT(src.device().type() == TORCH_DEVICE_NPU and src.is_contiguous() and src.numel() > 0);
            const auto num_shard_bytes = static_cast<int64_t>(src.nbytes());
            const auto src_offset = reinterpret_cast<int64_t>(src.data_ptr()) - buffer_ptr;
            int64_t bucket_offset = src_offset - rank_idx * num_shard_bytes;
            if (not dsts.empty()) {
                const auto& dst = dsts[bucket_idx];
                EP_HOST_ASSERT(dst.device() == src.device() and dst.is_contiguous() and dst.scalar_type() == src.scalar_type());
                EP_HOST_ASSERT(dst.nbytes() == num_ranks * num_shard_bytes);
                bucket_offset = reinterpret_cast<int64_t>(dst.data_ptr()) - buffer_ptr;
            }
            EP_HOST_ASSERT(bucket_offset >= 0 and bucket_offset <= num_storage_bytes and
                           num_shard_bytes <= (num_storage_bytes - bucket_offset) / num_ranks and
                           "All-gather outputs must belong to this BucketBuffer");
            EP_HOST_ASSERT(src_offset == bucket_offset + rank_idx * num_shard_bytes or
                           src_offset + num_shard_bytes <= bucket_offset or
                           src_offset >= bucket_offset + num_ranks * num_shard_bytes);
            buckets[bucket_idx] = {.offset = bucket_offset, .num_bytes = num_shard_bytes};
            if (src_buckets.has_value())
                (*src_buckets)[bucket_idx] = {.offset = src_offset, .num_bytes = num_shard_bytes};
            gathered.push_back(storage.narrow(0, bucket_offset, num_ranks * num_shard_bytes).view(src.scalar_type()));
        }

        // Stream control
        const auto compute_stream = c10_npu::getCurrentNPUStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);

        // Remote copy exits immediately after issuing
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context->workspace);
        launch_all_gather(&workspace.signals[context_idx], context->hccl_context->buffer, context->buffer,
                          context->hccl_context->jetty_device_ptrs,
                          num_buckets, buckets, src_buckets,
                          rank_idx, num_ranks,
                          context->num_gpu_timeout_cycles, comm_stream);

        // Local copy
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto num_bytes = buckets[bucket_idx].num_bytes;
            auto* dst = math::advance_ptr<void>(context->buffer, buckets[bucket_idx].offset + rank_idx * num_bytes);
            const auto* src = srcs[bucket_idx].data_ptr();
            if (src != dst)
                ACL_CHECK(aclrtMemcpyAsync(dst, num_bytes, src, num_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream));
        }

        auto event = EventHandle(comm_stream);

        // URMA reads sources after the issue kernel exits; no NPU stream tracks that traffic.
        // Capture the sources until the epilogue queues its drain; record_stream cannot replace this.
        std::function<pybind11::object()> epilogue = [context, srcs, gathered = std::move(gathered)]() {
            EP_HOST_ASSERT(deep_jit::get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 1 and
                           "URMA all-gather requires EP_AVOID_RECORD_STREAM=1");
            // Order shared barrier and Jetty state with all issued communication.
            comm::barrier(*context, false, true, true);
            return pybind11::cast(gathered);
        };
        return pybind11::make_tuple(event, epilogue);
    }
};

}  // namespace deep_ep
