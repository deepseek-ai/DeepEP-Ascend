#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/layout/bucket/send_recv.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../kernels/bucket/all_gather.hpp"
#include "../kernels/bucket/all_reduce.hpp"
#include "../kernels/bucket/reduce_scatter.hpp"
#include "../kernels/bucket/ring_send_recv.hpp"
#include "../kernels/bucket/send_recv.hpp"
#include "../utils/event.hpp"
#include "../utils/tensor.hpp"

namespace deep_ep {

class BucketBuffer: public BufferBase {
    int64_t num_storage_bytes;
    int num_gpu_timeout_secs;
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory;

    void check_available() const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(is_active() and "the buffer is not active; call reconfigure first");
    }

    void clear_workspace() {
        ACL_CHECK(aclrtMemset(symmetric_memory->ptr, sizeof(layout::BucketWorkspace),
                             0, sizeof(layout::BucketWorkspace)));
    }

public:
    std::vector<std::shared_ptr<comm::Context>> contexts;
    torch::Tensor storage;

    BucketBuffer(const int64_t& num_storage_bytes, const bool& use_huge1g_page,
                 const int& num_gpu_timeout_secs, const bool& explicitly_destroy):
        BufferBase(explicitly_destroy), num_storage_bytes(num_storage_bytes), num_gpu_timeout_secs(num_gpu_timeout_secs) {
        EP_HOST_ASSERT(num_storage_bytes > 0 and num_storage_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_gpu_timeout_secs > 0);
        constexpr auto num_workspace_bytes = static_cast<int64_t>(sizeof(layout::BucketWorkspace));
        symmetric_memory = std::make_shared<symmetric::SymmetricMemory>(
            num_workspace_bytes + num_storage_bytes, use_huge1g_page);
        auto* buffer = math::advance_ptr<void>(symmetric_memory->ptr, num_workspace_bytes);
        // Tensor views retain the allocation independently of the communication contexts.
        storage = torch::from_blob(buffer, {num_storage_bytes}, [symmetric_memory = symmetric_memory](void*) {},
                                   torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU));
    }

    ~BucketBuffer() noexcept(false) override {
        destroy_on_destruction("bucket");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        if (is_active())
            for (const auto& context: contexts)
                comm::barrier(*context, true);
        for (const auto& context: contexts)
            context->finalize();
        storage = torch::Tensor();
        contexts.clear();
        main_context = nullptr;
        // Explicit destroy invalidates views even if their storage objects survive.
        symmetric_memory->free();
        destroyed = true;
    }

    void clear_error_status() const override {
        EP_HOST_ASSERT(not destroyed and "error record unavailable after destroy");
        for (const auto& context: contexts)
            context->clear_error_status();
    }

    std::vector<layout::ErrorCode> get_error_status() const override {
        EP_HOST_ASSERT(not destroyed and "error record unavailable after destroy");
        std::vector<layout::ErrorCode> status;
        for (const auto& context: contexts)
            status.push_back(context->get_error_status());
        return status;
    }

    bool is_active() const {
        return not contexts.empty() and std::all_of(
            contexts.begin(), contexts.end(), [](const auto& context) { return context->is_active(); });
    }

    // Drop the old peers' channels and remote mappings. Non-collective: the failed ranks are gone.
    void invalidate(const int& num_timeout_secs) {
        check_available();
        for (const auto& context: contexts)
            context->invalidate(num_timeout_secs);
    }

    void configure(const std::vector<comm::ProcessGroup>& groups) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not groups.empty() and groups.size() <= kNumMaxContexts);
        EP_HOST_ASSERT(std::none_of(
            contexts.begin(), contexts.end(), [](const auto& context) { return context->is_active(); }) and
            "call invalidate before configure");
        const bool first_configuration = contexts.empty();
        EP_HOST_ASSERT(first_configuration or groups.size() == contexts.size());

        for (size_t context_idx = 0; context_idx < groups.size(); ++ context_idx) {
            const auto& group = groups[context_idx];
            EP_HOST_ASSERT(group != nullptr);
            EP_HOST_ASSERT(first_configuration or group->getSize() == contexts[context_idx]->num_ranks);
            // The shared allocation has already selected its page size.
            auto context = std::make_shared<comm::Context>(
                group, group->getRank(), group->getSize(), sizeof(layout::BucketWorkspace),
                num_storage_bytes, 0, false, 300, num_gpu_timeout_secs, symmetric_memory);
            if (first_configuration) {
                contexts.emplace_back(std::move(context));
            } else {
                auto& old_context = contexts[context_idx];
                EP_HOST_ASSERT(context->get_physical_domain_size() == old_context->get_physical_domain_size() and
                               "configure must preserve the topology");
                old_context->finalize();
                old_context = std::move(context);
            }
        }

        clear_workspace();
        auto& workspace = *static_cast<layout::BucketWorkspace*>(symmetric_memory->ptr);
        for (size_t context_idx = 0; context_idx < contexts.size(); ++ context_idx)
            contexts[context_idx]->set_barrier_signals(&workspace.signals[context_idx].barrier_signals);
        main_context = contexts.front();
    }

    pybind11::tuple send_recv(
        const std::vector<std::tuple<torch::Tensor, int, int64_t>>& send_ops,
        const int& context_idx,
        const int& num_sms) const {
        check_available();
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        EP_HOST_ASSERT(num_sms == 0);
        const auto& context = contexts[context_idx];
        EP_HOST_ASSERT(send_ops.size() <= layout::kNumMaxSendRecvEntries and
                       "send_ops exceeds the maximum number of send-recv entries");

        const auto check_tensor = [this](const torch::Tensor& tensor) {
            EP_HOST_ASSERT(tensor.device().type() == TORCH_DEVICE_NPU and tensor.device() == storage.device() and tensor.is_contiguous());
            const auto offset = reinterpret_cast<int64_t>(tensor.data_ptr()) - reinterpret_cast<int64_t>(storage.data_ptr());
            const auto num_bytes = static_cast<int64_t>(tensor.nbytes());
            EP_HOST_ASSERT(offset >= 0 and offset <= storage.numel() and num_bytes <= storage.numel() - offset);
            EP_HOST_ASSERT(offset % kNumRDMAAlignmentBytes == 0 and num_bytes % kNumRDMAAlignmentBytes == 0);
        };

        int num_entries = 0;
        layout::SendRecvList entries;
        for (const auto& [tensor, dst_rank_idx, dst_offset]: send_ops) {
            // Empty sends are no-ops
            if (tensor.numel() == 0)
                continue;

            check_tensor(tensor);
            EP_HOST_ASSERT(dst_rank_idx >= 0 and dst_rank_idx < context->num_ranks);
            const auto num_bytes = static_cast<int64_t>(tensor.nbytes());
            EP_HOST_ASSERT(dst_offset >= 0 and num_bytes <= storage.numel() - dst_offset);
            EP_HOST_ASSERT(dst_offset % kNumRDMAAlignmentBytes == 0);
            // Storage uses the first context's mapping, so address conversion is needed
            const auto src_offset = reinterpret_cast<int64_t>(tensor.data_ptr()) - reinterpret_cast<int64_t>(storage.data_ptr());
            if (dst_rank_idx == context->rank_idx)
                EP_HOST_ASSERT((src_offset == dst_offset or src_offset + num_bytes <= dst_offset or
                                dst_offset + num_bytes <= src_offset) and
                               "Self-copy source and destination must not partially overlap");
            entries.entries[num_entries ++] = {src_offset, dst_offset, num_bytes, dst_rank_idx};
        }

        // Group by destination rank in cyclic order.
        std::stable_sort(entries.entries, entries.entries + num_entries,
                         [&context](const auto& lhs, const auto& rhs) {
            const auto key = [&context](const auto& entry) {
                return (entry.dst_rank_idx - context->rank_idx + context->num_ranks) % context->num_ranks;
            };
            return key(lhs) < key(rhs);
        });

        const auto compute_stream = c10_npu::getCurrentNPUStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context->workspace);

        launch_send_recv(&workspace.signals[context_idx], context->hcomm_context->buffer, context->buffer,
                         context->hcomm_context->device_context, context->rank_idx, context->num_ranks,
                         context->hcomm_context->num_scaleup_ranks, context->hcomm_context->num_scaleout_jetties,
                         num_entries, entries, context->num_gpu_timeout_cycles, comm_stream);

        auto event = EventHandle(comm_stream);
        std::function epilogue = [context, send_ops]() {
            EP_HOST_ASSERT(deep_jit::get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 1 and
                           "URMA send-recv requires EP_AVOID_RECORD_STREAM=1");
            comm::barrier(*context, false, true, true);
        };
        return pybind11::make_tuple(event, epilogue, num_sms);
    }

    pybind11::tuple reduce_scatter(const std::vector<torch::Tensor>& srcs,
                                  const int& context_idx, const int& num_sms,
                                  const std::string& comm_precision, const float& scale) const {
        check_available();
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < contexts.size());
        EP_HOST_ASSERT(num_sms == 0);
        EP_HOST_ASSERT(comm_precision == "fp32");
        EP_HOST_ASSERT(not srcs.empty() and srcs.size() <= comm::kNumMaxBuckets);
        const auto& context = contexts[context_idx];
        const auto rank_idx = context->rank_idx, num_ranks = context->num_ranks;

        const auto num_buckets = static_cast<int>(srcs.size());
        comm::BucketList src_buckets;
        std::vector<torch::Tensor> reduced;
        const auto buffer_ptr = reinterpret_cast<int64_t>(storage.data_ptr());
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& src = srcs[bucket_idx];
            EP_HOST_ASSERT(src.device().type() == TORCH_DEVICE_NPU and src.is_contiguous() and src.numel() > 0);
            EP_HOST_ASSERT(src.device() == storage.device());
            EP_HOST_ASSERT(src.scalar_type() == torch::kFloat32 and src.numel() % num_ranks == 0);
            const auto num_shard_bytes = static_cast<int64_t>(src.nbytes()) / num_ranks;
            const auto src_offset = reinterpret_cast<int64_t>(src.data_ptr()) - buffer_ptr;
            EP_HOST_ASSERT(src_offset >= 0 and src_offset <= num_storage_bytes and
                           static_cast<int64_t>(src.nbytes()) <= num_storage_bytes - src_offset and
                           "Reduce-scatter sources must belong to this BucketBuffer");
            src_buckets[bucket_idx] = {.offset = src_offset, .num_bytes = num_shard_bytes};
            const auto num_shard_elems = src.numel() / num_ranks;
            reduced.push_back(src.view({-1}).narrow(0, rank_idx * num_shard_elems, num_shard_elems));
        }

        // Stream control
        const auto compute_stream = c10_npu::getCurrentNPUStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);

        // Remote reads accumulate into the local shard asynchronously.
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context->workspace);
        launch_reduce_scatter(&workspace.signals[context_idx], context->hcomm_context->buffer, context->buffer,
                              context->hcomm_context->device_context,
                              num_buckets, src_buckets, rank_idx, num_ranks,
                              context->hcomm_context->num_scaleup_ranks,
                              context->num_gpu_timeout_cycles, comm_stream);

        auto event = EventHandle(comm_stream);

        // The epilogue drains URMA and synchronize source reuse, then scale the local shards.
        std::function<pybind11::object()> epilogue = [context, srcs, reduced = std::move(reduced),
                                                    num_buckets, src_buckets, scale]() {
            context->check_available();
            EP_HOST_ASSERT(deep_jit::get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 1 and
                           "URMA reduce-scatter requires EP_AVOID_RECORD_STREAM=1");
            const auto stream = c10_npu::getCurrentNPUStream();
            const auto comm_stream = comm::get_comm_stream();
            comm::stream_wait(stream, comm_stream);
            launch_reduce_scatter_epilogue(context->barrier_signals, context->hcomm_context->buffer, context->buffer,
                                           context->hcomm_context->device_context,
                                           context->rank_idx, context->num_ranks,
                                           context->hcomm_context->num_scaleup_ranks,
                                           context->hcomm_context->num_scaleup_jetties,
                                           context->hcomm_context->num_scaleout_jetties,
                                           num_buckets, src_buckets, scale,
                                           context->num_gpu_timeout_cycles, stream);
            // A subsequent exchange must wait for reads, reductions and scaling to complete.
            comm::stream_wait(comm_stream, stream);
            // Rejoin comm_stream into the capture stream so ACLGraph capture does not leave it dangling.
            if (comm::is_stream_capturing(stream))
                comm::stream_wait(stream, comm_stream);
            return pybind11::cast(reduced);
        };
        return pybind11::make_tuple(event, epilogue);
    }

    pybind11::tuple all_reduce(const std::vector<torch::Tensor>& srcs,
                              const int& context_idx, const int& num_sms, const float& scale) const {
        check_available();
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < contexts.size());
        EP_HOST_ASSERT(num_sms == 0 and "All-reduce uses AI cores only in its epilogue");
        EP_HOST_ASSERT(not srcs.empty() and srcs.size() <= comm::kNumMaxBuckets);
        const auto& context = contexts[context_idx];
        const auto num_buckets = static_cast<int>(srcs.size());
        comm::BucketList buckets;
        const auto buffer_ptr = reinterpret_cast<int64_t>(storage.data_ptr());
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& src = srcs[bucket_idx];
            EP_HOST_ASSERT(src.device() == storage.device() and src.is_contiguous() and src.numel() > 0);
            EP_HOST_ASSERT(src.scalar_type() == torch::kFloat32);
            const auto offset = reinterpret_cast<int64_t>(src.data_ptr()) - buffer_ptr;
            EP_HOST_ASSERT(offset >= 0 and offset <= num_storage_bytes and
                           static_cast<int64_t>(src.nbytes()) <= num_storage_bytes - offset and
                           "All-reduce sources must belong to this BucketBuffer");
            buckets[bucket_idx] = {.offset = offset, .num_bytes = static_cast<int64_t>(src.nbytes())};
        }

        // Stream control
        const auto compute_stream = c10_npu::getCurrentNPUStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);

        auto& workspace = *static_cast<layout::BucketWorkspace*>(context->workspace);
        const auto& hcomm_context = context->hcomm_context;
        const auto num_timeout_ns =
            context->num_gpu_timeout_cycles / comm::kNumSystemCyclesPerSecond * 1'000'000'000LL;
        launch_all_reduce(&workspace.signals[context_idx], hcomm_context->buffer, context->buffer,
                          hcomm_context->device_context, context->rank_idx, context->num_ranks,
                          hcomm_context->num_scaleup_ranks, hcomm_context->num_scaleup_jetties,
                          hcomm_context->num_scaleout_jetties, num_buckets, buckets,
                          num_timeout_ns, comm_stream);
        auto event = EventHandle(comm_stream);

        // The epilogue drains URMA and synchronize source reuse, then scale the whole buckets.
        // Scaling is not done by AICPU due to its limited processing bandwidth.
        std::function<pybind11::object()> epilogue = [context, srcs, num_buckets, buckets, scale]() {
            context->check_available();
            const auto& hcomm_context = context->hcomm_context;
            EP_HOST_ASSERT(deep_jit::get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 1 and
                           "URMA all-reduce requires EP_AVOID_RECORD_STREAM=1");
            const auto stream = c10_npu::getCurrentNPUStream();
            const auto comm_stream = comm::get_comm_stream();
            comm::stream_wait(stream, comm_stream);
            launch_all_reduce_epilogue(context->barrier_signals, hcomm_context->buffer, context->buffer,
                                       hcomm_context->device_context, context->rank_idx, context->num_ranks,
                                       hcomm_context->num_scaleup_ranks, hcomm_context->num_scaleup_jetties,
                                       hcomm_context->num_scaleout_jetties, num_buckets, buckets, scale,
                                       context->num_gpu_timeout_cycles, stream);
            comm::stream_wait(comm_stream, stream);
            // Rejoin comm_stream into the capture stream so ACLGraph capture does not leave it dangling.
            if (comm::is_stream_capturing(stream))
                comm::stream_wait(stream, comm_stream);
            return pybind11::cast(srcs);
        };
        return pybind11::make_tuple(event, epilogue);
    }

    pybind11::tuple all_gather(const std::vector<torch::Tensor>& srcs,
                              const std::vector<torch::Tensor>& dsts,
                              const int& context_idx, const int& num_sms) const {
        check_available();
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < contexts.size());
        EP_HOST_ASSERT(num_sms >= 0 and num_sms <= runtime->get_num_aicore_cores());
        EP_HOST_ASSERT(not srcs.empty() and srcs.size() <= comm::kNumMaxBuckets);
        EP_HOST_ASSERT(dsts.empty() or dsts.size() == srcs.size());
        const auto& context = contexts[context_idx];
        const auto rank_idx = context->rank_idx, num_ranks = context->num_ranks;

        // Validate gathered outputs and optional external sources
        const auto num_buckets = static_cast<int>(srcs.size());
        comm::BucketList buckets;
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
            gathered.push_back(storage.narrow(0, bucket_offset, num_ranks * num_shard_bytes).view(src.scalar_type()));
        }

        // Stream control and local staging
        const auto compute_stream = c10_npu::getCurrentNPUStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);
        if (not dsts.empty()) {
            for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
                const auto num_bytes = buckets[bucket_idx].num_bytes;
                auto* dst = math::advance_ptr<void>(context->buffer, buckets[bucket_idx].offset + rank_idx * num_bytes);
                const auto* src = srcs[bucket_idx].data_ptr();
                if (src != dst)
                    ACL_CHECK(aclrtMemcpyAsync(dst, num_bytes, src, num_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, comm_stream));
            }
        }

        const auto& hcomm_context = context->hcomm_context;
        const auto num_scaleup_ranks = hcomm_context->num_scaleup_ranks;
        const auto num_scaleup_jetties = std::min(num_scaleup_ranks - 1, hcomm_context->num_scaleup_jetties);
        const int num_scaleout_jetties = num_ranks > num_scaleup_ranks ? 1 : 0;
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context->workspace);
        if (num_sms > 0) {
            const auto num_vec_cores = num_sms * runtime->get_num_vec_cores_per_ai_core();
            launch_all_gather_aiv(&workspace.signals[context_idx], hcomm_context->buffer, context->buffer,
                                  hcomm_context->device_context, rank_idx, num_ranks, num_scaleup_ranks,
                                  num_vec_cores, num_scaleup_jetties, num_scaleout_jetties,
                                  num_buckets, buckets, context->num_gpu_timeout_cycles, comm_stream);
        } else {
            const auto num_timeout_ns =
                context->num_gpu_timeout_cycles / comm::kNumSystemCyclesPerSecond * 1'000'000'000LL;
            launch_all_gather_aicpu(&workspace.signals[context_idx], hcomm_context->buffer, context->buffer,
                                    hcomm_context->device_context, rank_idx, num_ranks, num_scaleup_ranks,
                                    num_scaleup_jetties, num_scaleout_jetties, num_buckets, buckets,
                                    num_timeout_ns, comm_stream);
        }

        auto event = EventHandle(comm_stream);

        // The AIV path defers CQ drain and barrier to wait(); the AICPU path finishes them on comm_stream.
        std::function<pybind11::object()> epilogue = [context, srcs, num_sms, gathered = std::move(gathered)]() {
            EP_HOST_ASSERT(deep_jit::get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 1 and
                           "URMA all-gather requires EP_AVOID_RECORD_STREAM=1");
            if (num_sms > 0)
                comm::barrier(*context, false, true, true);
            return pybind11::cast(gathered);
        };
        return pybind11::make_tuple(event, epilogue);
    }

    pybind11::tuple ring_send_recv(const std::vector<torch::Tensor>& srcs,
                                   const std::vector<torch::Tensor>& dsts,
                                   const int& shift, const bool& cyclic,
                                   const int& context_idx, const int& num_sms) const {
        check_available();
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        EP_HOST_ASSERT(num_sms == 0);
        EP_HOST_ASSERT(not srcs.empty() and srcs.size() <= comm::kNumMaxBuckets);
        EP_HOST_ASSERT(dsts.size() == srcs.size());
        const auto& context = contexts[context_idx];
        const auto rank_idx = context->rank_idx, num_ranks = context->num_ranks;
        const auto& hcomm_context = context->hcomm_context;
        const auto num_scaleup_ranks = hcomm_context->num_scaleup_ranks;
        EP_HOST_ASSERT(num_ranks > 1 and num_ranks == num_scaleup_ranks and
                       "Ring send-recv currently supports scale-up only");
        EP_HOST_ASSERT(cyclic ? (shift % num_ranks != 0) :
                                (shift != 0 and shift > -num_ranks and shift < num_ranks));

        const auto num_buckets = static_cast<int>(srcs.size());
        comm::BucketList src_buckets, dst_buckets;
        const auto buffer_ptr = reinterpret_cast<int64_t>(storage.data_ptr());
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& src = srcs[bucket_idx];
            const auto& dst = dsts[bucket_idx];
            EP_HOST_ASSERT(src.device() == storage.device() and src.is_contiguous() and src.numel() > 0);
            EP_HOST_ASSERT(dst.device() == storage.device() and dst.is_contiguous() and
                           dst.scalar_type() == src.scalar_type() and dst.nbytes() == src.nbytes());
            const auto num_bytes = static_cast<int64_t>(src.nbytes());
            const auto src_offset = reinterpret_cast<int64_t>(src.data_ptr()) - buffer_ptr;
            const auto dst_offset = reinterpret_cast<int64_t>(dst.data_ptr()) - buffer_ptr;
            EP_HOST_ASSERT(src_offset >= 0 and src_offset <= num_storage_bytes and
                           num_bytes <= num_storage_bytes - src_offset and
                           "Ring send-recv sources must belong to this BucketBuffer");
            EP_HOST_ASSERT(dst_offset >= 0 and dst_offset <= num_storage_bytes and
                           num_bytes <= num_storage_bytes - dst_offset and
                           "Ring send-recv destinations must belong to this BucketBuffer");
            EP_HOST_ASSERT((src_offset + num_bytes <= dst_offset or dst_offset + num_bytes <= src_offset) and
                           "Ring send-recv source and destination must not overlap");
            src_buckets[bucket_idx] = {.offset = src_offset, .num_bytes = num_bytes};
            dst_buckets[bucket_idx] = {.offset = dst_offset, .num_bytes = num_bytes};
        }

        const int norm_shift = cyclic ? ((shift % num_ranks) + num_ranks) % num_ranks : shift;
        const int raw_dst_rank_idx = cyclic ? (rank_idx + norm_shift) % num_ranks : (rank_idx + shift);
        const int dst_rank_idx = (raw_dst_rank_idx >= 0 and raw_dst_rank_idx < num_ranks) ? raw_dst_rank_idx : -1;

        const auto compute_stream = c10_npu::getCurrentNPUStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);

        constexpr int kNumRingSendRecvJetties = 4;
        const int num_jetties = std::min(kNumRingSendRecvJetties, hcomm_context->num_scaleup_jetties);
        const auto num_timeout_ns =
            context->num_gpu_timeout_cycles / comm::kNumSystemCyclesPerSecond * 1'000'000'000LL;
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context->workspace);
        launch_ring_send_recv(&workspace.signals[context_idx], hcomm_context->buffer, context->buffer,
                              hcomm_context->device_context, rank_idx, dst_rank_idx, num_ranks,
                              num_jetties, num_buckets, src_buckets, dst_buckets, num_timeout_ns, comm_stream);

        auto event = EventHandle(comm_stream);

        std::function<pybind11::object()> epilogue = [dsts]() {
            EP_HOST_ASSERT(deep_jit::get_env<int>("EP_AVOID_RECORD_STREAM", 0) == 1 and
                           "URMA ring send-recv requires EP_AVOID_RECORD_STREAM=1");
            return pybind11::cast(dsts);
        };
        return pybind11::make_tuple(event, epilogue);
    }
};

}  // namespace deep_ep
