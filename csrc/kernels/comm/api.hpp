#pragma once

#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/layout/common/context.hpp>
#include <deep_ep/layout/common/error.hpp>

#include "hcomm.hpp"
#include "../../comm/stream.hpp"
#include "../../comm/process_group.hpp"
#include "../../runtime/jit.hpp"

namespace deep_ep::comm {

class Context {
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory;
    bool finalized = false;
    std::tuple<int, int> physical_domain_size;

public:
    std::shared_ptr<HCommContext> hcomm_context;
    ProcessGroup group;
    int rank_idx, num_ranks;
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;
    int64_t num_workspace_bytes;
    int64_t num_gpu_buffer_bytes;
    int64_t num_rdma_storage_bytes;
    bool use_huge1g_page;
    bool use_cpu_rdma_storage = false;
    void* workspace;
    void* barrier_signals = nullptr;
    void* buffer;
    void* local_rdma_storage_ptr;
    void* host_error_status = nullptr;
    void* mapped_host_error_status = nullptr;

    Context(const ProcessGroup& group, const int& rank_idx, const int& num_ranks,
            const int64_t& num_workspace_bytes, const int64_t& num_gpu_buffer_bytes,
            const int64_t& num_rdma_storage_bytes, const bool& use_huge1g_page,
            const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
            const std::shared_ptr<symmetric::SymmetricMemory>& shared_memory = nullptr):
        group(group),
        rank_idx(rank_idx), num_ranks(num_ranks),
        num_cpu_timeout_secs(num_cpu_timeout_secs),
        num_gpu_timeout_cycles(static_cast<int64_t>(num_gpu_timeout_secs) * kNumSystemCyclesPerSecond),
        num_workspace_bytes(num_workspace_bytes), num_gpu_buffer_bytes(num_gpu_buffer_bytes),
        num_rdma_storage_bytes(num_rdma_storage_bytes), use_huge1g_page(use_huge1g_page) {
        EP_HOST_ASSERT(group != nullptr);
        EP_HOST_ASSERT(num_ranks > 0 and rank_idx >= 0 and rank_idx < num_ranks);
        EP_HOST_ASSERT(num_cpu_timeout_secs > 0 and num_gpu_timeout_secs > 0);
        EP_HOST_ASSERT(num_workspace_bytes > 0 and num_workspace_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_gpu_buffer_bytes >= 0 and num_rdma_storage_bytes >= 0);
        const auto num_bytes = num_workspace_bytes + math::align(num_gpu_buffer_bytes + num_rdma_storage_bytes,
                                                                kNumAllocationAlignmentBytes);
        const int num_scaleup_jetties = runtime->get_num_vec_cores();
        const int num_scaleout_jetties = runtime->get_num_vec_cores();
        hcomm_context = std::make_shared<HCommContext>(
            group, rank_idx, num_ranks, num_bytes, num_scaleup_jetties, num_scaleout_jetties,
            num_cpu_timeout_secs, use_huge1g_page, shared_memory);
        physical_domain_size = std::make_tuple(hcomm_context->num_scaleout_ranks, hcomm_context->num_scaleup_ranks);
        symmetric_memory = hcomm_context->memory;
        workspace = symmetric_memory->ptr;
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(workspace) % kNumAllocationAlignmentBytes == 0);
        buffer = math::advance_ptr<void>(symmetric_memory->ptr, num_workspace_bytes);
        local_rdma_storage_ptr = math::advance_ptr<void>(buffer, num_gpu_buffer_bytes);
        init_error_status();
    }


    void init_error_status() {
        if (host_error_status == nullptr) {
            ACL_CHECK(aclrtMallocHost(&host_error_status, sizeof(layout::ErrorStatusSlot)));
            const auto result = aclrtHostRegister(host_error_status, sizeof(layout::ErrorStatusSlot),
                                                  ACL_HOST_REGISTER_MAPPED, &mapped_host_error_status);
            if (result != ACL_SUCCESS) {
                aclrtFreeHost(host_error_status);
                host_error_status = nullptr;
                mapped_host_error_status = nullptr;
                ACL_CHECK(result);
            }
            std::memset(host_error_status, 0, sizeof(layout::ErrorStatusSlot));
        }

        if (hcomm_context != nullptr and mapped_host_error_status != nullptr) {
            const auto address = reinterpret_cast<uint64_t>(mapped_host_error_status);
            ACL_CHECK(aclrtMemcpy(&hcomm_context->device_context->error_status_addr, sizeof(uint64_t),
                                  &address, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }

    void clear_error_status() {
        if (host_error_status == nullptr)
            return;
        std::memset(host_error_status, 0, sizeof(layout::ErrorStatusSlot));
    }

    layout::ErrorCode get_error_status() const {
        EP_HOST_ASSERT(mapped_host_error_status != nullptr and "error record unavailable after finalize");
        return *static_cast<const volatile layout::ErrorCode*>(host_error_status);
    }

    void set_barrier_signals(void* ptr) {
        check_available();
        const auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(workspace);
        EP_HOST_ASSERT(ptr != nullptr and offset >= 0 and offset <= num_workspace_bytes);
        EP_HOST_ASSERT(sizeof(layout::BarrierSignals) <= num_workspace_bytes - offset);
        EP_HOST_ASSERT(offset % alignof(layout::BarrierSignals) == 0);
        barrier_signals = ptr;

        // Each group has its own peer table and barrier state in the shared workspace.
        // The UBMEM table contains only this scale-up domain's peers, including self.
        auto scaleup_peer_ptrs = hcomm_context->scaleup_peer_ptrs;
        for (auto& peer_ptr: scaleup_peer_ptrs)
            if (peer_ptr != 0)
                peer_ptr += offset;
        ACL_CHECK(aclrtMemcpy(hcomm_context->device_context->scaleup_peer_ptrs,
                              sizeof(hcomm_context->device_context->scaleup_peer_ptrs),
                              scaleup_peer_ptrs.data(), sizeof(uint64_t) * scaleup_peer_ptrs.size(), ACL_MEMCPY_HOST_TO_DEVICE));

        const auto num_scaleout_ranks = hcomm_context->num_scaleout_ranks;
        if (num_scaleout_ranks > 1) {
            auto scaleout_peer_ptrs = hcomm_context->scaleout_peer_ptrs;
            for (auto& peer_ptr: scaleout_peer_ptrs)
                if (peer_ptr != 0)
                    peer_ptr += offset;
            ACL_CHECK(aclrtMemcpy(hcomm_context->device_context->scaleout_peer_ptrs,
                                  sizeof(hcomm_context->device_context->scaleout_peer_ptrs),
                                  scaleout_peer_ptrs.data(), sizeof(uint64_t) * num_scaleout_ranks,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }

    std::tuple<int, int> get_physical_domain_size() const {
        return physical_domain_size;
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return get_physical_domain_size();
    }

    bool is_active() const {
        return hcomm_context != nullptr;
    }

    void check_available() const {
        EP_HOST_ASSERT(not finalized);
        EP_HOST_ASSERT(is_active() and "the buffer is not active; call reconfigure first");
    }

    // Callers stop old work first; keep error records until reconfigure or destroy.
    void invalidate(const int& num_timeout_secs) {
        EP_HOST_ASSERT(hcomm_context != nullptr);
        EP_HOST_ASSERT(num_timeout_secs > 0);

        const auto comm_stream = get_comm_stream();
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(comm_stream, num_timeout_secs * 1000));
        hcomm_context.reset();
    }

    void resize(const int64_t& num_gpu_buffer_bytes,
                const int64_t& num_rdma_storage_bytes,
                const std::vector<int64_t>& old_buffer_bytes,
                const std::vector<int64_t>& new_buffer_bytes) {
        check_available();
        EP_HOST_ASSERT(num_gpu_buffer_bytes >= 0 and num_rdma_storage_bytes >= 0);
        EP_HOST_ASSERT(old_buffer_bytes.size() == new_buffer_bytes.size());
        // Context and its HCommContext each own one reference; other contexts add more.
        EP_HOST_ASSERT(symmetric_memory.use_count() == 2 and "Resizing shared symmetric memory is not supported");
        const auto num_bytes = num_workspace_bytes + math::align(num_gpu_buffer_bytes + num_rdma_storage_bytes,
                                                                 kNumAllocationAlignmentBytes);
        auto new_hcomm_context = std::make_shared<HCommContext>(
            group, rank_idx, num_ranks, num_bytes, runtime->get_num_vec_cores(), runtime->get_num_vec_cores(),
            num_cpu_timeout_secs, use_huge1g_page,
            nullptr);

        // Copy old buffer contents to new buffer
        auto new_symmetric_memory = new_hcomm_context->memory;
        int64_t old_offset = num_workspace_bytes, new_offset = num_workspace_bytes;
        ACL_CHECK(aclrtMemcpy(new_symmetric_memory->ptr, num_workspace_bytes, symmetric_memory->ptr,
                              num_workspace_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
        for (size_t i = 0; i < old_buffer_bytes.size(); ++ i) {
            if (old_buffer_bytes[i] > 0 and new_buffer_bytes[i] > 0) {
                ACL_CHECK(aclrtMemcpy(
                    math::advance_ptr<void>(new_symmetric_memory->ptr, new_offset), new_buffer_bytes[i],
                    math::advance_ptr<void>(symmetric_memory->ptr, old_offset),
                    std::min(old_buffer_bytes[i], new_buffer_bytes[i]), ACL_MEMCPY_DEVICE_TO_DEVICE));
            }
            old_offset += old_buffer_bytes[i];
            new_offset += new_buffer_bytes[i];
        }

        this->num_gpu_buffer_bytes = num_gpu_buffer_bytes;
        this->num_rdma_storage_bytes = num_rdma_storage_bytes;
        hcomm_context = new_hcomm_context;
        symmetric_memory = new_symmetric_memory;
        workspace = symmetric_memory->ptr;
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(workspace) % kNumAllocationAlignmentBytes == 0);
        buffer = math::advance_ptr<void>(symmetric_memory->ptr, num_workspace_bytes);
        local_rdma_storage_ptr = math::advance_ptr<void>(buffer, num_gpu_buffer_bytes);
        init_error_status();
    }

    void finalize() {
        EP_HOST_ASSERT(not finalized);
        if (host_error_status != nullptr) {
            ACL_CHECK(aclrtHostUnregister(host_error_status));
            ACL_CHECK(aclrtFreeHost(host_error_status));
            host_error_status = nullptr;
            mapped_host_error_status = nullptr;
        }
        hcomm_context.reset();
        symmetric_memory.reset();
        finalized = true;
    }
};

}  // namespace deep_ep::comm
