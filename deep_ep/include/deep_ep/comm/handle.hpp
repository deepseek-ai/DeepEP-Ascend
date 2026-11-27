#pragma once

#include <cstddef>

#include <deep_ep/common/compiled.hpp>
#include <hcomm/hcomm.h>
#ifdef EP_AICPU
#include <cstring>
#include <deep_ep/common/aicpu.hpp>
#else
#include <kernel_operator.h>
#include <simt_api/device_functions.h>
#include <c_api/asc_simd.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>
#endif

#include <deep_ep/common/error.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/layout/common/error.hpp>

namespace deep_ep::handle {

static constexpr int kNumMaxSGEPairsPerSQE = 6;
static constexpr int kNumWQEBBBytes = 64;
static constexpr uint32_t kSQDepth = 32768;

// sqe_bb_idx[15:0] = 0, odr[18:16] = 0, cqe[21] = 0,
// owner[31] = 0 (filled per WQEBB), opcode[47:40] = 0x11.
static constexpr uint64_t kNopSQEHeaderTemplate = 0x0000110000000000ULL;

// sqe_bb_idx[15:0] = 0, odr[18:16] = 1, fence[19] = 0, se[20] = 0,
// cqe[21] = 0, inline_en[22] = 0, udf_flag[23] = 0, nf[27] = 0,
// token_en[28] = 1, remote_jetty_type[30:29] = 1, owner[31] = 0 (filled per WQEBB),
// target_hint[39:32] = 0, opcode[47:40] = 0x3, inline_msg_len[63:54] = 0.
static constexpr uint64_t kWriteSQEHeaderTemplate = 0x0000030030010000ULL;
// The final payload WRITE uses odr=5, fence=1 and cqe=1 to cover prior SQEs.
static constexpr uint64_t kWriteSQEFinalFlags = (1ULL << 18) | (1ULL << 19) | (1ULL << 21);

#ifdef EP_AICORE
struct alignas(8) HcommPeerInfo {
    // SQE bytes 8-15
    // [23:0] tp_id, [31:24] num_sges, [51:32] remote_token_id
    uint64_t tp_id_num_sges_remote_token_id;

    // SQE bytes 16-31
    uint64_t remote_eid[2];

    // SQE bytes 32-47
    // [31:0] remote_token_value, [63:32] UDF fields
    uint64_t remote_token_value_udf;
    uint64_t remote_base_addr;
    uint64_t remote_buffer_size;

    // One SIMT thread loads one peer; callers choose the peer-to-thread mapping.
    static __simt_callee__ __forceinline__ void load_peer_info(
            __ubuf__ HcommPeerInfo* peer_info, __gm__ const uint64_t* jetty_ptrs, const int jetty_idx,
            __gm__ layout::ErrorStatusSlot* error_status) {
        const auto jetty_value = jetty_ptrs[jetty_idx];
        if (jetty_value == 0)
            record_error_and_trap_simt(error_status, layout::kInvalidCommMetadata);
        const auto channel = reinterpret_cast<__gm__ const AscendC::ChannelEntity*>(jetty_value);
        if (channel->remoteBufferNum != 1)
            record_error_and_trap_simt(error_status, layout::kInvalidCommMetadata);
        // HCOMM's generic pointers must be loaded as GM addresses in SIMT code.
        const auto channel_words = reinterpret_cast<__gm__ const uint64_t*>(channel);
        const auto remote_buffer = reinterpret_cast<__gm__ const AscendC::RegedBufferEntity*>(
            channel_words[offsetof(AscendC::ChannelEntity, remoteBufferAddr) / sizeof(uint64_t)]);
        const auto sq_context = reinterpret_cast<__gm__ const AscendC::SqContext*>(
            channel_words[offsetof(AscendC::ChannelEntity, sqContextAddr) / sizeof(uint64_t)]) +
            AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
        peer_info->remote_token_value_udf = remote_buffer->bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        peer_info->remote_base_addr = remote_buffer->bufferInfo.rma.addr;
        peer_info->remote_buffer_size = remote_buffer->bufferInfo.rma.size;
        peer_info->tp_id_num_sges_remote_token_id =
            static_cast<uint64_t>(sq_context->contextInfo.ubJfs.tpID) |
            (static_cast<uint64_t>(remote_buffer->bufferInfo.rma.protectionInfo.memInfo.ub.tokenId) << 32);
        const auto eid = reinterpret_cast<__gm__ const uint64_t*>(sq_context->contextInfo.ubJfs.remoteEID);
        peer_info->remote_eid[0] = eid[0];
        peer_info->remote_eid[1] = eid[1];
    }
};
EP_STATIC_ASSERT(sizeof(HcommPeerInfo) == 48);
EP_STATIC_ASSERT(alignof(HcommPeerInfo) == 8);

struct alignas(8) HcommJettyInfo {
    uint64_t sq_base_addr;
    uint64_t sq_head_addr;
    uint64_t completion_tail_addr;
    uint64_t sq_doorbell_addr;
    uint64_t cq_base_addr;
    uint64_t cq_tail_addr;
    uint64_t cq_doorbell_addr;
    // packed_head[31:0] = current SQ WQEBB head, [63:32] = expected completion count.
    uint64_t packed_head;
    // Match the SQ/CQ context words so each SIMT lane can copy one aligned u64.
    uint32_t reserved_jfs_id;
    uint32_t num_wqebb_bytes;
    uint32_t reserved_jfc_id;
    uint32_t num_cqe_bytes;
    uint32_t cq_depth;
    uint32_t reserved;

    // One warp loads one Jetty; lanes [0, 11) each copy one u64, including the live SQ head.
    static __simt_callee__ __forceinline__ void load_jetty_info(
            __ubuf__ HcommJettyInfo* jetty_info, __gm__ const uint64_t* jetty_ptrs,
            const int jetty_idx, const int lane_idx, __gm__ layout::ErrorStatusSlot* error_status) {
        if (lane_idx >= sizeof(HcommJettyInfo) / sizeof(uint64_t))
            return;
        const auto jetty_value = jetty_ptrs[jetty_idx];
        if (jetty_value == 0)
            record_error_and_trap_simt(error_status, layout::kInvalidCommMetadata);
        const auto channel = reinterpret_cast<__gm__ const AscendC::ChannelEntity*>(jetty_value);
        if (channel->sqNum == 0 or channel->cqNum == 0)
            record_error_and_trap_simt(error_status, layout::kInvalidCommMetadata);
        const auto channel_words = reinterpret_cast<__gm__ const uint64_t*>(channel);
        const auto sq_context = reinterpret_cast<__gm__ AscendC::SqContext*>(
            channel_words[offsetof(AscendC::ChannelEntity, sqContextAddr) / sizeof(uint64_t)]) +
            AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
        const auto cq_context = reinterpret_cast<__gm__ AscendC::CqContext*>(
            channel_words[offsetof(AscendC::ChannelEntity, cqContextAddr) / sizeof(uint64_t)]) +
            AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
        const auto sq_words = reinterpret_cast<__gm__ uint64_t*>(&sq_context->contextInfo.ubJfs);
        const auto cq_words = reinterpret_cast<__gm__ uint64_t*>(&cq_context->contextInfo.ubJfc);
        __gm__ uint64_t* src;
        if (lane_idx < 4)
            src = sq_words + lane_idx;
        else if (lane_idx < 7)
            src = cq_words + (lane_idx == 4 ? 0 : lane_idx - 3);
        else if (lane_idx == 7)
            src = reinterpret_cast<__gm__ uint64_t*>(sq_words[1]);
        else
            src = lane_idx < 9 ? sq_words + lane_idx - 4 : cq_words + lane_idx - 5;
        // Use one common load instruction, with a field-specific address in each lane.
        reinterpret_cast<__ubuf__ uint64_t*>(jetty_info)[lane_idx] = asc_ldcg(src);
    }
};
static_assert(sizeof(HcommJettyInfo) == 88);
static_assert(alignof(HcommJettyInfo) == 8);

struct HcommPeer {
    __ubuf__ HcommPeerInfo* peer_info = nullptr;

    __aicore__ __forceinline__ explicit HcommPeer(__ubuf__ HcommPeerInfo* peer_info):
        peer_info(peer_info) {}

    __aicore__ __forceinline__ HcommPeer(
            __ubuf__ HcommPeerInfo* peer_info,
            __gm__ void* jetty_ptrs, const int jetty_idx):
        peer_info(peer_info) {
        // Load the remote registered buffer
        const auto jetty_table = static_cast<__gm__ uint64_t*>(jetty_ptrs);
        const auto jetty_value = AscendC::ReadGmByPassDCache(jetty_table + jetty_idx);
        EP_DEVICE_ASSERT(jetty_value != 0);
        const auto jetty = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(jetty_value);
        EP_DEVICE_ASSERT(jetty->remoteBufferNum == 1);
        const auto remote_buffer = jetty->remoteBufferAddr[0];
        peer_info->remote_token_value_udf = remote_buffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        peer_info->remote_base_addr = remote_buffer.bufferInfo.rma.addr;
        peer_info->remote_buffer_size = remote_buffer.bufferInfo.rma.size;

        // Load the remote Jetty transport metadata
        const auto sq_context = jetty->sqContextAddr[AscendC::HCOMM_URMA_DEFAULT_QP_IDX];
        peer_info->tp_id_num_sges_remote_token_id =
            static_cast<uint64_t>(sq_context.contextInfo.ubJfs.tpID) |
            (static_cast<uint64_t>(remote_buffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId) << 32);
        peer_info->remote_eid[0] = 0;
        peer_info->remote_eid[1] = 0;
        for (int byte_idx = 0; byte_idx < 8; ++ byte_idx) {
            peer_info->remote_eid[0] |=
                static_cast<uint64_t>(sq_context.contextInfo.ubJfs.remoteEID[byte_idx]) << (byte_idx * 8);
            peer_info->remote_eid[1] |=
                static_cast<uint64_t>(sq_context.contextInfo.ubJfs.remoteEID[byte_idx + 8]) << (byte_idx * 8);
        }
    }
};

struct HcommJetty {
    static constexpr uint32_t kNumWriteWQEBBs =
        (sizeof(AscendC::HcommUrmaSqeCtx) + sizeof(AscendC::HcommUrmaSgeCtx) + kNumWQEBBBytes - 1) /
        kNumWQEBBBytes;
    static constexpr uint32_t kNumReadWQEBBs = kNumWriteWQEBBs;
    static constexpr uint32_t kNumWriteWithNotifyWQEBBs =
        (sizeof(AscendC::HcommUrmaSqeCtx) + sizeof(AscendC::HcommUrmaNotifyCtx) +
         sizeof(AscendC::HcommUrmaSgeCtx) + kNumWQEBBBytes - 1) / kNumWQEBBBytes;
    static constexpr uint32_t kNumMaxSQEBytes = kNumWriteWithNotifyWQEBBs * kNumWQEBBBytes;

    __ubuf__ HcommJettyInfo* jetty_info = nullptr;
    __ubuf__ uint8_t* ub_sqe = nullptr;
    __gm__ layout::ErrorStatusSlot* error_status = nullptr;

    __aicore__ __forceinline__ HcommJetty(
            __ubuf__ HcommJettyInfo* jetty_info,
            __ubuf__ uint8_t* ub_sqe,
            __gm__ void* jetty_ptrs, const int jetty_idx,
            __gm__ layout::ErrorStatusSlot* error_status = nullptr):
        jetty_info(jetty_info),
        ub_sqe(ub_sqe),
        error_status(error_status) {

        // TODO(HUAWEI): enable loopback Jetty
        if (jetty_ptrs == nullptr)
            return;

        // Load shared Jetty info
        const auto jetty_table = static_cast<__gm__ uint64_t*>(jetty_ptrs);
        const auto jetty_value = AscendC::ReadGmByPassDCache(jetty_table + jetty_idx);
        EP_DEVICE_ASSERT(jetty_value != 0);
        const auto jetty = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(jetty_value);
        EP_DEVICE_ASSERT(jetty->sqNum > 0 and jetty->cqNum > 0);

        // Load queue resources
        const auto sq_context = jetty->sqContextAddr[AscendC::HCOMM_URMA_DEFAULT_QP_IDX];
        const auto cq_context = jetty->cqContextAddr[AscendC::HCOMM_URMA_DEFAULT_QP_IDX];
        jetty_info->sq_base_addr = sq_context.contextInfo.ubJfs.sqVa;
        jetty_info->sq_head_addr = sq_context.contextInfo.ubJfs.headAddr;
        jetty_info->completion_tail_addr = sq_context.contextInfo.ubJfs.tailAddr;
        jetty_info->sq_doorbell_addr = sq_context.contextInfo.ubJfs.dbVa;
        jetty_info->cq_base_addr = cq_context.contextInfo.ubJfc.scqVa;
        jetty_info->cq_tail_addr = cq_context.contextInfo.ubJfc.tailAddr;
        jetty_info->cq_doorbell_addr = cq_context.contextInfo.ubJfc.dbVa;
        jetty_info->packed_head = AscendC::ReadGmByPassDCache(
            reinterpret_cast<__gm__ uint64_t*>(jetty_info->sq_head_addr));
        jetty_info->num_wqebb_bytes = sq_context.contextInfo.ubJfs.wqeSize;
        jetty_info->num_cqe_bytes = cq_context.contextInfo.ubJfc.cqeSize;
        jetty_info->cq_depth = cq_context.contextInfo.ubJfc.cqDepth;
    }

    template <int64_t kNumTimeoutCycles>
    __aicore__ __forceinline__ void poll_cq_when_sq_overflow() {
        // Trigger CQ polling before the completion ring reaches its wraparound threshold.
        static constexpr uint32_t kNumPollCQThreshold = 10;
        // Reclaim a bounded batch to free CQ slots without draining all in-flight writes.
        static constexpr uint32_t kNumCQEsPerPoll = 100;
        const auto num_completions = get_expected_cq_tail();
        const auto completion_tail = AscendC::ReadGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->completion_tail_addr));

        if ((num_completions + kNumPollCQThreshold) % jetty_info->cq_depth ==
            completion_tail % jetty_info->cq_depth) {
            poll_cq<kNumTimeoutCycles>(
                math::min(completion_tail + kNumCQEsPerPoll, num_completions));
        }
    }

    template <bool kIsRead, bool kDoReduce, bool kWithNotify, bool kStrongPlacementOrder,
              bool kInline = false, typename InlineT = int>
    __aicore__ __forceinline__ void build_sqe(
            const HcommPeer& peer, __gm__ void* local_base,
            __gm__ void* remote_sym_ptr, __gm__ void* local_ptr, uint64_t num_bytes,
            const uint32_t current_head,
            __gm__ void* notify_sym_ptr = nullptr, const uint64_t notify_value = 0,
            const bool request_cqe = true, const InlineT inline_data = {}) {
        static_assert(!kIsRead || !kWithNotify, "URMA READ with notify is not supported by this builder");

        // false: ODR=5, relaxed placement; later strong requests cannot overtake it.
        // true: ODR=6, strong placement; in source-Jetty ordering mode, waits for
        // all responses to prior relaxed SQEs. Both settings require ordered CQEs;
        // neither alone guarantees execution order for subsequent relaxed SQEs.
        static constexpr uint32_t kOrdering = kStrongPlacementOrder ? 6U : 5U;
        static constexpr uint32_t kFence = 1U << 3;
        static constexpr uint32_t kRequestCQE = 1U << 5;
        static constexpr uint32_t kUdfFlag = 1U << 7;
        static constexpr uint32_t kInlineFlag = 1U << 6;  // inline_en[22]

        if constexpr (kInline)
            num_bytes = sizeof(InlineT);
        // Reduction always interprets the payload and destination as FP32 elements.
        if constexpr (kDoReduce)
            EP_DEVICE_ASSERT(num_bytes % sizeof(float) == 0);

        const auto peer_info = peer.peer_info;
        // READ uses the remote address as its source; WRITE uses it as its destination.
        const auto remote_offset = reinterpret_cast<uint64_t>(remote_sym_ptr) - reinterpret_cast<uint64_t>(local_base);
        EP_DEVICE_ASSERT(
            remote_offset <= peer_info->remote_buffer_size and num_bytes <= peer_info->remote_buffer_size - remote_offset);
        const auto remote_addr = peer_info->remote_base_addr + remote_offset;

        // Fill the SQE header
        const auto sqe = reinterpret_cast<__ubuf__ AscendC::HcommUrmaSqeCtx*>(ub_sqe);
        sqe->sqeBbIdx = 0;
        sqe->flag = kOrdering | kFence | (request_cqe ? kRequestCQE : 0) | (kDoReduce ? kUdfFlag : 0) | (kInline ? kInlineFlag : 0);
        sqe->rsv0 = 0;
        sqe->nf = 0;
        sqe->tokenEn = 1;
        sqe->rmtJettyType = 1;
        sqe->owner = (current_head & kSQDepth) == 0 ? 1 : 0;
        sqe->targetHint = 0;
        sqe->opcode = static_cast<uint32_t>(kIsRead ? AscendC::HcommUrmaOpCode::READ :
            (kWithNotify ? AscendC::HcommUrmaOpCode::WRITE_WITH_NOTIFY : AscendC::HcommUrmaOpCode::WRITE));
        sqe->rsv1 = 0;
        sqe->inlineMsgLen = kInline ? sizeof(InlineT) : 0;
        sqe->tpId = static_cast<uint32_t>(peer_info->tp_id_num_sges_remote_token_id);
        sqe->sgeNum = kInline ? 0 : 1;
        sqe->rmtJettyOrSegId = static_cast<uint32_t>(peer_info->tp_id_num_sges_remote_token_id >> 32);
        sqe->rsv2 = 0;
        sqe->rmtEidL = peer_info->remote_eid[0];
        sqe->rmtEidH = peer_info->remote_eid[1];
        sqe->rmtTokenValue = static_cast<uint32_t>(peer_info->remote_token_value_udf);
        sqe->udfType = 0;
        sqe->reduceDataType = kDoReduce ? 0x7U : 0;  // URMA FP32 encoding
        sqe->reduceOpcode = kDoReduce ? 0xAU : 0;    // URMA SUM encoding
        sqe->rsv3 = 0;
        sqe->rmtAddrLOrTokenId = static_cast<uint32_t>(remote_addr);
        sqe->rmtAddrHOrTokenValue = static_cast<uint32_t>(remote_addr >> 32);

        // Fill optional notify metadata
        __ubuf__ uint8_t* ub_sge = ub_sqe + sizeof(AscendC::HcommUrmaSqeCtx);
        if constexpr (kWithNotify) {
            EP_DEVICE_ASSERT(notify_sym_ptr != nullptr);
            const auto notify_offset =
                reinterpret_cast<uint64_t>(notify_sym_ptr) - reinterpret_cast<uint64_t>(local_base);
            EP_DEVICE_ASSERT(notify_offset <= peer_info->remote_buffer_size and
                             sizeof(uint64_t) <= peer_info->remote_buffer_size - notify_offset);
            const auto remote_notify_addr = peer_info->remote_base_addr + notify_offset;
            const auto notify = reinterpret_cast<__ubuf__ AscendC::HcommUrmaNotifyCtx*>(ub_sge);
            notify->notifyTokenId = static_cast<uint32_t>(peer_info->tp_id_num_sges_remote_token_id >> 32);
            notify->rsv = 0;
            notify->notifyTokenValue = static_cast<uint32_t>(peer_info->remote_token_value_udf);
            notify->notifyAddrL = static_cast<uint32_t>(remote_notify_addr);
            notify->notifyAddrH = static_cast<uint32_t>(remote_notify_addr >> 32);
            notify->notifyDataL = static_cast<uint32_t>(notify_value);
            notify->notifyDataH = static_cast<uint32_t>(notify_value >> 32);
            notify->rsv2[0] = 0;
            notify->rsv2[1] = 0;
            ub_sge += sizeof(AscendC::HcommUrmaNotifyCtx);
        }

        if constexpr (kInline) {
            // Keep inline data within a single WQEBB.
            EP_STATIC_ASSERT(
                sizeof(InlineT) <= kNumWQEBBBytes - sizeof(AscendC::HcommUrmaSqeCtx),
                "inline data exceeds single-WQEBB capacity");
            *reinterpret_cast<__ubuf__ InlineT*>(ub_sge) = inline_data;
        } else {
            // The local SGE is the READ destination or WRITE source.
            const auto sge = reinterpret_cast<__ubuf__ AscendC::HcommUrmaSgeCtx*>(ub_sge);
            sge->len = static_cast<uint32_t>(num_bytes);
            sge->tokenId = 0;
            sge->va = reinterpret_cast<uint64_t>(local_ptr);
        }
    }

    __aicore__ __forceinline__ void copy_sqe_to_sq(
            const uint32_t current_head, const uint32_t num_wqebbs) {
        // READ/WRITE use one WQEBB; WRITE_WITH_NOTIFY uses two and may wrap the SQ.
        const auto sq_idx = current_head & (kSQDepth - 1);
        const auto num_first_wqebbs = math::min<uint32_t>(num_wqebbs, kSQDepth - sq_idx);

        // Copy the first contiguous SQ segment
        asc_sync_notify(PIPE_S, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_S, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2gm_align(
            reinterpret_cast<__gm__ uint8_t*>(
                jetty_info->sq_base_addr + static_cast<uint64_t>(jetty_info->num_wqebb_bytes) * sq_idx),
            ub_sqe,
            /* n_burst= */ 1, /* len_burst= */ jetty_info->num_wqebb_bytes * num_first_wqebbs,
            /* l2_cache_mode= */ static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NOTALLOC_CI),
            /* dst_gap= */ 0, /* src_gap= */ 0);

        // Wrap the remaining WQEBBs to SQ base
        if (num_first_wqebbs < num_wqebbs) {
            asc_copy_ub2gm_align(
                reinterpret_cast<__gm__ uint8_t*>(jetty_info->sq_base_addr),
                ub_sqe + jetty_info->num_wqebb_bytes * num_first_wqebbs,
                /* n_burst= */ 1,
                /* len_burst= */ jetty_info->num_wqebb_bytes * (num_wqebbs - num_first_wqebbs),
                /* l2_cache_mode= */ static_cast<uint8_t>(ST_L2CacheType::L2_CACHE_HINT_NOTALLOC_CI),
                /* dst_gap= */ 0, /* src_gap= */ 0);
        }
        asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

    __aicore__ __forceinline__ void advance_sq(const uint32_t num_wqebbs, const uint32_t num_completions = 1) {
        const auto new_head = static_cast<uint32_t>(jetty_info->packed_head) + num_wqebbs;
        const auto new_num_completions = get_expected_cq_tail() + num_completions;
        jetty_info->packed_head = static_cast<uint64_t>(new_head) |
            (static_cast<uint64_t>(new_num_completions) << 32);
    }

    __aicore__ __forceinline__ void set_sq_head(const uint32_t new_head) {
        jetty_info->packed_head = (jetty_info->packed_head & 0xFFFFFFFF00000000ULL) | new_head;
    }

    __aicore__ __forceinline__ void ring_doorbell() {
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint64_t*>(jetty_info->sq_head_addr),
            jetty_info->packed_head);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->sq_doorbell_addr),
            static_cast<uint32_t>(jetty_info->packed_head));
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    template <int64_t kNumTimeoutCycles, bool kDoReduce, bool kWithNotify, bool kDoCommit, bool kStrongPlacementOrder,
              bool kInline = false, typename InlineT = int>
    __aicore__ __forceinline__ void post_write(
            const HcommPeer& peer, __gm__ void* local_base,
            __gm__ void* dst_sym_ptr, __gm__ void* src_ptr, const uint64_t num_bytes,
            __gm__ void* notify_sym_ptr = nullptr, const uint64_t notify_value = 0,
            const bool request_cqe = true, const InlineT inline_data = {}) {
        const auto num_wqebbs = kWithNotify ? kNumWriteWithNotifyWQEBBs : kNumWriteWQEBBs;
        const auto current_head = static_cast<uint32_t>(jetty_info->packed_head);

        // Reclaim CQ entries and prepare the SQE
        poll_cq_when_sq_overflow<kNumTimeoutCycles>();
        build_sqe<false, kDoReduce, kWithNotify, kStrongPlacementOrder, kInline, InlineT>(
            peer, local_base, dst_sym_ptr, src_ptr, num_bytes,
            current_head, notify_sym_ptr, notify_value, request_cqe, inline_data);

        // Publish the SQE and optionally ring the doorbell
        copy_sqe_to_sq(current_head, num_wqebbs);
        advance_sq(num_wqebbs, request_cqe ? 1 : 0);
        if constexpr (kDoCommit)
            ring_doorbell();
    }

    template <int64_t kNumTimeoutCycles, bool kDoReduce, bool kWithNotify, bool kDoCommit, bool kStrongPlacementOrder>
    __aicore__ __forceinline__ void post_read(
            const HcommPeer& peer, __gm__ void* local_base,
            __gm__ void* dst_ptr, __gm__ void* src_sym_ptr, const uint64_t num_bytes) {
        const auto current_head = static_cast<uint32_t>(jetty_info->packed_head);

        // Reclaim CQ entries and prepare the SQE
        poll_cq_when_sq_overflow<kNumTimeoutCycles>();
        build_sqe<true, kDoReduce, kWithNotify, kStrongPlacementOrder>(
            peer, local_base, src_sym_ptr, dst_ptr, num_bytes, current_head);

        // Publish the SQE and optionally ring the doorbell
        copy_sqe_to_sq(current_head, kNumReadWQEBBs);
        advance_sq(kNumReadWQEBBs);
        if constexpr (kDoCommit)
            ring_doorbell();
    }

    template <int64_t kNumTimeoutCycles, bool kDoReduce = false, bool kDoCommit = true, bool kStrongPlacementOrder = false>
    __aicore__ __forceinline__ void put(
            const HcommPeer& peer, __gm__ void* local_base,
            __gm__ void* dst_sym_ptr, __gm__ void* src_ptr,
            const uint64_t num_bytes) {
        post_write<kNumTimeoutCycles, kDoReduce, false, kDoCommit, kStrongPlacementOrder>(
            peer, local_base, dst_sym_ptr, src_ptr, num_bytes);
    }

    template <int64_t kNumTimeoutCycles, bool kDoReduce = false, bool kDoCommit = true, bool kStrongPlacementOrder = false>
    __aicore__ __forceinline__ void get(
            const HcommPeer& peer, __gm__ void* local_base,
            __gm__ void* dst_ptr, __gm__ void* src_sym_ptr,
            const uint64_t num_bytes) {
        post_read<kNumTimeoutCycles, kDoReduce, false, kDoCommit, kStrongPlacementOrder>(
            peer, local_base, dst_ptr, src_sym_ptr, num_bytes);
    }

    template <int64_t kNumTimeoutCycles>
    __aicore__ __forceinline__ void put_with_notify(
            const HcommPeer& peer, __gm__ void* local_base,
            __gm__ void* dst_sym_ptr, __gm__ void* src_ptr, const uint64_t num_bytes,
            __gm__ void* notify_sym_ptr, const uint64_t notify_value) {
        post_write<kNumTimeoutCycles, false, true, true, false>(
            peer, local_base, dst_sym_ptr, src_ptr, num_bytes,
            notify_sym_ptr, notify_value);
    }

    template <int64_t kNumTimeoutCycles>
    __aicore__ __forceinline__ void poll_cq(const uint32_t expected_tail) {
        static constexpr uint32_t kNumCQDoorbellBits = 24;
        static constexpr uint32_t kCQEOwnerBit = 2;

        // Poll expected completions
        auto current_tail = get_current_cq_tail();
        while (current_tail != expected_tail) {
            const auto cqe = reinterpret_cast<__gm__ uint32_t*>(
                jetty_info->cq_base_addr +
                static_cast<uint64_t>(jetty_info->num_cqe_bytes) *
                    (current_tail & (jetty_info->cq_depth - 1)));
            const auto tail_phase = (current_tail / jetty_info->cq_depth) & 1;
            const auto start_cycle = static_cast<uint64_t>(asc_get_system_cycle());
            uint32_t cqe_header = 0;
            while (true) {
                cqe_header = AscendC::ReadGmByPassDCache(cqe);
                const auto cqe_owner = (cqe_header >> kCQEOwnerBit) & 1;
                // Hardware flips the owner bit when publishing a new CQE; matching the tail phase means stale data.
                if (cqe_owner != tail_phase)
                    break;

                if (static_cast<uint64_t>(asc_get_system_cycle()) - start_cycle >= kNumTimeoutCycles) {
                    AscendC::printf(
                        "DeepEP HCOMM scalar CQ poll timeout, current tail: %u, expected tail: %u, cqe: 0x%x\n",
                        current_tail, expected_tail, cqe_header);
                    record_error_and_trap(error_status, layout::kCQEPollTimeout);
                }
                AscendC::Nop<300>();
            }

            const auto substatus = static_cast<uint8_t>(cqe_header >> 16);
            const auto status = static_cast<uint8_t>(cqe_header >> 24);
            if (status != 0 or substatus != 0) {
                AscendC::printf(
                    "DeepEP HCOMM scalar CQE failed, current tail: %u, status: %u, substatus: %u\n",
                    current_tail, status, substatus);
                record_error_and_trap(error_status, layout::kCQEError);
            }
            ++ current_tail;
        }

        // Publish reclaimed CQ and SQ tails
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->cq_tail_addr), current_tail);
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->cq_doorbell_addr),
            current_tail & ((1U << kNumCQDoorbellBits) - 1));
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->completion_tail_addr), current_tail);
    }

    // Reclaim the ready CQEs without blocking. Return the exclusive completion tail.
    __aicore__ __forceinline__ uint32_t try_poll_cq(const uint32_t expected_tail) {
        static constexpr uint32_t kNumCQDoorbellBits = 24;
        static constexpr uint32_t kCQEOwnerBit = 2;

        // Reclaim the completed entries without waiting for the next CQE
        auto current_tail = get_current_cq_tail();
        const auto initial_tail = current_tail;
        while (static_cast<int32_t>(expected_tail - current_tail) > 0) {
            const auto cqe = reinterpret_cast<__gm__ uint32_t*>(
                jetty_info->cq_base_addr +
                static_cast<uint64_t>(jetty_info->num_cqe_bytes) *
                    (current_tail & (jetty_info->cq_depth - 1)));
            const auto tail_phase = (current_tail / jetty_info->cq_depth) & 1;
            const auto cqe_header = AscendC::ReadGmByPassDCache(cqe);
            // A nonblocking poll reclaims only the contiguous completed prefix.
            if (((cqe_header >> kCQEOwnerBit) & 1) == tail_phase)
                break;

            const auto substatus = static_cast<uint8_t>(cqe_header >> 16);
            const auto status = static_cast<uint8_t>(cqe_header >> 24);
            if (status != 0 or substatus != 0) {
                AscendC::printf(
                    "DeepEP HCOMM scalar CQE failed, current tail: %u, status: %u, substatus: %u\n",
                    current_tail, status, substatus);
                record_error_and_trap(error_status, layout::kCQEError);
            }
            ++ current_tail;
        }

        // Publish reclaimed CQ tails
        if (current_tail == initial_tail)
            return current_tail;

        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->cq_tail_addr), current_tail);
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->cq_doorbell_addr),
            current_tail & ((1U << kNumCQDoorbellBits) - 1));
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(jetty_info->completion_tail_addr), current_tail);
        return current_tail;
    }

    // Preserve the four-WQEBB boundary required by dispatch/combine on these same queues.
    template <int64_t kNumTimeoutCycles>
    __aicore__ __forceinline__ void align_sq() {
        const auto head = static_cast<uint32_t>(jetty_info->packed_head);
        const auto num_padding = math::align(head, static_cast<uint32_t>(kSQEBBAlignment)) - head;
        if (num_padding == 0)
            return;
        poll_cq_when_sq_overflow<kNumTimeoutCycles>();
        for (uint32_t i = 0; i < num_padding; ++ i) {
            const auto idx = head + i;
            const auto owner = static_cast<uint64_t>((idx & kSQDepth) == 0) << 31;
            // Only the last NOP requests an ordered completion.
            const auto flags = i + 1 == num_padding ? (2ULL << 16) | (1ULL << 21) : 0;
            const auto sq_idx = idx & (kSQDepth - 1);
            AscendC::WriteGmByPassDCache(
                reinterpret_cast<__gm__ uint64_t*>(jetty_info->sq_base_addr + sq_idx * kNumWQEBBBytes),
                static_cast<uint64_t>(kNopSQEHeaderTemplate | flags | owner | sq_idx));
        }
        advance_sq(num_padding);
        ring_doorbell();
    }

    // Exclusive completion count for all SQEs posted on this Jetty.
    __aicore__ __forceinline__ uint32_t get_expected_cq_tail() const {
        return static_cast<uint32_t>(jetty_info->packed_head >> 32);
    }

    // Exclusive completion count already reclaimed by CQ polling.
    __aicore__ __forceinline__ uint32_t get_current_cq_tail() const {
        return AscendC::ReadGmByPassDCache(reinterpret_cast<__gm__ uint32_t*>(jetty_info->cq_tail_addr));
    }

    template <int64_t kNumTimeoutCycles>
    __aicore__ __forceinline__ void drain() {
        poll_cq<kNumTimeoutCycles>(get_expected_cq_tail());
    }
};
#endif

#ifdef EP_AICPU
namespace aicpu {

namespace memory = ::deep_ep::aicpu;

// Assumes that HCOMM tables and descriptors stay immutable while in use.
EP_INLINE const AscendC::ChannelEntity* load_channel(const void* jetty_ptrs, const int jetty_idx) {
    const auto value = static_cast<const uint64_t*>(jetty_ptrs)[jetty_idx];
    EP_DEVICE_ASSERT(value != 0);
    const auto channel = reinterpret_cast<const AscendC::ChannelEntity*>(value);
    return channel;
}

struct HcommPeer {
    uint64_t remote_base_addr, remote_buffer_size;
    uint64_t remote_eid[2];
    uint32_t tp_id, remote_token_id, remote_token_value;

    EP_INLINE HcommPeer(const void* jetty_ptrs, const int peer_idx) {
        const auto channel = load_channel(jetty_ptrs, peer_idx);
        EP_DEVICE_ASSERT(channel->remoteBufferNum == 1 and channel->sqNum > 0);
        const auto buffer = channel->remoteBufferAddr;
        const auto sq = channel->sqContextAddr + AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
        remote_base_addr = buffer->bufferInfo.rma.addr;
        remote_buffer_size = buffer->bufferInfo.rma.size;
        remote_token_id = buffer->bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        remote_token_value = buffer->bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        tp_id = sq->contextInfo.ubJfs.tpID;
        std::memcpy(remote_eid, sq->contextInfo.ubJfs.remoteEID, sizeof(remote_eid));
    }
};

struct HcommJetty {
private:
    uint8_t* sq_base;
    uint8_t* cq_base;
    uint64_t* sq_head_ptr;
    uint32_t* sq_doorbell;
    uint32_t* cq_tail_ptr;
    uint32_t* cq_doorbell;
    uint32_t* completion_tail_ptr;
    uint32_t cq_depth, num_cqe_bytes, max_outstanding;
    // committed_head is the first SQ WQEBB not yet published to the device.
    uint32_t sq_head, expected_tail, current_tail, committed_head;
    void* error_status = nullptr;

    template <int64_t kNumTimeoutNs>
    EP_INLINE void wait_for_room() {
        if (expected_tail - current_tail < max_outstanding)
            return;
        // Commit pending requests before waiting for their completions.
        const auto num_to_reclaim = math::min<uint32_t>(100, expected_tail - current_tail);
        poll_cq<kNumTimeoutNs>(current_tail + num_to_reclaim);
    }

    template <int64_t kNumTimeoutNs, bool kIsRead, bool kDoReduce, bool kWithNotify,
              bool kDoCommit, bool kStrongPlacementOrder, bool kInline = false, typename InlineT = uint64_t>
    EP_INLINE uint32_t post(
            const HcommPeer& peer, void* local_base, void* remote_sym_ptr, void* local_ptr,
            uint64_t num_bytes, void* notify_sym_ptr = nullptr, const uint64_t notify_value = 0,
            const InlineT inline_data = {}, const bool request_cqe = true) {
        EP_STATIC_ASSERT(!kIsRead or (!kWithNotify and !kInline));
        EP_STATIC_ASSERT(!kInline or (!kDoReduce and !kWithNotify));
        if constexpr (kInline) {
            EP_STATIC_ASSERT(sizeof(InlineT) == 4 or sizeof(InlineT) == 8);
            num_bytes = sizeof(InlineT);
        }
        EP_DEVICE_ASSERT(num_bytes <= UINT32_MAX);
        if constexpr (kDoReduce)
            EP_DEVICE_ASSERT(num_bytes % sizeof(float) == 0);
        const auto offset = reinterpret_cast<uintptr_t>(remote_sym_ptr) - reinterpret_cast<uintptr_t>(local_base);
        EP_DEVICE_ASSERT(offset <= peer.remote_buffer_size and num_bytes <= peer.remote_buffer_size - offset);
        wait_for_room<kNumTimeoutNs>();

        // 48-byte header and 16-byte SGE. Requested completions are ordered.
        EP_STATIC_ASSERT(sizeof(AscendC::HcommUrmaSqeCtx) == 48 and sizeof(AscendC::HcommUrmaSgeCtx) == 16);
        const uint64_t flags = (kStrongPlacementOrder ? 6U : 5U) | (1U << 3) | (request_cqe ? 1U << 5 : 0) |
            (kDoReduce ? 1U << 7 : 0) | (kInline ? 1U << 6 : 0);
        const auto opcode = kIsRead ? AscendC::HcommUrmaOpCode::READ :
            (kWithNotify ? AscendC::HcommUrmaOpCode::WRITE_WITH_NOTIFY : AscendC::HcommUrmaOpCode::WRITE);
        alignas(kNumWQEBBBytes) uint64_t sqe[kWithNotify ? 16 : 8]{};
        sqe[0] = (sq_head & (kSQDepth - 1)) | (flags << 16) | (1ULL << 28) | (1ULL << 29) |
            (static_cast<uint64_t>((sq_head & kSQDepth) == 0) << 31) | (static_cast<uint64_t>(opcode) << 40) |
            (kInline ? static_cast<uint64_t>(sizeof(InlineT)) << 54 : 0);
        sqe[1] = peer.tp_id | (kInline ? 0 : 1ULL << 24) | (static_cast<uint64_t>(peer.remote_token_id) << 32);
        sqe[2] = peer.remote_eid[0];
        sqe[3] = peer.remote_eid[1];
        sqe[4] = peer.remote_token_value | (kDoReduce ? (0x7ULL << 40) | (0xAULL << 44) : 0);
        sqe[5] = peer.remote_base_addr + offset;
        if constexpr (kWithNotify) {
            const auto notify_offset =
                reinterpret_cast<uintptr_t>(notify_sym_ptr) - reinterpret_cast<uintptr_t>(local_base);
            EP_DEVICE_ASSERT(notify_offset <= peer.remote_buffer_size and
                             sizeof(uint64_t) <= peer.remote_buffer_size - notify_offset);
            sqe[6] = peer.remote_token_id | (static_cast<uint64_t>(peer.remote_token_value) << 32);
            sqe[7] = peer.remote_base_addr + notify_offset;
            sqe[8] = notify_value;
        }
        constexpr int kSgeWord = kWithNotify ? 10 : 6;
        if constexpr (kInline) {
            std::memcpy(sqe + kSgeWord, &inline_data, sizeof(InlineT));
        } else {
            sqe[kSgeWord] = num_bytes;
            sqe[kSgeWord + 1] = reinterpret_cast<uintptr_t>(local_ptr);
        }

        constexpr uint32_t kNumWQEBBs = kWithNotify ? 2 : 1;
        for (uint32_t bb = 0; bb < kNumWQEBBs; ++ bb) {
            auto* dst = reinterpret_cast<volatile uint64_t*>(
                sq_base + ((sq_head + bb) & (kSQDepth - 1)) * kNumWQEBBBytes);
            for (int word = 1; word < 8; ++ word)
                dst[word] = sqe[bb * 8 + word];
            dst[0] = sqe[bb * 8];
        }
        sq_head += kNumWQEBBs;
        expected_tail += request_cqe;
        if constexpr (kDoCommit)
            ring_doorbell();
        return expected_tail;
    }

public:
    EP_INLINE HcommJetty(const void* jetty_ptrs, const int jetty_idx, void* error_status = nullptr):
        error_status(error_status) {
        const auto channel = load_channel(jetty_ptrs, jetty_idx);
        EP_DEVICE_ASSERT(channel->sqNum > 0 and channel->cqNum > 0);
        const auto sq = channel->sqContextAddr + AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
        const auto cq = channel->cqContextAddr + AscendC::HCOMM_URMA_DEFAULT_QP_IDX;
        const auto& sq_info = sq->contextInfo.ubJfs;
        const auto& cq_info = cq->contextInfo.ubJfc;
        sq_base = reinterpret_cast<uint8_t*>(sq_info.sqVa);
        sq_head_ptr = reinterpret_cast<uint64_t*>(sq_info.headAddr);
        completion_tail_ptr = reinterpret_cast<uint32_t*>(sq_info.tailAddr);
        sq_doorbell = reinterpret_cast<uint32_t*>(sq_info.dbVa);
        cq_base = reinterpret_cast<uint8_t*>(cq_info.scqVa);
        cq_tail_ptr = reinterpret_cast<uint32_t*>(cq_info.tailAddr);
        cq_doorbell = reinterpret_cast<uint32_t*>(cq_info.dbVa);
        // Use the same fixed SQ depth and owner-bit convention as AICore HcommJetty.
        cq_depth = cq_info.cqDepth;
        num_cqe_bytes = cq_info.cqeSize;
        // Read fresh queue cursors
        memory::refresh_and_load_gm(
            [&] {
                memory::refresh_gm<false>(sq_head_ptr);
                memory::refresh_gm<false>(cq_tail_ptr);
            },
            [&] {
                const auto packed_head = *reinterpret_cast<const volatile uint64_t*>(sq_head_ptr);
                sq_head = committed_head = static_cast<uint32_t>(packed_head);
                expected_tail = static_cast<uint32_t>(packed_head >> 32);
                current_tail = *reinterpret_cast<const volatile uint32_t*>(cq_tail_ptr);
            });
        // At most four WQEBBs per CQE, including the NOPs used by align_sq().
        max_outstanding = math::min<uint32_t>(cq_depth - 1, kSQDepth / kSQEBBAlignment - 1);
    }

    EP_INLINE void ring_doorbell() {
        if (sq_head == committed_head)
            return;
        const auto num_pending = sq_head - committed_head;
        const auto sq_idx = committed_head & (kSQDepth - 1);
        const auto first = math::min<uint32_t>(num_pending, kSQDepth - sq_idx);
        *reinterpret_cast<volatile uint64_t*>(sq_head_ptr) = static_cast<uint64_t>(sq_head) |
                                                           (static_cast<uint64_t>(expected_tail) << 32);
        memory::publish_gm<false>(sq_base + sq_idx * kNumWQEBBBytes, first * kNumWQEBBBytes);
        if (first < num_pending)
            memory::publish_gm<false>(sq_base, (num_pending - first) * kNumWQEBBBytes);
        memory::publish_gm<false>(sq_head_ptr);
        memory::fence();
        *reinterpret_cast<volatile uint32_t*>(sq_doorbell) = sq_head;
        memory::fence();
        committed_head = sq_head;
    }

    template <int64_t kNumTimeoutNs, bool kDoReduce = false, bool kDoCommit = true, bool kStrongPlacementOrder = false>
    EP_INLINE uint32_t put(const HcommPeer& peer, void* local_base, void* dst_sym_ptr, void* src_ptr,
                          const uint64_t num_bytes) {
        return post<kNumTimeoutNs, false, kDoReduce, false, kDoCommit, kStrongPlacementOrder>(
            peer, local_base, dst_sym_ptr, src_ptr, num_bytes);
    }

    template <int64_t kNumTimeoutNs, bool kDoReduce = false, bool kDoCommit = true, bool kStrongPlacementOrder = false>
    EP_INLINE uint32_t get(const HcommPeer& peer, void* local_base, void* dst_ptr, void* src_sym_ptr,
                          const uint64_t num_bytes) {
        return post<kNumTimeoutNs, true, kDoReduce, false, kDoCommit, kStrongPlacementOrder>(
            peer, local_base, src_sym_ptr, dst_ptr, num_bytes);
    }

    template <int64_t kNumTimeoutNs, bool kDoCommit = true, bool kStrongPlacementOrder = false>
    EP_INLINE uint32_t put_with_notify(const HcommPeer& peer, void* local_base, void* dst_sym_ptr, void* src_ptr,
                                      const uint64_t num_bytes, void* notify_sym_ptr, const uint64_t notify_value) {
        return post<kNumTimeoutNs, false, false, true, kDoCommit, kStrongPlacementOrder>(
            peer, local_base, dst_sym_ptr, src_ptr, num_bytes, notify_sym_ptr, notify_value);
    }

    // Without a CQE, the caller must bound outstanding SQEs and establish completion through its protocol.
    template <int64_t kNumTimeoutNs, bool kDoCommit = true, bool kStrongPlacementOrder = true, typename dtype_t>
    EP_INLINE uint32_t put_value(const HcommPeer& peer, void* local_base, void* dst_sym_ptr, const dtype_t value,
                                 const bool request_cqe = true) {
        return post<kNumTimeoutNs, false, false, false, kDoCommit, kStrongPlacementOrder, true, dtype_t>(
            peer, local_base, dst_sym_ptr, nullptr, sizeof(value), nullptr, 0, value, request_cqe);
    }

    EP_INLINE uint32_t get_expected_cq_tail() const { return expected_tail; }
    EP_INLINE uint32_t get_current_cq_tail() const { return current_tail; }

    // Consume only the ready CQEs. Return the exclusive completion tail.
    EP_INLINE uint32_t try_poll_cq(const uint32_t target) {
        const auto initial_tail = current_tail;
        while (static_cast<int32_t>(target - current_tail) > 0) {
            const auto cqe = reinterpret_cast<const uint32_t*>(
                cq_base + static_cast<uint64_t>(current_tail & (cq_depth - 1)) * num_cqe_bytes);
            const auto header = memory::refresh_and_load_gm(cqe);
            if (((header >> 2) & 1) == ((current_tail / cq_depth) & 1))
                break;
            if (header >> 16) {
                record_error(error_status, layout::kCQEError);
                throw EPExceptionWithLineInfo("URMA CQE", "tail=" + std::to_string(current_tail) +
                    ", status=" + std::to_string(header >> 24) +
                    ", substatus=" + std::to_string((header >> 16) & 255));
            }
            ++ current_tail;
        }
        if (current_tail != initial_tail) {
            // Batch local GM cursors and the CQ software doorbell between ordering fences.
            memory::store_and_publish_gm(
                [&] {
                    *reinterpret_cast<volatile uint32_t*>(cq_tail_ptr) = current_tail;
                    *reinterpret_cast<volatile uint32_t*>(cq_doorbell) = current_tail & 0xFFFFFFU;
                    *reinterpret_cast<volatile uint32_t*>(completion_tail_ptr) = current_tail;
                },
                [&] {
                    memory::publish_gm<false>(cq_tail_ptr);
                    memory::publish_gm<false>(cq_doorbell);
                    memory::publish_gm<false>(completion_tail_ptr);
                });
        }
        return current_tail;
    }

    template <int64_t kNumTimeoutNs>
    EP_INLINE void poll_cq(const uint32_t target) {
        ring_doorbell();
        if (static_cast<int32_t>(target - current_tail) <= 0)
            return;
        memory::timeout_while<kNumTimeoutNs>([&](const bool timeout) {
            if (static_cast<int32_t>(target - try_poll_cq(target)) <= 0)
                return true;
            if (timeout) {
                record_error(error_status, layout::kCQEPollTimeout);
                throw EPExceptionWithLineInfo("URMA CQ timeout", "current tail=" + std::to_string(current_tail) +
                    ", expected tail=" + std::to_string(target));
            }
            return false;
        });
    }

    // Preserve the four-WQEBB boundary before handing this queue to an AICore kernel.
    template <int64_t kNumTimeoutNs>
    EP_INLINE void align_sq() {
        const auto num_padding = (-sq_head) & (kSQEBBAlignment - 1);
        if (num_padding == 0)
            return;
        wait_for_room<kNumTimeoutNs>();
        for (uint32_t i = 0; i < num_padding; ++ i) {
            const auto idx = sq_head + i;
            const auto owner = static_cast<uint64_t>((idx & kSQDepth) == 0) << 31;
            const auto flags = i + 1 == num_padding ? (2ULL << 16) | (1ULL << 21) : 0;
            auto* dst = reinterpret_cast<uint64_t*>(sq_base + (idx & (kSQDepth - 1)) * kNumWQEBBBytes);
            *reinterpret_cast<volatile uint64_t*>(dst) = kNopSQEHeaderTemplate | flags | owner |
                (idx & (kSQDepth - 1));
        }
        sq_head += num_padding;
        ++ expected_tail;
        ring_doorbell();
    }

    template <int64_t kNumTimeoutNs>
    EP_INLINE void drain() { poll_cq<kNumTimeoutNs>(expected_tail); }
};

}  // namespace aicpu
#endif

}  // namespace deep_ep::handle
