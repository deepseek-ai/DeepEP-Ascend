#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/error.hpp>
#include <deep_ep/layout/common/context.hpp>
#include <deep_ep/layout/common/error.hpp>
#include <deep_ep/common/exception.hpp>

#ifdef EP_AICPU
#include <deep_ep/common/aicpu.hpp>
#include <deep_ep/comm/handle.hpp>
#else
#include <deep_ep/common/simt.hpp>
#ifdef __CCE__
#include <deep_ep/comm/handle.hpp>

#include <kernel_operator.h>
#include <simt_api/asc_simt.h>
#include <c_api/sync/sync.h>
#include <c_api/asc_simd.h>
#include <utils/debug/asc_time.h>
#endif
#endif

namespace deep_ep::comm {

// TODO: no hardcode, obtain by runtime
static constexpr int64_t kNumSystemCyclesPerSecond = 1'000'000'000;
static constexpr int64_t kNumSimtCyclesPerSecond = 1'650'000'000;

#ifdef EP_AICORE
namespace simt {

template <int64_t kNumTimeoutCycles, uint32_t kNumPollPeriodCycles = 300, typename Func>
__simt_callee__ __forceinline__ void timeout_while(const Func& func, uint64_t start_clock = 0,
    __gm__ layout::ErrorStatusSlot* error_status = nullptr) {
    using namespace __cce_simt;
    static constexpr int64_t kNumTimeoutSimtCycles =
        kNumTimeoutCycles / kNumSystemCyclesPerSecond * kNumSimtCyclesPerSecond;
    if (start_clock == 0)
        start_clock = clock64();

    while (true) {
        const bool timeout = clock64() - start_clock >= kNumTimeoutSimtCycles;
        if (func(timeout))
            break;

        // NOTES: NPU will break down if too many requests fall at the same address
        deep_ep::simt::nop<kNumPollPeriodCycles>();
        if (timeout) {
            start_clock = clock64();
            while (clock64() - start_clock < kNumSimtCyclesPerSecond) {}
            record_error_and_trap_simt(error_status, layout::kBarrierTimeout);
        }
    }
}

template <int kNumScaleupRanks>
__simt_vf__ __aicore__ __launch_bounds__(kNumMaxScaleupRanks) void remote_signal(
    __gm__ layout::BarrierSignals* signals, __gm__ layout::DeviceContext* device_context,
    const int phase, const int delta, const int rank_idx) {
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    if (const auto thread_idx = static_cast<int>(threadIdx.x); thread_idx < kNumScaleupRanks) {
        const auto& base_signals = *reinterpret_cast<__gm__ const layout::BarrierSignals*>(
            device_context->scaleup_peer_ptrs[rank_idx % kNumScaleupRanks]);
        const auto remote_signal = layout::get_sym_ptr(base_signals, *device_context,
                                                       &signals->scaleup_barrier_signals[phase],
                                                       thread_idx);
        asc_atomic_add(remote_signal, delta);
    }
}

template <int kNumVecCores>
__simt_callee__ void grid_sync(
    __gm__ uint32_t* counter_ptr, const int vec_core_idx) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    EP_STATIC_ASSERT(kNumVecCores > 0);

    asc_syncthreads();
    if (threadIdx.x == 0) {
        const auto old_value = asc_atomic_add(
            counter_ptr, vec_core_idx == 0 ? kFinishSumTag - (kNumVecCores - 1) : 1u);
        while (true) {
            const auto new_value = *reinterpret_cast<volatile __gm__ int*>(counter_ptr);
            if (((new_value ^ old_value) & kFinishSumTag) != 0)
                break;

            // NOTES: NPU will break down if too many requests fall at the same address
            // TODO: figure out a better number (equals to the HBM latency)
            deep_ep::simt::nop<300>();
        }
    }
    asc_syncthreads();
}

}  // namespace simt

namespace scalar {

template <int64_t kNumTimeoutCycles, uint32_t kNumPollPeriodCycles = 300, typename Func>
__aicore__ __forceinline__ void timeout_while(const Func& func, uint64_t start_clock = 0,
    __gm__ layout::ErrorStatusSlot* error_status = nullptr) {
    if (start_clock == 0)
        start_clock = static_cast<uint64_t>(asc_get_system_cycle());

    while (true) {
        const bool timeout = static_cast<uint64_t>(asc_get_system_cycle()) - start_clock >= kNumTimeoutCycles;
        if (func(timeout))
            break;

        #pragma unroll 1
        for (int i = 0; i < kNumPollPeriodCycles; ++ i) {
            // AscendC::Nop has a pipe_barrier(PIPE_ALL), which is not necessary here and may cause performance degradation.
            asm volatile("nop");
        }
        if (timeout) {
            start_clock = static_cast<uint64_t>(asc_get_system_cycle());
            while (static_cast<uint64_t>(asc_get_system_cycle()) - start_clock < kNumSystemCyclesPerSecond) {}
            record_error_and_trap(error_status, layout::kBarrierTimeout);
        }
    }
}

template <int kNumScaleupRanks, int64_t kNumTimeoutCycles>
__aicore__ __forceinline__ void scaleup_barrier_wo_local_sync(
    __gm__ layout::BarrierSignals& signals, __gm__ layout::DeviceContext& device_context,
    const int vec_core_idx, const int rank_idx) {
    const auto error_status = layout::DeviceContext::get_error_status(device_context);
    __gm__ int64_t* counter_ptr = &signals.scaleup_barrier_counter;

    // Get status (must bypass DCache because counter is written by previous barrier invocations)
    const auto counter = AscendC::ReadGmByPassDCache(counter_ptr);
    const auto status = static_cast<int>(counter & 3);
    const auto phase = status & 1, sign = status >> 1;
    const auto delta = sign ? -1 : 1;
    const auto target = sign ? 0 : kNumScaleupRanks;

    // Increment local counter
    AscendC::WriteGmByPassDCache(counter_ptr, counter + 1);

    // Remote atomic add: parallel SIMT write to each peer's signal[phase] via UBMEM.
    __gm__ int* signal_ptr = &signals.scaleup_barrier_signals[phase];
    asc_vf_call<simt::remote_signal<kNumScaleupRanks>>(
        dim3(kNumScaleupRanks, 1, 1),
        &signals, &device_context, phase, delta, rank_idx
    );

    // Spin-wait until local signal reaches target
    // TODO(HUAWEI): why even with `volatile`, the compiler cannot generate bypass automatically?
    timeout_while<kNumTimeoutCycles>([&](const bool& timeout) __aicore__ {
        const auto signal = AscendC::ReadGmByPassDCache(signal_ptr);
        if (signal == target)
            return true;

        if (timeout) {
            AscendC::printf(
                "DeepEP barrier timeout, core: %d, status: %d, signal: %d, phase: %d, target: %d, counter: %ld\n",
                vec_core_idx, status, signal, phase, target, counter);
        }
        return false;
    }, 0, error_status);
}

// AIVs split peer notifications across their own jetties; each worker waits for its own peers' slots.
template <int kNumRanks, int kNumScaleupRanks, int64_t kNumTimeoutCycles,
          bool kWaitCompletion = false>
__aicore__ __forceinline__ void scaleout_barrier_wo_local_sync(
    __gm__ layout::BarrierSignals& signals, __gm__ void* local_base,
    __gm__ layout::DeviceContext& device_context,
    const int rank_idx, const int64_t epoch,
    const int num_scaleout_vec_cores, const int scaleout_vec_core_idx) {
    const auto error_status = layout::DeviceContext::get_error_status(device_context);
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumScaleoutRanks > 1);
    const auto phase = static_cast<int>((epoch - 1) & 1);
    const auto scaleout_rank_idx = rank_idx / kNumScaleupRanks;
    const auto num_peers = kNumScaleoutRanks - 1;
    if (scaleout_vec_core_idx >= num_scaleout_vec_cores or scaleout_vec_core_idx >= num_peers)
        return;

    alignas(8) __ubuf__ uint8_t ub_jetty_info[sizeof(handle::HcommJettyInfo)];
    alignas(8) __ubuf__ uint8_t ub_peer_info[sizeof(handle::HcommPeerInfo)];
    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];
    handle::HcommJetty jetty(reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(ub_jetty_info), ub_sqe,
                             device_context.scaleout_jetties, scaleout_vec_core_idx, error_status);

    for (int peer_idx = scaleout_vec_core_idx; peer_idx < num_peers; peer_idx += num_scaleout_vec_cores) {
        // The channel table keeps non-self peers in order; every peer owns one phase slot.
        const handle::HcommPeer peer(reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(ub_peer_info),
                                     device_context.scaleout_jetties, peer_idx);
        jetty.template post_write<kNumTimeoutCycles, false, false, false, false, true, int64_t>(
            peer, local_base, &signals.scaleout_barrier_slots[phase][scaleout_rank_idx].value,
            nullptr, sizeof(epoch), nullptr, 0, kWaitCompletion, epoch);
    }
    jetty.ring_doorbell();

    // Before destroy, wait until this AIV's own writes have completed with CQEs.
    if constexpr (kWaitCompletion)
        jetty.template drain<kNumTimeoutCycles>();

    // Each worker waits only for the peers it wrote to, reading their phase slots.
    for (int peer_idx = scaleout_vec_core_idx; peer_idx < num_peers; peer_idx += num_scaleout_vec_cores) {
        const auto peer_scaleout_rank_idx = peer_idx + (peer_idx >= scaleout_rank_idx ? 1 : 0);
        auto* const local_value = &signals.scaleout_barrier_slots[phase][peer_scaleout_rank_idx].value;
        timeout_while<kNumTimeoutCycles, 16>([&](const bool& timeout) __aicore__ {
            const auto value = AscendC::ReadGmByPassDCache(local_value);
            if (value == epoch)
                return true;
            if (timeout) {
                AscendC::printf(
                    "DeepEP scaleout slot barrier timeout, phase: %d, peer: %d, value: %ld, epoch: %ld\n",
                    phase, peer_scaleout_rank_idx, value, epoch);
            }
            return false;
        }, 0, error_status);
    }
}

template <int kNumRanks, int kNumScaleupRanks, int64_t kNumTimeoutCycles,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true,
          int kNumVecCores = 1, int kNumScaleupJetties = 0, int kNumScaleoutJetties = 0,
          bool kSequential = true, bool kWaitCompletion = false>
__aicore__ __forceinline__ void barrier(
    __gm__ layout::BarrierSignals& signals, __gm__ layout::DeviceContext& device_context,
    const int vec_core_idx, const int rank_idx, __gm__ void* local_base = nullptr) {
    const auto error_status = layout::DeviceContext::get_error_status(device_context);
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks % kNumScaleupRanks == 0);
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumScaleoutRanks > 0 and kNumScaleoutRanks <= kNumMaxScaleoutRanks);
    constexpr bool kParallel = not kSequential and kNumScaleoutRanks > 1 and kNumScaleupRanks > 1;
    EP_STATIC_ASSERT(not kParallel or kNumVecCores >= 2);
    constexpr bool kDrainJetties = kFlushStores and kNumRanks > 1 and
        (kNumScaleupJetties > 0 or kNumScaleoutJetties > 0);

    // Wait for MTE3 stores before launching the SIMT signals
    if constexpr (kFlushStores) {
        asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    if constexpr (kSyncAtStart)
        AscendC::SyncAll<true>();

    if constexpr (kDrainJetties) {
        // Distribute scalar jetty drains across all AIVs.
        if constexpr (kNumScaleupRanks > 1) {
            for (int jetty_idx = vec_core_idx; jetty_idx < kNumScaleupJetties; jetty_idx += kNumVecCores) {
                handle::HcommJetty scaleup_jetty(
                    reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(0),
                    nullptr,
                    device_context.scaleup_jetties, jetty_idx, error_status);
                scaleup_jetty.template drain<kNumTimeoutCycles>();
            }
        }
        if constexpr (kNumScaleoutRanks > 1) {
            for (int jetty_idx = vec_core_idx; jetty_idx < kNumScaleoutJetties; jetty_idx += kNumVecCores) {
                handle::HcommJetty scaleout_jetty(
                    reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(0),
                    nullptr,
                    device_context.scaleout_jetties, jetty_idx, error_status);
                scaleout_jetty.template drain<kNumTimeoutCycles>();
            }
        }
    }

    int64_t epoch = 0;
    if constexpr (kNumScaleoutRanks > 1) {
        EP_STATIC_ASSERT(kNumScaleoutJetties > 0, "scale-out barrier needs at least one Jetty");
        // All AIVs read the shared round before one AIV writes it.
        const auto previous = AscendC::ReadGmByPassDCache(&signals.scaleout_barrier_counter);
        EP_DEVICE_ASSERT(previous < INT64_MAX);
        epoch = previous + 1;
    }

    // Reuse the drain sync as the read-before-write sync for the shared scale-out round.
    if constexpr (kDrainJetties or kNumScaleoutRanks > 1)
        AscendC::SyncAll<true>();

    if constexpr (kNumScaleoutRanks > 1) {
        if (vec_core_idx == 0)
            AscendC::WriteGmByPassDCache(&signals.scaleout_barrier_counter, epoch);

        // Sequential: every AIV owns a share of the scale-out queue. Parallel: AIV0 keeps scale-up and
        // the remaining AIVs own the scale-out queues, starting at Jetty index 0.
        constexpr int kScaleoutVecCoreStartIdx = kParallel ? 1 : 0;
        constexpr int kNumAvailableScaleoutVecCores = kNumVecCores - kScaleoutVecCoreStartIdx;
        constexpr int kNumScaleoutVecCores = math::min(kNumAvailableScaleoutVecCores, kNumScaleoutJetties);
        const auto scaleout_vec_core_idx = vec_core_idx - kScaleoutVecCoreStartIdx;

        if constexpr (kParallel) {
            if (vec_core_idx == 0) {
                scaleup_barrier_wo_local_sync<kNumScaleupRanks, kNumTimeoutCycles>(
                    signals, device_context, vec_core_idx, rank_idx);
            } else {
                scaleout_barrier_wo_local_sync<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles,
                    kWaitCompletion>(
                    signals, local_base, device_context, rank_idx, epoch, kNumScaleoutVecCores,
                    scaleout_vec_core_idx);
            }
        } else {
            scaleout_barrier_wo_local_sync<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles,
                kWaitCompletion>(
                signals, local_base, device_context, rank_idx, epoch, kNumScaleoutVecCores,
                scaleout_vec_core_idx);
            if constexpr (kNumScaleupRanks > 1) {
                AscendC::SyncAll<true>();
                if (vec_core_idx == 0)
                    scaleup_barrier_wo_local_sync<kNumScaleupRanks, kNumTimeoutCycles>(
                        signals, device_context, vec_core_idx, rank_idx);
            }
        }
    } else if constexpr (kNumScaleupRanks > 1) {
        if (vec_core_idx == 0)
            scaleup_barrier_wo_local_sync<kNumScaleupRanks, kNumTimeoutCycles>(
                signals, device_context, vec_core_idx, rank_idx);
    }

    if constexpr (kSyncAtEnd)
        AscendC::SyncAll<true>();
}

}  // namespace scalar
#endif

#ifdef EP_AICPU
namespace aicpu {

namespace memory = ::deep_ep::aicpu;

template <int kNumSlots, int64_t kNumTimeoutNs, typename Slot>
EP_INLINE void wait_barrier_slots(const Slot* slots, const int self, const int64_t epoch, const int rank_idx,
                                  void* error_status, const char* domain) {
    EP_STATIC_ASSERT(kNumSlots > 0);
    int peer = 0;
    memory::timeout_while<kNumTimeoutNs>([&](const bool timeout) {
        for (int slot_idx = peer; slot_idx < kNumSlots; ++ slot_idx)
            if (slot_idx != self)
                memory::refresh_gm<false>(&slots[slot_idx].value);
        memory::fence();
        for (; peer < kNumSlots; ++ peer) {
            if (peer == self)
                continue;
            const auto value = *reinterpret_cast<const volatile int64_t*>(&slots[peer].value);
            if (value != epoch) {
                if (timeout) {
                    record_error(error_status, layout::kBarrierTimeout);
                    throw EPExceptionWithLineInfo("AICPU barrier timeout", "domain=" + std::string(domain) +
                        ", rank=" + std::to_string(rank_idx) + ", slot=" + std::to_string(peer) + ", value=" + std::to_string(value) +
                        ", epoch=" + std::to_string(epoch));
                }
                return false;
            }
        }
        memory::fence();
        return true;
    });
}

template <int kNumScaleupRanks>
EP_INLINE void notify_scaleup_barrier(layout::BarrierSignals& signals, const layout::DeviceContext& device_context,
                                      const int rank_idx, const int64_t epoch) {
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    const auto phase = static_cast<int>((epoch - 1) & 1);
    const auto domain_start = layout::get_scaleup_domain_start_rank<kNumScaleupRanks>(rank_idx);
    const auto scaleup_rank_idx = rank_idx - domain_start;
    const auto* my_slot = &signals.scaleup_barrier_slots[phase][scaleup_rank_idx].value;
    const auto offset = reinterpret_cast<uintptr_t>(my_slot) - reinterpret_cast<uintptr_t>(&signals);

    // Batch independent UBMEM notifications between fences
    memory::fence();
    for (int peer_rank_idx = 0; peer_rank_idx < kNumScaleupRanks; ++ peer_rank_idx) {
        if (peer_rank_idx == scaleup_rank_idx)
            continue;
        auto* const remote_slot = reinterpret_cast<int64_t*>(
            device_context.scaleup_peer_ptrs[peer_rank_idx] + offset);
        *reinterpret_cast<volatile int64_t*>(remote_slot) = epoch;
    }
    memory::fence();
}

template <int kNumRanks, int kNumScaleupRanks, int64_t kNumTimeoutNs, bool kWaitCompletion>
EP_INLINE void notify_scaleout_barrier(layout::BarrierSignals& signals, void* local_base,
                                       const layout::DeviceContext& device_context, const int rank_idx,
                                       const int64_t epoch, handle::aicpu::HcommJetty& jetty) {
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    const auto phase = static_cast<int>((epoch - 1) & 1);
    const auto scaleout_rank_idx = rank_idx / kNumScaleupRanks;
    for (int peer_idx = 0; peer_idx < kNumScaleoutRanks - 1; ++ peer_idx) {
        const handle::aicpu::HcommPeer peer(device_context.scaleout_jetties, peer_idx);
        // Independent peer slots need no strong placement ordering.
        jetty.template put_value<kNumTimeoutNs, false, false>(
            peer, local_base, &signals.scaleout_barrier_slots[phase][scaleout_rank_idx].value, epoch, kWaitCompletion);
    }
    jetty.ring_doorbell();
}

// Share the scale-out round and URMA slots with AICore. Scale-up has separate AICPU receive slots.
template <int kNumRanks, int kNumScaleupRanks, int64_t kNumTimeoutNs,
          bool kFlushStores = true, int kNumScaleupJetties = 0, int kNumScaleoutJetties = 0,
          bool kSequential = true, bool kWaitCompletion = false>
EP_INLINE void barrier(layout::BarrierSignals& signals, const layout::DeviceContext& device_context,
                       const int rank_idx, void* local_base = nullptr) {
    EP_STATIC_ASSERT(kNumScaleupRanks > 0 and kNumScaleupRanks <= kNumMaxScaleupRanks);
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumRanks % kNumScaleupRanks == 0);
    EP_STATIC_ASSERT(kNumScaleupJetties >= 0 and kNumScaleoutJetties >= 0);
    constexpr int kNumScaleoutRanks = kNumRanks / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumScaleoutRanks > 0 and kNumScaleoutRanks <= kNumMaxScaleoutRanks);
    constexpr bool kParallel = not kSequential and kNumScaleoutRanks > 1 and kNumScaleupRanks > 1;
    if constexpr (kNumRanks == 1)
        return;

    auto* const error_status = reinterpret_cast<void*>(device_context.error_status_addr);

    if constexpr (kFlushStores) {
        if constexpr (kNumScaleupRanks > 1) {
            for (int jetty_idx = 0; jetty_idx < kNumScaleupJetties; ++ jetty_idx) {
                handle::aicpu::HcommJetty jetty(device_context.scaleup_jetties, jetty_idx, error_status);
                jetty.template drain<kNumTimeoutNs>();
            }
        }
        if constexpr (kNumScaleoutRanks > 1) {
            for (int jetty_idx = 0; jetty_idx < kNumScaleoutJetties; ++ jetty_idx) {
                handle::aicpu::HcommJetty jetty(device_context.scaleout_jetties, jetty_idx, error_status);
                jetty.template drain<kNumTimeoutNs>();
            }
        }
    }

    const auto previous = memory::refresh_and_load_gm(&signals.scaleout_barrier_counter);
    EP_DEVICE_ASSERT(previous < INT64_MAX);
    const auto phase = static_cast<int>(previous & 1);
    const auto epoch = previous + 1;
    memory::store_and_publish_gm(&signals.scaleout_barrier_counter, epoch);

    if constexpr (kNumScaleoutRanks > 1) {
        EP_STATIC_ASSERT(kNumScaleoutJetties > 0, "scale-out barrier needs at least one Jetty");
        EP_DEVICE_ASSERT(local_base != nullptr);
        handle::aicpu::HcommJetty jetty(device_context.scaleout_jetties, 0, error_status);
        notify_scaleout_barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutNs, kWaitCompletion>(
            signals, local_base, device_context, rank_idx, epoch, jetty);
        if constexpr (kParallel)
            notify_scaleup_barrier<kNumScaleupRanks>(signals, device_context, rank_idx, epoch);
        wait_barrier_slots<kNumScaleoutRanks, kNumTimeoutNs>(
            signals.scaleout_barrier_slots[phase], rank_idx / kNumScaleupRanks, epoch, rank_idx, error_status, "scale-out");
        if constexpr (kWaitCompletion)
            jetty.template drain<kNumTimeoutNs>();
    }
    if constexpr (kNumScaleupRanks > 1) {
        if constexpr (not kParallel)
            notify_scaleup_barrier<kNumScaleupRanks>(signals, device_context, rank_idx, epoch);
        wait_barrier_slots<kNumScaleupRanks, kNumTimeoutNs>(
            signals.scaleup_barrier_slots[phase], rank_idx % kNumScaleupRanks, epoch, rank_idx, error_status, "scale-up");
    }
}

}  // namespace aicpu
#endif

}  // namespace deep_ep::comm
