#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/common/signals.hpp>
#include <deep_ep/common/simt.hpp>
#include <deep_ep/common/exception.hpp>

#ifdef __CCE__
#include <deep_ep/comm/handle.hpp>

#include <kernel_operator.h>
#include <simt_api/asc_simt.h>
#include <c_api/sync/sync.h>
#include <c_api/asc_simd.h>
#include <utils/debug/asc_time.h>
#endif

namespace deep_ep::comm {

// TODO: no hardcode, obtain by runtime
static constexpr int64_t kNumSystemCyclesPerSecond = 1'000'000'000;
static constexpr int64_t kNumSimtCyclesPerSecond = 1'650'000'000;

#ifdef __CCE__
namespace simt {

template <int64_t kNumTimeoutCycles, uint32_t kNumPollPeriodCycles = 300, typename Func>
__simt_callee__ __forceinline__ void timeout_while(const Func& func, uint64_t start_clock = 0) {
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
            __asc_simt_vf::__trap();
        }
    }
}

template <int kNumRanks>
__simt_vf__ __aicore__ __launch_bounds__(kNumMaxRanks) void remote_signal(
    __gm__ layout::CommonSignals* signals, const int phase, const int delta) {
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRanks);
    if (const auto thread_idx = static_cast<int>(threadIdx.x); thread_idx < kNumRanks) {
        const auto remote_signal = layout::get_sym_ptr(*signals, &signals->barrier_signals[phase], thread_idx);
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
__aicore__ __forceinline__ void timeout_while(const Func& func, uint64_t start_clock = 0) {
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
            trap();
        }
    }
}

template <int kNumRanks, int64_t kNumTimeoutCycles,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true,
          int kNumVecCores = 1, int kNumJetties = 0>
__aicore__ __forceinline__ void barrier(
    __gm__ layout::CommonSignals& signals, const int vec_core_idx,
    __gm__ void* jetty_ptrs = nullptr) {
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRanks);

    // Wait for MTE3 stores before launching the SIMT signals
    if constexpr (kFlushStores) {
        asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    if constexpr (kSyncAtStart)
        AscendC::SyncAll<true>();

    if constexpr (kFlushStores and kNumRanks > 1 and kNumJetties > 0) {
        if (jetty_ptrs != nullptr) {
            // Distribute scalar jetty drains across all AIVs, then synchronize them.
            for (int jetty_idx = vec_core_idx; jetty_idx < kNumJetties; jetty_idx += kNumVecCores) {
                handle::HcommJetty jetty(
                    reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(0),
                    nullptr,
                    jetty_ptrs, jetty_idx);
                jetty.template drain<kNumTimeoutCycles>();
            }
            AscendC::SyncAll<true>();
        }
    }

    if (vec_core_idx == 0) {
        __gm__ int64_t* counter_ptr = &signals.barrier_counter;

        // Get status (must bypass DCache because counter is written by previous barrier invocations)
        const auto counter = AscendC::ReadGmByPassDCache(counter_ptr);
        const auto status = static_cast<int>(counter & 3);
        const auto phase = status & 1, sign = status >> 1;
        const auto delta = sign ? -1 : 1;
        const auto target = sign ? 0 : kNumRanks;

        // Increment local counter
        AscendC::WriteGmByPassDCache(counter_ptr, counter + 1);

        // Remote atomic add: parallel SIMT write to each peer's signal[phase] via UBMEM.
        __gm__ int* signal_ptr = &signals.barrier_signals[phase];
        asc_vf_call<simt::remote_signal<kNumRanks>>(
            dim3(kNumRanks, 1, 1),
            &signals, phase, delta
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
        });
    }

    if constexpr (kSyncAtEnd)
        AscendC::SyncAll<true>();
}

}  // namespace scalar
#endif

}  // namespace deep_ep::comm
