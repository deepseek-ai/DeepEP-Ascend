#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/common/context.hpp>
#include <deep_ep/common/exception.hpp>

#include <kernel_operator.h>
#include <simt_api/device_atomic_functions.h>

namespace deep_ep {

template <int kNumRanks, int kNumScaleupRanks, int kNumVecCores, int kNumScaleupJetties, int kNumScaleoutJetties,
          int64_t kNumTimeoutCycles, bool kSequential, bool kWaitCompletion = false>
__global__ __vector__ void barrier_impl(__gm__ void* barrier_signals, __gm__ void* local_base,
                                        __gm__ void* device_context_ptr, const int rank_idx) {
    AscendC::InitSocState();

    auto& signals = *static_cast<__gm__ layout::BarrierSignals*>(barrier_signals);
    auto* device_context = static_cast<__gm__ layout::DeviceContext*>(device_context_ptr);
    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    comm::scalar::barrier<kNumRanks, kNumScaleupRanks, kNumTimeoutCycles, true, false, false,
                          kNumVecCores, kNumScaleupJetties, kNumScaleoutJetties, kSequential, kWaitCompletion>(
        signals, *device_context, vec_core_idx, rank_idx, local_base);
}

}  // namespace deep_ep
