#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/common/error.hpp>

#ifdef EP_AICORE
#include <kernel_operator.h>
#include <simt_api/asc_simt.h>
#include <simt_api/device_functions.h>
#include <simt_api/device_sync_functions.h>
#include <utils/debug/asc_assert.h>

namespace deep_ep {

// Record an error into host-mapped memory, then trap.
__aicore__ __forceinline__ void record_error_and_trap(
    __gm__ layout::ErrorStatusSlot* host_status, const layout::ErrorStatus status) {
    if (host_status != nullptr) {
        AscendC::WriteGmByPassDCache(
            reinterpret_cast<__gm__ uint32_t*>(&host_status->code), static_cast<uint32_t>(status));
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    trap();
}

// SIMT variant.
__simt_callee__ __forceinline__ void record_error_and_trap_simt(
    __gm__ layout::ErrorStatusSlot* host_status, const layout::ErrorStatus status) {
    if (host_status != nullptr) {
        asc_stcg(
            reinterpret_cast<__gm__ uint32_t*>(&host_status->code), static_cast<uint32_t>(status));
        asc_threadfence();
    }
    __asc_simt_vf::__trap();
}

}  // namespace deep_ep
#endif

#ifdef EP_AICPU

#include <deep_ep/common/aicpu.hpp>

namespace deep_ep {

EP_INLINE void record_error(void* host_status, const layout::ErrorStatus status) {
    if (host_status == nullptr)
        return;
    *static_cast<volatile uint32_t*>(host_status) = static_cast<uint32_t>(status);
    aicpu::fence();
}

}  // namespace deep_ep
#endif
