#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>

#ifdef EP_AICPU
#include <ctime>
#include <type_traits>

namespace deep_ep::aicpu {

// Order memory accesses and data cache maintenance, including remote UBMEM and MMIO.
EP_INLINE void fence() {
    asm volatile("dmb sy" ::: "memory");
}

// Publish AICPU writes to local GM.
// Use only local, cacheable memory, never remote UBMEM or MMIO.
template <bool kDoFenceAfter = true>
EP_INLINE void publish_gm(const void* ptr, const uint64_t num_bytes) {
    if (num_bytes == 0)
        return;
    const auto begin = reinterpret_cast<uintptr_t>(ptr), end = begin + num_bytes;
    for (auto address = begin & ~(kNumAicpuCacheLineBytes - 1); address < end; address += kNumAicpuCacheLineBytes)
        asm volatile("dc cvac, %0" :: "r"(address) : "memory");
    if constexpr (kDoFenceAfter)
        fence();
}

// Publish one object to local GM.
// Use only local, cacheable memory, never remote UBMEM or MMIO.
template <bool kDoFenceAfter = true, typename dtype_t>
EP_INLINE void publish_gm(const dtype_t* ptr) {
    if constexpr (alignof(dtype_t) >= kNumAicpuCacheLineBytes or sizeof(dtype_t) == alignof(dtype_t)) {
        const auto begin = reinterpret_cast<uintptr_t>(ptr);
        #pragma unroll
        for (uint64_t offset = 0; offset < sizeof(dtype_t); offset += kNumAicpuCacheLineBytes)
            asm volatile("dc cvac, %0" :: "r"(begin + offset) : "memory");
        if constexpr (kDoFenceAfter)
            fence();
    } else {
        publish_gm<kDoFenceAfter>(ptr, sizeof(dtype_t));
    }
}

// Refresh cached local GM after a device/peer may have written it.
// DC IVAC is illegal in AICPU user mode, so we use CIVAC, which also writes back dirty lines.
// Callers must keep the entire affected cache lines clean before and during device/peer writes.
// Otherwise, the writeback may overwrite the device/peer's updates.
template <bool kDoFenceAfter = true>
EP_INLINE void refresh_gm(const void* ptr, const uint64_t num_bytes) {
    if (num_bytes == 0)
        return;
    const auto begin = reinterpret_cast<uintptr_t>(ptr), end = begin + num_bytes;
    for (auto address = begin & ~(kNumAicpuCacheLineBytes - 1); address < end; address += kNumAicpuCacheLineBytes)
        asm volatile("dc civac, %0" :: "r"(address) : "memory");
    if constexpr (kDoFenceAfter)
        fence();
}

// Refresh one object after a device/peer may have written it.
// Callers must keep the object clean before and during device/peer writes.
template <bool kDoFenceAfter = true, typename dtype_t>
EP_INLINE void refresh_gm(const dtype_t* ptr) {
    if constexpr (alignof(dtype_t) >= kNumAicpuCacheLineBytes or sizeof(dtype_t) == alignof(dtype_t)) {
        const auto begin = reinterpret_cast<uintptr_t>(ptr);
        #pragma unroll
        for (uint64_t offset = 0; offset < sizeof(dtype_t); offset += kNumAicpuCacheLineBytes)
            asm volatile("dc civac, %0" :: "r"(begin + offset) : "memory");
        if constexpr (kDoFenceAfter)
            fence();
    } else {
        refresh_gm<kDoFenceAfter>(ptr, sizeof(dtype_t));
    }
}

// Refresh and load multiple objects in a batch to amortize fence overhead.
template <typename RefreshFunc, typename LoadFunc>
EP_INLINE auto refresh_and_load_gm(const RefreshFunc& refresh_func, const LoadFunc& load_func) {
    refresh_func();
    fence();
    if constexpr (std::is_void_v<decltype(load_func())>) {
        load_func();
        fence();
    } else {
        auto value = load_func();
        fence();
        return value;
    }
}

// Refresh and load an object in local, cacheable GM.
template <typename dtype_t>
EP_INLINE dtype_t refresh_and_load_gm(const dtype_t* ptr) {
    return refresh_and_load_gm([&] { refresh_gm<false>(ptr); },
                               [&] { return *reinterpret_cast<const volatile dtype_t*>(ptr); });
}

// Store and publish multiple objects in a batch to amortize fence overhead.
template <typename StoreFunc, typename PublishFunc>
EP_INLINE void store_and_publish_gm(const StoreFunc& store_func, const PublishFunc& publish_func) {
    fence();
    store_func();
    publish_func();
    fence();
}

// Store and publish an object in local, cacheable GM.
template <typename dtype_t>
EP_INLINE void store_and_publish_gm(dtype_t* ptr, const dtype_t value) {
    store_and_publish_gm([&] { *reinterpret_cast<volatile dtype_t*>(ptr) = value; },
                         [&] { publish_gm<false>(ptr); });
}

// Remote UBMEM mappings are accessed directly (non-cacheable).
template <typename dtype_t>
EP_INLINE dtype_t load_ubmem(const dtype_t* ptr) {
    const auto value = *reinterpret_cast<const volatile dtype_t*>(ptr);
    fence();
    return value;
}

template <typename dtype_t>
EP_INLINE void store_ubmem(dtype_t* ptr, const dtype_t value) {
    fence();
    *reinterpret_cast<volatile dtype_t*>(ptr) = value;
    fence();
}

// Monotonic wall time, including time descheduled by Linux (in nanoseconds)
EP_INLINE uint64_t get_time_ns() {
    timespec ts{};
    EP_DEVICE_ASSERT(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

// Similar to comm::scalar::timeout_while, but the timeout including descheduled time
template <int64_t kNumTimeoutNs, typename Func>
EP_INLINE void timeout_while(const Func& func, uint64_t start_time_ns = 0) {
    EP_STATIC_ASSERT(kNumTimeoutNs > 0);
    if (start_time_ns == 0)
        start_time_ns = get_time_ns();
    while (true) {
        const bool timeout = get_time_ns() - start_time_ns >= static_cast<uint64_t>(kNumTimeoutNs);
        if (func(timeout))
            return;
        if (timeout)
            throw EPExceptionWithLineInfo("Timeout", "AICPU polling timed out");
    }
}

}  // namespace deep_ep::aicpu
#endif
