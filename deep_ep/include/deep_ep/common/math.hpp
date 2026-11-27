#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>

namespace deep_ep::math {

template <typename dtype_t>
constexpr EP_INLINE dtype_t min(dtype_t a, dtype_t b) {
    return a < b ? a : b;
}

template <typename dtype_t>
constexpr EP_INLINE dtype_t max(dtype_t a, dtype_t b) {
    return a > b ? a : b;
}

template <typename dtype_t>
constexpr EP_INLINE bool is_power_of_2(dtype_t value) {
    return value > 0 and (value & (value - 1)) == 0;
}

template <typename dtype_t>
constexpr EP_INLINE dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t, bool kDoCeilAlignment = true>
constexpr EP_INLINE dtype_t align(dtype_t a, dtype_t b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename dst_t, typename src_t>
constexpr EP_INLINE dst_t* advance_ptr(src_t* ptr, const int64_t num_bytes) {
    const auto addr = __builtin_bit_cast(uintptr_t, ptr) + num_bytes;
    return __builtin_bit_cast(dst_t*, addr);
}

}  // namespace deep_ep
