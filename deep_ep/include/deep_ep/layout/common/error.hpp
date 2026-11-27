#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>

namespace deep_ep::layout {

// Recoverable errors carry this bit; unrecoverable ones do not. `kNone` is checked on its own,
// because a zero is neither.
static constexpr uint32_t kRecoverableMask = 1u << 31;

enum ErrorStatus : uint32_t {
    kNone                = 0,

    kCQEPollTimeout      = kRecoverableMask | 1,
    kCQEError            = kRecoverableMask | 2,
    kBarrierTimeout      = kRecoverableMask | 3,

    kInvalidCommMetadata = 4,
    kSQCapacityExceeded  = 5,
    kSQAlignmentError    = 6,
};
EP_STATIC_ASSERT(static_cast<uint32_t>(kNone) == 0);

// Stored and passed around as a plain code; the enum above names its values. A typedef instead of
// the enum type keeps the value convertible through pybind11, which only maps enums to int when
// they are registered as a Python enum class.
using ErrorCode = uint32_t;

// Host-mapped status of the last error, kept within one cache line.
struct alignas(64) ErrorStatusSlot {
    ErrorCode code;
};
EP_STATIC_ASSERT(sizeof(ErrorStatusSlot) <= 64);

}  // namespace deep_ep::layout
