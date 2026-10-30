// Host-only policy checks. No Torch, CANN or NPU dependencies.
// c++ -std=c++17 -I. tests/ep/low_latency_policy_test.cpp -o /tmp/ep-policy
#include <cassert>
#include <csrc/kernels/ep/low_latency_policy.hpp>

using deep_ep::prefer_low_latency_transport;

static_assert(prefer_low_latency_transport(3510, 128, 1, 1, 7168, 8, 128, 64, false));
static_assert(prefer_low_latency_transport(3510, 128, 32, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 33, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 0, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 15, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 257, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 256, 7168, 8, 128, 64, false, 190 * 1024 - 1));
static_assert(prefer_low_latency_transport(3510, 128, 16, 256, 7168, 8, 128, 64, false, 190 * 1024));
static_assert(!prefer_low_latency_transport(3500, 128, 16, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 256, 4096, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 256, 7168, 6, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 256, 7168, 8, 256, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 256, 7168, 8, 128, 32, false));
static_assert(!prefer_low_latency_transport(3510, 128, 16, 256, 7168, 8, 128, 64, true));

int main() {
    const int worlds[] = {1, 2, 4, 8, 16, 32, 56, 64, 128, 256};
    const int capacities[] = {1, 2, 16, 32, 33, 64, 256, 257, 4096};
    const int large_batches[] = {512, 1024, 2048, 4096};
    for (const int world : worlds) {
        for (const int capacity : capacities) {
            for (int bs = 0; bs <= 257; ++bs) {
                const bool expected = world == 128 and bs >= 1 and bs <= 32 and
                                      bs <= capacity and capacity <= 256;
                assert(prefer_low_latency_transport(3510, world, bs, capacity, 7168, 8, 128, 64, false)
                       == expected);
            }
        }
        for (const int bs : large_batches)
            assert(!prefer_low_latency_transport(3510, world, bs, bs, 7168, 8, 128, 64, false));
    }
    // Different local B values may choose different transports, so protocol
    // interoperability is a separate multi-rank correctness requirement.
    assert(prefer_low_latency_transport(3510, 128, 2, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(3510, 128, 128, 256, 7168, 8, 128, 64, false));
    return 0;
}
