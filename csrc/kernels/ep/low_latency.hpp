#pragma once

#include <deep_ep/impls/low_latency/dispatch/low_latency/launch.hpp>
#include <deep_ep/impls/low_latency/combine/low_latency/launch.hpp>
#include "../../runtime/jit.hpp"

namespace deep_ep {

inline void launch_dispatch_low_latency(const deepep::EpServerDedupKernelArgs& a) {
    const auto kernel = jit->compile("dispatch_low_latency", R"(
#include <deep_ep/impls/low_latency/dispatch.hpp>
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&dispatch_server_dedup_kernel<>);
}
)");
    jit->launch(kernel, {.num_blocks = 64, .num_ubuf_bytes = 0},
                a.commArgs, a.x, a.expertIds, a.expertScales, a.expandXOut, a.dynamicScalesOut,
                a.expertTokenNumsOut, a.sendCountsOut, a.tokenTypeOut, a.destinationIndexOut,
                a.relayReadIndexOut, a.sourceMaskOut, a.expertScalesOut, a.tilingData,
                a.reservedMagic, a.inputScales);
}

inline void launch_combine_low_latency(const deepep::EpServerDedupCombineKernelArgs& a) {
    const auto kernel = jit->compile("combine_low_latency", R"(
#include <deep_ep/impls/low_latency/combine.hpp>
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&combine_server_dedup_kernel<>);
}
)");
    jit->launch(kernel, {.num_blocks = 64, .num_ubuf_bytes = 0},
                a.commArgs, a.expertOut, a.destinationIndex, a.sendCounts,
                a.tokenType, a.relayReadIndex, a.sourceMask, a.yOut, a.tilingData);
}

}  // namespace deep_ep
