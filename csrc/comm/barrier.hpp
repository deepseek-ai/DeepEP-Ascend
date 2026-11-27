#pragma once

#include "stream.hpp"
#include "../kernels/comm/barrier.hpp"
#include "../kernels/comm/api.hpp"

namespace deep_ep::comm {

static void barrier(const Context& context, const bool& with_cpu_sync = false,
                    const bool& sequential = true, const bool& wait_comm_stream = false,
                    const bool& wait_completion = false) {
    context.check_available();
    const auto stream = c10_npu::getCurrentNPUStream();

    // Sync
    const bool wait_comm = wait_comm_stream and
        not (is_stream_capturing(stream) and not is_stream_capturing(get_comm_stream()));
    if (with_cpu_sync) {
        c10_npu::npuSynchronizeDevice();
    } else if (wait_comm) {
        stream_wait(stream, get_comm_stream());
    }

    // Launch
    EP_HOST_ASSERT(context.barrier_signals != nullptr);
    launch_barrier(context.barrier_signals, context.hcomm_context->buffer,
                   context.hcomm_context->device_context,
                   context.rank_idx, context.num_ranks, context.hcomm_context->num_scaleup_ranks,
                   context.hcomm_context->num_scaleup_jetties, context.hcomm_context->num_scaleout_jetties,
                   sequential, context.num_gpu_timeout_cycles, wait_completion, stream);

    // Sync
    if (with_cpu_sync) {
        c10_npu::npuSynchronizeDevice();
    } else if (wait_comm) {
        stream_wait(get_comm_stream(), stream);
        // Rejoin comm_stream into the capture stream so ACLGraph capture does not leave it dangling.
        if (is_stream_capturing(stream))
            stream_wait(stream, get_comm_stream());
    }
}

}  // namespace deep_ep::comm
