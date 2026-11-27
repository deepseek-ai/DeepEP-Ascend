#pragma once

#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

#include "../utils/event.hpp"

namespace deep_ep::comm {

static c10_npu::NPUStream get_comm_stream() {
    static const auto comm_stream = c10_npu::getStreamFromPool(true, -1);
    return comm_stream;
}

static c10_npu::NPUEvent create_event(const c10_npu::NPUStream& stream) {
    auto event = c10_npu::NPUEvent();
    event.record(stream);
    return event;
}

static void stream_wait(const c10_npu::NPUStream& stream, const c10_npu::NPUStream& event_stream) {
    if (stream != event_stream) {
        auto event = create_event(event_stream);
        event.block(stream);
    }
}

static void stream_wait(const c10_npu::NPUStream& stream, const EventHandle& event) {
    event.event->block(stream);
}

static bool is_stream_capturing(const c10_npu::NPUStream& stream) {
    return c10_npu::isStreamCapturingMayInitCtx(stream.stream());
}

}  // namespace deep_ep::comm
