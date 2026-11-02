#pragma once

#include <map>
#include <mutex>

#include "../utils/event.hpp"

namespace deep_ep::comm {

static c10_npu::NPUStream get_comm_stream() {
    static std::mutex mutex;
    static std::map<c10::DeviceIndex, c10_npu::NPUStream> streams;
    const auto device = c10_npu::getCurrentNPUStream().device_index();
    const std::lock_guard lock(mutex);
    auto stream = streams.find(device);
    if (stream == streams.end())
        stream = streams.emplace(device, c10_npu::getStreamFromPool(true, device)).first;
    return stream->second;
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

}  // namespace deep_ep::comm
