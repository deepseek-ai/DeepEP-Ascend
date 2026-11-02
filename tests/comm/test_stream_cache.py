"""Exercise the native stream cache without requiring a CANN installation."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class StreamCacheTest(unittest.TestCase):
    def test_streams_follow_the_current_device(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('A C++20 compiler is required')
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            headers = {
                'hccl/hccl.h': '#pragma once\n',
                'torch/python.h': '#pragma once\n',
                'torch_npu/csrc/aten/common/from_blob.h': '#pragma once\n',
                'torch_npu/csrc/core/npu/NPUStream.h': r'''
#pragma once
#include <array>
namespace c10 { using DeviceIndex = int; }
namespace c10_npu {
inline thread_local int current_device = 0;
inline std::array<int, 2> allocations{};
struct NPUStream {
    int device, id;
    int device_index() const { return device; }
    bool operator==(const NPUStream&) const = default;
};
inline NPUStream getCurrentNPUStream() { return {current_device, 0}; }
inline NPUStream getStreamFromPool(bool, int device) {
    if (device == -1) device = current_device;
    return {device, ++allocations.at(device)};
}
}
''',
                'torch_npu/csrc/core/npu/NPUEvent.h': r'''
#pragma once
#include "NPUStream.h"
namespace c10_npu {
struct NPUEvent {
    void record(const NPUStream&) {}
    void block(const NPUStream&) const {}
};
}
''',
            }
            for name, content in headers.items():
                target = path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            source = path / 'test.cpp'
            source.write_text(r'''
#include <iostream>
#include <thread>
#include "csrc/comm/stream.hpp"
int main() {
    const auto first = deep_ep::comm::get_comm_stream();
    if (first.device_index() != 0) return 1;
    c10_npu::current_device = 1;
    const auto second = deep_ep::comm::get_comm_stream();
    if (second.device_index() != 1) {
        std::cerr << "current device is 1, but the communication stream belongs to "
                  << second.device_index() << '\n';
        return 2;
    }
    c10_npu::current_device = 0;
    if (deep_ep::comm::get_comm_stream() != first) return 3;
    bool same = false;
    std::thread worker([&] {
        c10_npu::current_device = 1;
        same = deep_ep::comm::get_comm_stream() == second;
    });
    worker.join();
    if (!same || c10_npu::allocations != std::array<int, 2>{1, 1}) return 4;
}
''')
            executable = path / 'test'
            subprocess.run([compiler, '-std=c++20', '-pthread', '-I', str(path), '-I', str(ROOT),
                            str(source), '-o', str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    unittest.main()
