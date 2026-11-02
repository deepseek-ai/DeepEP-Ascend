"""Exercise HCCLContext's remote-memory selection with host API doubles."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class RemoteMemoryTest(unittest.TestCase):
    def test_workspace_is_selected_by_its_registration_tag(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('A C++20 compiler is required')
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            # The real HCCLContext header is compiled below. Only the external
            # runtime APIs and unrelated device helpers are replaced.
            headers = [
                'acl/acl.h', 'hccl/hccl.h', 'hccl/hccl_comm.h',
                'hccl/hccl_res.h', 'hccl/hccl_rank_graph.h', 'hcomm/hcomm_res.h',
                'torch_npu/csrc/core/npu/NPUStream.h',
                'deep_ep/common/math.hpp', 'deep_ep/common/exception.hpp',
            ]
            for name in headers:
                target = path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('#pragma once\n#include "runtime.hpp"\n')
            (path / 'runtime.hpp').write_text(r'''
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
using HcclComm = void*;
using HcclMemHandle = void*;
using HcommMemHandle = void*;
using EndpointHandle = uint64_t;
using ChannelHandle = uint64_t;
using HcommChannelConfig = void*;
constexpr int HCCL_SUCCESS = 0, HCCL_E_AGAIN = 1;
constexpr int COMM_MEM_TYPE_DEVICE = 1;
constexpr int COMM_PROTOCOL_UB_MEM = 1, COMM_PROTOCOL_UBC_CTP = 2;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0, ACL_MEM_MALLOC_HUGE1G_ONLY = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0, ACL_MEMCPY_DEVICE_TO_HOST = 0;
constexpr int HCCL_DATA_TYPE_UINT16 = 0;
constexpr int HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0;
constexpr int HCOMM_SOCKET_ROLE_SERVER = 0, HCOMM_SOCKET_ROLE_CLIENT = 1;
constexpr int HCOMM_CHANNEL_NAME_MAX_LEN = 255;
enum class CommEngine { COMM_ENGINE_AIV };
struct CommMem { int type; void* addr; uint64_t size; };
struct Endpoint {
    int protocol = COMM_PROTOCOL_UBC_CTP;
    struct { int type = 0; char raws[8]{}; } commAddr;
    struct { int locType = 0; char raws[8]{}; } loc;
};
struct CommLink {
    struct { int linkProtocol; } linkAttr;
    Endpoint srcEndpointDesc, dstEndpointDesc;
};
struct HcclChannelDesc {
    uint32_t remoteRank;
    int channelProtocol, notifyNum, memHandleNum;
    Endpoint localEndpoint, remoteEndpoint;
    HcclMemHandle* memHandles;
};
struct HcommChannelDesc {
    Endpoint remoteEndpoint;
    int notifyNum, memHandleNum, role;
    HcommMemHandle* memHandles;
    uint16_t port;
    const char* channelName;
};
#define EP_HOST_ASSERT(cond) do { if (!(cond)) throw std::runtime_error(#cond); } while (0)
#define ACL_CHECK(cmd) EP_HOST_ASSERT((cmd) == 0)
#define HCCL_CHECK(cmd) EP_HOST_ASSERT((cmd) == 0)
#define HCOMM_CHECK(cmd) EP_HOST_ASSERT((cmd) == 0)
namespace deep_ep::math {
template <typename T> T max(T a, T b) { return std::max(a, b); }
template <typename T> T* advance_ptr(void* ptr, int64_t bytes) {
    return reinterpret_cast<T*>(static_cast<char*>(ptr) + bytes);
}
}
namespace c10_npu { inline void* getCurrentNPUStream() { return nullptr; } }
inline int aclrtMalloc(void** ptr, uint64_t size, int) { *ptr = std::malloc(size); return *ptr ? 0 : 1; }
inline int aclrtFree(void* ptr) { std::free(ptr); return 0; }
inline int aclrtMemset(void* ptr, uint64_t, int value, uint64_t size) { std::memset(ptr, value, size); return 0; }
inline int aclrtMemcpy(void* dst, uint64_t, const void* src, uint64_t size, int) { std::memcpy(dst, src, size); return 0; }
inline int aclrtSynchronizeStreamWithTimeout(void*, int) { return 0; }
namespace fixture {
inline std::string registered_tag;
inline CommMem memories[3];
inline char* tags[3];
}
inline int HcclCommGetHandleWithName(const char*, HcclComm* comm) { *comm = nullptr; return 0; }
inline int HcclCommMemReg(HcclComm, const char* tag, CommMem*, HcclMemHandle*) {
    fixture::registered_tag = tag; return 0;
}
inline int HcclRankGraphGetLayers(HcclComm, uint32_t** layers, uint32_t* count) {
    static uint32_t values[]{0, 1}; *layers = values; *count = 2; return 0;
}
inline int HcclRankGraphGetLinks(HcclComm, uint32_t, int, int, CommLink** links, uint32_t* count) {
    static CommLink values[]{{{COMM_PROTOCOL_UB_MEM}, {}, {}}, {{COMM_PROTOCOL_UBC_CTP}, {}, {}}};
    *links = values; *count = 2; return 0;
}
inline void HcclChannelDescInit(HcclChannelDesc* desc, int) { *desc = {}; }
inline int HcclChannelAcquire(HcclComm, CommEngine, HcclChannelDesc*, uint32_t, ChannelHandle*) { return 0; }
inline int HcclChannelGetRemoteMems(HcclComm, ChannelHandle, uint32_t* count, CommMem** memories, char*** tags) {
    *count = 3; *memories = fixture::memories; *tags = fixture::tags; return 0;
}
inline int HcommEndpointCreate(const Endpoint*, EndpointHandle* endpoint) { *endpoint = 1; return 0; }
inline int HcommMemReg(EndpointHandle, const char*, CommMem*, HcommMemHandle*) { return 0; }
inline int HcommEndpointGetListenPort(EndpointHandle, uint32_t* port) { *port = 1234; return 0; }
inline int HcommChannelConfigCreate(HcommChannelConfig* config) { *config = nullptr; return 0; }
inline int HcommChannelConfigSetInt(HcommChannelConfig, int, int) { return 0; }
inline int HcommChannelDescInit(HcommChannelDesc* desc, int) { *desc = {}; return 0; }
inline int HcommChannelCreateWithConfig(EndpointHandle, CommEngine, HcommChannelDesc*, int, HcommChannelConfig, ChannelHandle*) { return 0; }
inline int HcommChannelConfigDestroy(HcommChannelConfig) { return 0; }
inline int HcommChannelGetStatus(const ChannelHandle*, uint32_t count, int32_t* states) { std::fill_n(states, count, 0); return 0; }
inline int HcommChannelDestroy(ChannelHandle*, uint32_t) { return 0; }
inline int HcommMemUnreg(EndpointHandle, HcommMemHandle) { return 0; }
inline int HcommEndpointDestroy(EndpointHandle) { return 0; }
inline int HcclAllGather(const void* src, void* dst, int, int, HcclComm, void*) {
    std::fill_n(static_cast<uint16_t*>(dst), 2, *static_cast<const uint16_t*>(src)); return 0;
}
''')
            source = path / 'test.cpp'
            source.write_text(r'''
#include <iostream>
#include "csrc/kernels/comm/hccl.hpp"
int main(int argc, char** argv) {
    const std::string scenario = argv[1];
    char regions[3][64]{};
    std::string names[]{"default HcclBuffer", "unrelated buffer", "other buffer"};
    const int target = scenario == "first" ? 0 : scenario == "last" ? 2 : 1;
    if (scenario != "missing") names[target] = "deep_ep/group/0";
    for (int i = 0; i < 3; ++i) {
        fixture::memories[i] = {COMM_MEM_TYPE_DEVICE, regions[i], 64};
        fixture::tags[i] = names[i].data();
    }
    if (scenario == "wrong-size") fixture::memories[target].size = 32;
    try {
        deep_ep::HCCLContext context("group", 0, 2, 64, 2, 1, false);
        if (fixture::registered_tag != "deep_ep/group/0") return 1;
        if (scenario == "missing" || scenario == "wrong-size") {
            std::cerr << "accepted an absent or mismatched workspace: " << scenario << '\n';
            return 2;
        }
        if (context.peer_ptrs[1] != reinterpret_cast<int64_t>(regions[target])) {
            std::cerr << "selected another buffer when matching workspace is " << scenario << '\n';
            return 3;
        }
    } catch (const std::runtime_error& error) {
        if (scenario != "missing" && scenario != "wrong-size") {
            std::cerr << error.what() << '\n';
            return 4;
        }
    }
}
''')
            executable = path / 'test'
            subprocess.run([compiler, '-std=c++20', '-I', str(path), '-I', str(ROOT),
                            str(source), '-o', str(executable)], check=True)
            for scenario in ('first', 'middle', 'last', 'missing', 'wrong-size'):
                with self.subTest(scenario=scenario):
                    result = subprocess.run([str(executable), scenario], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
