#pragma once

#include <cstdint>

#include <acl/acl.h>

#include <deep_ep/common/exception.hpp>

namespace deep_ep::symmetric {

class SymmetricMemory {
public:
    void* ptr = nullptr;
    int64_t num_bytes;

    SymmetricMemory(const int64_t& num_bytes, const bool& use_huge1g_page): num_bytes(num_bytes) {
        EP_HOST_ASSERT(num_bytes > 0);
        ACL_CHECK(aclrtMalloc(&ptr, num_bytes,
                              use_huge1g_page ? ACL_MEM_MALLOC_HUGE1G_ONLY : ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(ptr, num_bytes, 0, num_bytes));
    }

    ~SymmetricMemory() noexcept(false) {
        free();
    }

    void free() {
        if (ptr != nullptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr = nullptr;
        }
    }

    SymmetricMemory(const SymmetricMemory&) = delete;
    SymmetricMemory& operator=(const SymmetricMemory&) = delete;
};

}  // namespace deep_ep::symmetric
