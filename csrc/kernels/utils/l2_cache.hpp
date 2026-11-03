#pragma once

#include "../../runtime/jit.hpp"

namespace deep_ep {

static constexpr int kL2CacheCleanUbBytes = 64 * 1024;

static void launch_l2_cache_clean(void* buffer, const int64_t& num_bytes,
                                  const c10_npu::NPUStream& stream) {
    const auto kernel = jit->compile("l2_cache_clean", R"(
#include <deep_ep/impls/utils/l2_cache.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&l2_cache_clean_impl);
}
)");

    jit->launch(
        kernel,
        {
            .stream = stream.stream(),
            .num_blocks = runtime->get_num_vec_cores(),
            .num_ubuf_bytes = kL2CacheCleanUbBytes,
        },
        buffer, num_bytes);
}

}  // namespace deep_ep
