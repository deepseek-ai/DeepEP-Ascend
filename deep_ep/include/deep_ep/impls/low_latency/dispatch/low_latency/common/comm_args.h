// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_DEDUP_COMM_ARGS_H
#define DISPATCH_DEDUP_COMM_ARGS_H
#include <cstdint>
#include <cstddef>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE 1
#else
#define DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE 0
#endif

#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE
#include <kernel_operator.h>
#endif

#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

#if DISPATCH_DEDUP_ASCENDC_AICORE_COMPILE && !defined(FORCE_INLINE_AICORE)
#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#endif

namespace DispatchDedup {
constexpr int DISPATCH_DEDUP_MAX_RANK_SIZE = 256;

// Private host/device ABI for the UBMEM low-latency kernels. Both sides are
// built from this header; no source-repository runtime ABI is required.
struct CommArgs {
    int32_t rank = 0;
    int32_t rankSize = 0;
    GM_ADDR peerMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};
};

static_assert(offsetof(CommArgs, rankSize) == 4, "rank count ABI offset");
static_assert(offsetof(CommArgs, peerMems) == 8, "peer address ABI offset");
static_assert(sizeof(CommArgs) == 2056, "communication ABI size");

}  // namespace DispatchDedup
#endif  // DISPATCH_DEDUP_COMM_ARGS_H
