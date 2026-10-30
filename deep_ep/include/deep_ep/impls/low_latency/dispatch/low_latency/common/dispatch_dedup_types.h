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
#ifndef DISPATCH_DEDUP_TYPES_H
#define DISPATCH_DEDUP_TYPES_H

namespace DispatchDedup {
constexpr int DISPATCH_DEDUP_SUCCESS = 0;
constexpr int DISPATCH_DEDUP_ERROR_PARA_CHECK_FAIL = -3;
constexpr int DISPATCH_DEDUP_FLAG_BUFF_BYTES = 4 * 1024 * 1024;

// Preserve the source kernel's dtype codes used in device tiling.
enum DispatchDedupDataType {
    DISPATCH_DEDUP_DATA_TYPE_FP16 = 3,
    DISPATCH_DEDUP_DATA_TYPE_BFP16 = 11,
    DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 = 15,
    DISPATCH_DEDUP_DATA_TYPE_FP8E5M2 = 16,
    DISPATCH_DEDUP_DATA_TYPE_RESERVED = 255
};

}  // namespace DispatchDedup
#endif  // DISPATCH_DEDUP_TYPES_H
