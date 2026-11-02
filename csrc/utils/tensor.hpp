#pragma once

// Load CANN HCCL before torch_npu's bundled HCCL headers.
#include <hccl/hccl.h>

#include <torch/python.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#define TORCH_DEVICE_NPU c10::DeviceType::PrivateUse1
