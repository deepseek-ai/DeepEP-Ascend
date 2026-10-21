# NOTES: this CMake is only for debugging/IDE indexing; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_ep LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

# CXX configs
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_STANDARD 20)

# Dependency
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# NOTES: CLion can not index linked directories
# CANN lays out arch-specific headers under aarch64-linux/ or x86_64-linux/
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(ASCEND_ARCH_SUBDIR aarch64-linux)
else()
    set(ASCEND_ARCH_SUBDIR x86_64-linux)
endif()
include_directories(
    csrc deep_ep/include
    third-party/clangd-ascend/include
    third-party/deep_jit/include
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/include
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/include/ascendc/host_apiascendc/include/highlevel_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/impl/adv_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/impl/basic_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/impl/c_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/impl/micro_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/impl/simt_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/impl/utils
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/adv_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/adv_api/matmul
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/aicpu_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/basic_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/c_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/interface
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/micro_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/simt_api
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/tiling
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_SUBDIR}/asc/include/utils
    ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
    ${TORCH_NPU_INSTALL_PREFIX}/include ${TORCH_NPU_INSTALL_PREFIX}/include/third_party/acl/inc
)
link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${TORCH_INSTALL_PREFIX}/lib
    ${TORCH_NPU_INSTALL_PREFIX}/lib
)

pybind11_add_module(_C csrc/python_api.cpp)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python ascendcl hccl hcomm torch_npu)
target_link_options(_C PRIVATE "-Wl,-Bsymbolic-functions")

# For CLion indexing only (not compiled in production)
add_executable(indexing EXCLUDE_FROM_ALL csrc/indexing/main.cpp)
target_include_directories(indexing PRIVATE
    ${ASCEND_HOME_PATH}/tools/bisheng_compiler/lib/clang/15.0.5/include)
