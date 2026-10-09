# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
#
# ExecuTorch on the Seeed XIAO ESP32S3 (Sense) under Zephyr.
#
# The model is embedded as a const array and only the operators it uses are
# linked (selective build), the same approach as the upstream
# zephyr/samples/hello-executorch sample but without any Arm-specific pieces,
# since the ESP32-S3 is an Xtensa LX7 and runs the portable kernels.

cmake_minimum_required(VERSION 3.24)

set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)

set(ET_PTE_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/model/mnist_tiny.pte"
    CACHE FILEPATH "Path to the ExecuTorch .pte model to embed"
)
if(NOT IS_ABSOLUTE "${ET_PTE_FILE_PATH}")
  get_filename_component(
    ET_PTE_FILE_PATH "${ET_PTE_FILE_PATH}" ABSOLUTE BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(FATAL_ERROR "Model not found at ET_PTE_FILE_PATH=${ET_PTE_FILE_PATH}. "
                      "Run train/train_mnist.py first or pass -DET_PTE_FILE_PATH=<file>.")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_xiao_esp32s3)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -ffunction-sections -fdata-sections"
)

if(NOT DEFINED EXECUTORCH_DIR)
  if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
    set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
  else()
    message(FATAL_ERROR "ExecuTorch Zephyr module not found. Check west.yml and run west update.")
  endif()
endif()
message(STATUS "EXECUTORCH_DIR=${EXECUTORCH_DIR}")

include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
# arm_runner_create_selected_ops_lib() is a thin, target-agnostic wrapper over
# ExecuTorch's codegen helpers; it works unchanged for Xtensa.
include(${EXECUTORCH_DIR}/backends/arm/cmake/ArmRunnerUtils.cmake)

# Portable kernels target must exist even though the full portable ops
# library is disabled in prj.conf (CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n).
# On ExecuTorch v1.5.1 the KERNELS_ONLY flag is not yet honoured, so the
# all-ops registration library is still compiled; it is never linked (check
# zephyr.map: only the op_*.cpp.obj for ops in the .pte appear).
if(NOT TARGET portable_kernels)
  set(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY ON)
  add_subdirectory(
    ${EXECUTORCH_DIR}/kernels/portable
    ${CMAKE_CURRENT_BINARY_DIR}/executorch/kernels/portable
  )
  unset(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY)
endif()

# Generate a kernel registration library containing only the ops in the .pte.
arm_runner_create_selected_ops_lib(
  LIB_NAME "xiao_portable_ops_lib"
  FUNCTIONS_YAML "${EXECUTORCH_DIR}/kernels/portable/functions.yaml"
  OPS_FROM_MODEL "${ET_PTE_FILE_PATH}"
  KERNEL_LIBS portable_kernels
  DEPS executorch
  DTYPE_SELECTIVE_BUILD ""
)

# flatcc builds a host tool as part of the ExecuTorch tree; give it a writable
# copy inside the build directory (same workaround as the upstream sample).
set(_local_flatcc_root ${CMAKE_BINARY_DIR}/flatcc_src)
if(NOT EXISTS ${_local_flatcc_root}/CMakeLists.txt)
  file(MAKE_DIRECTORY ${_local_flatcc_root})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${EXECUTORCH_DIR}/third-party/flatcc ${_local_flatcc_root}
  )
endif()
set(EXECUTORCH_FLATCC_SOURCE_ROOT ${_local_flatcc_root} CACHE PATH "" FORCE)
set(EXECUTORCH_FLATCC_INSTALL_ROOT ${_local_flatcc_root} CACHE PATH "" FORCE)

# Embed the model.
set(_model_pte_header ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h)
add_custom_command(
  OUTPUT ${_model_pte_header}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
          --pte ${ET_PTE_FILE_PATH} --out ${_model_pte_header}
  DEPENDS ${ET_PTE_FILE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
  COMMENT "Converting ${ET_PTE_FILE_PATH} to model_pte.h"
)
add_custom_target(gen_model_header DEPENDS ${_model_pte_header})
add_dependencies(app gen_model_header)

target_sources(app PRIVATE src/main.cpp)
target_sources_ifdef(CONFIG_APP_CAMERA app PRIVATE src/camera.c)

target_compile_definitions(
  app PRIVATE
  APP_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE}
  APP_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE}
)

target_link_libraries(app PRIVATE libexecutorch xiao_portable_ops_lib)
target_include_directories(app PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
