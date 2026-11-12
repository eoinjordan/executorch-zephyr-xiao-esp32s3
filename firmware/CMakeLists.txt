# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
#
# ExecuTorch on the Seeed XIAO ESP32S3 (Sense) under Zephyr.
#
# Two programs of the same network are embedded as const arrays and only the
# operators they use are linked (selective build):
#
#   model/mnist_tiny.pte           fp32, portable kernels
#   model/mnist_tiny_cmsis_nn.pte  int8, cortex_m operators on CMSIS-NN
#
# The ESP32-S3 is an Xtensa LX7, not an Arm core. CMSIS-NN has a plain C code
# path next to its DSP and Helium ones, and that path builds for any target,
# so the quantised kernels run here too. -DAPP_CMSIS_NN=OFF leaves them and
# the int8 program out.
#
# With CONFIG_APP_LEGO (lego.conf) the application is the brick detector
# instead, src/lego.cpp, and the two programs are those of BrickNet:
#
#   model/lego_bricknet.pte           fp32, portable kernels
#   model/lego_bricknet_cmsis_nn.pte  int8, cortex_m operators on CMSIS-NN

cmake_minimum_required(VERSION 3.24)

set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)

option(APP_CMSIS_NN "Also embed the int8 program and build the CMSIS-NN kernels" ON)

set(ET_PTE_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/model/mnist_tiny.pte"
    CACHE FILEPATH "Path to the fp32 ExecuTorch .pte model to embed"
)
set(ET_PTE_Q_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/model/mnist_tiny_cmsis_nn.pte"
    CACHE FILEPATH "Path to the int8 (cortex_m) ExecuTorch .pte model to embed"
)
set(LEGO_PTE_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/model/lego_bricknet.pte"
    CACHE FILEPATH "Path to the fp32 BrickNet .pte (CONFIG_APP_LEGO)"
)
set(LEGO_PTE_Q_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/model/lego_bricknet_cmsis_nn.pte"
    CACHE FILEPATH "Path to the int8 (cortex_m) BrickNet .pte (CONFIG_APP_LEGO)"
)
foreach(_var ET_PTE_FILE_PATH ET_PTE_Q_FILE_PATH LEGO_PTE_FILE_PATH LEGO_PTE_Q_FILE_PATH)
  if(NOT IS_ABSOLUTE "${${_var}}")
    get_filename_component(
      ${_var} "${${_var}}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
endforeach()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(FATAL_ERROR "Model not found at ET_PTE_FILE_PATH=${ET_PTE_FILE_PATH}. "
                      "Run train/train_mnist.py first or pass -DET_PTE_FILE_PATH=<file>.")
endif()
if(APP_CMSIS_NN AND NOT EXISTS "${ET_PTE_Q_FILE_PATH}")
  message(FATAL_ERROR "Model not found at ET_PTE_Q_FILE_PATH=${ET_PTE_Q_FILE_PATH}. "
                      "Run train/export_cmsis_nn.py or pass -DAPP_CMSIS_NN=OFF.")
endif()

# The ExecuTorch Zephyr module builds its Cortex-M backend (the cortex_m
# kernels and CMSIS-NN) only when CONFIG_CPU_CORTEX_M is set. Nothing in that
# backend needs an Arm core, so ask for it here, before the module is read.
if(APP_CMSIS_NN)
  set(EXECUTORCH_BUILD_CORTEX_M ON CACHE BOOL "" FORCE)
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_xiao_esp32s3)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The application mode is a Kconfig choice, known only from here on.
if(CONFIG_APP_LEGO)
  set(ET_PTE_FILE_PATH "${LEGO_PTE_FILE_PATH}")
  set(ET_PTE_Q_FILE_PATH "${LEGO_PTE_Q_FILE_PATH}")
  set(_lego_models "${ET_PTE_FILE_PATH}")
  if(APP_CMSIS_NN)
    list(APPEND _lego_models "${ET_PTE_Q_FILE_PATH}")
  endif()
  foreach(_pte ${_lego_models})
    if(NOT EXISTS "${_pte}")
      message(FATAL_ERROR "Model not found at ${_pte}. Run train/train_lego.py and "
                          "train/export_lego.py.")
    endif()
  endforeach()
endif()

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

set(_app_cmsis_nn OFF)
if(APP_CMSIS_NN)
  if(TARGET cortex_m_kernels)
    set(_app_cmsis_nn ON)
  else()
    message(FATAL_ERROR "APP_CMSIS_NN is ON but the ExecuTorch module did not build "
                        "cortex_m_kernels. Start from a clean build directory (west build -p).")
  endif()
endif()

# One registration library per kernel library, each with only the operators
# the embedded programs use. An operator may be registered once, so the
# portable library takes the union over both programs: the int8 program is
# all cortex_m operators except for a layout copy that the portable kernels
# provide.
set(_app_models "${ET_PTE_FILE_PATH}")
if(_app_cmsis_nn)
  list(APPEND _app_models "${ET_PTE_Q_FILE_PATH}")
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_ops.py
          --exclude cortex_m:: ${_app_models}
  OUTPUT_VARIABLE _app_portable_ops
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _app_pte_ops_result
)
if(NOT _app_pte_ops_result EQUAL 0 OR _app_portable_ops STREQUAL "")
  message(FATAL_ERROR "scripts/pte_ops.py could not list the operators of ${_app_models}. "
                      "It needs the executorch Python package (requirements.txt).")
endif()
message(STATUS "Portable operators: ${_app_portable_ops}")

set(_selected_ops_libs)
arm_runner_create_selected_ops_lib(
  LIB_NAME "xiao_portable_ops_lib"
  FUNCTIONS_YAML "${EXECUTORCH_DIR}/kernels/portable/functions.yaml"
  OP_LIST "${_app_portable_ops}"
  KERNEL_LIBS portable_kernels
  DEPS executorch
  DTYPE_SELECTIVE_BUILD ""
)
list(APPEND _selected_ops_libs xiao_portable_ops_lib)

if(_app_cmsis_nn)
  arm_runner_create_selected_ops_lib(
    LIB_NAME "xiao_cortex_m_ops_lib"
    CUSTOM_OPS_YAML "${EXECUTORCH_DIR}/backends/cortex_m/ops/operators.yaml"
    OPS_FROM_MODEL "${ET_PTE_Q_FILE_PATH}"
    KERNEL_LIBS cortex_m_kernels
    DEPS executorch
  )
  list(APPEND _selected_ops_libs xiao_cortex_m_ops_lib)
endif()

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

# Embed the models.
function(app_embed_pte pte header symbol)
  add_custom_command(
    OUTPUT ${header}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
            --pte ${pte} --out ${header} --symbol ${symbol}
    DEPENDS ${pte} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
    COMMENT "Converting ${pte} to ${header}"
  )
endfunction()

set(_model_headers ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h)
app_embed_pte(${ET_PTE_FILE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h model_pte)
if(_app_cmsis_nn)
  list(APPEND _model_headers ${CMAKE_CURRENT_BINARY_DIR}/model_q_pte.h)
  app_embed_pte(${ET_PTE_Q_FILE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/model_q_pte.h model_q_pte)
endif()
add_custom_target(gen_model_header DEPENDS ${_model_headers})
add_dependencies(app gen_model_header)

# The ExecuTorch module's runtime includes <zephyr/kernel.h>, which needs
# Zephyr's generated headers. Order it explicitly so a parallel build does not
# compile it before they exist.
if(TARGET zephyr_generated_headers)
  foreach(_t executorch_core executorch portable_kernels kernels_util_all_deps cortex_m_kernels)
    if(TARGET ${_t})
      add_dependencies(${_t} zephyr_generated_headers)
    endif()
  endforeach()
endif()

if(CONFIG_APP_LEGO)
  target_sources(app PRIVATE src/lego.cpp)
else()
  target_sources(app PRIVATE src/main.cpp)
endif()
target_sources_ifdef(CONFIG_APP_CAMERA app PRIVATE src/camera.c)
target_sources_ifdef(CONFIG_APP_SCREEN app PRIVATE src/screen.c)

target_compile_definitions(
  app PRIVATE
  APP_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE}
  APP_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE}
  APP_Q_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_APP_CMSIS_NN_METHOD_POOL_SIZE}
  APP_Q_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_APP_CMSIS_NN_TEMP_POOL_SIZE}
)
if(_app_cmsis_nn)
  target_compile_definitions(app PRIVATE APP_CMSIS_NN=1)
endif()

target_link_libraries(app PRIVATE libexecutorch ${_selected_ops_libs})
target_include_directories(app PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
