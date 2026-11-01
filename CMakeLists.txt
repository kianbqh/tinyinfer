cmake_minimum_required(VERSION 3.24)
project(inference_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(IK_ENABLE_CUDA "Build CUDA kernels (auto-off if no nvcc)" ON)
option(IK_BUILD_PYTHON "Build pybind11 module" ON)
option(IK_NATIVE "Compile CPU code with -march=native" ON)
set(IK_CUDA_ARCH "86" CACHE STRING "CUDA arch (86 = RTX 30xx / Ampere)")

find_package(OpenMP REQUIRED)

# ---------------- CPU library ----------------
add_library(ik_cpu STATIC src/cpu/matmul.cpp src/cpu/softmax.cpp src/cpu/layernorm.cpp)
target_include_directories(ik_cpu PUBLIC include)
target_link_libraries(ik_cpu PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(ik_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(ik_cpu PRIVATE /O2 /fp:fast /arch:AVX2)
else()
  target_compile_options(ik_cpu PRIVATE -O3 -Wall -Wextra -ffp-contract=fast -fno-math-errno)
  if(IK_NATIVE)
    target_compile_options(ik_cpu PRIVATE -march=native)
  endif()
endif()

# ---------------- CUDA library (optional) ----------------
set(IK_HAVE_CUDA OFF)
if(IK_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(IK_HAVE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES ${IK_CUDA_ARCH})
    add_library(ik_cuda STATIC src/cuda/matmul.cu src/cuda/softmax.cu src/cuda/layernorm.cu src/cuda/bench_api.cu)
    target_include_directories(ik_cuda PUBLIC include)
    target_link_libraries(ik_cuda PUBLIC CUDA::cudart)
    target_compile_options(ik_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>")
    set_target_properties(ik_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(ik_cuda PUBLIC IK_HAVE_CUDA=1)
  else()
    message(WARNING "nvcc not found: building CPU-only")
  endif()
endif()
message(STATUS "IK_HAVE_CUDA=${IK_HAVE_CUDA}")

# ---------------- Tests ----------------
enable_testing()
# GoogleTest is built from source with the same compiler (avoids ABI mismatches with system builds).
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
                     URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
                     DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(test_cpu tests/test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE ik_cpu GTest::gtest_main)
gtest_discover_tests(test_cpu)

add_executable(bench_cpu bench/bench_cpu.cpp)
target_link_libraries(bench_cpu PRIVATE ik_cpu)

if(IK_HAVE_CUDA)
  add_executable(test_cuda tests/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE ik_cuda ik_cpu GTest::gtest_main)
  gtest_discover_tests(test_cuda DISCOVERY_MODE PRE_TEST)
  add_executable(bench_cuda bench/bench_cuda.cpp)
  target_link_libraries(bench_cuda PRIVATE ik_cuda)
endif()

# ---------------- Python bindings ----------------
if(IK_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  find_package(pybind11 CONFIG REQUIRED HINTS ${PYBIND11_CMAKE_DIR})
  pybind11_add_module(ikernels python/bindings.cpp)
  target_link_libraries(ikernels PRIVATE ik_cpu)
  if(IK_HAVE_CUDA)
    target_link_libraries(ikernels PRIVATE ik_cuda)
  endif()
endif()
