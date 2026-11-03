cmake_minimum_required(VERSION 3.24)
project(tinyinfer_libfuzzer LANGUAGES CXX)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "LLVM Clang with compiler-rt/libFuzzer is required; no automatic install")
endif()
get_filename_component(TINYINFER_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_library(ti_fuzz_production STATIC
  "${TINYINFER_ROOT}/backend/host_codec.cpp"
  "${TINYINFER_ROOT}/kernels/src/cpu/matmul.cpp"
  "${TINYINFER_ROOT}/kernels/src/cpu/softmax.cpp"
  "${TINYINFER_ROOT}/kernels/src/cpu/layernorm.cpp")
target_include_directories(ti_fuzz_production PUBLIC "${TINYINFER_ROOT}/backend/include" "${TINYINFER_ROOT}/kernels/include")
target_compile_features(ti_fuzz_production PUBLIC cxx_std_17)
target_compile_options(ti_fuzz_production PRIVATE -O1 -g -fno-omit-frame-pointer -fno-fast-math
  -fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined -Wno-unknown-pragmas)
foreach(target IN ITEMS json artifact host_validation cpu_ops codec)
  add_executable("fuzz_${target}" "harnesses/${target}.cpp")
  target_link_libraries("fuzz_${target}" PRIVATE ti_fuzz_production)
  target_include_directories("fuzz_${target}" PRIVATE "${TINYINFER_ROOT}/llm/src")
  target_compile_options("fuzz_${target}" PRIVATE -O1 -g -fno-omit-frame-pointer -fno-fast-math
    -fsanitize=fuzzer-no-link,address,undefined -fno-sanitize-recover=undefined)
  target_link_options("fuzz_${target}" PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined)
endforeach()
