option(TINYC_V2_CUDA "Build additive device-pointer v2 CUDA ABI" OFF)
set(TINYC_V2_GEMM_CONFIG "-1" CACHE STRING "v2 GEMM allow-list ID (-1=all, 0..11=single candidate)")
if(TINYC_V2_CUDA)
  if(NOT TINYC_CUDA)
    message(FATAL_ERROR "TINYC_V2_CUDA requires TINYC_CUDA=ON")
  endif()
  add_library(tinyinfer_v2 OBJECT
    backend/runtime_v2.cu
    kernels/src/cuda/tc/gemm.cu
    kernels/src/cuda/tc/attention.cu
    kernels/src/cuda/tc/elementwise.cu)
  target_include_directories(tinyinfer_v2 PRIVATE backend/include kernels/src/cuda)
  if(NOT TINYC_V2_GEMM_CONFIG MATCHES "^(-1|[0-9]|10|11)$")
    message(FATAL_ERROR "TINYC_V2_GEMM_CONFIG must be -1 or an allow-listed ID0..11")
  endif()
  target_compile_definitions(tinyinfer_v2 PRIVATE TI_V2_ONLY_GEMM_CONFIG=${TINYC_V2_GEMM_CONFIG})
  target_link_libraries(tinyinfer_v2 PRIVATE CUDA::cudart)
  set_target_properties(tinyinfer_v2 PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 86)
  # Keep the v2 object's numerical flags independent of the original kernels.
  target_compile_options(tinyinfer_v2 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-allow-unsupported-compiler>")
  target_sources(tinyinfer_backend PRIVATE $<TARGET_OBJECTS:tinyinfer_v2>)
  add_executable(profile_v2 bench/v2/kernels/profile.cu)
  target_link_libraries(profile_v2 PRIVATE tinyinfer_backend CUDA::cudart)
  set_target_properties(profile_v2 PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 86)
  target_compile_options(profile_v2 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-allow-unsupported-compiler>")
endif()
