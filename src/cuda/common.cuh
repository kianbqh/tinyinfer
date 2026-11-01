#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define IK_CUDA_CHECK(expr)                                                            \
  do {                                                                                 \
    cudaError_t _e = (expr);                                                           \
    if (_e != cudaSuccess) {                                                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, \
                   __LINE__);                                                          \
      std::abort();                                                                    \
    }                                                                                  \
  } while (0)

namespace ik::cuda::detail {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

}  // namespace ik::cuda::detail
