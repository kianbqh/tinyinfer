#include "common.cuh"
#include "ik/cuda_kernels.hpp"

namespace ik::cuda {
namespace {

__global__ void k_layernorm(const float* __restrict__ x, const float* __restrict__ g,
                            const float* __restrict__ b, float* __restrict__ y, int cols,
                            float eps) {
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  // Subtract a nearby value before reducing squared deviations. This keeps
  // variance accurate for rows with a large mean and a small spread.
  const float shift = xr[0];
  float s = 0.f, ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) { const float v = xr[c] - shift; s += v; ss += v * v; }
  s = detail::warp_sum(s);
  ss = detail::warp_sum(ss);
  __shared__ float sh_s[32], sh_ss[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if (lane == 0) { sh_s[wid] = s; sh_ss[wid] = ss; }
  __syncthreads();
  if (wid == 0) {
    s = lane < nw ? sh_s[lane] : 0.f;
    ss = lane < nw ? sh_ss[lane] : 0.f;
    s = detail::warp_sum(s);
    ss = detail::warp_sum(ss);
    if (lane == 0) { sh_s[0] = s; sh_ss[0] = ss; }
  }
  __syncthreads();
  const float mean_d = sh_s[0] / cols;
  const float mean = shift + mean_d;
  const float var = fmaxf(sh_ss[0] / cols - mean_d * mean_d, 0.f);
  const float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = (xr[c] - mean) * rstd * g[c] + b[c];
}

}  // namespace

void layernorm_fused(const float* dx, const float* dg, const float* db, float* dy, int rows,
                     int cols, float eps) {
  k_layernorm<<<rows, 256>>>(dx, dg, db, dy, cols, eps);
  IK_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ik::cuda
