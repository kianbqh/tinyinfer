#include "common.cuh"
#include "ik/cuda_kernels.hpp"

namespace ik::cuda {
namespace {

// Merge two partial (max, sum-of-exp) states; empty states (sum == 0) are ignored.
__device__ __forceinline__ void merge(float& m, float& s, float om, float os) {
  if (os == 0.f) return;
  if (s == 0.f) { m = om; s = os; return; }
  const float nm = fmaxf(m, om);
  s = s * __expf(m - nm) + os * __expf(om - nm);
  m = nm;
}

// One block (256 threads) per row. Online softmax: each thread keeps (max, sum) and merges,
// so the input is read once for statistics and once for the output (no extra passes over HBM
// beyond what is needed, and no intermediate buffer).
__global__ void k_softmax(const float* __restrict__ x, float* __restrict__ y, int cols) {
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  float m = -INFINITY, s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = xr[c];
    if (v == -INFINITY) continue;  // Masked logits contribute zero probability.
    const float nm = fmaxf(m, v);
    s = s * __expf(m - nm) + __expf(v - nm);
    m = nm;
  }
  // Merge (m, s) across the warp.
  for (int o = 16; o > 0; o >>= 1) {
    const float om = __shfl_xor_sync(0xffffffff, m, o);
    const float os = __shfl_xor_sync(0xffffffff, s, o);
    merge(m, s, om, os);
  }
  __shared__ float sm[32], ss[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if (lane == 0) { sm[wid] = m; ss[wid] = s; }
  __syncthreads();
  if (wid == 0) {
    m = lane < nw ? sm[lane] : -INFINITY;
    s = lane < nw ? ss[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      const float om = __shfl_xor_sync(0xffffffff, m, o);
      const float os = __shfl_xor_sync(0xffffffff, s, o);
      merge(m, s, om, os);
    }
    if (lane == 0) { sm[0] = m; ss[0] = s; }
  }
  __syncthreads();
  const float gm = sm[0], inv = 1.f / ss[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = __expf(xr[c] - gm) * inv;
}

}  // namespace

void softmax_fused(const float* dx, float* dy, int rows, int cols) {
  k_softmax<<<rows, 256>>>(dx, dy, cols);
  IK_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ik::cuda
