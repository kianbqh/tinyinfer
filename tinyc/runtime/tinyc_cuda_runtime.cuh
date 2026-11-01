// tinyc CUDA runtime: kernels used by generated .cu code (sm_70+; tuned for nothing in particular yet).
#pragma once
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

#define TC_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

namespace tc {
constexpr int TS = 16;
// C[M,N] = A[M,K] B[K,N]; 16x16 shared-memory tiles; epilogue functor writes the result.
template <class Epi>
__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B, int M, int K, int N, Epi epi) {
  __shared__ float As[TS][TS], Bs[TS][TS + 1];
  const int i = blockIdx.y * TS + threadIdx.y, j = blockIdx.x * TS + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TS) {
    As[threadIdx.y][threadIdx.x] = (i < M && k0 + threadIdx.x < K) ? A[(size_t)i * K + k0 + threadIdx.x] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (j < N && k0 + threadIdx.y < K) ? B[(size_t)(k0 + threadIdx.y) * N + j] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TS; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (i < M && j < N) epi(i, j, acc);
}
template <class Epi>
inline void gemm(const float* A, const float* B, int M, int K, int N, Epi epi, cudaStream_t s = 0) {
  dim3 blk(TS, TS), grd((N + TS - 1) / TS, (M + TS - 1) / TS);
  gemm_tiled<<<grd, blk, 0, s>>>(A, B, M, K, N, epi);
}

// batched im2col: x[B,C,H,W] -> col[C*k*k, B*OH*OW]
__global__ void im2col_k(const float* x, int B, int C, int H, int W, int k, int s, int p, int OH, int OW, float* col) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x, P = (long)OH * OW, BP = B * P;
  if (idx >= (long)C * k * k * BP) return;
  const int q = idx / BP, bp = idx % BP, b = bp / P, pix = bp % P, oy = pix / OW, ox = pix % OW;
  const int c = q / (k * k), ky = (q / k) % k, kx = q % k;
  const int iy = oy * s - p + ky, ix = ox * s - p + kx;
  col[idx] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[(((size_t)b * C + c) * H + iy) * W + ix] : 0.f;
}
__global__ void maxpool_k(const float* x, int BC, int H, int W, int k, int s, int OH, int OW, float* y) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)BC * OH * OW) return;
  const int bc = idx / (OH * OW), oy = (idx / OW) % OH, ox = idx % OW;
  const float* xi = x + (size_t)bc * H * W;
  float m = -FLT_MAX;
  for (int ky = 0; ky < k; ++ky)
    for (int kx = 0; kx < k; ++kx) m = fmaxf(m, xi[(oy * s + ky) * W + ox * s + kx]);
  y[idx] = m;
}
__global__ void batchnorm_k(const float* x, long n, int C, int HW, const float* sc, const float* sh, float* y) {
  const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) { const int c = (i / HW) % C; y[i] = x[i] * sc[c] + sh[c]; }
}
inline int blocks(long n, int t = 256) { return int((n + t - 1) / t); }
}  // namespace tc
