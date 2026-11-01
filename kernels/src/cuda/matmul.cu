#include "common.cuh"
#include "ik/cuda_kernels.hpp"

namespace ik::cuda {
namespace {

__global__ void k_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = acc;
}

constexpr int T = 32;
__global__ void k_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T + 1];  // +1 pad avoids bank conflicts
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// 64x64 C tile per block, 16x16 threads, each thread owns a 4x4 micro-tile in registers.
constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
__global__ void k_matmul_tiled_reg(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[BK][BM];  // stored transposed: As[k][m]
  __shared__ float Bs[BK][BN];
  const int tx = threadIdx.x, ty = threadIdx.y;  // 16 x 16
  const int tid = ty * 16 + tx;
  const int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;
  float acc[TM][TN] = {};
  float a_reg[TM], b_reg[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile 64x16 (1024 elems, 4 per thread) and B tile 16x64.
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int idx = tid + l * 256;
      const int am = idx / BK, ak = idx % BK;
      const int gr = rowBase + am, gk = k0 + ak;
      As[ak][am] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : 0.f;
      const int bk = idx / BN, bn = idx % BN;
      const int gk2 = k0 + bk, gc = colBase + bn;
      Bs[bk][bn] = (gk2 < K && gc < N) ? B[(size_t)gk2 * N + gc] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) a_reg[i] = As[k][ty * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = rowBase + ty * TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = colBase + tx * TN + j;
      if (c < N) C[(size_t)r * N + c] = acc[i][j];
    }
  }
}

}  // namespace

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
  k_matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
  IK_CUDA_CHECK(cudaGetLastError());
}
void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(T, T), grid((N + T - 1) / T, (M + T - 1) / T);
  k_matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
  IK_CUDA_CHECK(cudaGetLastError());
}
void matmul_tiled_reg(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16), grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  k_matmul_tiled_reg<<<grid, block>>>(A, B, C, M, N, K);
  IK_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ik::cuda
