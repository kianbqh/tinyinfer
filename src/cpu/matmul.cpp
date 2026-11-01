#include "ik/cpu_kernels.hpp"
#include <algorithm>
#include <cstring>

namespace ik::cpu {

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = acc;
    }
}

namespace {
constexpr int BM = 64;   // rows of A / C per tile
constexpr int BK = 256;  // shared dimension per tile (A tile = 64 KB fits L2)
constexpr int BN = 256;  // cols of B / C per tile (B tile 256x256 floats = 256 KB in L2)
}  // namespace

void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, sizeof(float) * (size_t)M * N);
#pragma omp parallel for collapse(2) schedule(static)
  for (int i0 = 0; i0 < M; i0 += BM) {
    for (int j0 = 0; j0 < N; j0 += BN) {
      const int iMax = std::min(i0 + BM, M), jMax = std::min(j0 + BN, N);
      for (int k0 = 0; k0 < K; k0 += BK) {
        const int kMax = std::min(k0 + BK, K);
        for (int i = i0; i < iMax; ++i) {
          float* __restrict__ c = C + (size_t)i * N;
          const float* __restrict__ a = A + (size_t)i * K;
          for (int k = k0; k < kMax; ++k) {
            const float aik = a[k];
            const float* __restrict__ b = B + (size_t)k * N;
#pragma omp simd
            for (int j = j0; j < jMax; ++j) c[j] += aik * b[j];
          }
        }
      }
    }
  }
}

}  // namespace ik::cpu
