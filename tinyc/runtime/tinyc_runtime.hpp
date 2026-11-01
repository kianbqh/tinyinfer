// tinyc CPU runtime: the few kernels generated code calls. Header-only, C++17, OpenMP + omp simd.
#pragma once
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include "tinyinfer_backend.h"

namespace tc {
template <class Epi>
inline void gemm_hand(float* tmp, const float* a, const float* b, int m, int k, int n, Epi epi) {
  ti_cpu_gemm(a, b, tmp, m, k, n);
#pragma omp parallel for schedule(static) if ((long)m * n > 16384)
  for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) epi(i, j, tmp[(size_t)i*n+j]);
}
inline void softmax(const float* x,float* y,int rows,int cols) {
#pragma omp parallel for schedule(static)
  for(int r=0;r<rows;++r) { const float* p=x+(size_t)r*cols; float* q=y+(size_t)r*cols; float m=-FLT_MAX; double sum=0;
    for(int c=0;c<cols;++c)m=std::max(m,p[c]);
    for(int c=0;c<cols;++c){q[c]=std::exp(p[c]-m);sum+=q[c];}
    for(int c=0;c<cols;++c)q[c]/=sum;
  }
}
inline void layernorm(const float* x,const float* g,const float* b,float* y,int rows,int cols,float eps) {
#pragma omp parallel for schedule(static)
  for(int r=0;r<rows;++r) {const float* p=x+(size_t)r*cols;float* q=y+(size_t)r*cols;double mean=0,var=0;
    for(int c=0;c<cols;++c)mean+=p[c]; mean/=cols;
    for(int c=0;c<cols;++c){double d=p[c]-mean;var+=d*d;} float s=1/std::sqrt(var/cols+eps);
    for(int c=0;c<cols;++c)q[c]=(p[c]-mean)*s*g[c]+b[c];
  }
}

// C[M,N] = A[M,K] * B[K,N], then out = epi(i, j, acc). Register/cache tiling: 4 rows x 64 cols of
// accumulators per task; the k-loop streams contiguous 64-float rows of B (vectorized by omp simd).
// micro-kernel: MR x NB accumulator tile kept in registers (MR*NB/16 zmm on AVX-512), k-loop streams rows of B.
template <int R, int MR, int NB>
inline void tile_full(const float* __restrict A, const float* __restrict B, int K, int N, int i0, int j0,
                      float (&acc)[MR][NB]) {
  for (int k = 0; k < K; ++k) {
    const float* b = B + (size_t)k * N + j0;
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) {
      const float a = A[(size_t)(i0 + r) * K + k];
#pragma omp simd
      for (int j = 0; j < NB; ++j) acc[r][j] += a * b[j];
    }
  }
}

// C[M,N] = A[M,K] * B[K,N], then out = epi(i, j, acc). Full tiles use the fixed-size micro-kernel;
// ragged edge tiles use a generic loop. Tiles are distributed over OpenMP threads.
template <class Epi>
inline void gemm(const float* __restrict A, const float* __restrict B, int M, int K, int N, Epi epi) {
  constexpr int MR = 6, NB = 64;
  static_assert(MR == 6, "switch below assumes MR == 6");
  const int mt = (M + MR - 1) / MR, nt = (N + NB - 1) / NB;
#pragma omp parallel for schedule(static) collapse(2) if ((long)M * N * K > 32768)
  for (int bi = 0; bi < mt; ++bi)
    for (int bj = 0; bj < nt; ++bj) {
      const int i0 = bi * MR, j0 = bj * NB;
      const int mr = std::min(MR, M - i0), nb = std::min(NB, N - j0);
      alignas(64) float acc[MR][NB] = {};
      if (nb == NB) {   // full-width tile: fixed-size register kernel for every row count
        switch (mr) {
          case 6: tile_full<6>(A, B, K, N, i0, j0, acc); break;
          case 5: tile_full<5>(A, B, K, N, i0, j0, acc); break;
          case 4: tile_full<4>(A, B, K, N, i0, j0, acc); break;
          case 3: tile_full<3>(A, B, K, N, i0, j0, acc); break;
          case 2: tile_full<2>(A, B, K, N, i0, j0, acc); break;
          default: tile_full<1>(A, B, K, N, i0, j0, acc); break;
        }
      } else {
        for (int k = 0; k < K; ++k) {
          const float* b = B + (size_t)k * N + j0;
          for (int r = 0; r < mr; ++r) {
            const float a = A[(size_t)(i0 + r) * K + k];
            for (int j = 0; j < nb; ++j) acc[r][j] += a * b[j];
          }
        }
      }
      for (int r = 0; r < mr; ++r)
        for (int j = 0; j < nb; ++j) epi(i0 + r, j0 + j, acc[r][j]);
    }
}

// batched im2col: x[B,C,H,W] -> col[C*k*k, B*OH*OW] so one GEMM covers the whole batch
inline void im2col(const float* x, int B, int C, int H, int W, int k, int s, int p, int OH, int OW, float* col) {
  const long P = (long)OH * OW, BP = B * P;
#pragma omp parallel for collapse(2) schedule(static) if ((long)C * k * k * BP > 65536)
  for (int q = 0; q < C * k * k; ++q)
    for (int b = 0; b < B; ++b) {
      const int c = q / (k * k), ky = (q / k) % k, kx = q % k;
      const float* xi = x + ((size_t)b * C + c) * H * W;
      float* dst = col + (size_t)q * BP + (size_t)b * P;
      for (int oy = 0; oy < OH; ++oy) {
        const int iy = oy * s - p + ky;
        for (int ox = 0; ox < OW; ++ox) {
          const int ix = ox * s - p + kx;
          dst[oy * OW + ox] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? xi[iy * W + ix] : 0.f;
        }
      }
    }
}

inline void maxpool2d(const float* x, int B, int C, int H, int W, int k, int s, int OH, int OW, float* y) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c) {
      const float* xi = x + ((size_t)b * C + c) * H * W;
      float* yo = y + ((size_t)b * C + c) * OH * OW;
      for (int oy = 0; oy < OH; ++oy)
        for (int ox = 0; ox < OW; ++ox) {
          float m = -FLT_MAX;
          for (int ky = 0; ky < k; ++ky)
            for (int kx = 0; kx < k; ++kx) m = std::max(m, xi[(oy * s + ky) * W + ox * s + kx]);
          yo[oy * OW + ox] = m;
        }
    }
}

inline void batchnorm(const float* x, int B, int C, int HW, const float* sc, const float* sh, float* y) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c) {
      const float* xi = x + ((size_t)b * C + c) * HW;
      float* yo = y + ((size_t)b * C + c) * HW;
#pragma omp simd
      for (int i = 0; i < HW; ++i) yo[i] = xi[i] * sc[c] + sh[c];
    }
}
}  // namespace tc
