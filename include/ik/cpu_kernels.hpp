// CPU reference + optimized kernels. All matrices are row-major float32.
#pragma once
#include <cstddef>

namespace ik::cpu {

// C[M,N] = A[M,K] @ B[K,N]. Textbook i-j-k triple loop (strided B access), single thread.
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);

// Cache-blocked (M/K/N tiles 64x256x256), OpenMP over output tiles, SIMD inner loop.
void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K);

// Row-wise softmax over the last dim. Naive: 3 passes (max, sum exp, normalize).
void softmax_naive(const float* x, float* y, int rows, int cols);
// Fused exp/sum pass between max and normalize passes; OpenMP over rows and SIMD reductions.
void softmax_fused(const float* x, float* y, int rows, int cols);

// LayerNorm over the last dim with affine gamma/beta.
void layernorm_naive(const float* x, const float* gamma, const float* beta, float* y,
                     int rows, int cols, float eps);
// Fused: shifted float sum/sum-of-squares + affine, OpenMP + SIMD.
void layernorm_fused(const float* x, const float* gamma, const float* beta, float* y,
                     int rows, int cols, float eps);

}  // namespace ik::cpu
