// Host-callable launchers for CUDA kernels. Pointers are DEVICE pointers, row-major float32.
#pragma once

namespace ik::cuda {

void matmul_naive(const float* dA, const float* dB, float* dC, int M, int N, int K);
// 32x32 shared-memory tiling, coalesced loads.
void matmul_tiled(const float* dA, const float* dB, float* dC, int M, int N, int K);
// 64x64 block tile, each thread computes a 4x4 register micro-tile (shared + register blocking).
void matmul_tiled_reg(const float* dA, const float* dB, float* dC, int M, int N, int K);

// One block per row; online max/sum with warp shuffles; two input reads and one output write.
void softmax_fused(const float* dx, float* dy, int rows, int cols);
// One block per row; fused mean/var reduction + affine in one kernel.
void layernorm_fused(const float* dx, const float* dgamma, const float* dbeta, float* dy,
                     int rows, int cols, float eps);

bool device_available();
void synchronize();

// CUDA-event timing of repeated launches with device-resident data. Return mean ms per call.
// variant for matmul: 0 = naive, 1 = tiled, 2 = tiled_reg
double bench_matmul(int variant, int M, int N, int K, int iters);
double bench_softmax(int rows, int cols, int iters);
double bench_layernorm(int rows, int cols, int iters);

}  // namespace ik::cuda
