#include <vector>
#include <random>
#include "common.cuh"
#include "ik/cuda_kernels.hpp"

namespace ik::cuda {

bool device_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
void synchronize() { IK_CUDA_CHECK(cudaDeviceSynchronize()); }

namespace {
float* dev_random(size_t n, unsigned seed) {
  std::vector<float> h(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto& v : h) v = d(rng);
  float* p = nullptr;
  IK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
  IK_CUDA_CHECK(cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  return p;
}
template <class F>
double time_ms(F&& f, int iters) {
  for (int i = 0; i < 3; ++i) f();  // warmup
  cudaEvent_t a, b;
  IK_CUDA_CHECK(cudaEventCreate(&a));
  IK_CUDA_CHECK(cudaEventCreate(&b));
  IK_CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) f();
  IK_CUDA_CHECK(cudaEventRecord(b));
  IK_CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  IK_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}
}  // namespace

double bench_matmul(int variant, int M, int N, int K, int iters) {
  float *A = dev_random((size_t)M * K, 1), *B = dev_random((size_t)K * N, 2), *C = nullptr;
  IK_CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * sizeof(float)));
  auto fn = [&] {
    if (variant == 0) matmul_naive(A, B, C, M, N, K);
    else if (variant == 1) matmul_tiled(A, B, C, M, N, K);
    else matmul_tiled_reg(A, B, C, M, N, K);
  };
  const double ms = time_ms(fn, iters);
  cudaFree(A); cudaFree(B); cudaFree(C);
  return ms;
}
double bench_softmax(int rows, int cols, int iters) {
  float *x = dev_random((size_t)rows * cols, 3), *y = nullptr;
  IK_CUDA_CHECK(cudaMalloc(&y, (size_t)rows * cols * sizeof(float)));
  const double ms = time_ms([&] { softmax_fused(x, y, rows, cols); }, iters);
  cudaFree(x); cudaFree(y);
  return ms;
}
double bench_layernorm(int rows, int cols, int iters) {
  float *x = dev_random((size_t)rows * cols, 4), *g = dev_random(cols, 5), *b = dev_random(cols, 6), *y = nullptr;
  IK_CUDA_CHECK(cudaMalloc(&y, (size_t)rows * cols * sizeof(float)));
  const double ms = time_ms([&] { layernorm_fused(x, g, b, y, rows, cols, 1e-5f); }, iters);
  cudaFree(x); cudaFree(g); cudaFree(b); cudaFree(y);
  return ms;
}

}  // namespace ik::cuda
