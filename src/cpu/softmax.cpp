#include "ik/cpu_kernels.hpp"
#include <cmath>
#include <limits>

namespace ik::cpu {

void softmax_naive(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (size_t)r * cols;
    float* yr = y + (size_t)r * cols;
    float m = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < cols; ++c) m = std::fmax(m, xr[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) { yr[c] = std::exp(xr[c] - m); s += yr[c]; }
    for (int c = 0; c < cols; ++c) yr[c] /= s;
  }
}

void softmax_fused(const float* x, float* y, int rows, int cols) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* __restrict__ xr = x + (size_t)r * cols;
    float* __restrict__ yr = y + (size_t)r * cols;
    // Pass 1: max (vectorized reduction).
    float m = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : m)
    for (int c = 0; c < cols; ++c) m = xr[c] > m ? xr[c] : m;
    // Pass 2: exp + sum fused, write unnormalized result.
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int c = 0; c < cols; ++c) { const float e = std::exp(xr[c] - m); yr[c] = e; s += e; }
    const float inv = 1.f / s;
#pragma omp simd
    for (int c = 0; c < cols; ++c) yr[c] *= inv;
  }
}

}  // namespace ik::cpu
