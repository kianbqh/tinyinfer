#include "ik/cpu_kernels.hpp"
#include <cmath>

namespace ik::cpu {

void layernorm_naive(const float* x, const float* gamma, const float* beta, float* y,
                     int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (size_t)r * cols;
    float* yr = y + (size_t)r * cols;
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) mean += xr[c];
    mean /= cols;
    double var = 0.0;
    for (int c = 0; c < cols; ++c) { const double d = xr[c] - mean; var += d * d; }
    var /= cols;
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) yr[c] = float((xr[c] - mean) * rstd) * gamma[c] + beta[c];
  }
}

void layernorm_fused(const float* x, const float* gamma, const float* beta, float* y,
                     int rows, int cols, float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* __restrict__ xr = x + (size_t)r * cols;
    float* __restrict__ yr = y + (size_t)r * cols;
    // Single pass: shifted sums (shift = first element) for numerical stability.
    const float shift = xr[0];
    float s = 0.f, ss = 0.f;
#pragma omp simd reduction(+ : s, ss)
    for (int c = 0; c < cols; ++c) { const float d = xr[c] - shift; s += d; ss += d * d; }
    const float mean_d = s / cols;
    const float var = ss / cols - mean_d * mean_d;
    const float mean = shift + mean_d;
    const float rstd = 1.f / std::sqrt((var > 0.f ? var : 0.f) + eps);
#pragma omp simd
    for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
  }
}

}  // namespace ik::cpu
