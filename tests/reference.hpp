// Double-precision references used as ground truth by tests.
#pragma once
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace ref {

inline std::vector<float> random_vec(size_t n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

inline std::vector<float> matmul(const std::vector<float>& A, const std::vector<float>& B, int M, int N, int K) {
  std::vector<float> C((size_t)M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = (float)acc;
    }
  return C;
}

inline std::vector<float> softmax(const std::vector<float>& x, int rows, int cols) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    double m = -INFINITY, s = 0;
    for (int c = 0; c < cols; ++c) m = std::fmax(m, (double)x[(size_t)r * cols + c]);
    for (int c = 0; c < cols; ++c) s += std::exp((double)x[(size_t)r * cols + c] - m);
    for (int c = 0; c < cols; ++c) y[(size_t)r * cols + c] = (float)(std::exp((double)x[(size_t)r * cols + c] - m) / s);
  }
  return y;
}

inline std::vector<float> layernorm(const std::vector<float>& x, const std::vector<float>& g,
                                    const std::vector<float>& b, int rows, int cols, double eps) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    double mean = 0, var = 0;
    for (int c = 0; c < cols; ++c) mean += x[(size_t)r * cols + c];
    mean /= cols;
    for (int c = 0; c < cols; ++c) { double d = x[(size_t)r * cols + c] - mean; var += d * d; }
    var /= cols;
    for (int c = 0; c < cols; ++c)
      y[(size_t)r * cols + c] = (float)((x[(size_t)r * cols + c] - mean) / std::sqrt(var + eps) * g[c] + b[c]);
  }
  return y;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<float>::infinity();
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

}  // namespace ref
