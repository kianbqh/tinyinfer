#include <gtest/gtest.h>
#include <numeric>
#include <tuple>
#include "ik/cpu_kernels.hpp"
#include "reference.hpp"

using namespace ik;

TEST(Reference, NonFiniteOutputCannotPassComparison) {
  EXPECT_TRUE(std::isinf(ref::max_abs_diff({NAN}, {0.f})));
  EXPECT_TRUE(std::isinf(ref::max_abs_diff({INFINITY}, {INFINITY})));
}

class MatmulShapes : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MatmulShapes, NaiveAndBlockedMatchFp64Reference) {
  auto [M, N, K] = GetParam();
  auto A = ref::random_vec((size_t)M * K, 1), B = ref::random_vec((size_t)K * N, 2);
  auto R = ref::matmul(A, B, M, N, K);
  std::vector<float> C1((size_t)M * N), C2((size_t)M * N);
  cpu::matmul_naive(A.data(), B.data(), C1.data(), M, N, K);
  cpu::matmul_blocked(A.data(), B.data(), C2.data(), M, N, K);
  const float tol = 1e-5f * K + 1e-5f;
  EXPECT_LT(ref::max_abs_diff(C1, R), tol);
  EXPECT_LT(ref::max_abs_diff(C2, R), tol);
}

// Includes non-multiples of the tile sizes (64/256) and degenerate shapes.
INSTANTIATE_TEST_SUITE_P(Shapes, MatmulShapes,
                         ::testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(7, 13, 5),
                                           std::make_tuple(64, 256, 256), std::make_tuple(65, 257, 300),
                                           std::make_tuple(128, 96, 513), std::make_tuple(300, 1, 77)));

TEST(Softmax, MatchesReferenceAndRowsSumToOne) {
  for (int cols : {1, 31, 128, 1000, 4097}) {
    const int rows = 17;
    auto x = ref::random_vec((size_t)rows * cols, cols, -8.f, 8.f);
    auto r = ref::softmax(x, rows, cols);
    std::vector<float> y1(x.size()), y2(x.size());
    cpu::softmax_naive(x.data(), y1.data(), rows, cols);
    cpu::softmax_fused(x.data(), y2.data(), rows, cols);
    EXPECT_LT(ref::max_abs_diff(y1, r), 1e-6f) << "cols=" << cols;
    EXPECT_LT(ref::max_abs_diff(y2, r), 1e-6f) << "cols=" << cols;
    for (int i = 0; i < rows; ++i) {
      double s = std::accumulate(y2.begin() + (size_t)i * cols, y2.begin() + (size_t)(i + 1) * cols, 0.0);
      EXPECT_NEAR(s, 1.0, 1e-4);
    }
  }
}

TEST(Softmax, NumericallyStableForLargeLogits) {
  const int rows = 4, cols = 513;
  auto x = ref::random_vec((size_t)rows * cols, 9, 1000.f, 1010.f);  // exp would overflow without max-shift
  std::vector<float> y(x.size());
  cpu::softmax_fused(x.data(), y.data(), rows, cols);
  for (float v : y) ASSERT_TRUE(std::isfinite(v));
  EXPECT_LT(ref::max_abs_diff(y, ref::softmax(x, rows, cols)), 1e-6f);
}

TEST(LayerNorm, MatchesReference) {
  for (int cols : {2, 64, 768, 1024, 4099}) {
    const int rows = 9;
    auto x = ref::random_vec((size_t)rows * cols, cols, -3.f, 5.f);
    auto g = ref::random_vec(cols, 11, 0.5f, 1.5f), b = ref::random_vec(cols, 12);
    auto r = ref::layernorm(x, g, b, rows, cols, 1e-5);
    std::vector<float> y1(x.size()), y2(x.size());
    cpu::layernorm_naive(x.data(), g.data(), b.data(), y1.data(), rows, cols, 1e-5f);
    cpu::layernorm_fused(x.data(), g.data(), b.data(), y2.data(), rows, cols, 1e-5f);
    EXPECT_LT(ref::max_abs_diff(y1, r), 1e-5f) << "cols=" << cols;
    EXPECT_LT(ref::max_abs_diff(y2, r), 1e-4f) << "cols=" << cols;
  }
}

TEST(LayerNorm, LargeOffsetInputIsStable) {
  // Shifted single-pass variance must not suffer catastrophic cancellation at mean ~ 1e4.
  const int rows = 3, cols = 1024;
  auto x = ref::random_vec((size_t)rows * cols, 21, 1e4f - 1.f, 1e4f + 1.f);
  std::vector<float> g(cols, 1.f), b(cols, 0.f), y(x.size());
  cpu::layernorm_fused(x.data(), g.data(), b.data(), y.data(), rows, cols, 1e-5f);
  EXPECT_LT(ref::max_abs_diff(y, ref::layernorm(x, g, b, rows, cols, 1e-5)), 5e-3f);
}

TEST(LayerNorm, ConstantRowGivesBeta) {
  const int cols = 256;
  std::vector<float> x(cols, 3.25f), g(cols, 2.f), b(cols, 0.5f), y(cols);
  cpu::layernorm_fused(x.data(), g.data(), b.data(), y.data(), 1, cols, 1e-5f);
  for (float v : y) EXPECT_NEAR(v, 0.5f, 1e-6f);
}
