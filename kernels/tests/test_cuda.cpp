#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <tuple>
#include "ik/cuda_kernels.hpp"
#include "reference.hpp"

using namespace ik;

namespace {
struct Dev {
  float* p = nullptr;
  explicit Dev(const std::vector<float>& h) {
    cudaMalloc(&p, h.size() * sizeof(float));
    cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  }
  explicit Dev(size_t n) { cudaMalloc(&p, n * sizeof(float)); }
  ~Dev() { cudaFree(p); }
  std::vector<float> host(size_t n) const {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
  }
};
#define REQUIRE_GPU() if (!cuda::device_available()) GTEST_SKIP() << "no CUDA device"
}  // namespace

class CudaMatmul : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(CudaMatmul, AllVariantsMatchReference) {
  REQUIRE_GPU();
  auto [M, N, K] = GetParam();
  auto A = ref::random_vec((size_t)M * K, 1), B = ref::random_vec((size_t)K * N, 2);
  auto R = ref::matmul(A, B, M, N, K);
  Dev dA(A), dB(B), dC((size_t)M * N);
  const float tol = 1e-5f * K + 1e-4f;
  cuda::matmul_naive(dA.p, dB.p, dC.p, M, N, K);
  EXPECT_LT(ref::max_abs_diff(dC.host((size_t)M * N), R), tol);
  cuda::matmul_tiled(dA.p, dB.p, dC.p, M, N, K);
  EXPECT_LT(ref::max_abs_diff(dC.host((size_t)M * N), R), tol);
  cuda::matmul_tiled_reg(dA.p, dB.p, dC.p, M, N, K);
  EXPECT_LT(ref::max_abs_diff(dC.host((size_t)M * N), R), tol);
}
INSTANTIATE_TEST_SUITE_P(Shapes, CudaMatmul,
                         ::testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(33, 65, 17),
                                           std::make_tuple(64, 64, 64), std::make_tuple(257, 130, 515),
                                           std::make_tuple(512, 512, 512)));

TEST(CudaSoftmax, MatchesReference) {
  REQUIRE_GPU();
  for (int cols : {1, 31, 256, 1000, 4097}) {
    const int rows = 37;
    auto x = ref::random_vec((size_t)rows * cols, cols, -8.f, 8.f);
    Dev dx(x), dy(x.size());
    cuda::softmax_fused(dx.p, dy.p, rows, cols);
    EXPECT_LT(ref::max_abs_diff(dy.host(x.size()), ref::softmax(x, rows, cols)), 1e-5f) << cols;
  }
}

TEST(CudaLayerNorm, MatchesReference) {
  REQUIRE_GPU();
  for (int cols : {2, 64, 768, 4099}) {
    const int rows = 19;
    auto x = ref::random_vec((size_t)rows * cols, cols, -3.f, 5.f);
    auto g = ref::random_vec(cols, 11, 0.5f, 1.5f), b = ref::random_vec(cols, 12);
    Dev dx(x), dg(g), db(b), dy(x.size());
    cuda::layernorm_fused(dx.p, dg.p, db.p, dy.p, rows, cols, 1e-5f);
    EXPECT_LT(ref::max_abs_diff(dy.host(x.size()), ref::layernorm(x, g, b, rows, cols, 1e-5)), 1e-3f) << cols;
  }
}

TEST(CudaSoftmax, NegativeInfinityMaskHasZeroProbability) {
  REQUIRE_GPU();
  for (int cols : {31, 513}) {
    const int rows = 3;
    std::vector<float> x((size_t)rows * cols, -INFINITY);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; c += 7) x[(size_t)r * cols + c] = float(c % 11) - 5.f;
    Dev dx(x), dy(x.size());
    cuda::softmax_fused(dx.p, dy.p, rows, cols);
    auto actual = dy.host(x.size());
    EXPECT_LT(ref::max_abs_diff(actual, ref::softmax(x, rows, cols)), 1e-6f);
    for (size_t i = 0; i < x.size(); ++i)
      if (x[i] == -INFINITY) EXPECT_EQ(actual[i], 0.f);
  }
}

TEST(CudaLayerNorm, LargeOffsetInputIsStable) {
  REQUIRE_GPU();
  const int rows = 3, cols = 1024;
  auto x = ref::random_vec((size_t)rows * cols, 21, 1e4f - 1.f, 1e4f + 1.f);
  std::vector<float> g(cols, 1.f), b(cols, 0.f);
  Dev dx(x), dg(g), db(b), dy(x.size());
  cuda::layernorm_fused(dx.p, dg.p, db.p, dy.p, rows, cols, 1e-5f);
  EXPECT_LT(ref::max_abs_diff(dy.host(x.size()), ref::layernorm(x, g, b, rows, cols, 1e-5)), 5e-3f);
}

TEST(CudaLayerNorm, ConstantRowGivesBeta) {
  REQUIRE_GPU();
  const int rows = 2, cols = 513;
  std::vector<float> x((size_t)rows * cols, 1e4f), g(cols, 2.f), b(cols, 0.5f);
  Dev dx(x), dg(g), db(b), dy(x.size());
  cuda::layernorm_fused(dx.p, dg.p, db.p, dy.p, rows, cols, 1e-5f);
  for (float value : dy.host(x.size())) EXPECT_NEAR(value, 0.5f, 1e-6f);
}
