// CPU benchmark: naive vs optimized kernels. Prints a markdown table and writes CSV.
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "ik/cpu_kernels.hpp"

using namespace ik;

static std::vector<float> rnd(size_t n, unsigned s) {
  std::mt19937 g(s);
  std::uniform_real_distribution<float> d(-1, 1);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// Median of `reps` timed runs after one warmup; returns ms.
static double time_ms(const std::function<void()>& f, int reps) {
  f();
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    auto a = std::chrono::steady_clock::now();
    f();
    auto b = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  const std::string csv = argc > 1 ? argv[1] : "results/bench_cpu.csv";
  FILE* out = std::fopen(csv.c_str(), "w");
  std::fprintf(out, "op,shape,variant,ms,gflops_or_gbps,speedup\n");
  std::printf("threads=%d\n\n", omp_get_max_threads());
  std::printf("| op | shape | naive ms | optimized ms | optimized throughput | speedup |\n|---|---|---|---|---|---|\n");

  for (int n : {256, 512, 1024}) {
    auto A = rnd((size_t)n * n, 1), B = rnd((size_t)n * n, 2);
    std::vector<float> C((size_t)n * n);
    const int reps = n >= 1024 ? 3 : 7;
    double t0 = time_ms([&] { cpu::matmul_naive(A.data(), B.data(), C.data(), n, n, n); }, reps);
    double t1 = time_ms([&] { cpu::matmul_blocked(A.data(), B.data(), C.data(), n, n, n); }, reps * 3);
    const double gf = 2.0 * n * n * n / (t1 * 1e6);
    const double gf0 = 2.0 * n * n * n / (t0 * 1e6);
    std::printf("| matmul | %dx%dx%d | %.2f | %.2f | %.1f GFLOP/s | %.1fx |\n", n, n, n, t0, t1, gf, t0 / t1);
    std::fprintf(out, "matmul,%dx%dx%d,naive,%.4f,%.2f,1\n", n, n, n, t0, gf0);
    std::fprintf(out, "matmul,%dx%dx%d,blocked_omp_simd,%.4f,%.2f,%.2f\n", n, n, n, t1, gf, t0 / t1);
  }

  struct S { int r, c; };
  for (S s : {S{4096, 1024}, S{8192, 4096}}) {
    auto x = rnd((size_t)s.r * s.c, 3), g = rnd(s.c, 4), b = rnd(s.c, 5);
    std::vector<float> y(x.size());
    const double bytes = 2.0 * x.size() * sizeof(float);  // read + write
    double t0 = time_ms([&] { cpu::softmax_naive(x.data(), y.data(), s.r, s.c); }, 7);
    double t1 = time_ms([&] { cpu::softmax_fused(x.data(), y.data(), s.r, s.c); }, 15);
    std::printf("| softmax | %dx%d | %.2f | %.2f | %.1f GB/s | %.1fx |\n", s.r, s.c, t0, t1, bytes / (t1 * 1e6), t0 / t1);
    std::fprintf(out, "softmax,%dx%d,naive,%.4f,%.2f,1\n", s.r, s.c, t0, bytes / (t0 * 1e6));
    std::fprintf(out, "softmax,%dx%d,fused_omp_simd,%.4f,%.2f,%.2f\n", s.r, s.c, t1, bytes / (t1 * 1e6), t0 / t1);
    t0 = time_ms([&] { cpu::layernorm_naive(x.data(), g.data(), b.data(), y.data(), s.r, s.c, 1e-5f); }, 7);
    t1 = time_ms([&] { cpu::layernorm_fused(x.data(), g.data(), b.data(), y.data(), s.r, s.c, 1e-5f); }, 15);
    std::printf("| layernorm | %dx%d | %.2f | %.2f | %.1f GB/s | %.1fx |\n", s.r, s.c, t0, t1, bytes / (t1 * 1e6), t0 / t1);
    std::fprintf(out, "layernorm,%dx%d,naive,%.4f,%.2f,1\n", s.r, s.c, t0, bytes / (t0 * 1e6));
    std::fprintf(out, "layernorm,%dx%d,fused_omp_simd,%.4f,%.2f,%.2f\n", s.r, s.c, t1, bytes / (t1 * 1e6), t0 / t1);
  }
  std::fclose(out);
  std::printf("\nwrote %s\n", csv.c_str());
}
