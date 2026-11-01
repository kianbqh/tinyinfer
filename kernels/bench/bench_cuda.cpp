// GPU benchmark (CUDA events over repeated launches, transfers excluded).
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include "ik/cuda_kernels.hpp"

int main(int argc, char** argv) {
  if (!ik::cuda::device_available()) { std::printf("No CUDA device found.\n"); return 1; }
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  const std::string csv = argc > 1 ? argv[1] : "results/bench_cuda.csv";
  FILE* out = std::fopen(csv.c_str(), "w");
  std::fprintf(out, "op,shape,variant,ms,throughput\n");
  std::printf("device=%s\n\n| op | shape | variant | ms | throughput |\n|---|---|---|---|---|\n", p.name);
  const char* names[] = {"naive", "smem_tiled_32", "smem_reg_tiled_64x64_4x4"};
  for (int n : {1024, 2048, 4096}) {
    for (int v = 0; v < 3; ++v) {
      const double ms = ik::cuda::bench_matmul(v, n, n, n, n >= 4096 ? 5 : 20);
      const double tf = 2.0 * n * n * n / (ms * 1e9);
      std::printf("| matmul | %d^3 | %s | %.3f | %.2f TFLOP/s |\n", n, names[v], ms, tf);
      std::fprintf(out, "matmul,%d^3,%s,%.5f,%.3f\n", n, names[v], ms, tf);
    }
  }
  for (auto rc : {std::pair{4096, 1024}, std::pair{8192, 4096}}) {
    const double bytes = 2.0 * rc.first * rc.second * 4;
    double ms = ik::cuda::bench_softmax(rc.first, rc.second, 50);
    std::printf("| softmax | %dx%d | fused_online | %.3f | %.1f GB/s |\n", rc.first, rc.second, ms, bytes / (ms * 1e6));
    std::fprintf(out, "softmax,%dx%d,fused_online,%.5f,%.2f\n", rc.first, rc.second, ms, bytes / (ms * 1e6));
    ms = ik::cuda::bench_layernorm(rc.first, rc.second, 50);
    std::printf("| layernorm | %dx%d | fused | %.3f | %.1f GB/s |\n", rc.first, rc.second, ms, bytes / (ms * 1e6));
    std::fprintf(out, "layernorm,%dx%d,fused,%.5f,%.2f\n", rc.first, rc.second, ms, bytes / (ms * 1e6));
  }
  std::fclose(out);
  std::printf("\nwrote %s\n", csv.c_str());
}
