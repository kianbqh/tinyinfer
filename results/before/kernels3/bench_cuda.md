device=NVIDIA GeForce RTX 3080

| op | shape | variant | ms | throughput |
|---|---|---|---|---|
| matmul | 1024^3 | naive | 1.574 | 1.36 TFLOP/s |
| matmul | 1024^3 | smem_tiled_32 | 1.106 | 1.94 TFLOP/s |
| matmul | 1024^3 | smem_reg_tiled_64x64_4x4 | 0.369 | 5.81 TFLOP/s |
| matmul | 2048^3 | naive | 11.062 | 1.55 TFLOP/s |
| matmul | 2048^3 | smem_tiled_32 | 8.477 | 2.03 TFLOP/s |
| matmul | 2048^3 | smem_reg_tiled_64x64_4x4 | 1.988 | 8.64 TFLOP/s |
| matmul | 4096^3 | naive | 83.635 | 1.64 TFLOP/s |
| matmul | 4096^3 | smem_tiled_32 | 66.511 | 2.07 TFLOP/s |
| matmul | 4096^3 | smem_reg_tiled_64x64_4x4 | 16.652 | 8.25 TFLOP/s |
| softmax | 4096x1024 | fused_online | 0.047 | 712.7 GB/s |
| layernorm | 4096x1024 | fused | 0.055 | 607.7 GB/s |
| softmax | 8192x4096 | fused_online | 0.615 | 436.3 GB/s |
| layernorm | 8192x4096 | fused | 0.605 | 444.0 GB/s |

wrote results/bench_cuda.csv
