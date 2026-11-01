threads=8

| op | shape | naive ms | optimized ms | optimized throughput | speedup |
|---|---|---|---|---|---|
| matmul | 256x256x256 | 8.06 | 0.27 | 125.1 GFLOP/s | 30.0x |
| matmul | 512x512x512 | 117.36 | 1.88 | 143.1 GFLOP/s | 62.6x |
| matmul | 1024x1024x1024 | 2607.75 | 15.79 | 136.0 GFLOP/s | 165.2x |
| softmax | 4096x1024 | 11.76 | 1.82 | 18.4 GB/s | 6.5x |
| layernorm | 4096x1024 | 4.58 | 0.45 | 75.3 GB/s | 10.3x |
| softmax | 8192x4096 | 96.41 | 15.78 | 17.0 GB/s | 6.1x |
| layernorm | 8192x4096 | 38.07 | 7.27 | 36.9 GB/s | 5.2x |

wrote results/bench_cpu.csv
