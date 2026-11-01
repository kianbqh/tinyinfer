| Model | Batch | CUDA eager (µs) | tinyc O0 (µs) | tinyc O2 (µs) | O2 max abs error |
|---|---:|---:|---:|---:|---:|
| mlp | 1 | 128.04 | 93.18 | 59.04 | 4.17e-07 |
| mlp | 64 | 135.57 | 118.20 | 70.52 | 5.96e-07 |
| cnn | 1 | 311.21 | 129.08 | 70.76 | 5.22e-08 |
| cnn | 32 | 288.29 | 186.47 | 131.92 | 1.42e-07 |

7 repeats × 200 forwards. Inputs and custom activation buffers are device-resident. CUDA events surround a C++ enqueue loop for tinyc and a Python eager enqueue loop for PyTorch, including inter-launch gaps; transfers and compilation are outside the window. tinyc warms up 10 forwards per repeat; PyTorch warms up 20 before all repeats. PyTorch matmul and cuDNN convolution use IEEE FP32 settings with TF32 disabled; tinyc uses nvcc --use_fast_math and absolute error tolerance 1e-4.

The legacy `kernels` field counts IR compute nodes. CUDA convolutions each issue im2col and GEMM, so the CNN has 18→9 actual CUDA launches for O0→O2 (16→7 IR compute nodes).
