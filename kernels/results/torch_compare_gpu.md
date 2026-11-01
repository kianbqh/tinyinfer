CPU: x86_64 | torch 2.15.0+cu132 | threads 8 | GPU: NVIDIA GeForce RTX 3080

| device | op | shape | ours ms | torch ms | ours / torch time | max abs err vs torch |
|---|---|---|---|---|---|---|
| CPU | matmul | 256^3 | 0.286 | 0.043 | 6.62x | 0.00e+00 |
| CPU | matmul | 512^3 | 1.607 | 0.532 | 3.02x | 8.39e-05 |
| CPU | matmul | 1024^3 | 13.896 | 3.908 | 3.56x | 2.29e-04 |
| CPU | softmax | 4096x1024 | 2.117 | 1.162 | 1.82x | 5.22e-08 |
| CPU | layernorm | 4096x1024 | 0.776 | 0.713 | 1.09x | 7.39e-06 |
| CPU | softmax | 8192x4096 | 19.898 | 14.481 | 1.37x | 6.33e-08 |
| CPU | layernorm | 8192x4096 | 12.635 | 13.195 | 0.96x | 1.57e-05 |
| GPU | matmul_smem_tiled | 1024^3 | 1.214 | 0.157 | 7.75x | n/a (tested in gtest) |
| GPU | matmul_reg_tiled | 1024^3 | 0.367 | 0.157 | 2.34x | n/a (tested in gtest) |
| GPU | matmul_smem_tiled | 2048^3 | 9.104 | 1.000 | 9.10x | n/a (tested in gtest) |
| GPU | matmul_reg_tiled | 2048^3 | 1.981 | 1.000 | 1.98x | n/a (tested in gtest) |
| GPU | matmul_smem_tiled | 4096^3 | 67.113 | 7.467 | 8.99x | n/a (tested in gtest) |
| GPU | matmul_reg_tiled | 4096^3 | 16.707 | 7.467 | 2.24x | n/a (tested in gtest) |
| GPU | softmax | 4096x1024 | 0.058 | 0.045 | 1.28x | n/a (tested in gtest) |
| GPU | layernorm | 4096x1024 | 0.045 | 0.083 | 0.54x | n/a (tested in gtest) |
| GPU | softmax | 8192x4096 | 0.612 | 0.422 | 1.45x | n/a (tested in gtest) |
| GPU | layernorm | 8192x4096 | 0.615 | 0.641 | 0.96x | n/a (tested in gtest) |
