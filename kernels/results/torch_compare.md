CPU: x86_64 | torch 2.15.0+cu132 | threads 8

| device | op | shape | ours ms | torch ms | ours / torch time | max abs err vs torch |
|---|---|---|---|---|---|---|
| CPU | matmul | 256^3 | 0.266 | 0.035 | 7.59x | 0.00e+00 |
| CPU | matmul | 512^3 | 1.884 | 0.518 | 3.64x | 8.39e-05 |
| CPU | matmul | 1024^3 | 15.228 | 3.874 | 3.93x | 2.29e-04 |
| CPU | softmax | 4096x1024 | 2.177 | 1.187 | 1.83x | 5.22e-08 |
| CPU | layernorm | 4096x1024 | 0.868 | 0.786 | 1.10x | 7.39e-06 |
| CPU | softmax | 8192x4096 | 20.181 | 15.157 | 1.33x | 6.33e-08 |
| CPU | layernorm | 8192x4096 | 11.894 | 12.651 | 0.94x | 1.57e-05 |
