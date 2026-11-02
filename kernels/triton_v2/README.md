# Triton comparison

`ops.py` implements tiled FP16 GEMM with FP32 accumulation and a fused last-axis FP16 softmax with FP32 reductions. It uses a small fixed configuration allow-list, masks edge tiles, allocates output on the input CUDA device and launches on the current PyTorch stream. It is inference only.

Run the new `bench.v2.kernels.run` driver from the root after the CUDA correctness gate. It compares each Triton path with the custom CUDA path and PyTorch, storing raw samples/configuration/environment under a new `results/v2/<run-id>`. JIT compilation happens in warmup/correctness, outside the timed samples; cold compile cost is not a steady-state latency claim. All caches must point to the coordinated private G-backed runtime.

## Measured results

Measured in WSL2 on an i7-12700KF and RTX 3080 12 GB, with PyTorch 2.15.0+cu132, Triton 3.9.0 and a standalone CUDA 12.4.131 build at source `3edcfeb`. Displayed decimals are rounded; linked JSON retains full precision. Seed `20261101`; concurrency 1; 10 warmups, 50 CUDA-event samples and 20 separately synchronized host samples per case. Events surround one device-resident Python wrapper call and include enqueue gaps. Host samples include wrapper dispatch, allocation, stream handling and synchronization. Attention also uploads validated lengths/offsets on each call. These are separate timing windows, not hardware-only kernel latency.

The [GEMM comparison](../src/cuda/tc/README.md#gemm) holds c0 fixed alongside cuBLAS, WMMA and the legacy FP32 adapter. [All configurations and raw GEMM samples](../../results/v2/20261102T132951Z-m2-native-headers-gemm-cpath/run.json) retain c0/c1/c2 without selecting the fastest configuration after measurement.

| Shape | Softmax path | Event p50 (ms) | Host p50 (ms) |
| --- | --- | --- | --- |
| 128×128 | PyTorch | 0.013360 | 0.029576 |
| 128×128 | Custom CUDA | 0.069632 | 0.071710 |
| 128×128 | Triton | 0.072656 | 0.048528 |
| 128×512 | PyTorch | 0.020480 | 0.030346 |
| 128×512 | Custom CUDA | 0.028672 | 0.043818 |
| 128×512 | Triton | 0.040960 | 0.047194 |
| 128×2048 | PyTorch | 0.017408 | 0.038586 |
| 128×2048 | Custom CUDA | 0.028512 | 0.045108 |
| 128×2048 | Triton | 0.031744 | 0.050560 |
| 128×4096 | PyTorch | 0.016224 | 0.035743 |
| 128×4096 | Custom CUDA | 0.029696 | 0.045192 |
| 128×4096 | Triton | 0.037888 | 0.048824 |

[All 12 softmax cases and raw samples](../../results/v2/20261102T132951Z-m2-native-headers-softmax-cpath/run.json). The completed GEMM retry fixed a direct GCC `Python.h` search failure with existing child-only `CPATH` headers; [the original partial failure](../../results/v2/20261102T132951Z-m2-native-headers-gemm/run.json) is retained. JIT warmup is outside steady-state samples.

## Limitations

Fused softmax is limited to last-axis sizes 1..32768; GEMM supports only contiguous row-major FP16 input. The custom and Triton softmax paths lost to PyTorch on every tested shape; the complete raw distributions remain linked above.
