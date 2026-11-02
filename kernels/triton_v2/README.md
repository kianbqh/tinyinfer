# Triton comparison

`ops.py` implements tiled FP16 GEMM with FP32 accumulation and a fused last-axis FP16 softmax with FP32 reductions. It uses a small fixed configuration allow-list, masks edge tiles, allocates output on the input CUDA device and launches on the current PyTorch stream. It is inference only.

Run the new `bench.v2.kernels.run` driver from the root after the CUDA correctness gate. It compares each Triton path with the custom CUDA path and PyTorch, storing raw samples/configuration/environment under a new `results/v2/<run-id>`. JIT compilation happens in warmup/correctness, outside the timed samples; cold compile cost is not a steady-state latency claim. All caches must point to the coordinated private G-backed runtime.

## Limitations

This code has not been imported, JIT-compiled or GPU-executed during private preparation. Fused softmax is limited to last-axis sizes 1..32768; GEMM supports only contiguous row-major FP16 input. No performance result is implied by implementation or installed Triton metadata.
