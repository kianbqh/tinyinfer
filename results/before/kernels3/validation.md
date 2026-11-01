# Local validation

Run on 2026-10-31 in Ubuntu 24.04 under WSL2, with GCC 13.3.0, CUDA Toolkit 12.4.131, and an NVIDIA GeForce RTX 3080 (compute capability 8.6).

| Build | Check | Result |
|---|---|---|
| CPU only | `ctest --test-dir build --output-on-failure` | 12 / 12 passed |
| CPU only | `python -m pytest -q --capture=sys tests/test_python.py` | 16 passed in 4.90 s |
| CUDA enabled | `ctest --test-dir build-gpu --output-on-failure` | 22 / 22 passed: 12 CPU and 10 CUDA |
| CUDA enabled | `IK_BUILD_DIR=build-gpu python -m pytest -q --capture=sys tests/test_python.py` | 16 passed in 1.06 s |

All 10 CUDA tests executed on the GPU, with no skipped tests. They compare all three matmul implementations with an FP64 reference, including non-tile-aligned dimensions, and check masked softmax, large-offset LayerNorm, and constant rows. The Python suite exercises the NumPy/CPU operations in both builds and checks them against PyTorch.

The CPU C++ suite also checks that the reference comparison rejects NaN and infinity. The local run used eight OpenMP threads and the Release configuration. The scripts in [`scripts/`](../scripts/) reproduce the build, correctness checks, and benchmarks.
