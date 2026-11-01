# Hand-written kernels inside tinyinfer

These C++/CUDA kernels are built into the root shared backend and are called from generated model code through the C ABI in `backend/include/tinyinfer_backend.h`. The same implementation is exposed by the root pybind11 module and tested by the unified suite.

| Operation | CPU | CUDA |
|---|---|---|
| GEMM | Cache blocking, OpenMP and SIMD | Shared-memory and register tiling |
| Softmax | Row parallelism and fused passes | Block reductions and warp shuffles |
| LayerNorm | Shifted statistics and affine transform | Fused reduction, normalization and affine transform |

## Inputs and results

Inputs are seeded synthetic FP32 tensors, including ragged matrix dimensions, constant rows, large offsets, masks and non-finite edge cases. The integrated model experiments use seed-0 synthetic inputs and fixed random model parameters, with non-trivial LayerNorm affine values. There is no training dataset. Tests measure numerical agreement; benchmarks measure execution time and preserve raw repeats.

The current dispatch decisions and results are in the [root reports](../results/) and [README](../README.md). Generated code calls the backend only where the recorded policy selects it. The three-path model comparison uses identical inputs and weights. Graph-level GEMM results measure all substituted calls together.

Build and test from the repository root:

```bash
./scripts/run_gpu.sh
```

`./scripts/test.sh` runs the same unified correctness suite without timing. The root [workflow](https://github.com/kianbqh/tinyinfer/actions/workflows/tinyinfer.yml) owns CI. This module's former independent build and workflow have been replaced by the root build; its original history is preserved by the non-squashed subtree merge.
