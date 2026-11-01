# History and source inputs

The compiler checkout was cloned locally with full history from `12_tiny_ml_compiler/tiny-ml-compiler` at `4aa6f4c99d383c3e51e0d9fd1364c4eea6343a85`:

```text
git clone --local --no-hardlinks G:/ResumeProject/12_tiny_ml_compiler/tiny-ml-compiler G:/ResumeProject/platforms/tinyinfer
git subtree add --prefix=kernels G:/ResumeProject/3_cpp_cuda_inference_kernels/cpp-cuda-inference-kernels main
```

The subtree command did not use squash. Its merge commit is `5020d7d35be684ed41f86b5787649f481a1ca267`; the imported kernel source is `2dc7ffae9a66897f71fd3f7a007dba18f500355b`. Both original main commits remain ancestors, verified with `git merge-base --is-ancestor`. Their original timestamps and authors were retained. Integration changes follow that merge normally.

Earlier measurements are archived under `results/before/`, with kernel measurements under `results/before/kernels3/`. They are provenance snapshots, separate from the new integrated runs. In particular, the original eight-thread CPU GEMM 1024³ record is 15.228 ms custom versus 3.874 ms Torch/MKL, a 3.93× time ratio; the archived 256³/512³ ratios are 7.59×/3.64×. There is no fixed sixfold claim.

Inputs are synthetic FP32 tensors and seeded model parameters; integration does not download a dataset or perform training. Fresh reports record input and parameter SHA256 values to establish that all three execution paths used identical values.
