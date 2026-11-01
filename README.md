# tiny-ml-compiler

With tinyc, I can follow a small PyTorch model from its graph to the C++ or CUDA code it runs. The compiler captures a model with `torch.fx`, lowers it to a static IR, applies a handful of optimization passes, and generates shared libraries.

The two examples deliberately give the passes something to do: an MLP with an unused branch and a constant residual gate, and a CNN with BatchNorm and elementwise chains. Their weights are initialized with fixed seeds, so I can compare the generated code directly against PyTorch.

## How it works

| Stage | Implementation |
|---|---|
| Capture | FX tracing and shape propagation; parameters become IR constants |
| Optimize | BatchNorm folding, constant folding, dead-code elimination, GEMM epilogue fusion, elementwise fusion, weight layout, and liveness-based buffer reuse |
| C++ | OpenMP/SIMD GEMM, im2col convolution, generated epilogues, and a reusable activation arena |
| CUDA | 16×16 shared-memory GEMM tiles, im2col, generated epilogues and elementwise kernels |
| Run | A compiled library loaded through ctypes, with independent state for each model handle |

Compilation specializes FP32 inference to one input shape. Supported operations include Linear, Conv2d, BatchNorm2d, MaxPool2d, flatten, and common elementwise functions with scalar or same-shape operands. The implementation lives in `tinyc/`; the two runtime headers are under `tinyc/runtime/`.

## Run

Use Python with PyTorch, NumPy, SciPy and pytest, plus a C++17 compiler with OpenMP. CUDA execution also needs an NVIDIA GPU and nvcc.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch numpy scipy pytest
TINYC_THREADS=4 TINYC_CXX=g++ ./scripts/run_all.sh
./scripts/run_gpu.sh
```

`TINYC_CACHE` selects the build cache. `TINYC_NVCC`, `TINYC_CXX` and `TINYC_CUDA_ARCH` select CUDA tools and the target architecture; the default architecture is `sm_86`.

```python
import torch
from tinyc import compile_model
from tinyc.models import MLP

model = MLP().eval()
x = torch.randn(1, 784)
compiled = compile_model(model, x, opt=True)  # backend="cuda" selects CUDA
print(compiled(x))
print(compiled.info["source"])
```

## Local results

Measured October 31, 2026 on an Intel Core i7-12700KF and RTX 3080 12 GB, using Ubuntu 24.04 under WSL2, GCC 13.3.0, nvcc 12.4.131, driver 621.01, and PyTorch 2.15.0+cu132. **All 29 checks passed**: the 27 original tests, including CUDA compilation, plus two regressions confirming that same-graph model handles with different weights stay independent on C++ and CUDA.

Each benchmark uses seed 0, seven repeats and 200 forwards per repeat. Every compiled configuration was checked against the same CPU PyTorch reference with absolute tolerance `1e-4`; the largest observed error across CPU and GPU configurations was `5.96e-7`.

### CPU

Four threads for PyTorch and OpenMP. After 20 warmup calls, a host timer surrounds each call; the table reports the median of the seven per-repeat medians. PyTorch eager returns a new Tensor. tinyc's ctypes `run_into` reuses a preallocated output and does not copy a return value. Inputs are allocated before timing in both cases.

| Model | Batch | Eager (µs) | tinyc O0 (µs) | tinyc O2 (µs) | Eager/O2 |
|---|---:|---:|---:|---:|---:|
| MLP 784→512→512→10 | 1 | 41.7 | 39.9 | 18.6 | 2.24× |
| MLP | 64 | 357.5 | 509.7 | 366.3 | 0.98× |
| CNN | 1 | 108.6 | 80.7 | 62.8 | 1.73× |
| CNN | 32 | 996.8 | 1041.7 | 951.3 | 1.05× |

The pass log reduces MLP compute nodes from 10 to 3 and CNN nodes from 16 to 7. Activation storage falls from 14,464 to 4,096 bytes for MLP batch 1 and from 265,472 to 62,720 bytes for CNN batch 1. In the included CNN batch-1 ablations, removing epilogue fusion changed 62.8 to 71.1 µs; removing elementwise fusion changed it to 67.6 µs. Full ranges and both ablations are in [the CPU table](results/bench_cpu.md).

### GPU

| Model | Batch | CUDA eager (µs) | tinyc O0 (µs) | tinyc O2 (µs) |
|---|---:|---:|---:|---:|
| MLP | 1 | 128.04 | 93.18 | 59.04 |
| MLP | 64 | 135.57 | 118.20 | 70.52 |
| CNN | 1 | 311.21 | 129.08 | 70.76 |
| CNN | 32 | 288.29 | 186.47 | 131.92 |

CUDA events surround repeated forwards with inputs already on the device. tinyc enqueues them in a C++ loop; PyTorch uses a Python eager loop. The event windows include gaps between launches and exclude compilation and host/device transfers. tinyc warms up 10 forwards before each repeat; PyTorch warms up 20 before the repeats. Custom activation buffers are preallocated; eager executes its usual Tensor operations. These are per-forward event-window averages, separate from the Python `compiled(x)` call, which transfers input and output.

PyTorch matmul and cuDNN convolution use IEEE FP32 settings with TF32 disabled. tinyc is compiled with nvcc `--use_fast_math` and checked against the reference using the tolerance above. The legacy JSON field `kernels` counts IR compute nodes; each CUDA convolution issues both im2col and GEMM, making the CNN's actual launch count 18→9 for O0→O2.

Raw samples, numerical errors and timing metadata are in [CPU JSON](results/bench_cpu.json) and [GPU JSON](results/bench_gpu.json). The generated pass log is in [run_all.log](results/run_all.log).

## Checks

The tests cover capture and pass equivalence, dead-code and constant removal, BatchNorm folding, buffer reuse, C++ output for both optimization levels and fresh inputs, individual-pass ablations, ragged GEMMs, CUDA compilation, and model-instance isolation. The GPU benchmark additionally executes both CUDA optimization levels for all four model/batch cases. CI runs the CPU checks; the CUDA execution results above come from the local GPU.
