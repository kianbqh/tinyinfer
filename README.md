# cpp-cuda-inference-kernels

[![CPU tests](https://github.com/kianbqh/cpp-cuda-inference-kernels/actions/workflows/tests.yml/badge.svg)](https://github.com/kianbqh/cpp-cuda-inference-kernels/actions/workflows/tests.yml)

I wanted to get a better feel for what happens underneath a tensor operation. So I wrote matmul, softmax, and LayerNorm in C++ and CUDA, starting with plain loops and adding tiling, parallelism, and fused reductions. This repo keeps the simple versions around so I can measure what each change buys me.

The kernels work on row-major FP32 arrays. A pybind11 module connects the CPU implementations to NumPy, and the benchmark compares them with PyTorch on the same machine.

## The implementations

| Operation | CPU | CUDA |
|---|---|---|
| Matmul | Plain `i-j-k` loops; then 64×256×256 cache tiles, `i-k-j` loop order, OpenMP, and SIMD | One output per thread; then 32×32 shared-memory tiles; then 64×64 output tiles with a 4×4 register tile per thread |
| Softmax | Max reduction, a fused exp/sum pass, and normalization across parallel rows | Online max/sum states merged with warp shuffles; one block per row |
| LayerNorm | FP64 two-pass baseline; shifted FP32 sums with OpenMP and SIMD | Shifted sums, warp/block reductions, and affine output in one kernel |

Two details were especially useful to test: subtracting a nearby value before accumulating variance keeps LayerNorm accurate around a large mean, and masked `-inf` logits need to contribute zero to the softmax reduction. The CUDA tests cover both, along with odd dimensions and constant rows.

## Run it

I use Linux through WSL2 on Windows. The CPU build needs a C++17 compiler with OpenMP, CMake 3.24+, and Python 3.10+ with its development headers. On Ubuntu, the system packages are `build-essential python3-dev python3-venv`. CMake downloads GoogleTest during configuration.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

bash scripts/run_cpu.sh
~~~

That builds the C++ library and Python module, runs both test suites, and writes the CPU benchmark tables to `results/`. The script uses eight OpenMP threads by default; use `OMP_NUM_THREADS=4 bash scripts/run_cpu.sh` to change it.

For CUDA, install a toolkit with `nvcc` and a CUDA build of PyTorch. With an RTX 3080:

~~~bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
IK_CUDA_ARCH=86 bash scripts/run_gpu.sh
~~~

The GPU script builds into `build-gpu`, runs the CUDA correctness tests, benchmarks all three matmul variants, and compares the optimized kernels with PyTorch. Set `CUDACXX=/path/to/cuda/bin/nvcc` when the toolkit is outside `PATH`.

After the environment is ready in WSL, PowerShell can launch either path:

~~~powershell
.\run.ps1 -Distribution Ubuntu -Mode cpu
.\run.ps1 -Distribution Ubuntu -Mode gpu -CudaHome /path/to/cuda
~~~

Replace `Ubuntu` with your installed WSL distribution. Keep the project on a drive mounted under `/mnt/<drive>`.

## Results from my machine

Measured on 2026-10-31 with an Intel Core i7-12700KF, eight OpenMP threads, and an RTX 3080 12 GB under Ubuntu 24.04 / WSL2. The builds use GCC 13.3, CUDA Toolkit 12.4, and PyTorch 2.15.0+cu132. The full configuration is in [`results/environment.json`](results/environment.json).

| Kernel | Shape | Plain version | Optimized version | Speedup |
|---|---|---:|---:|---:|
| CPU matmul | 1024³ | 2607.75 ms | 15.79 ms | 165.2× |
| CPU softmax | 8192×4096 | 96.41 ms | 15.78 ms | 6.1× |
| CPU LayerNorm | 8192×4096 | 38.07 ms | 7.27 ms | 5.2× |
| CUDA matmul | 4096³ | 83.635 ms | 16.652 ms | 5.0× |

The register-tiled CUDA matmul reaches 8.64 TFLOP/s at 2048³ and 8.25 TFLOP/s at 4096³. Keeping each thread's 4×4 output tile in registers makes a much larger difference here than shared-memory tiling alone.

I also timed the Python module against PyTorch, using the same thread count on CPU:

| Device / operation | Shape | Custom | PyTorch | Custom / PyTorch time |
|---|---|---:|---:|---:|
| CPU matmul | 1024³ | 15.228 ms | 3.874 ms | 3.93× |
| CPU softmax | 8192×4096 | 20.181 ms | 15.157 ms | 1.33× |
| CPU LayerNorm | 8192×4096 | 11.894 ms | 12.651 ms | 0.94× |
| CUDA register-tiled matmul | 4096³ | 16.707 ms | 7.467 ms | 2.24× |
| CUDA softmax | 8192×4096 | 0.612 ms | 0.422 ms | 1.45× |
| CUDA LayerNorm | 4096×1024 | 0.045 ms | 0.083 ms | 0.54× |
| CUDA LayerNorm | 8192×4096 | 0.615 ms | 0.641 ms | 0.96× |

A lower time ratio is faster. PyTorch wins the matmul and softmax comparisons; the fused LayerNorm is close on the larger shape and faster on the smaller GPU shape. I keep both comparisons because improving a plain loop and matching a tuned tensor library are different milestones.

CPU timings are medians after warmup. The C++ benchmarks reuse allocated outputs; the Python CPU comparison includes output allocation on both sides. The GPU tables use average CUDA-event time over repeated launches, with inputs already on the GPU and transfers excluded. The custom GPU benchmark enqueues launches from a C++ loop, while PyTorch uses a Python loop; the CUDA-event interval includes the gaps between those launches. FP32 matmul comparison uses TF32 disabled in PyTorch. Effective bandwidth counts one FP32 input read and one output write; the actual kernels make their own passes through those arrays.

The generated tables and CSV files live in [`results/`](results/). `bench_cpu` compares plain and optimized C++; `bench_cuda` compares the CUDA variants; `torch_compare` reports the custom/PyTorch time ratio for the same shapes.

## Correctness

The C++ tests compare against FP64 references and reject non-finite outputs. They cover non-tile-aligned matmuls, large logits, masked softmax, large-offset LayerNorm, and constant rows. Python tests compare the CPU module with PyTorch and check its shape validation.

The local CPU build passed 12 C++ tests and 16 Python tests. The CUDA build passed 22 C++ tests, including all 10 GPU tests, and the same 16 Python tests. [`results/validation.md`](results/validation.md) records the commands and counts.

GitHub Actions runs the CPU C++ and Python tests on every push. The CUDA tests run through `scripts/run_gpu.sh` on the local GPU.

## Around the repo

~~~text
include/ik/       Public CPU and CUDA interfaces
src/cpu/          Baseline and optimized C++ kernels
src/cuda/         CUDA kernels and event timing helpers
tests/            GoogleTest and PyTorch comparisons
bench/            CPU and GPU benchmark executables
python/           pybind11 module and PyTorch timing script
scripts/          CPU and GPU run scripts
results/          Measured timings and environment details
~~~
