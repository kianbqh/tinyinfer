#!/usr/bin/env bash
# Run on a machine with an NVIDIA GPU (e.g. RTX 3080 under WSL2 with the CUDA toolkit installed).
# Optional env: PYTHON (default python3), TINYC_CUDA_ARCH (default sm_86), TINYC_NVCC, TINYC_CXX (host compiler for nvcc).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
nvidia-smi --query-gpu=name,driver_version --format=csv
"$PY" bench/bench_gpu.py
echo "wrote results/bench_gpu.json"
