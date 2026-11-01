#!/usr/bin/env bash
# For a machine with an NVIDIA GPU (e.g. RTX 3080 = sm_86) and CUDA toolkit >= 11.8.
# Requires a CUDA build of PyTorch for the torch comparison:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
PY=${PYTHON:-python3}
ARCH=${IK_CUDA_ARCH:-86}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OMP_PROC_BIND=close
export XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" results
"$PY" -c 'import torch; assert torch.cuda.is_available(), "Install a CUDA-enabled PyTorch build to run the GPU comparison"'
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DIK_ENABLE_CUDA=ON -DIK_CUDA_ARCH="$ARCH" \
      -DPython3_EXECUTABLE="$(command -v "$PY")"
cmake --build build-gpu -j "${IK_BUILD_JOBS:-8}"
(cd build-gpu && ctest --output-on-failure)          # CUDA tests run (not skipped) when a GPU is present
./build-gpu/bench_cuda results/bench_cuda.csv | tee results/bench_cuda.md
IK_BUILD_DIR=build-gpu "$PY" -m pytest -q --capture=sys tests/test_python.py
"$PY" python/bench_vs_torch.py --build build-gpu --out results/torch_compare_gpu.md
"$PY" scripts/record_environment.py --build build-gpu
