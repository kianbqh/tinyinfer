#!/usr/bin/env bash
# Build CPU kernels, run tests, and compare timings with PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OMP_PROC_BIND=close
export XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" results
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DIK_ENABLE_CUDA=OFF -DPython3_EXECUTABLE="$(command -v "$PY")" "$@"
cmake --build build -j "${IK_BUILD_JOBS:-8}"
(cd build && ctest --output-on-failure)
"$PY" -m pytest -q --capture=sys tests/test_python.py
./build/bench_cpu results/bench_cpu.csv | tee results/bench_cpu.md
"$PY" python/bench_vs_torch.py --out results/torch_compare.md
"$PY" scripts/record_environment.py --build build
