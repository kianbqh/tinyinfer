#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo=$(pwd -P)
PYTHON=${PYTHON:-python3}
BUILD_DIR=${BUILD_DIR:-$repo/build}
mkdir -p "$BUILD_DIR"
BUILD_DIR=$(cd "$BUILD_DIR" && pwd -P)
args=(-S "$repo" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON" -DTINYC_CUDA="${TINYC_CUDA:-OFF}")
if [[ ${TINYC_CUDA:-OFF} == ON ]]; then
  args+=(-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler)
fi
cmake "${args[@]}"
cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-4}"
export PYTHONPATH="$BUILD_DIR:$repo${PYTHONPATH:+:$PYTHONPATH}" IK_BUILD_DIR="$BUILD_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} TINYC_THREADS=${TINYC_THREADS:-8}
