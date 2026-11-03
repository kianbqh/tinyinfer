#!/usr/bin/env bash
set -euo pipefail
source /mnt/g/ResumeProject/.runtime/all-modules-v2/m2/runtime/actual_runtime_env_3ed.sh
export M2_RUN_ID=20261102T230756Z-m2-autotune-transportfix
export M2_CHECKPOINT=91ae0821a1a1e1129beeabbed0aa6ff1d2e4e947
export M2_SCRATCH="$M2_ROOT/runs/$M2_RUN_ID"
export M2_LOGS="$M2_SCRATCH/commands"
export CPATH=/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include/python3.12:/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include
export CXXFLAGS=-I/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include
cd "$M2_REPO"
# Exactly one manually supplied recorder command; no queue, retry or daemon.
m2_record_runtime "$@"
