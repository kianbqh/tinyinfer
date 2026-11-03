#!/usr/bin/env bash
set -euo pipefail
source /mnt/g/ResumeProject/.runtime/all-modules-v2/m2/runtime/actual_runtime_env_3ed.sh
export M2_RUN_ID=20261102T234500Z-m2-autotune-compact-history
export M2_CHECKPOINT=6b1b9f665052e05715b15427f0a9a4777d0414fb
if [[ ! "$M2_CHECKPOINT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Root final clean checkpoint and explicit execution release required" >&2
  exit 78
fi
export M2_SCRATCH="$M2_ROOT/runs/$M2_RUN_ID"
export M2_LOGS="$M2_SCRATCH/commands"
export CPATH=/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include/python3.12:/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include
export CXXFLAGS=-I/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include
cd "$M2_REPO"
# Exactly one manually supplied recorder command; no queue, retry or daemon.
m2_record_runtime "$@"
