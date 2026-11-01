#!/usr/bin/env bash
source "$(dirname "$0")/build_backend.sh"
if [[ ${TINYC_CUDA:-OFF} == ON ]]; then
  "$PYTHON" -c 'import torch,ikernels; assert torch.cuda.is_available() and ikernels.has_cuda and ikernels.cuda_device_available(), "Actual CUDA hardware is required"'
fi
ctest --test-dir "$BUILD_DIR" --output-on-failure
"$PYTHON" -m pytest --capture=sys -q tests kernels/tests/test_python.py
