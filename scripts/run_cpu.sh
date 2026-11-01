#!/usr/bin/env bash
source "$(dirname "$0")/build_backend.sh"
if [[ ${TINYC_SKIP_TESTS:-0} != 1 ]]; then
  ctest --test-dir "$BUILD_DIR" --output-on-failure
  "$PYTHON" -m pytest --capture=sys -q tests kernels/tests/test_python.py
fi
"$PYTHON" bench/integrated.py --device cpu
