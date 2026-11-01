#!/usr/bin/env bash
# CPU pipeline: tests (incl. CUDA compile check if nvcc is on PATH) -> pass log -> CPU benchmark. Writes results/.
# Env: PYTHON (needs torch, numpy, scipy, pytest), TINYC_CXX (C++ compiler, default g++), TINYC_THREADS (default 4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
mkdir -p results
{
  echo "== $(date -Is)  $(nproc) cpus  load: $(cut -d' ' -f1-3 /proc/loadavg)"
  "$PY" -c "import torch; print('torch', torch.__version__)"
  echo "== pytest"
  "$PY" -m pytest --capture=sys -q tests
  echo "== pass log"
  "$PY" - <<'PYEOF'
import torch
from tinyc import capture, optimize
from tinyc.models import MODELS, randomize_bn
torch.manual_seed(0)
for name, (M, shp) in MODELS.items():
    m = randomize_bn(M()).eval(); x = torch.randn(1, *shp)
    g = capture(m, x); log = []; go = optimize(g, log=log)
    print(f"--- {name}: captured ops {g.op_counts()}")
    for p, a, b in log: print(f"  {p:18s} nodes {a:3d} -> {b:3d}")
    print(go)
PYEOF
  echo "== CPU benchmark"
  "$PY" bench/bench.py --repeats "${REPEATS:-7}" --n "${N:-200}"
  cat results/bench_cpu.md
} 2>&1 | tee results/run_all.log
