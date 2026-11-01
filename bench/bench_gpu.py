"""GPU numerical checks and CUDA-event timing (NVIDIA GPU + nvcc).

For each model/batch: build the generated .cu into a .so, check numerics vs PyTorch on CPU, then time
device-only forward with CUDA events (tinyc_bench) and, if torch has CUDA, torch eager on the same GPU."""
from __future__ import annotations

import ctypes
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tinyc.compile import compile_model  # noqa: E402
from tinyc.models import MODELS, randomize_bn  # noqa: E402


def main(out="results/bench_gpu.json", repeats=7, iters=200):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA execution is required for this benchmark")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.manual_seed(0)
    rows = []
    for name, bs in [("mlp", 1), ("mlp", 64), ("cnn", 1), ("cnn", 32)]:
        Model, shp = MODELS[name]
        m = randomize_bn(Model()).eval()
        x = torch.randn(bs, *shp)
        with torch.no_grad():
            ref = m(x).numpy()
        r = {"model": name, "batch": bs}
        for opt in (False, True):
            cm = compile_model(m, x, opt=opt, tag=f"gpu_{name}_b{bs}", backend="cuda")
            cm.lib.tinyc_bench.restype = ctypes.c_float
            cm.lib.tinyc_bench.argtypes = [ctypes.c_int]
            actual = cm(x)
            np.testing.assert_allclose(actual, ref, rtol=0, atol=1e-4)
            err = float(np.abs(actual - ref).max())
            ts = [cm.lib.tinyc_bench(iters) * 1e3 for _ in range(repeats)]
            assert all(np.isfinite(t) and t > 0 for t in ts), ts
            r["O2" if opt else "O0"] = {"median_us": statistics.median(ts), "min_us": min(ts), "max_us": max(ts),
                                        "max_abs_err": err, "kernels": cm.info["kernels"], "repeats_us": ts}
        if torch.cuda.is_available():
            md, xd = m.cuda(), x.cuda()
            with torch.inference_mode():
                torch_actual = md(xd).cpu().numpy()
            np.testing.assert_allclose(torch_actual, ref, rtol=0, atol=1e-4)
            ts = []
            with torch.inference_mode():
                for _ in range(20):
                    md(xd)
                for _ in range(repeats):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(iters):
                        md(xd)
                    b.record(); torch.cuda.synchronize()
                    ts.append(a.elapsed_time(b) / iters * 1e3)
            assert all(np.isfinite(t) and t > 0 for t in ts), ts
            r["torch_cuda_eager"] = {"median_us": statistics.median(ts), "min_us": min(ts), "max_us": max(ts),
                                     "max_abs_err": float(np.abs(torch_actual - ref).max()), "repeats_us": ts}
            m.cpu()
        rows.append(r)
        print(json.dumps(r))
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "seed": 0,
            "repeats": repeats, "iterations": iters, "tinyc_warmup_per_repeat": 10, "torch_warmup": 20,
            "tinyc_fast_math": True,
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "protocol": "Device-resident FP32 inputs; CUDA events surround C++ run_graph enqueue loops for tinyc and Python eager enqueue loops for torch; includes inter-launch gaps, excludes host/device transfers and compilation"}
    Path(out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
