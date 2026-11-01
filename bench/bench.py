"""CPU latency: torch eager vs tinyc O0 (no optimization passes) vs tinyc O2 (all passes), plus pass ablation.

Each config: R repeats x N timed calls after warmup; we report the median per-call latency of each repeat,
then the median / min / max across repeats. Same thread count for torch and OpenMP."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

THREADS = int(os.environ.get("TINYC_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tinyc.compile import compile_model  # noqa: E402
from tinyc.models import MODELS, randomize_bn  # noqa: E402
from tinyc.passes import PASSES  # noqa: E402

torch.set_num_threads(THREADS)


def time_fn(fn, n, repeats, warmup=20):
    for _ in range(warmup):
        fn()
    meds = []
    for _ in range(repeats):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
        meds.append(statistics.median(ts) / 1e3)
    return {"median_us": statistics.median(meds), "min_us": min(meds), "max_us": max(meds), "repeats": meds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/bench_cpu.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    loadavg_start = os.getloadavg()
    configs = [("mlp", 1), ("mlp", 64), ("cnn", 1), ("cnn", 32)]
    rows = []
    for name, bs in configs:
        Model, in_shape = MODELS[name]
        m = randomize_bn(Model()).eval()
        x = torch.randn(bs, *in_shape)
        xn = x.numpy().copy()
        with torch.no_grad():
            ref = m(x).numpy()
        out = np.empty_like(ref)
        res = {"model": name, "batch": bs}
        with torch.inference_mode():
            res["torch_eager"] = time_fn(lambda: m(x), a.n, a.repeats)
        variants = {"tinyc_O0": dict(opt=False), "tinyc_O2": dict(opt=True)}
        for drop in ("fuse_epilogue", "fuse_elementwise"):     # ablation: O2 minus one pass
            variants[f"O2_no_{drop}"] = dict(opt=True, passes=[p for p in PASSES if p.__name__ != drop])
        for vname, kw in variants.items():
            cm = compile_model(m, x, tag=f"{name}_b{bs}_{vname}", **kw)
            actual = cm(xn)
            np.testing.assert_allclose(actual, ref, rtol=0, atol=1e-4)
            err = float(np.abs(actual - ref).max())
            r = time_fn(lambda: cm.run_into(xn, out), a.n, a.repeats)
            r.update(max_abs_err=err, kernels=cm.info["kernels"], arena_bytes=cm.info["arena_bytes"])
            res[vname] = r
        rows.append(res)
        e, o0, o2 = res["torch_eager"]["median_us"], res["tinyc_O0"]["median_us"], res["tinyc_O2"]["median_us"]
        print(f"{name:4s} bs={bs:3d}  eager {e:9.1f}us  O0 {o0:9.1f}us  O2 {o2:9.1f}us  "
              f"O2 vs eager {e / o2:5.2f}x  O2 vs O0 {o0 / o2:5.2f}x  "
              f"kernels {res['tinyc_O0']['kernels']}->{res['tinyc_O2']['kernels']}  "
              f"arena {res['tinyc_O0']['arena_bytes']}->{res['tinyc_O2']['arena_bytes']} B  "
              f"err {res['tinyc_O2']['max_abs_err']:.1e}", flush=True)
    meta = {"threads": THREADS, "torch": torch.__version__, "cpu": os.popen("lscpu | grep 'Model name'").read().strip(),
            "loadavg_start": loadavg_start, "repeats": a.repeats, "n": a.n, "warmup": 20, "seed": 0,
            "protocol": "Host perf_counter around each Python eager call or ctypes run_into; FP32 input preallocated for both; eager returns a new Tensor, tinyc reuses a preallocated output without copying the return value"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    # markdown table
    md = ["| model | batch | torch eager (us) | tinyc O0 (us) | tinyc O2 (us) | O2 no epilogue-fusion (us) | O2 no ew-fusion (us) | O2 vs eager | O2 vs O0 | kernels O0->O2 | arena O0->O2 (KB) | max abs err |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r: f"{r['median_us']:.1f} [{r['min_us']:.1f}-{r['max_us']:.1f}]"
    for r in rows:
        md.append(f"| {r['model']} | {r['batch']} | {f(r['torch_eager'])} | {f(r['tinyc_O0'])} | {f(r['tinyc_O2'])} | "
                  f"{f(r['O2_no_fuse_epilogue'])} | {f(r['O2_no_fuse_elementwise'])} | "
                  f"{r['torch_eager']['median_us'] / r['tinyc_O2']['median_us']:.2f}x | "
                  f"{r['tinyc_O0']['median_us'] / r['tinyc_O2']['median_us']:.2f}x | "
                  f"{r['tinyc_O0']['kernels']}->{r['tinyc_O2']['kernels']} | "
                  f"{r['tinyc_O0']['arena_bytes'] / 1024:.0f}->{r['tinyc_O2']['arena_bytes'] / 1024:.0f} | "
                  f"{r['tinyc_O2']['max_abs_err']:.1e} |")
    md.append(f"\nthreads={THREADS}, {a.repeats} repeats x {a.n} calls; cell = median of per-repeat medians [min-max]. "
              f"loadavg at start {meta['loadavg_start']}.")
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
