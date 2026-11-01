"""Compare the hand-written kernels (via pybind11 module `ikernels`) against PyTorch.

CPU: both sides take/return host tensors; PyTorch uses MKL/oneDNN with the same thread count.
GPU: CUDA-event time over repeated launches, with inputs already on the device.
Usage: python python/bench_vs_torch.py [--build build] [--out results/torch_compare.md]
"""
import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch


def median_ms(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def torch_cuda_ms(fn, iters):
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--out", default="results/torch_compare.md")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.build))
    import ikernels as ik

    rng = np.random.default_rng(0)
    rows = []
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    hdr = f"CPU: {platform.processor() or platform.machine()} | torch {torch.__version__} | threads {torch.get_num_threads()}"

    # ---------- CPU ----------
    for n in (256, 512, 1024):
        A = rng.standard_normal((n, n), dtype=np.float32)
        B = rng.standard_normal((n, n), dtype=np.float32)
        tA, tB = torch.from_numpy(A), torch.from_numpy(B)
        err = float(np.abs(ik.matmul(A, B) - (tA @ tB).numpy()).max())
        ours = median_ms(lambda: ik.matmul(A, B), 15)
        ref = median_ms(lambda: tA @ tB, 15)
        rows.append(("CPU", "matmul", f"{n}^3", ours, ref, err))
    for r, c in ((4096, 1024), (8192, 4096)):
        x = rng.standard_normal((r, c), dtype=np.float32)
        g = rng.uniform(0.5, 1.5, c).astype(np.float32)
        b = rng.standard_normal(c, dtype=np.float32)
        tx, tg, tb = torch.from_numpy(x), torch.from_numpy(g), torch.from_numpy(b)
        err = float(np.abs(ik.softmax(x) - torch.softmax(tx, -1).numpy()).max())
        rows.append(("CPU", "softmax", f"{r}x{c}", median_ms(lambda: ik.softmax(x), 15),
                     median_ms(lambda: torch.softmax(tx, -1), 15), err))
        ln = lambda: torch.nn.functional.layer_norm(tx, (c,), tg, tb, 1e-5)
        err = float(np.abs(ik.layernorm(x, g, b) - ln().numpy()).max())
        rows.append(("CPU", "layernorm", f"{r}x{c}", median_ms(lambda: ik.layernorm(x, g, b), 15),
                     median_ms(ln, 15), err))

    # ---------- GPU ----------
    gpu = ik.has_cuda and ik.cuda_device_available() and torch.cuda.is_available()
    if gpu:
        hdr += f" | GPU: {torch.cuda.get_device_name(0)}"
        torch.backends.cuda.matmul.allow_tf32 = False  # fair FP32 comparison
        for n in (1024, 2048, 4096):
            tA = torch.randn(n, n, device="cuda")
            tB = torch.randn(n, n, device="cuda")
            it = 5 if n >= 4096 else 20
            ref = torch_cuda_ms(lambda: tA @ tB, it)
            for v, name in ((1, "matmul_smem_tiled"), (2, "matmul_reg_tiled")):
                rows.append(("GPU", name, f"{n}^3", ik.cuda_bench_matmul(v, n, n, n, it), ref, float("nan")))
        for r, c in ((4096, 1024), (8192, 4096)):
            tx = torch.randn(r, c, device="cuda")
            tg, tb = torch.ones(c, device="cuda"), torch.zeros(c, device="cuda")
            rows.append(("GPU", "softmax", f"{r}x{c}", ik.cuda_bench_softmax(r, c, 50),
                         torch_cuda_ms(lambda: torch.softmax(tx, -1), 50), float("nan")))
            rows.append(("GPU", "layernorm", f"{r}x{c}", ik.cuda_bench_layernorm(r, c, 50),
                         torch_cuda_ms(lambda: torch.nn.functional.layer_norm(tx, (c,), tg, tb), 50), float("nan")))

    lines = [hdr, "", "| device | op | shape | ours ms | torch ms | ours / torch time | max abs err vs torch |",
             "|---|---|---|---|---|---|---|"]
    for d, op, shp, o, t, e in rows:
        es = "n/a (tested in gtest)" if e != e else f"{e:.2e}"
        lines.append(f"| {d} | {op} | {shp} | {o:.3f} | {t:.3f} | {o / t:.2f}x | {es} |")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
