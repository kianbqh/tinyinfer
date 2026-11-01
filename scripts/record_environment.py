"""Save the machine and toolchain details alongside a benchmark run."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch


def version(command):
    return subprocess.check_output(command, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build-gpu")
    parser.add_argument("--out", default="results/environment.json")
    args = parser.parse_args()
    cache = {}
    for line in (Path(args.build) / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    cpu = platform.processor() or platform.machine()
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    cuda_compiler = cache.get("CMAKE_CUDA_COMPILER", "")
    cuda_build = cache.get("IK_ENABLE_CUDA") == "ON" and bool(cuda_compiler) and not cuda_compiler.endswith("NOTFOUND")
    nvcc = cache.get("CMAKE_CUDA_COMPILER") if cuda_build else None
    report = {
        "run_date": datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat(),
        "platform": platform.platform(),
        "cpu": cpu,
        "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "close"),
        "compiler": version([cache.get("CMAKE_CXX_COMPILER", "g++"), "--version"]).splitlines()[0],
        "cmake": version([cache.get("CMAKE_COMMAND", "cmake"), "--version"]).splitlines()[0],
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pybind11", "pytest")},
        "torch_build": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": version([nvcc, "--version"]) if nvcc else None,
        "build": {
            "directory": args.build,
            "type": cache.get("CMAKE_BUILD_TYPE"),
            "native_cpu": cache.get("IK_NATIVE") == "ON",
            "cuda_enabled": cuda_build,
            "cuda_architectures": cache.get("IK_CUDA_ARCH") if cuda_build else None,
        },
        "timing": {
            "cpu": "Median wall-clock latency after warmup",
            "gpu": "CUDA events over repeated launches; device-resident inputs; transfers excluded",
            "gpu_submission": "Custom launches use a C++ loop; PyTorch launches use a Python loop; event intervals include gaps between launches",
            "torch_matmul_tf32": False,
        },
    }
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        report.update({
            "gpu": gpu.name,
            "gpu_memory_bytes": gpu.total_memory,
            "compute_capability": f"{gpu.major}.{gpu.minor}",
        })
        smi = shutil.which("nvidia-smi")
        if smi:
            report["nvidia_driver"] = version([smi, "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
