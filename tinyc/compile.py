"""Build generated code into a shared library and wrap it in a callable."""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import fcntl
from pathlib import Path

import numpy as np
import torch

from .capture import capture
from .codegen_cpp import generate_cpp
from .codegen_cuda import generate_cuda
from .passes import PASSES, lower_required, optimize
from .dispatch import apply_dispatch

RUNTIME = Path(__file__).parent / "runtime"
CACHE = Path(os.environ.get("TINYC_CACHE", Path(__file__).resolve().parent.parent / "build" / "kernels"))


def cxx() -> str:
    return os.environ.get("TINYC_CXX") or os.environ.get("CXX") or shutil.which("g++") or "c++"


class CompiledModel:
    def __init__(self, lib_path: Path, blob: np.ndarray, in_shape, out_shape, info: dict, graph):
        # dlopen reuses a previously loaded file, including its generated globals.
        # A real copy gives each handle independent weights and activation storage.
        # On Linux the mapped library remains valid after its temporary file is removed.
        with tempfile.TemporaryDirectory(prefix="loaded_", dir=lib_path.parent) as directory:
            instance_path = Path(directory) / lib_path.name
            shutil.copyfile(lib_path, instance_path)
            self.lib = ctypes.CDLL(str(instance_path))
        self.blob = np.ascontiguousarray(blob, np.float32)   # must outlive the library handle
        self.in_shape, self.out_shape, self.info, self.graph = tuple(in_shape), tuple(out_shape), info, graph
        self.lib.tinyc_init.argtypes = [ctypes.c_void_p]
        self.lib.tinyc_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.tinyc_forward.restype = None
        if self.lib.tinyc_init(self.blob.ctypes.data) != 0:
            raise RuntimeError("tinyc_init failed")
        self._out = np.empty(self.out_shape, np.float32)

    def __call__(self, x) -> np.ndarray:
        x = np.ascontiguousarray(x.numpy() if isinstance(x, torch.Tensor) else x, np.float32)
        assert x.shape == self.in_shape, (x.shape, self.in_shape)
        self.lib.tinyc_forward(x.ctypes.data, self._out.ctypes.data)
        return self._out.copy()

    def run_into(self, x: np.ndarray, out: np.ndarray):           # zero-copy path for benchmarking
        self.lib.tinyc_forward(x.ctypes.data, out.ctypes.data)


def nvcc() -> str:
    return os.environ.get("TINYC_NVCC") or shutil.which("nvcc") or "nvcc"


def nvcc_cmd(cu: Path, out: Path, compile_only: bool = False) -> list[str]:
    arch = os.environ.get("TINYC_CUDA_ARCH", "sm_86")      # RTX 3080 = Ampere sm_86
    cmd = [nvcc(), "-O3", f"-arch={arch}", "-std=c++17", "--use_fast_math", f"-I{RUNTIME}", str(cu), "-o", str(out)]
    if os.environ.get("TINYC_CXX"):
        cmd += ["-ccbin", os.environ["TINYC_CXX"], "-allow-unsupported-compiler"]
    return cmd + (["-c"] if compile_only else ["-shared", "-Xcompiler", "-fPIC"])


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compile failed:\n" + " ".join(cmd) + "\n" + r.stderr[-4000:])


def build(src: str, tag: str, backend: str = "cpp", compile_only: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    tool = cxx() if backend == "cpp" else nvcc()
    rt = "".join(f.read_text() for f in sorted(RUNTIME.iterdir()) if f.is_file())   # runtime edits invalidate cache
    h = hashlib.sha1((src + rt + tool + str(compile_only) + Path(__file__).read_text()).encode()).hexdigest()[:12]
    ext = ".so"
    so = CACHE / f"{tag}_{h}{ext}"
    if not so.exists():
        srcf = CACHE / f"{tag}_{h}.{'cpp' if backend == 'cpp' else 'cu'}"
        srcf.write_text(src)
        import ikernels
        library = Path(ikernels.__file__).parent / 'libtinyinfer_backend.so'
        if not library.is_file():
            raise RuntimeError('Build the unified CMake backend before compiling a model')
        build_dir = CACHE / f'cmake_{tag}_{h}'
        # Both the native runtime and generated models use the repository's single CMake project.
        so = so.with_suffix('.so')
        cmd = [shutil.which('cmake') or 'cmake', '-S', str(RUNTIME.parents[1]), '-B', str(build_dir),
               '-DCMAKE_BUILD_TYPE=Release', f'-DTINYC_JIT_SOURCE={srcf}', f'-DTINYC_JIT_NAME={so.stem}',
               f'-DTINYC_JIT_OUTPUT={CACHE}', f'-DTINYC_BACKEND_LIBRARY={library}',
               f'-DCMAKE_CXX_COMPILER={cxx()}', f'-DTINYC_CUDA={"ON" if backend == "cuda" else "OFF"}']
        if backend == 'cuda':
            cmd += [f'-DCMAKE_CUDA_COMPILER={nvcc()}', '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler']
        # Reuse compiler discovery across shapes; serialize reconfiguration/builds.
        shared_build = CACHE / f'cmake_{backend}'
        cmd[cmd.index('-B') + 1] = str(shared_build)
        with (CACHE / f'cmake_{backend}.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.exists():
                _run(cmd)
                _run([shutil.which('cmake') or 'cmake', '--build', str(shared_build), '-j', '2'])
    return so


def lower(model, example, opt=True, passes=None, log=None):
    g = capture(model, example)
    if opt:
        g = optimize(g, passes, log)
        if passes is not None:          # ablations: lower whatever the chosen passes left unfused
            g = lower_required(g)
    else:
        g = lower_required(g)
    return g


def compile_model(model: torch.nn.Module, example: torch.Tensor, opt: bool = True, passes=None, tag="model",
                  log: list | None = None, backend: str = "cpp", compile_only: bool = False, implementation: str = 'codegen'):
    """backend="cpp" -> OpenMP C++ .so; backend="cuda" -> nvcc .so (needs a GPU to run).
    compile_only=True checks compilation/linking without loading or executing the generated shared library;
    it returns that shared-library artifact path."""
    g = lower(model, example, opt, passes, log)
    decisions = apply_dispatch(g, backend, implementation)
    gen = generate_cpp if backend == "cpp" else generate_cuda
    src, blob, info = gen(g, reuse_memory=opt)
    so = build(src, f"{tag}_{backend}" + ("_O2" if opt else "_O0"), backend, compile_only)
    info["source"] = src
    info['dispatch'] = decisions
    info['implementation'] = implementation
    if compile_only:
        return so, info
    return CompiledModel(so, blob, example.shape, g.node(g.outputs[0]).shape if g.outputs[0] in [n.name for n in g.nodes]
                         else g.consts[g.outputs[0]].shape, info, g)
