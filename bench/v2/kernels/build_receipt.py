"""Capture an already completed build; does not configure, compile or run CUDA.

Caller supplies the actual build command and retained complete successful log.
Configure CMake with CMAKE_EXPORT_COMPILE_COMMANDS=ON before building.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
from .protocol import build_source_paths,sha256
from .run import command,git_info,write


def main():
    parser=argparse.ArgumentParser()
    for argument in ("library","build-directory","build-log","build-command","nvcc","cxx","output"):
        parser.add_argument("--"+argument,required=True)
    parser.add_argument("--build-exit-code",required=True,type=int)
    args=parser.parse_args()
    if args.build_exit_code!=0:raise ValueError("a failed build cannot produce an accepted library receipt; retain its log separately")
    root=Path(__file__).resolve().parents[3]
    path=Path(args.output).resolve()
    if path.exists():raise FileExistsError(path)
    library=Path(args.library).resolve(strict=True);build=Path(args.build_directory).resolve(strict=True)
    compile_path=build/"compile_commands.json"
    commands=json.loads(compile_path.read_text(encoding="utf-8"))
    v2=[row for row in commands if any(name in row["file"].replace("\\","/") for name in (
        "backend/runtime_v2.cu","kernels/src/cuda/tc/gemm.cu","kernels/src/cuda/tc/attention.cu","kernels/src/cuda/tc/elementwise.cu"))]
    if len(v2)!=4:raise ValueError("expected exactly four additive v2 CUDA compilation records")
    sources=build_source_paths(root)
    receipt={"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"git":git_info(root),
             "library_path":str(library),"library_sha256":sha256(library),
             "cuda_toolkit_version":command([args.nvcc,"--version"]),"compiler_version":command([args.cxx,"--version"]),
             "build_directory":str(build),"build_command":args.build_command,"build_exit_code":args.build_exit_code,
             "build_log_path":str(Path(args.build_log).resolve(strict=True)),"build_log_sha256":sha256(args.build_log),
             "compile_commands_sha256":sha256(compile_path),"compile_commands":v2,
             "cmake_cache_sha256":sha256(build/"CMakeCache.txt"),
             "source_sha256":{str(p.relative_to(root)):sha256(p) for p in sorted(sources)}}
    path.parent.mkdir(parents=True,exist_ok=True);write(path,receipt)


if __name__=="__main__":main()
