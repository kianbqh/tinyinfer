"""Compile one allow-listed GEMM choice in a unique private scratch directory.

This driver never accepts model-generated commands, C++/CUDA code or compiler
flags. GPU correctness/timing is a subsequent explicit runner stage. Every
configure/build attempt (including failures/timeouts) is retained.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from llm.native_ops import GEMM_CATALOG
from .protocol import build_source_paths,canonical_hash,sha256
from .run import command,git_info,write


def validate_proposal(value):
    if not isinstance(value,dict) or set(value)!={"config_id","tile","stages","epilogue"}:
        raise ValueError("proposal schema: config_id,tile,stages,epilogue only")
    identifier=value["config_id"]
    if type(identifier) is not int or not 0<=identifier<len(GEMM_CATALOG):raise ValueError("config_id")
    expected=GEMM_CATALOG[identifier]
    if (not isinstance(value["tile"],list) or any(type(x) is not int for x in value["tile"]) or
        tuple(value["tile"])!=expected["tile"] or type(value["stages"]) is not int or
        value["stages"]!=expected["stages"] or value["epilogue"]!=expected["epilogue"]):
        raise ValueError("proposal tuple must match the exact compiled catalog ID")
    return identifier


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--proposal",required=True)
    for field in ("scratch","cmake","ninja","nvcc","cxx","python","python-include","gtest-source"):
        parser.add_argument("--"+field,required=True)
    parser.add_argument("--timeout-seconds",type=int,default=900)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    proposal_path=Path(args.proposal).resolve(strict=True)
    proposal=json.loads(proposal_path.read_text(encoding="utf-8"));identifier=validate_proposal(proposal)
    scratch=Path(args.scratch).resolve()
    # The execution environment is WSL with a G-backed native /opt disk or /mnt/g.
    # Native Windows use is unsupported; no fallback to a C:-hosted temp directory.
    if os.name!="posix" or not (str(scratch).startswith("/opt/codex-resume/all-modules-v2/m2/") or str(scratch).startswith("/mnt/g/ResumeProject/.runtime/all-modules-v2/m2/")):
        raise ValueError("candidate scratch must be inside the coordinated G-backed M2 runtime")
    scratch.mkdir(parents=True,exist_ok=False)
    build=scratch/"build";build.mkdir();tmp=scratch/"tmp";tmp.mkdir()
    journal={"status":"started","timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),
             "proposal":proposal,"proposal_sha256":sha256(proposal_path),"proposal_canonical_sha256":canonical_hash(proposal),
             "git":git_info(root),"candidate_id":identifier,"attempts":[]}
    destination=scratch/"candidate.json";write(destination,journal)
    started=time.perf_counter();environment=dict(os.environ,TMPDIR=str(tmp))
    nvcc=Path(args.nvcc).resolve()
    configure=[str(Path(args.cmake).resolve()),"-S",str(root),"-B",str(build),"-G","Ninja",
               "-DCMAKE_BUILD_TYPE=Release","-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
               f"-DCMAKE_MAKE_PROGRAM={Path(args.ninja).resolve()}",
               "-DTINYC_CUDA=ON","-DTINYC_V2_CUDA=ON",f"-DTINYC_V2_GEMM_CONFIG={identifier}",
               f"-DCMAKE_CUDA_COMPILER={nvcc}",f"-DCUDAToolkit_ROOT={nvcc.parent.parent}",
               f"-DCMAKE_CXX_COMPILER={Path(args.cxx).resolve()}",
               f"-DPython3_EXECUTABLE={Path(args.python).resolve()}",
               f"-DPython3_INCLUDE_DIR={Path(args.python_include).resolve()}",
               "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
               f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={Path(args.gtest_source).resolve()}"]
    build_command=[configure[0],"--build",str(build),"--target","tinyinfer_backend","--parallel","2"]
    try:
        source_paths=build_source_paths(root)
        journal["source_sha256"]={str(p.relative_to(root)):sha256(p) for p in sorted(source_paths)}
        for phase,argv in (("configure",configure),("build",build_command)):
            record={"phase":phase,"argv":argv,"status":"started"};journal["attempts"].append(record);write(destination,journal)
            before=time.perf_counter();log=scratch/(phase+".log")
            try:
                with log.open("w",encoding="utf-8") as handle:
                    result=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT,env=environment,timeout=args.timeout_seconds,check=False)
                record.update({"exit_code":result.returncode,"status":"completed" if result.returncode==0 else "failed"})
                if result.returncode:raise RuntimeError(f"{phase} failed with exit code {result.returncode}")
            finally:
                record["wall_seconds"]=time.perf_counter()-before
                if log.exists():record["log_sha256"]=sha256(log)
                write(destination,journal)
        library=build/"libtinyinfer_backend.so"
        commands_path=build/"compile_commands.json"
        commands=json.loads(commands_path.read_text(encoding="utf-8"))
        selected=[r for r in commands if "/tc/" in r["file"].replace("\\","/") or r["file"].endswith("runtime_v2.cu")]
        if len(selected)!=4:raise RuntimeError("expected exactly four v2 compile records")
        final_source={str(p.relative_to(root)):sha256(p) for p in sorted(source_paths)}
        if final_source!=journal["source_sha256"]:raise RuntimeError("kernel/CMake source changed while candidate was building")
        receipt={"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"git":journal["git"],
                 "candidate":proposal,"library_sha256":sha256(library),"library_path":str(library),
                 "cuda_toolkit_version":command([str(nvcc),"--version"]),"compiler_version":command([args.cxx,"--version"]),
                 "build_command":build_command,"build_exit_code":0,"build_log_sha256":sha256(scratch/"build.log"),
                 "compile_commands":selected,"compile_commands_sha256":sha256(commands_path),
                 "cmake_cache_sha256":sha256(build/"CMakeCache.txt"),"source_sha256":final_source}
        write(scratch/"build-manifest.json",receipt)
        journal["status"]="compiled_not_gpu_tested"
    except Exception as error:
        journal.update({"status":"failed","error_type":type(error).__name__,"error":str(error)})
        for record in journal["attempts"]:
            if record["status"]=="started":record["status"]="failed"
    finally:
        journal["wall_seconds"]=time.perf_counter()-started
        journal["finished_timestamp"]=dt.datetime.now(dt.timezone.utc).isoformat();write(destination,journal)
    return 0 if journal["status"]=="compiled_not_gpu_tested" else 1


if __name__=="__main__":raise SystemExit(main())
