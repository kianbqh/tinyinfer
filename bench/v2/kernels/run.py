"""Config-driven, non-overwriting v2 GPU kernel evidence runner.

Run with python -m bench.v2.kernels.run --config ... --library ...
--output results/v2/<unique-id>. Never call the frozen v1 runners for v2.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .protocol import build_source_paths,canonical_hash,sha256,summary,unique_output,validate_config


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def command(argv,cwd=None):
    return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()


def git_info(root):
    executable=os.environ.get("TINYC_GIT") or shutil.which("git")
    if not executable:raise RuntimeError("Set TINYC_GIT to an existing Git executable; no unverified SHA fallback")
    directory=str(root)
    if executable.lower().endswith(".exe") and platform.system()=="Linux":
        directory=command(["wslpath","-w",directory])
    base=[executable,"-C",directory]
    return {"executable":executable,"version":command([executable,"--version"]),
            "sha":command(base+["rev-parse","HEAD"]),"status_porcelain":command(base+["status","--porcelain"])}


def environment(torch,root,library,build_manifest):
    versions={}
    for name in ("torch","triton","numpy","pybind11"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    source_paths=build_source_paths(root)+[root/"llm/native_ops.py",root/"kernels/src/cuda/matmul.cu"]
    for relative in ("kernels/src/cuda/tc","kernels/triton_v2","bench/v2/kernels"):
        source_paths.extend(p for p in (root/relative).rglob("*") if p.suffix in (".py",".cu",".cuh",".h"))
    receipt=json.loads(Path(build_manifest).read_text(encoding="utf-8"))
    required=("library_sha256","cuda_toolkit_version","compiler_version","build_command","compile_commands","build_log_sha256","source_sha256")
    if any(not receipt.get(key) for key in required) or receipt["library_sha256"]!=sha256(library):
        raise ValueError("build manifest must identify this exact library and actual toolkit/compiler/commands/log")
    for relative,digest in receipt["source_sha256"].items():
        source=(root/relative).resolve()
        if not source.is_relative_to(root) or sha256(source)!=digest:
            raise ValueError(f"kernel source no longer matches the measured build: {relative}")
    properties=torch.cuda.get_device_properties(torch.cuda.current_device())
    try:cpu=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError,StopIteration):cpu=platform.processor()
    runtime="Docker" if Path("/.dockerenv").exists() else ("WSL2" if "microsoft" in platform.release().lower() else platform.system())
    return {"environment":runtime,
            "platform":platform.platform(),"cpu":cpu,"python":sys.version,"python_executable":sys.executable,
            "versions":versions,"torch_version":torch.__version__,"torch_cuda":torch.version.cuda,
            "gpu":properties.name,"compute_capability":[properties.major,properties.minor],
            "device_total_bytes":properties.total_memory,
            "nvidia_smi":command(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]),
            "git":git_info(root),"library_path":str(library),"library_sha256":sha256(library),
            "build_manifest_path":str(Path(build_manifest).resolve()),"build_manifest_sha256":sha256(build_manifest),"build_manifest":receipt,
            "source_sha256":{str(p.relative_to(root)):sha256(p) for p in sorted(set(source_paths))},
            "current_device":torch.cuda.current_device(),"current_stream":torch.cuda.current_stream().cuda_stream,
            "float32_matmul_precision":torch.get_float32_matmul_precision(),
            "allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "allow_fp16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}


def measure(torch,fn,config):
    for _ in range(config["warmups"]):
        out=fn();del out
    torch.cuda.synchronize()
    pairs=[]
    # Events are allocated before timing; each interval encloses one dispatched call.
    for _ in range(config["repeats"]):
        pairs.append((torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)))
    for begin,end in pairs:
        begin.record();out=fn();end.record();del out
    torch.cuda.synchronize()
    device=[start.elapsed_time(end) for start,end in pairs]
    host=[]
    for _ in range(config["host_repeats"]):
        torch.cuda.synchronize();start=time.perf_counter_ns()
        out=fn();torch.cuda.synchronize()
        host.append((time.perf_counter_ns()-start)/1e6);del out
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    out=fn();torch.cuda.synchronize()
    memory={"baseline_torch_allocated_bytes":baseline,"peak_torch_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_additional_torch_allocated_bytes":torch.cuda.max_memory_allocated()-baseline}
    del out
    return {"cuda_event_ms_raw":device,"cuda_event":summary(device),"host_synchronized_ms_raw":host,
            "host_synchronized":summary(host),"memory":memory,
            "timer_includes":"CUDA event interval around one device-resident Python wrapper invocation; may include enqueue gaps; host timer includes wrapper, allocation, stream recording and device synchronization",
            "attention_metadata":"host validated lengths copied to CUDA on current stream in each custom call",
            "memory_scope":"additional peak Torch allocator bytes after input creation; excludes driver/context, compiled code, shared/register storage and non-Torch allocations",
            "concurrency":1,"warmups":config["warmups"],"repeats":config["repeats"],"host_repeats":config["host_repeats"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--library",required=True)
    parser.add_argument("--build-manifest",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--group",choices=("all","gemm","attention","softmax"),default="all")
    parser.add_argument("--correctness-only",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    config_path=Path(args.config).resolve();config=validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    out=unique_output(args.output,root)
    start=time.perf_counter()
    journal={"status":"started","timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"command":shlex.join(sys.argv),
             "config":config,"config_sha256":sha256(config_path),"config_canonical_sha256":canonical_hash(config),
             "group":args.group,"results":[]}
    write(out/"run.json",journal)
    try:
        import torch
        from llm.native_ops import NativeOps
        from .correctness import run_correctness
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend,sdpa_kernel
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        torch.manual_seed(config["seed"]);torch.cuda.manual_seed_all(config["seed"])
        torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        lib=Path(args.library).resolve(strict=True);ops=NativeOps(lib)
        journal["environment"]=environment(torch,root,lib,args.build_manifest)
        write(out/"run.json",journal)
        with torch.inference_mode():
            gates=run_correctness(ops,torch,rtol=config["rtol"],atol=config["atol"])
            write(out/"correctness.json",gates)
            if gates["status"]!="passed":raise RuntimeError("kernel correctness gate failed; benchmarks not run")
            triton_ops=None
            try:
                from kernels.triton_v2 import ops as triton_ops
            except Exception as error:
                journal["results"].append({"method":"triton_import","status":"failed","error":str(error)})

            def case(name,shape,fn,reference,*,optional=False,flops=None):
                entry={"method":name,"shape":shape,"status":"started","timestamp":dt.datetime.now(dt.timezone.utc).isoformat()}
                journal["results"].append(entry);write(out/"run.json",journal)
                before=time.perf_counter()
                try:
                    actual=fn();expected=reference()
                    torch.testing.assert_close(actual.float(),expected.float(),rtol=config["rtol"],atol=config["atol"])
                    entry["max_abs_error"]=(actual.float()-expected.float()).abs().max().item()
                    entry["rtol"],entry["atol"]=config["rtol"],config["atol"]
                    del actual,expected
                    entry.update(measure(torch,fn,config));entry["status"]="completed"
                    if flops is not None:entry["tflops_at_event_p50"]=flops/(entry["cuda_event"]["p50_ms"]*1e9)
                except Exception as error:
                    unsupported=optional and isinstance(error,RuntimeError) and any(
                        text in str(error).lower() for text in ("no available kernel","no viable backend","not compiled","not supported","not available"))
                    entry.update({"status":"not_run" if unsupported else "failed","error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
                finally:
                    entry["wall_seconds"]=time.perf_counter()-before;write(out/"run.json",journal)

            if not args.correctness_only and args.group in ("all","gemm"):
                for n in config["gemm_sizes"]:
                    a=torch.randn((n,n),device="cuda",dtype=torch.float16)*0.25;b=torch.randn_like(a)*0.25
                    af=a.float();bf=b.float();ref=lambda:af@bf
                    shape={"M":n,"N":n,"K":n}
                    case("cublas_torch_matmul_fp16",shape,lambda:a@b,ref,flops=2*n**3)
                    case("legacy_cuda_fp32_explicit_stream_adapter",shape,lambda:ops.legacy_gemm(af,bf),ref,flops=2*n**3)
                    for c in config["gemm_configs"]:
                        case(f"wmma_fp16_fp32acc_fp16out_c{c}",shape,lambda c=c:ops.gemm(a,b,config=c),ref,flops=2*n**3)
                        case(f"wmma_fp16_fp32acc_fp32out_c{c}",shape,lambda c=c:ops.gemm(a,b,config=c,output_dtype=torch.float32),ref,flops=2*n**3)
                        if triton_ops and c<3:
                            case(f"triton_fp16_fp32acc_fp16out_c{c}",shape,lambda c=c:triton_ops.gemm(a,b,config=c),ref,flops=2*n**3)
                    scales=bf.abs().amax(dim=0).div(127).clamp_min(1e-8)
                    quant=(bf/scales).round().clamp(-127,127).to(torch.int8)
                    dequant=(quant.float()*scales).half()
                    for c in config["gemm_configs"]:
                        case(f"wmma_int8_weightonly_fp32acc_c{c}",shape,lambda c=c:ops.gemm_int8(a,quant,scales,config=c),lambda:af@dequant.float(),flops=2*n**3)
                    del a,b,af,bf,scales,quant,dequant

            if not args.correctness_only and args.group in ("all","softmax"):
                for rows,cols in config["softmax_shapes"]:
                    x=torch.randn((rows,cols),device="cuda",dtype=torch.float16)
                    shape={"rows":rows,"cols":cols};ref=lambda:torch.softmax(x.float(),dim=-1)
                    case("torch_softmax_fp16",shape,lambda:torch.softmax(x,dim=-1),ref)
                    case("cuda_softmax_fp16",shape,lambda:ops.softmax(x),ref)
                    if triton_ops:case("triton_softmax_fp16",shape,lambda:triton_ops.softmax(x),ref)
                    del x

            if not args.correctness_only and args.group in ("all","attention"):
                for d in config["head_dims"]:
                    for n in config["attention_lengths"]:
                        shape={"B":config["attention_batch"],"H":config["attention_heads"],"Q":n,"K":n,"D":d,"causal":True}
                        q=torch.randn((shape["B"],shape["H"],n,d),device="cuda",dtype=torch.float16)*0.5
                        k=torch.randn_like(q)*0.5;v=torch.randn_like(q)
                        def sdpa(backend):
                            with sdpa_kernel(backend):return F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
                        def ref():return sdpa(SDPBackend.MATH)
                        mask=torch.ones((n,n),device=q.device,dtype=torch.bool).triu(1)
                        def naive():
                            scores=(q@k.transpose(-1,-2))*(d**-0.5)
                            scores.masked_fill_(mask,-float("inf"))
                            return torch.softmax(scores,dim=-1)@v
                        case("sdpa_forced_math",shape,ref,ref)
                        case("sdpa_forced_flash",shape,lambda:sdpa(SDPBackend.FLASH_ATTENTION),ref,optional=True)
                        case("sdpa_forced_efficient",shape,lambda:sdpa(SDPBackend.EFFICIENT_ATTENTION),ref,optional=True)
                        case("naive_materialized_fp16_attention",shape,naive,ref)
                        for c in config["attention_configs"]:
                            case(f"cuda_online_attention_c{c}",shape,lambda c=c:ops.attention(q,k,v,config=c),ref)
                        del q,k,v,mask
        for row in journal["results"]:
            if row.get("status")=="completed" and "M" in row.get("shape",{}):
                for baseline,label in (("cublas_torch_matmul_fp16","percent_of_cublas_fp16"),
                                       ("legacy_cuda_fp32_explicit_stream_adapter","speedup_vs_legacy_fp32")):
                    match=next((r for r in journal["results"] if r["method"]==baseline and r.get("shape")==row["shape"] and r.get("status")=="completed"),None)
                    if match:
                        ratio=match["cuda_event"]["p50_ms"]/row["cuda_event"]["p50_ms"]
                        row[label]=ratio*(100 if label.startswith("percent") else 1)
        journal["status"]="completed" if all(r["status"] in ("completed","not_run") for r in journal["results"]) else "partial_failure"
    except Exception as error:
        journal.update({"status":"failed","error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()})
    finally:
        journal["finished_timestamp"]=dt.datetime.now(dt.timezone.utc).isoformat()
        journal["wall_seconds"]=time.perf_counter()-start;write(out/"run.json",journal)
    return 0 if journal["status"]=="completed" else 1


if __name__=="__main__":raise SystemExit(main())
