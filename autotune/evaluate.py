"""Actual one-candidate CUDA evaluation. Never imported for CPU gate tests."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from server.metadata import digest,now,write
from bench.v2.kernels.candidate_build import validate_proposal
from bench.v2.kernels.run import environment,measure
from bench.v2.kernels.correctness import run_correctness


def main():
    p=argparse.ArgumentParser()
    for name in ("config","proposal","build-manifest","output"):p.add_argument("--"+name,required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.config).read_text(encoding="utf-8"));value=json.loads(Path(a.proposal).read_text(encoding="utf-8"))
    identifier=validate_proposal(value);receipt=json.loads(Path(a.build_manifest).read_text(encoding="utf-8"))
    record=dict(status="started",timestamp=now(),proposal=value,correctness_pass=False,config_sha256=digest(a.config),build_manifest_sha256=digest(a.build_manifest))
    write(out/"evaluation.json",record);started=time.perf_counter()
    try:
        if receipt.get("candidate")!=value:raise ValueError("compiled library does not match candidate")
        import torch
        from llm.native_ops import NativeOps
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.manual_seed(config["fixture_seed"]);torch.cuda.manual_seed_all(config["fixture_seed"])
        ops=NativeOps(receipt["library_path"])
        if ops.catalog["only_config"]!=identifier:raise ValueError("candidate library must contain exactly one configuration")
        record["environment"]=environment(torch,Path(__file__).resolve().parents[1],Path(receipt["library_path"]),a.build_manifest)
        write(out/"evaluation.json",record)
        with torch.inference_mode():
            correctness=run_correctness(ops,torch,rtol=config["rtol"],atol=config["atol"])
            write(out/"correctness.json",correctness)
            if correctness["status"]!="passed":
                record.update(status="test_failed",error="full native correctness gate failed");return 1
            # Reset generator after gates: fixture bytes must match every config,
            # method, candidate, selection and independent remeasurement.
            torch.manual_seed(config["fixture_seed"]);torch.cuda.manual_seed_all(config["fixture_seed"])
            m,n,k=config["M"],config["N"],config["K"]
            lhs=torch.randn((m,k),device="cuda",dtype=torch.float16)*0.25
            rhs=torch.randn((k,n),device="cuda",dtype=torch.float16)*0.25
            bias=torch.randn(n,device="cuda",dtype=torch.float16)*0.1
            if bias.count_nonzero().item()==0:raise ValueError("fusion comparison requires a real nonzero bias")
            hashes={name:hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest() for name,tensor in (("lhs",lhs),("rhs",rhs),("bias",bias))}
            record["fixture"]={"M":m,"N":n,"K":k,"seed":config["fixture_seed"],"sha256":hashes,"bias":"FP16 nonzero N-vector; every candidate computes lhs@rhs+bias, FP32 accumulate, FP16 output"}
            ref=lhs.float()@rhs.float()+bias.float();actual=ops.gemm(lhs,rhs,bias=bias,config=identifier)
            torch.testing.assert_close(actual.float(),ref,rtol=config["rtol"],atol=config["atol"])
            record["fixture_max_abs_error"]=(actual.float()-ref).abs().max().item();del actual,ref
            timing=measure(torch,lambda:ops.gemm(lhs,rhs,bias=bias,config=identifier),config)
            record.update(status="passed",correctness_pass=True,samples_ms=timing["cuda_event_ms_raw"],timing=timing)
            del lhs,rhs,bias;torch.cuda.synchronize();gc.collect()
    except Exception as e:record.update(status="test_failed",error_type=type(e).__name__,error=str(e))
    finally:
        record.update(finished_timestamp=now(),wall_seconds=time.perf_counter()-started);write(out/"evaluation.json",record)
    return 0 if record["status"]=="passed" else 1

if __name__=="__main__":raise SystemExit(main())
