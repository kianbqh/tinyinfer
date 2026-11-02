"""Config-driven equal-budget search. Calls real bounded build + GPU evaluator."""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from .core import RandomProposer,gate,median,proposal,search
from .proposer import LocalLlamaProposer
from server.metadata import canonical,digest,metadata,now,output_directory,write

TOOLCHAIN={"cmake","ninja","nvcc","cxx","python","python_include","gtest_source"}


def validate(c):
    keys={"schema_version","seeds","proposal_budget","evaluation_budget","baseline_id","fixture","llama","toolchain","scratch_base","build_timeout_seconds","evaluation_timeout_seconds"}
    if set(c)!=keys or type(c["schema_version"]) is not int or c["schema_version"]!=1:raise ValueError("autotuner config schema")
    if not isinstance(c["seeds"],list) or len(set(c["seeds"]))<3 or len(c["seeds"])!=len(set(c["seeds"])) or any(type(x) is not int or not 0<=x<2**31 for x in c["seeds"]):raise ValueError("at least three unique fixed search seeds")
    for k,lo,hi in (("proposal_budget",1,10000),("evaluation_budget",1,11),("baseline_id",0,11),("build_timeout_seconds",1,86400),("evaluation_timeout_seconds",1,86400)):
        if type(c[k]) is not int or not lo<=c[k]<=hi:raise ValueError(k)
    if c["proposal_budget"]<c["evaluation_budget"]:raise ValueError("proposal budget")
    f=c["fixture"]
    if set(f)!={"fixture_seed","M","N","K","warmups","repeats","host_repeats","rtol","atol"}:raise ValueError("fixture schema")
    for k,lo,hi in (("fixture_seed",0,2**31-1),("M",1,4096),("N",1,4096),("K",1,4096),("warmups",1,1000),("repeats",7,10000),("host_repeats",7,10000)):
        if type(f[k]) is not int or not lo<=f[k]<=hi:raise ValueError(k)
    for k in ("rtol","atol"):
        if type(f[k]) not in (int,float) or not math.isfinite(f[k]) or not 0<f[k]<=0.1:raise ValueError(k)
    if set(c["toolchain"])!=TOOLCHAIN or any(not isinstance(v,str) or not v for v in c["toolchain"].values()):raise ValueError("toolchain paths")
    l=c["llama"]
    if set(l)!={"executable","model","executable_sha256","model_sha256","max_output_tokens","context_tokens","timeout_seconds"}:raise ValueError("llama schema")
    for key in ("executable_sha256","model_sha256"):
        if not isinstance(l[key],str) or len(l[key])!=64 or any(ch not in "0123456789abcdef" for ch in l[key]):raise ValueError(key)
    for key,lo,hi in (("max_output_tokens",32,1024),("context_tokens",4096,32768),("timeout_seconds",1,3600)):
        if type(l[key]) is not int or not lo<=l[key]<=hi:raise ValueError(key)
    for key in ("executable","model"):
        if not isinstance(l[key],str) or not l[key]:raise ValueError(key)
    scratch=str(c["scratch_base"])
    if not (scratch.startswith("/opt/codex-resume/all-modules-v2/m2/") or scratch.startswith("/mnt/g/ResumeProject/.runtime/all-modules-v2/m2/")):raise ValueError("G-backed coordinated M2 scratch required")
    # Resolve '..' before passing the allow-listed helper a unique scratch path.
    resolved=str(Path(scratch).resolve())
    if os.name=="posix" and not (resolved.startswith("/opt/codex-resume/all-modules-v2/m2/") or resolved.startswith("/mnt/g/ResumeProject/.runtime/all-modules-v2/m2/")):raise ValueError("scratch escapes coordinated root")
    return c


def retained_run(argv,directory,phase,timeout):
    log=directory/(phase+".log");record=dict(argv=argv,status="started",timestamp=now());write(directory/(phase+".json"),record)
    started=time.perf_counter()
    try:
        with log.open("w",encoding="utf-8") as handle:
            result=subprocess.run(argv,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        record.update(exit_code=result.returncode,status="completed" if not result.returncode else "failed")
        return result.returncode
    except Exception as e:
        record.update(status="failed",error_type=type(e).__name__,error=str(e));raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started,finished_timestamp=now())
        if log.exists():record["log_sha256"]=digest(log)
        write(directory/(phase+".json"),record)


class RealEvaluator:
    def __init__(self,config,out,run_name):
        self.config=config;self.out=Path(out);self.run_name=run_name
        self.run_root=next(p for p in self.out.parents if p.name==run_name and p.parent.name=="v2")
    def __call__(self,value,label):
        directory=self.out/("candidate-"+str(label));directory.mkdir(parents=True,exist_ok=False)
        path=directory/"proposal.json";write(path,value)
        scratch=Path(self.config["scratch_base"])/self.run_name/directory.relative_to(self.run_root)
        argv=[sys.executable,"-m","bench.v2.kernels.candidate_build","--proposal",str(path),"--scratch",str(scratch),
              "--timeout-seconds",str(self.config["build_timeout_seconds"])]
        for k,v in self.config["toolchain"].items():argv.extend(["--"+k.replace("_","-"),v])
        record=dict(proposal=value,status="compile_failed",correctness_pass=False,build_scratch=str(scratch),timestamp=now())
        try:
            code=retained_run(argv,directory,"build-helper",2*self.config["build_timeout_seconds"]+60)
            journal=scratch/"candidate.json"
            if journal.exists():record["build_journal"]=json.loads(journal.read_text(encoding="utf-8"))
            if code:return record
            record["status"]="test_failed"
            manifest=scratch/"build-manifest.json"
            fixture=directory/"fixture.json";write(fixture,self.config["fixture"])
            evaluation=directory/"gpu-evaluation"
            argv=[sys.executable,"-m","autotune.evaluate","--config",str(fixture),"--proposal",str(path),
                  "--build-manifest",str(manifest),"--output",str(evaluation)]
            code=retained_run(argv,directory,"gpu-helper",self.config["evaluation_timeout_seconds"])
            result_path=evaluation/"evaluation.json"
            if not result_path.exists():raise RuntimeError("GPU evaluator produced no terminal record")
            result=json.loads(result_path.read_text(encoding="utf-8"))
            if (not code)!=(result.get("status")=="passed"):raise RuntimeError("GPU evaluator exit/status mismatch")
            result["evaluation_record_path"]=str(result_path);result["evaluation_record_sha256"]=digest(result_path)
            result["build_scratch"]=str(scratch);record.update(result)
            return record
        except Exception as e:
            record.update(error_type=type(e).__name__,error=str(e));return record
        finally:write(directory/"driver.json",record)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    config_path=Path(a.config).resolve(strict=True);config=validate(json.loads(config_path.read_text(encoding="utf-8")))
    if os.name!="posix":raise ValueError("actual autotuning requires the coordinated WSL2/Linux execution window")
    root=Path(__file__).resolve().parents[1];out=output_directory(a.output,root);start=time.perf_counter()
    journal=dict(status="started",timestamp=now(),command_argv=sys.argv,config=config,config_sha256=digest(config_path),
                 config_canonical_sha256=canonical(config),environment=metadata(root,gpu=True),seeds=[],
                 budget_policy="per method: same proposal cap and same unique nonbaseline evaluation cap; invalid/duplicate/error proposals consume proposal attempts; compile/test failures consume unique evaluations; initial baseline and independent remeasurement are separately reported",
                 proposer_gpu_policy="local llama.cpp CPU process (-ngl 0), fully exits before each compile/test/timing process; no external LLM server allowed")
    write(out/"search.json",journal)
    try:
        for seed in config["seeds"]:
            directory=out/f"seed-{seed}";directory.mkdir()
            evaluate=RealEvaluator(config,directory,out.name)
            baseline=evaluate(proposal(config["baseline_id"]),"initial-baseline")
            if baseline.get("status")!="passed":raise RuntimeError("initial baseline failed correctness/build; search not meaningful")
            record=dict(seed=seed,baseline=baseline,methods={});journal["seeds"].append(record);write(out/"search.json",journal)
            # Alternate order by seed index to avoid always measuring LLM second.
            order=("random","llm") if len(journal["seeds"])%2 else ("llm","random")
            for method in order:
                method_dir=directory/method;method_dir.mkdir()
                proposer=RandomProposer(seed) if method=="random" else LocalLlamaProposer(config["llama"],method_dir/"proposals",seed)
                candidate_evaluator=RealEvaluator(config,method_dir,out.name)
                outcome=search(proposer,candidate_evaluator,baseline,proposal_budget=config["proposal_budget"],evaluation_budget=config["evaluation_budget"],
                               save=lambda value:write(method_dir/"search.json",value))
                record["methods"][method]=outcome;write(out/"search.json",journal)
                # Remeasure both matched initial baseline and selected winner
                # in fresh processes. These do not count as search evaluations.
                rebaseline=candidate_evaluator(baseline["proposal"],"remeasure-baseline")
                winner=candidate_evaluator(outcome["best"]["proposal"],"remeasure-winner")
                outcome["independent"]={"baseline":rebaseline,"winner":winner,"status":"failed"}
                if rebaseline.get("status")==winner.get("status")=="passed":
                    if rebaseline["fixture"]!=winner["fixture"] or baseline["fixture"]!=winner["fixture"]:raise ValueError("fixture changed across candidates")
                    outcome["independent"].update(status="passed",speedup=median(rebaseline["samples_ms"])/median(winner["samples_ms"]),
                                                  passes_3_percent_gate=gate(winner,rebaseline))
                write(method_dir/"search.json",outcome);write(out/"search.json",journal)
            record["comparable"]=all(r["budget_exhausted"] and r["independent"]["status"]=="passed" for r in record["methods"].values())
            write(out/"search.json",journal)
        journal["comparable"]=all(r["comparable"] for r in journal["seeds"])
        journal["summary"]={}
        if journal["comparable"]:
            for method in ("random","llm"):
                speeds=[r["methods"][method]["independent"]["speedup"] for r in journal["seeds"]]
                journal["summary"][method]={"independent_speedups":speeds,"mean":statistics.mean(speeds),"std_population":statistics.pstdev(speeds)}
            journal["summary"]["llm_mean_exceeds_random"]=journal["summary"]["llm"]["mean"]>journal["summary"]["random"]["mean"]
        journal["status"]="completed" if journal["comparable"] else "incomplete_incomparable"
    except Exception as e:journal.update(status="failed",error_type=type(e).__name__,error=str(e))
    finally:
        journal.update(finished_timestamp=now(),wall_seconds=time.perf_counter()-start);write(out/"search.json",journal)
    return 0 if journal["status"]=="completed" else 1

if __name__=="__main__":raise SystemExit(main())
