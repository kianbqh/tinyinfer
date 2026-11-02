"""Local llama.cpp CPU process. Outputs are data; model text is never executed."""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path
from .core import proposal
from server.metadata import digest,now,write


class LocalLlamaProposer:
    def __init__(self,config,directory,seed):
        self.config=config;self.directory=Path(directory);self.seed=seed
        self.directory.mkdir(parents=True,exist_ok=False)
    def __call__(self,history,attempt):
        out=self.directory/f"proposal-{attempt:04d}";out.mkdir(exist_ok=False)
        # Preserve previous raw correctness/timing outcomes; never substitute an
        # invented result table when the evaluator has failed.
        rows=[dict(proposal=history["baseline"]["proposal"],status="initial_baseline",samples_ms=history["baseline"]["samples_ms"])]
        for row in history["attempts"][:-1]:
            compact={k:row[k] for k in ("proposal","status","error") if k in row}
            evaluation=row.get("evaluation",{})
            compact.update({k:evaluation[k] for k in ("samples_ms","correctness_pass","error") if k in evaluation})
            rows.append(compact)
        prompt=("Choose one GEMM configuration using the previous measured outcomes. Return exactly one JSON object, no markdown, prose, code, commands or extra keys. "
                "Use config_id, tile, stages, epilogue exactly as one catalog entry. Avoid previously proposed IDs. "
                "Lower median samples_ms is better; correctness failures cannot be accepted. All candidates compute the same GEMM with a nonzero bias.\n"
                +"CATALOG="+json.dumps([proposal(i) for i in range(12)])+"\nPREVIOUS_RESULTS="+json.dumps(rows,allow_nan=False)+"\nJSON:\n")
        prompt_path=out/"prompt.txt";prompt_path.write_text(prompt,encoding="utf-8")
        executable=Path(self.config["executable"]).resolve(strict=True);model=Path(self.config["model"]).resolve(strict=True)
        # Fixed operator-owned flags only. No shell, remote endpoint, model code,
        # arbitrary grammar file or proposal-controlled argument is accepted.
        argv=[str(executable),"-m",str(model),"-ngl","0","--seed",str((self.seed+attempt)%(2**31)),
              "-n",str(self.config["max_output_tokens"]),"-c",str(self.config["context_tokens"]),
              "--temp","0.7","--no-display-prompt","--no-conversation","--log-disable","-f",str(prompt_path)]
        receipt=dict(status="started",timestamp=now(),argv=argv,prompt_sha256=digest(prompt_path),model_sha256=self.config["model_sha256"],
                     executable_sha256=self.config["executable_sha256"],gpu_policy="CPU proposer: fixed -ngl 0; subprocess fully exits before candidate build/GPU evaluation")
        write(out/"proposal.json",receipt);started=time.perf_counter()
        try:
            if digest(model)!=self.config["model_sha256"] or digest(executable)!=self.config["executable_sha256"]:raise ValueError("llama.cpp/model hash mismatch")
            # Disk output files retain complete stdout/stderr even after timeout.
            with (out/"stdout.txt").open("w",encoding="utf-8") as stdout,(out/"stderr.txt").open("w",encoding="utf-8") as stderr:
                completed=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=self.config["timeout_seconds"],check=False)
            receipt["exit_code"]=completed.returncode
            if completed.returncode:raise RuntimeError("local llama.cpp proposal process failed")
            if (out/"stdout.txt").stat().st_size>16384:raise ValueError("proposal output too large")
            raw=(out/"stdout.txt").read_text(encoding="utf-8").strip();receipt["status"]="returned_unvalidated"
            return raw
        except Exception as e:
            receipt.update(status="failed",error_type=type(e).__name__,error=str(e));raise
        finally:
            receipt.update(finished_timestamp=now(),wall_seconds=time.perf_counter()-started)
            for name in ("stdout.txt","stderr.txt"):
                if (out/name).exists():receipt[name+"_sha256"]=digest(out/name)
            write(out/"proposal.json",receipt)
