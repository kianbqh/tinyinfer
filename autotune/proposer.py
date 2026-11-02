"""Local llama.cpp CPU process. Outputs are data; model text is never executed."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from .core import proposal
from server.metadata import digest,now,write


# Exact llama.cpp ddf48f85c933c1f438212a02eb21e7b001d2d37d completion
# stdout framing: completion.cpp:927-930 emits EOG text; :949 emits two LFs.
# https://github.com/ggml-org/llama.cpp/blob/ddf48f85c933c1f438212a02eb21e7b001d2d37d/tools/completion/completion.cpp#L927-L949
COMPLETION_STDOUT_TRAILER = " [end of text]\n\n\n"


def decode_completion_stdout(text):
    """Remove at most one exact terminal CLI trailer; never interpret JSON."""
    removed = text.endswith(COMPLETION_STDOUT_TRAILER)
    return (text[:-len(COMPLETION_STDOUT_TRAILER)] if removed else text), removed


HISTORY_COLUMNS = ("attempt", "config_id", "status", "median_ms", "correctness_pass", "error_type")


def _history_median(record):
    samples = record.get("samples_ms")
    if (not isinstance(samples, list) or not samples or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in samples)):
        return None
    return statistics.median(samples)


def compact_history(history):
    """Keep every earlier attempt; full raw evidence stays in the journals."""
    baseline = history["baseline"]
    rows = [[-1, baseline["proposal"]["config_id"], "initial_baseline",
             _history_median(baseline), baseline.get("correctness_pass"), baseline.get("error_type")]]
    for attempt in history["attempts"][:-1]:
        evaluation = attempt.get("evaluation", {})
        value = attempt.get("proposal", {})
        rows.append([attempt["attempt"], value.get("config_id"), attempt["status"],
                     _history_median(evaluation), evaluation.get("correctness_pass"),
                     attempt.get("error_type", evaluation.get("error_type"))])
    return {"columns": list(HISTORY_COLUMNS), "rows": rows}


class LocalLlamaProposer:
    def __init__(self,config,directory,seed):
        self.config=config;self.directory=Path(directory);self.seed=seed
        self.directory.mkdir(parents=True,exist_ok=False)
    def __call__(self,history,attempt):
        out=self.directory/f"proposal-{attempt:04d}";out.mkdir(exist_ok=False)
        table=compact_history(history)
        prompt=("Choose one GEMM configuration using the previous measured outcomes. Return exactly one JSON object, no markdown, prose, code, commands or extra keys. "
                "Use config_id, tile, stages, epilogue exactly as one catalog entry. Avoid previously proposed IDs. "
                "Lower median_ms is better; correctness failures cannot be accepted. All candidates compute the same GEMM with a nonzero bias.\n"
                +"CATALOG="+json.dumps([proposal(i) for i in range(12)])+"\nPREVIOUS_RESULTS="+json.dumps(table,allow_nan=False,separators=(",",":"))+"\nJSON:\n")
        prompt_path=out/"prompt.txt";prompt_path.write_text(prompt,encoding="utf-8")
        executable=Path(self.config["executable"]).resolve(strict=True);model=Path(self.config["model"]).resolve(strict=True)
        # Fixed operator-owned flags only. No shell, remote endpoint, model code,
        # arbitrary grammar file or proposal-controlled argument is accepted.
        argv=[str(executable),"-m",str(model),"-ngl","0","--seed",str((self.seed+attempt)%(2**31)),
              "-n",str(self.config["max_output_tokens"]),"-c",str(self.config["context_tokens"]),
              "--temp","0.7","--no-display-prompt","--no-conversation","-f",str(prompt_path)]
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
            # Decode exact UTF-8 bytes: CRLF must not become the fixed LF trailer.
            payload=(out/"stdout.txt").read_bytes().decode("utf-8")
            payload,removed=decode_completion_stdout(payload)
            raw=payload.strip()
            receipt.update(status="returned_unvalidated",
                           stdout_transport=dict(contract="llama.cpp_completion_eog_stdout_v1",
                               upstream_revision="ddf48f85c933c1f438212a02eb21e7b001d2d37d",
                               exact_terminal_suffix=COMPLETION_STDOUT_TRAILER,
                               maximum_suffix_removals=1,cli_framing_removed=removed),
                           returned_payload_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest())
            return raw
        except Exception as e:
            receipt.update(status="failed",error_type=type(e).__name__,error=str(e));raise
        finally:
            receipt.update(finished_timestamp=now(),wall_seconds=time.perf_counter()-started)
            for name in ("stdout.txt","stderr.txt"):
                if (out/name).exists():receipt[name+"_sha256"]=digest(out/name)
            write(out/"proposal.json",receipt)
