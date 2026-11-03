"""Stdlib-only v2 benchmark contracts; safe to inspect/test without importing Torch."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_source_paths(root):
    """Include CMake fragments and shared host headers used by native builds."""
    root=Path(root)
    paths=[root/"CMakeLists.txt",root/"backend/runtime_v2.cu",root/"bench/v2/kernels/profile.cu"]
    for pattern in ("cmake/*.cmake","backend/include/*.h","backend/include/*.hpp","backend/*.cpp",
                    "backend/host_codec/CMakeLists.txt",
                    "kernels/src/cuda/tc/*.cu","kernels/src/cuda/tc/*.cuh"):
        paths.extend(root.glob(pattern))
    return sorted(set(paths))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def percentile(samples, percent):
    """Linear interpolation, same documented definition for every method."""
    if not samples or any(not math.isfinite(x) or x<0 for x in samples) or not 0<=percent<=100:
        raise ValueError("invalid samples/percentile")
    ordered=sorted(samples);index=(len(ordered)-1)*percent/100
    lo=math.floor(index);hi=math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def summary(samples):
    if any(x<=0 for x in samples):raise ValueError("latency samples must be positive")
    return {"count":len(samples),"mean_ms":statistics.mean(samples),
            "p50_ms":percentile(samples,50),"p95_ms":percentile(samples,95),
            "p99_ms":percentile(samples,99),"std_population_ms":statistics.pstdev(samples)}


def validate_config(config):
    required={"schema_version","seed","warmups","repeats","host_repeats","gemm_sizes",
              "attention_lengths","attention_heads","attention_batch","head_dims",
              "softmax_shapes","gemm_configs","attention_configs","rtol","atol"}
    if set(config)!=required or config["schema_version"]!=1:raise ValueError("config keys/schema mismatch")
    for key in ("seed","warmups","repeats","host_repeats","attention_heads","attention_batch"):
        if type(config[key]) is not int or config[key]<(0 if key in ("seed","warmups") else 1):
            raise ValueError(f"invalid {key}")
    for key in ("gemm_sizes","attention_lengths","head_dims"):
        if not config[key] or any(type(x) is not int or x<=0 for x in config[key]):raise ValueError(key)
        if len(config[key])!=len(set(config[key])):raise ValueError(f"duplicate {key}")
    if any(x not in (64,128) for x in config["head_dims"]):raise ValueError("head_dims")
    for key,allowed in (("gemm_configs",set(range(12))),("attention_configs",{0,1})):
        if not config[key] or len(config[key])!=len(set(config[key])) or any(type(x) is not int or x not in allowed for x in config[key]):raise ValueError(key)
    if not config["softmax_shapes"] or any(not isinstance(s,list) or len(s)!=2 or any(type(x) is not int or x<=0 for x in s) or s[1]>32768 for s in config["softmax_shapes"]):
        raise ValueError("softmax_shapes")
    for key in ("rtol","atol"):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or config[key]<=0:raise ValueError(key)
    return config


def attention_allowed(query,key,*,q_length,kv_length,offset,causal=True):
    return 0<=query<q_length and 0<=key<kv_length and (not causal or key<=offset+query)


def online_softmax(scores,values):
    """Small CPU equation oracle, not a replacement for GPU correctness."""
    if len(scores)!=len(values):raise ValueError("score/value length")
    if not scores:return 0.0
    m=-math.inf;l=0.0;out=0.0
    for score,value in zip(scores,values):
        nxt=max(m,score);alpha=0.0 if l==0 else math.exp(m-nxt);beta=math.exp(score-nxt)
        out=out*alpha+value*beta;l=l*alpha+beta;m=nxt
    return out/l


def unique_output(path,repo_root):
    root=(Path(repo_root)/"results"/"v2").resolve()
    out=Path(path).resolve()
    if out.parent!=root or not out.name or out.name.startswith("."):
        raise ValueError("output must be a direct unique results/v2/<run-id> child")
    out.mkdir(parents=True,exist_ok=False)
    return out
