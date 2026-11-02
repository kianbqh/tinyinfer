"""Config-driven launcher; captures actual binary/config/command/environment/logs."""
from __future__ import annotations
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from .metadata import digest,metadata,now,output_directory,write

FIELDS={"port":(1,65535),"max_batch":(1,32),"max_wait_us":(0,1000000),"capacity":(2,1024),
        "max_new_tokens":(1,1023),"max_pending":(1,100000),"max_connections":(1,100000),
        "per_connection":(1,100000),"max_output_bytes":(64,64*1024*1024),"max_ids":(1,1000000),"send_chunk_bytes":(1,1048576),
        "device":(0,64),"gemm_config":(0,11),"attention_config":(0,1)}


def validate(config):
    if set(config)!={"schema_version","host","model","manifest",*FIELDS} or type(config["schema_version"]) is not int or config["schema_version"]!=1:raise ValueError("server config schema")
    if not isinstance(config["manifest"],str):raise ValueError("manifest path")
    if config["model"] not in ("cpu","native") or config["host"]!="127.0.0.1":raise ValueError("local IPv4 endpoint/model required")
    for name,(lo,hi) in FIELDS.items():
        if type(config[name]) is not int or not lo<=config[name]<=hi:raise ValueError(name)
    if config["max_new_tokens"]>=config["capacity"] or config["per_connection"]>config["max_pending"]:raise ValueError("context/admission bounds")
    if config["model"]=="native" and not config["manifest"]:raise ValueError("native artifact manifest required")
    return config


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--binary",required=True)
    p.add_argument("--build-receipt",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];cfg_path=Path(a.config).resolve(strict=True)
    config=validate(json.loads(cfg_path.read_text(encoding="utf-8")));binary=Path(a.binary).resolve(strict=True)
    receipt=Path(a.build_receipt).resolve(strict=True)
    out=output_directory(a.output,root);start=time.perf_counter();proc=None
    journal={"status":"starting","timestamp":now(),"command_argv":sys.argv,"config":config,"config_sha256":digest(cfg_path),
             "binary":str(binary),"binary_sha256":digest(binary),"build_receipt":json.loads(receipt.read_text(encoding="utf-8")),"build_receipt_sha256":digest(receipt)}
    write(out/"server.json",journal)
    try:
        if journal["build_receipt"].get("binary_sha256")!=digest(binary):raise ValueError("build receipt does not identify exact server binary")
        for relative,expected in journal["build_receipt"].get("source_sha256",{}).items():
            source=(root/relative).resolve()
            if not source.is_relative_to(root) or digest(source)!=expected:raise ValueError("source no longer matches server build: "+relative)
        libraries=journal["build_receipt"].get("native_libraries",{})
        if config["model"]=="native" and set(libraries)!={"tinyinfer_llm","tinyinfer_backend"}:raise ValueError("native library build identities required")
        for library in libraries.values():
            if digest(library["path"])!=library["sha256"]:raise ValueError("native library changed after build receipt")
        if not journal["build_receipt"].get("source_sha256"):raise ValueError("build receipt needs exact compiled source hashes")
        journal["environment"]=metadata(root,gpu=config["model"]=="native")
        if config["model"]=="native":journal["artifact_manifest_sha256"]=digest(config["manifest"])
        argv=[str(binary)]
        for key,value in config.items():
            if key!="schema_version":argv.extend(["--"+key.replace("_","-"),str(value)])
        journal["server_argv"]=argv;write(out/"server.json",journal)
        with (out/"stdout.log").open("w",encoding="utf-8") as stdout,(out/"stderr.log").open("w",encoding="utf-8") as stderr:
            proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr)
            def stop(signum,frame):
                if proc.poll() is None:proc.send_signal(signal.SIGTERM)
            signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
            journal.update(status="running_startup_unverified",pid=proc.pid);write(out/"server.json",journal)
            journal["exit_code"]=proc.wait();journal["status"]="stopped" if proc.returncode==0 else "failed"
    except Exception as e:
        journal.update(status="failed",error_type=type(e).__name__,error=str(e))
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();journal["forced_kill"]=True
        journal.update(finished_timestamp=now(),wall_seconds=time.perf_counter()-start)
        for name in ("stdout.log","stderr.log"):
            if (out/name).exists():journal[name+"_sha256"]=digest(out/name)
        write(out/"server.json",journal)
    return 0 if journal["status"]=="stopped" else 1

if __name__=="__main__":raise SystemExit(main())
