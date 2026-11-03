"""Stdlib evidence helpers. No dependencies, model imports, or silent Git fallback."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def command(argv,cwd=None):
    return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()


def metadata(root,*,gpu=False):
    git=os.environ.get("TINYC_GIT") or shutil.which("git")
    if not git:raise RuntimeError("Set TINYC_GIT to the real Git executable")
    directory=str(root)
    if platform.system()=="Linux" and git.lower().endswith(".exe"):
        directory=command(["wslpath","-w",directory])
    base=[git,"-C",directory]
    cpu=platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu=next((v.split(":",1)[1].strip() for v in Path("/proc/cpuinfo").read_text().splitlines() if v.startswith("model name")),cpu)
    result={"environment":"Docker" if Path("/.dockerenv").exists() else ("WSL2" if "microsoft" in platform.release().lower() else platform.system()),
            "platform":platform.platform(),"cpu":cpu,"python":sys.version,"python_executable":sys.executable,
            "git":{"sha":command(base+["rev-parse","HEAD"]),"status_porcelain":command(base+["status","--porcelain"]),"version":command([git,"--version"])}}
    if gpu:
        result["nvidia_smi"]=command(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"])
        result["cuda_scope"]="driver/device reported by nvidia-smi; exact build toolkit belongs to supplied build receipt"
    else:result.update(gpu="not queried; CPU execution",cuda="not used")
    return result


def output_directory(path,root):
    path=Path(path).resolve();base=(Path(root)/"results/v2").resolve()
    if path.parent!=base or path.name.startswith("."):raise ValueError("unique results/v2/<run-id> required")
    path.mkdir(parents=True,exist_ok=False);return path
