"""Stdlib run records; no compiler, model, GPU or network work at import."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def storage(path):
    path=Path(path).resolve()
    if os.environ.get('GITHUB_ACTIONS')=='true':return path
    if os.name=='nt':okay=path.drive.upper()=='G:'
    else:okay=any(str(path).startswith(p) for p in ('/mnt/g/','/opt/codex-resume/all-modules-v2/'))
    if not okay:raise ValueError('use a task-local G-backed output/cache/build path')
    return path


def unique(parent, label):
    parent=storage(parent);parent.mkdir(parents=True,exist_ok=True)
    path=parent/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+label+'-'+uuid.uuid4().hex[:8])
    path.mkdir();return path


def sources():
    paths=[]
    for directory in ('fuzz','backend/include','llm/src'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and
                  (p.suffix in ('.py','.h','.hpp','.cpp','.json','.txt','.cmake') or p.name=='CMakeLists.txt') and
                  '__pycache__' not in p.parts]
    paths += [ROOT/'backend/host_codec.cpp',ROOT/'backend/runtime_v2.cu',ROOT/'llm/export_weights.py',
              ROOT/'llm/host_codec.py',ROOT/'kernels/src/cuda/tc/gemm.cu',ROOT/'kernels/src/cuda/tc/catalog.cuh']
    paths += list((ROOT/'kernels/src/cpu').glob('*.cpp'))+list((ROOT/'kernels/include/ik').glob('*.hpp'))
    paths += [ROOT/'CMakeLists.txt',ROOT/'backend/host_codec/CMakeLists.txt']
    paths += list((ROOT/'cmake').glob('*.cmake'))
    for directory in ('fuzz/corpus','fuzz/regressions'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file()]
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    def git(args):
        run=subprocess.run([os.environ.get('TINYC_GIT','git'),'-C',str(ROOT),*args],capture_output=True,text=True,timeout=10)
        return {'exit_code':run.returncode,'output':run.stdout,'stderr':run.stderr}
    return {'git_head':git(['rev-parse','HEAD']),'git_status':git(['status','--porcelain=v1']), 'files':files}


def run(argv,log,env=None,timeout=60):
    record={'argv':[str(x) for x in argv],'start_utc':utc()};start=time.monotonic()
    with Path(log).open('xb') as stream:
        try:
            child=subprocess.run(record['argv'],stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout)
            record['exit_code']=child.returncode
        except subprocess.TimeoutExpired:
            record.update(exit_code=None,timed_out=True)
        except OSError as error:
            record.update(exit_code=None,error=str(error))
    record.update(end_utc=utc(),wall_seconds=time.monotonic()-start,log=str(log),log_sha256=sha(log))
    return record


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def environment():
    return {'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
            'python':platform.python_version(),'environment':'GitHub-hosted' if os.environ.get('GITHUB_ACTIONS')=='true' else
            ('WSL2' if 'microsoft' in platform.release().lower() else 'native'),
            'cpu_threads':os.cpu_count(),'sanitizer_env':{'ASAN_OPTIONS':'abort_on_error=1:detect_leaks=1',
                'UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'},'cuda':'not used'}
