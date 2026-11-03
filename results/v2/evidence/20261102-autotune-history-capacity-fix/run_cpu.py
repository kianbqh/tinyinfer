"""Run private pre/post regression only after root confirmed full stop."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time

base=Path(__file__).resolve().parent;logs=base/'logs';logs.mkdir(exist_ok=True)
tmp=base/'tmp';tmp.mkdir(exist_ok=True)
results=[]
for stage in ['preimage','postimage']:
    argv=[sys.executable,'-B',str(base/'run_stage.py'),'--stage',stage]
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tmp))
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
    result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment,check=False,timeout=60)
    for stream in ['stdout','stderr']:(logs/(stage+'.'+stream+'.txt')).write_bytes(getattr(result,stream))
    record=dict(stage=stage,argv=argv,started_timestamp=start,finished_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                wall_seconds=time.perf_counter()-t,exit_code=result.returncode,
                stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
    results.append(record);print(json.dumps(record));print(result.stdout.decode(),end='');print(result.stderr.decode(),end='')
assert results[0]['exit_code']==1 and results[1]['exit_code']==0
out={'status':'preimage_fails_postimage_passes','interpreter':sys.executable,'results':results,'scope':'CPU fixtures only; no model, tokenizer, GPU or build'}
(base/'CPU_RESULT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
