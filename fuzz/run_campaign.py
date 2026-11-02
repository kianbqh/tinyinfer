"""Finite LLVM libFuzzer testing of our own local host code. No remote targets."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from metadata import ROOT, environment, run, sha, sources, unique, utc, write

TARGETS=('json','artifact','host_validation','cpu_ops','codec')


def config(path):
    value=json.loads(Path(path).read_text())
    if value.get('schema_version')!=1 or value.get('targets')!=list(TARGETS):raise ValueError('exact five-target config required')
    limits={'seed':(1,2**31-1),'runs':(1,1000000),'max_total_time_seconds_per_target':(1,600),
            'timeout_seconds_per_input':(1,10),'max_len':(1,65536),'rss_limit_mb':(128,2048),
            'malloc_limit_mb':(1,128),'workers':(1,1),'jobs':(0,0)}
    for name,(low,high) in limits.items():
        if type(value.get(name)) is not int or not low<=value[name]<=high:raise ValueError('invalid finite limit '+name)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--build-receipt',required=True)
    p.add_argument('--output-parent',required=True);a=p.parse_args();cfg=config(a.config)
    receipt=json.loads(Path(a.build_receipt).read_text())
    if receipt.get('status')!='built_not_run':raise ValueError('successful real build receipt required')
    start=sources()
    if start['files']!=receipt['sources_end']['files']:raise ValueError('current source differs from actual build')
    directory=unique(a.output_parent,'campaign');began=time.monotonic()
    record={'schema_version':1,'status':'started','start_utc':utc(),'config':cfg,'config_sha256':sha(a.config),
            'build_receipt':str(Path(a.build_receipt).resolve()),'build_receipt_sha256':sha(a.build_receipt),
            'sources_start':start,'environment':environment(),'targets':[],'confirmed_issues':None,'fixed_issues':None,
            'excluded_code_review_findings':['prior socket receive allowance and epoll registration identity fixes']}
    write(directory/'run.json',record)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             ASAN_OPTIONS='abort_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    try:
        for target in TARGETS:
            info=receipt['binaries'][target];binary=Path(info['path']).resolve(strict=True)
            if sha(binary)!=info['sha256']:raise ValueError('binary differs from build receipt')
            target_dir=directory/target;target_dir.mkdir();corpus=target_dir/'corpus';artifacts=target_dir/'artifacts';artifacts.mkdir()
            shutil.copytree(ROOT/'fuzz/corpus'/target,corpus)
            argv=[str(binary),str(corpus),'-seed='+str(cfg['seed']),'-runs='+str(cfg['runs']),
                  '-max_total_time='+str(cfg['max_total_time_seconds_per_target']),'-max_len='+str(cfg['max_len']),
                  '-timeout='+str(cfg['timeout_seconds_per_input']),'-rss_limit_mb='+str(cfg['rss_limit_mb']),
                  '-malloc_limit_mb='+str(cfg['malloc_limit_mb']),'-jobs=0','-workers=1',
                  '-artifact_prefix='+str(artifacts)+'/']
            item=run(argv,target_dir/'raw.log',env=env,
                     timeout=cfg['max_total_time_seconds_per_target']+cfg['timeout_seconds_per_input']+30)
            text=(target_dir/'raw.log').read_text(errors='replace')
            item.update(target=target,binary_sha256=sha(binary),
                        sanitizer_diagnostic_lines=[line for line in text.splitlines() if
                            re.search(r'ERROR: AddressSanitizer|SUMMARY: .*Sanitizer|runtime error:',line)],
                        artifacts=[{'path':str(f.relative_to(directory)),'sha256':sha(f),'bytes':f.stat().st_size}
                                   for f in sorted(artifacts.iterdir()) if f.is_file()],
                        corpus_files=[{'name':f.name,'sha256':sha(f),'bytes':f.stat().st_size}
                                      for f in sorted(corpus.iterdir()) if f.is_file()])
            record['targets'].append(item);write(directory/'run.json',record)
        record['sources_end']=sources()
        if record['sources_end']['files']!=start['files']:raise ValueError('source changed during campaign')
        failed=[t for t in record['targets'] if t['exit_code']!=0 or t['sanitizer_diagnostic_lines'] or t['artifacts']]
        record['status']='needs_triage' if failed else 'completed_no_diagnostics'
        # Zero is legitimate only for this completed finite run; no proof of absence.
        if not failed:record.update(confirmed_issues=0,fixed_issues=0)
    except Exception as error:record.update(status='failed',error=str(error))
    finally:
        record.update(end_utc=utc(),wall_seconds=time.monotonic()-began);write(directory/'run.json',record)
    print(directory/'run.json');return 0 if record['status']=='completed_no_diagnostics' else 1


if __name__=='__main__':sys.exit(main())
