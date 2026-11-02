"""Replay retained inputs through the same real sanitized binaries; finite only."""
import argparse
import json
import os
from pathlib import Path
import sys
from metadata import ROOT, run, sha, sources, unique, utc, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--build-receipt',required=True);p.add_argument('--output-parent',required=True)
    p.add_argument('--include-seeds',action='store_true');a=p.parse_args()
    receipt=json.loads(Path(a.build_receipt).read_text())
    if receipt.get('status')!='built_not_run' or sources()['files']!=receipt['sources_end']['files']:
        raise ValueError('matching successful real build required')
    cases=json.loads((ROOT/'fuzz/regressions/manifest.json').read_text())['cases']
    if a.include_seeds:
        seeds=json.loads((ROOT/'fuzz/corpus/MANIFEST.json').read_text())['files']
        cases += [{'target':x['target'],'path':'fuzz/corpus/'+x['path'],'sha256':x['sha256']} for x in seeds]
    if len(cases)>256:raise ValueError('regression runner cap is 256 files')
    output=unique(a.output_parent,'regression');record={'schema_version':1,'start_utc':utc(),'cases':[],
        'status':'started','include_seeds':a.include_seeds,'build_receipt_sha256':sha(a.build_receipt)}
    env=dict(os.environ,OMP_NUM_THREADS='1',ASAN_OPTIONS='abort_on_error=1:detect_leaks=1',
             UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    for index,case in enumerate(cases):
        sample=(ROOT/case['path']).resolve();sample.relative_to(ROOT/'fuzz')
        if sample.stat().st_size>65536 or sha(sample)!=case['sha256']:raise ValueError('invalid regression input')
        binary=receipt['binaries'][case['target']]
        if sha(binary['path'])!=binary['sha256']:raise ValueError('binary changed')
        item=run([binary['path'],str(sample),'-runs=1','-max_len=65536','-timeout=2','-rss_limit_mb=512'],
                 output/(str(index)+'.log'),env=env,timeout=15)
        item['input']=case;record['cases'].append(item);write(output/'replay.json',record)
    record.update(end_utc=utc(),status=('no_regressions_registered' if not cases else
        ('passed' if all(x['exit_code']==0 for x in record['cases']) else 'failed')))
    write(output/'replay.json',record);print(output/'replay.json')
    return 0 if record['status'] in ('passed','no_regressions_registered') else 1


if __name__=='__main__':sys.exit(main())
