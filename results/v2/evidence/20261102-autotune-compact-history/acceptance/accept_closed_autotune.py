"""Independent closed-run accounting and raw-sample review; never executes project code."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2] / 'platforms/tinyinfer'
HEAD = '6b1b9f665052e05715b15427f0a9a4777d0414fb'
reads = {}
evaluations = {}
source_hashes = {}

def require(value, message):
    if not value: raise AssertionError(message)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def read(path):
    path = Path(path)
    raw = path.read_bytes()
    reads[str(path)] = {'bytes': len(raw), 'sha256': digest(raw)}
    return json.loads(raw)

def number_equal(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)

def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position-left)

def check_stats(samples, summary, count):
    require(len(samples) == count, 'sample budget changed')
    require(all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in samples), 'nonpositive/invalid sample')
    expected = {'count':count,'mean_ms':statistics.mean(samples), 'std_population_ms':statistics.pstdev(samples),
                'p50_ms':quantile(samples,.5),'p95_ms':quantile(samples,.95),'p99_ms':quantile(samples,.99)}
    for name, value in expected.items():
        require(number_equal(summary[name],value), 'raw summary mismatch: '+name)

def local_path(value):
    prefix='/mnt/g/ResumeProject/'
    require(value.startswith(prefix), 'unknown mounted record path')
    return BASE.parents[2] / value[len(prefix):]

def check_proposal(value):
    require(set(value)=={'config_id','tile','stages','epilogue'}, 'proposal keys')
    identifier=value['config_id']
    require(type(identifier) is int and 0<=identifier<12, 'proposal id')
    tiles=((32,32,16,4),(64,32,32,8),(64,64,32,8))
    expected={'config_id':identifier,'tile':list(tiles[identifier%3]),
              'stages':2 if (identifier//3)%2==0 else 1,'epilogue':'fused' if identifier<6 else 'separate'}
    require(value==expected, 'catalog tuple mismatch')
    return identifier

def check_evaluation(row, fixture):
    require(row['status']=='passed' and row['correctness_pass'] is True, 'evaluation not passed')
    path=local_path(row['evaluation_record_path'])
    if str(path) in evaluations:
        require(evaluations[str(path)]['sha256']==row['evaluation_record_sha256'], 'conflicting evaluation reference')
        return
    actual=read(path)
    require(reads[str(path)]['sha256']==row['evaluation_record_sha256'], 'evaluation record hash')
    require(all(row[k]==v for k,v in actual.items()), 'embedded evaluation differs')
    require(actual['fixture']==fixture, 'fixture differs across candidates')
    check_proposal(actual['proposal'])
    timing=actual['timing']
    require((timing['warmups'],timing['repeats'],timing['host_repeats'],timing['concurrency'])==(20,101,21,1), 'timing protocol drift')
    require(actual['samples_ms']==timing['cuda_event_ms_raw'], 'selection samples differ')
    check_stats(actual['samples_ms'],timing['cuda_event'],101)
    check_stats(timing['host_synchronized_ms_raw'],timing['host_synchronized'],21)
    env=actual['environment']
    require(env['git']['sha']==HEAD, 'evaluation HEAD drift')
    require(env['allow_tf32'] is False and env['allow_fp16_reduced_precision_reduction'] is False, 'numeric mode drift')
    build=env['build_manifest']
    require(build['candidate']==actual['proposal'] and build['build_exit_code']==0, 'build candidate/exit mismatch')
    require(actual['build_manifest_sha256']==env['build_manifest_sha256'], 'build identity mismatch')
    correctness=read(path.parent/'correctness.json')
    require(correctness['status']=='passed', 'native correctness gate failed')
    for rel, expected in env['source_sha256'].items():
        if rel not in source_hashes:
            source_hashes[rel]=digest((REPO/rel).read_bytes())
        require(source_hashes[rel]==expected, 'current source mismatch '+rel)
    for label in ('build-helper','gpu-helper'):
        receipt=read(path.parents[1]/(label+'.json'))
        require(receipt['status']=='completed' and receipt['exit_code']==0, label+' failed')
        require(digest((path.parents[1]/(label+'.log')).read_bytes())==receipt['log_sha256'], label+' log mismatch')
    evaluations[str(path)]={'sha256':reads[str(path)]['sha256'],'config_id':actual['proposal']['config_id'],
                            'event_p50_ms':statistics.median(actual['samples_ms']), 'host_p50_ms':statistics.median(timing['host_synchronized_ms_raw'])}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run=Path(args.run)
    out=Path(args.output)
    require(not out.exists(), 'new acceptance output required')
    top=read(run/'search.json')
    require(top['status'] in ('completed','incomplete_incomparable'), 'search not successfully closed; inspect failure before accounting')
    config=top['config']
    require(config['seeds']==[17,41,73] and config['proposal_budget']==64 and config['evaluation_budget']==8, 'search protocol changed')
    require([x['seed'] for x in top['seeds']]==[17,41,73], 'missing seed')
    report=[]
    for seed in top['seeds']:
        baseline=seed['baseline']; fixture=baseline['fixture']
        require((fixture['M'],fixture['N'],fixture['K'],fixture['seed'])==(512,512,512,193), 'fixture protocol')
        check_evaluation(baseline,fixture)
        require(set(seed['methods'])=={'random','llm'}, 'missing method')
        for method, outcome in seed['methods'].items():
            require(read(run/f"seed-{seed['seed']}"/method/'search.json')==outcome, 'method/final journal mismatch')
            require(outcome['baseline']==baseline, 'methods do not share exact initial baseline')
            seen={check_proposal(baseline['proposal'])}; incumbent=baseline
            counts=Counter(); updates=0
            for index, attempt in enumerate(outcome['attempts']):
                require(attempt['attempt']==index, 'attempt order')
                status=attempt['status']; counts[status]+=1
                if status in ('invalid','proposal_error'): continue
                identifier=check_proposal(attempt['proposal'])
                if status=='duplicate':
                    require(identifier in seen, 'false duplicate'); continue
                require(identifier not in seen, 'duplicate was evaluated'); seen.add(identifier)
                row=attempt.get('evaluation')
                if status!='passed':
                    require(status in ('compile_failed','test_failed','evaluation_error'), 'unknown terminal attempt')
                    continue
                require(row['proposal']==attempt['proposal'], 'wrong evaluated tuple')
                check_evaluation(row,fixture)
                accept=statistics.median(row['samples_ms'])<=.97*statistics.median(incumbent['samples_ms'])
                require(attempt.get('accepted',False)==accept, 'incumbent gate mismatch')
                if accept: incumbent=row;updates+=1
            require(len(outcome['attempts'])<=64, 'proposal cap exceeded')
            require(outcome['proposal_attempts']==len(outcome['attempts']), 'proposal count')
            require(outcome['evaluations']==len(seen)-1<=8, 'unique evaluation count')
            for field,status in [('duplicates','duplicate'),('invalid','invalid'),('proposal_errors','proposal_error'),('compile_failures','compile_failed'),('test_failures','test_failed'),('evaluation_errors','evaluation_error')]:
                require(outcome[field]==counts[status], 'counter mismatch '+field)
            require(outcome['test_passes']==counts['passed'], 'passing count')
            require(outcome['valid_compiles']==counts['passed']+counts['test_failed'], 'compile count')
            require(outcome['accepted']==updates and outcome['best']==incumbent, 'selected incumbent mismatch')
            require(number_equal(outcome['selection_speedup_vs_initial_baseline'],statistics.median(baseline['samples_ms'])/statistics.median(incumbent['samples_ms'])), 'selection speedup')
            require(outcome['budget_exhausted']==(outcome['evaluations']==8), 'budget completion')
            independent=outcome['independent']
            if independent['status']=='passed':
                check_evaluation(independent['baseline'],fixture);check_evaluation(independent['winner'],fixture)
                require(independent['baseline']['proposal']==baseline['proposal'] and independent['winner']['proposal']==incumbent['proposal'], 'remeasured wrong tuple')
                base=statistics.median(independent['baseline']['samples_ms']);win=statistics.median(independent['winner']['samples_ms'])
                require(number_equal(independent['speedup'],base/win), 'independent speedup')
                require(independent['passes_3_percent_gate']==(win<=.97*base), 'independent gate')
            report.append({'seed':seed['seed'],'method':method,'proposals':len(outcome['attempts']),'evaluations':outcome['evaluations'],
                           'counts':dict(counts),'accepted':updates,'best_id':incumbent['proposal']['config_id'],
                           'selection_speedup':outcome['selection_speedup_vs_initial_baseline'],
                           'independent_speedup':independent.get('speedup'),'independent_gate':independent.get('passes_3_percent_gate')})
        comparable=all(x['budget_exhausted'] and x['independent']['status']=='passed' for x in seed['methods'].values())
        require(seed['comparable']==comparable, 'seed comparability')
    require(top['comparable']==all(x['comparable'] for x in top['seeds']), 'overall comparability')
    if top['comparable']:
        for method in ('random','llm'):
            speeds=[x['methods'][method]['independent']['speedup'] for x in top['seeds']]
            summary=top['summary'][method]
            require(summary['independent_speedups']==speeds, 'seed speedup list')
            require(number_equal(summary['mean'],statistics.mean(speeds)) and number_equal(summary['std_population'],statistics.pstdev(speeds)), 'seed summary')
        require(top['summary']['llm_mean_exceeds_random']==(top['summary']['llm']['mean']>top['summary']['random']['mean']), 'method comparison')
    else: require(top['summary']=={}, 'incomparable aggregate reported')
    result={'status':'passed','timestamp_utc':datetime.now(timezone.utc).isoformat(),'run_status':top['status'],
            'scope':'Independent raw-sample, source, build-identity, incumbent-gate and budget accounting. External native build receipt files and proposer process logs require separate provenance review.',
            'comparable':top['comparable'],'methods':report,'evaluations':evaluations,'source_hashes':source_hashes,'read_files':reads}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','comparability':top['comparable'],'unique_evaluation_records':len(evaluations),'output':str(out)}))

if __name__=='__main__': main()
