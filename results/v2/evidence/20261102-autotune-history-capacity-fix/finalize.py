"""Bind private CPU/tokenizer evidence and unchanged production control flow."""
from pathlib import Path
import ast, datetime, hashlib, json

BASE=Path(__file__).resolve().parent;REPO=BASE.parents[4]/'platforms/tinyinfer'
READS={}
def read(p):
    b=p.read_bytes();READS[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def obj(p):return json.loads(read(p))
manifest=obj(BASE/'CANDIDATE_MANIFEST.json')
old=ast.parse(read(BASE/'preimage/autotune/proposer.py').decode())
new=ast.parse(read(BASE/'candidate/autotune/proposer.py').decode())
def function(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
assert ast.dump(function(old,'decode_completion_stdout'))==ast.dump(function(new,'decode_completion_stdout'))
assert ast.dump(function(old,'__init__'))==ast.dump(function(new,'__init__'))
def tail(tree):
    body=function(tree,'__call__').body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt_path' for t in n.targets))
    return [ast.dump(n) for n in body[start:]]
assert tail(old)==tail(new)
def prose(tree):
    return next(n.value for n in ast.walk(function(tree,'__call__')) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('Choose one GEMM configuration'))
assert prose(old).replace('Lower median samples_ms is better','Lower median_ms is better')==prose(new)
states=[]
for f in manifest['files']:
    candidate=read(BASE/'candidate'/f['destination'])
    assert hashlib.sha256(candidate).hexdigest()==f['candidate']['sha256'];ast.parse(candidate.decode())
    actual=REPO/f['destination']
    if not actual.exists():assert f['preimage'] is None;states.append('new_file_not_imported');continue
    b=read(actual);h=hashlib.sha256(b).hexdigest()
    assert h in ([f['candidate']['sha256']] + ([f['preimage']['sha256']] if f['preimage'] else []))
    states.append('root_imported_exact_candidate' if h==f['candidate']['sha256'] else 'actual_preimage_unchanged')
for rel,expected in manifest['unchanged_dependencies'].items():assert hashlib.sha256(read(REPO/rel)).hexdigest()==expected['sha256']
for rel in ['autotune/run.py','autotune/core.py']:read(REPO/rel)
cpu=obj(BASE/'CPU_RESULT.json');capacity=obj(BASE/'TOKEN_CAPACITY_RESULT.json')
assert cpu['status']=='preimage_fails_postimage_passes'
assert [r['exit_code'] for r in cpu['results']]==[1,0]
for r in cpu['results']:
    for stream in ['stdout','stderr']:
        assert hashlib.sha256(read(BASE/'logs'/(r['stage']+'.'+stream+'.txt'))).hexdigest()==r[stream+'_sha256']
assert capacity['status']=='actual_8206_reproduced_candidates_fit'
assert capacity['records'][0]['token_count']==8206
assert all(r['exit_code']==0 for r in capacity['records'])
for r in capacity['records']:
    directory=BASE/'tokenizer-capacity'/r['name']
    actual=obj(directory/'command.json');assert actual==r
    for stream in ['stdout','stderr']:assert hashlib.sha256(read(directory/(stream+'.log'))).hexdigest()==r[stream+'_sha256']
    assert hashlib.sha256(read(Path(r['prompt_path'].replace('/mnt/g/ResumeProject/','G:/ResumeProject/')))).hexdigest()==r['prompt_sha256']
max_candidate=max(r['token_count'] for r in capacity['records'][2:]);assert max_candidate==4518
manifest.update(state='private_candidate_cpu_and_same_gguf_capacity_verified',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                actual_repository_file_states=states,
                unchanged_control={'decoder_ast':True,'constructor_ast':True,'all_call_statements_from_prompt_file_write_onward':True,
                                   'all_model_flags_seed_temperature_context_output_timeout_and_transport':True,
                                   'only_prose_change':'samples_ms to median_ms','catalog_and_avoid_previous_advice_unchanged':True,
                                   'gates_strict_schema_budget_accounting_source_hashes_unchanged':True},
                cpu_result={'sha256':hashlib.sha256(read(BASE/'CPU_RESULT.json')).hexdigest(),'preimage_new_capacity_failures':3,
                            'preimage_existing_transport_passes':8,'postimage_passes':11,'skips':0,'forbidden_ml_imports':[]},
                tokenizer_result={'sha256':hashlib.sha256(read(BASE/'TOKEN_CAPACITY_RESULT.json')).hexdigest(),
                                  'actual_closed_prompt_tokens':8206,'prompt_limit':8188,'context_tokens':8192,'fixed_output_tokens':384,
                                  'candidate_counts':{r['name']:r['token_count'] for r in capacity['records'][2:]},
                                  'largest_tested_conservative_envelope':max_candidate,'margin_to_prompt_limit':8188-max_candidate,
                                  'envelope_plus_output':max_candidate+384,
                                  'scope':'65 total rows (baseline + 64 previous attempts), full-width finite floats, known states and 32-character error-type envelope. Exact same GGUF/default BOS/special handling; fixture capacity results are not LLM quality or search results.'},
                pending=['Root source review/import/commit/CI','Both methods and all three seeds fresh under capacity-only interface repair'],read_files=READS)
(BASE/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':manifest['state'],'files':manifest['files'],'max_candidate_tokens':max_candidate,
                  'margin':8188-max_candidate,'actual_repository_file_states':states},indent=2))
