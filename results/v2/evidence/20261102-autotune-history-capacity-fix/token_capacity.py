"""Authorized same-GGUF tokenizer-only validation; never generates or uses GPU."""
from pathlib import Path
import ast, datetime, hashlib, importlib, json, os, subprocess, sys, time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]/'platforms/tinyinfer'
TOKENIZER=Path('/opt/codex-resume/llm-risk-classifier/llama/llama-b12024/llama-tokenize')
MODEL=Path('/opt/codex-resume/llm-risk-classifier/weights/qwen2.5-1.5b-instruct-q4_k_m.gguf')
EXPECTED_TOOL='18581ebb50c876bd317619b8eb12431b1ca4481c99ceb26fae07eab17ee4a690'
EXPECTED_MODEL='6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()
out=BASE/'tokenizer-capacity';out.mkdir(exist_ok=False)
assert digest(TOKENIZER)==EXPECTED_TOOL
assert digest(MODEL)==EXPECTED_MODEL
sys.path.insert(0,str(BASE/'postimage'));sys.path.insert(0,str(BASE/'postimage/autotune/tests'))
fixture=importlib.import_module('test_proposer_history')
history=fixture.wide_capacity_history()
for row in history['attempts'][:-1]:
    row['error_type']='QjZxVwKpRtYuBnMcAsDfGhLoErIqTyUi'
    row['proposal']=fixture.proposal(11)
    row['evaluation']['samples_ms']=[2.2250738585072014e-308]*101
envelope=fixture.capture_prompt(history)
envelope_path=BASE/'evidence/postimage-adversarial-envelope-prompt.txt'
envelope_path.write_bytes(envelope.encode())
(BASE/'evidence/postimage-adversarial-envelope-history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
closed=ROOT/'results/v2/20261102T230756Z-m2-autotune-transportfix/seed-17/llm/proposals/proposal-0026'
inputs=[('actual-closed-attempt26',closed/'prompt.txt'),
        ('preimage-long-history',BASE/'evidence/preimage-long-history-prompt.txt'),
        ('postimage-mixed-history',BASE/'evidence/postimage-long-history-prompt.txt'),
        ('postimage-wide-history',BASE/'evidence/postimage-wide-history-prompt.txt'),
        ('postimage-adversarial-envelope',envelope_path)]
records=[]
for name,prompt in inputs:
    directory=out/name;directory.mkdir()
    argv=[str(TOKENIZER),'-m',str(MODEL),'-f',str(prompt),'--ids','--show-count','--offline','-ngl','0']
    (directory/'argv.json').write_text(json.dumps(argv,indent=2)+'\n',encoding='utf-8')
    record=dict(name=name,argv=argv,prompt_path=str(prompt),prompt_sha256=digest(prompt),prompt_bytes=prompt.stat().st_size,
                tokenizer_sha256=EXPECTED_TOOL,model_sha256=EXPECTED_MODEL,started_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                default_bos=True,default_parse_special=True,scope='tokenizer-only; no generation; fixed -ngl 0')
    (directory/'command.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    t=time.perf_counter()
    with (directory/'stdout.log').open('wb') as stdout,(directory/'stderr.log').open('wb') as stderr:
        process=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=dict(os.environ,OMP_NUM_THREADS='1',LLAMA_ARG_OFFLINE='1'))
        record['pid']=process.pid
        try:record['exit_code']=process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();record['exit_code']=process.returncode;record['timeout']=True
    record.update(finished_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-t,
                  stdout_sha256=digest(directory/'stdout.log'),stderr_sha256=digest(directory/'stderr.log'))
    (directory/'command.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    assert record['exit_code']==0,(name,record['exit_code'])
    lists=[]
    for line in (directory/'stdout.log').read_text().splitlines():
        value=line.strip()
        if value.startswith('[') and value.endswith(']'):
            try:parsed=ast.literal_eval(value)
            except (ValueError,SyntaxError):continue
            if isinstance(parsed,list) and all(type(token) is int for token in parsed):lists.append(parsed)
    assert len(lists)==1,'Expected the documented single Python-parseable token ID list; raw outputs retained'
    record['token_count']=len(lists[0]);record['token_ids_sha256']=hashlib.sha256(json.dumps(lists[0],separators=(',',':')).encode()).hexdigest()
    record['prompt_limit']=8188;record['context_tokens']=8192;record['remaining_to_prompt_limit']=8188-record['token_count']
    (directory/'command.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    records.append(record);print(json.dumps({k:record[k] for k in ['name','pid','exit_code','prompt_bytes','token_count','remaining_to_prompt_limit','wall_seconds']}),flush=True)
    if name=='actual-closed-attempt26':assert record['token_count']==8206,'Tokenizer boundary must match actual completion CLI before candidate counts'
assert records[1]['token_count']>8188,'Old source capacity failure must remain reproducible'
assert all(r['token_count']<8188 for r in records[2:])
assert digest(TOKENIZER)==EXPECTED_TOOL
diagnostic=(closed/'stderr.txt').read_bytes()
result=dict(status='actual_8206_reproduced_candidates_fit',tokenizer_sha256=EXPECTED_TOOL,model_sha256=EXPECTED_MODEL,
            actual_closed_diagnostic_path=str(closed/'stderr.txt'),actual_closed_diagnostic_sha256=hashlib.sha256(diagnostic).hexdigest(),
            baseline_plus_previous_rows=65,conservative_error_type_envelope_chars=32,records=records,
            scope='Exact tokenizer counts for retained source prompts and declared conservative 64-prior-row fixtures. No model generation, GPU inference, benchmark, source/Git mutation or old-run formal audit.')
(BASE/'TOKEN_CAPACITY_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status']}))
