"""Private deterministic history-capacity candidate; no actual repo writes."""
from pathlib import Path
import hashlib, json

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[4]/'platforms/tinyinfer'
def rec(p):
    b=p.read_bytes();return {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
source=REPO/'autotune/proposer.py';before=source.read_bytes()
newline='\r\n' if b'\r\n' in before else '\n'
text=before.decode().replace('\r\n','\n')
text=text.replace('import json\n','import json\nimport math\nimport statistics\n',1)
helper='''HISTORY_COLUMNS = ("attempt", "config_id", "status", "median_ms", "correctness_pass", "error_type")


def _history_median(record):
    samples = record.get("samples_ms")
    if (not isinstance(samples, list) or not samples or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in samples)):
        return None
    return statistics.median(samples)


def compact_history(history):
    """Keep every earlier attempt; full raw evidence stays in the journals."""
    baseline = history["baseline"]
    rows = [[-1, baseline["proposal"]["config_id"], "initial_baseline",
             _history_median(baseline), baseline.get("correctness_pass"), baseline.get("error_type")]]
    for attempt in history["attempts"][:-1]:
        evaluation = attempt.get("evaluation", {})
        value = attempt.get("proposal", {})
        rows.append([attempt["attempt"], value.get("config_id"), attempt["status"],
                     _history_median(evaluation), evaluation.get("correctness_pass"),
                     attempt.get("error_type", evaluation.get("error_type"))])
    return {"columns": list(HISTORY_COLUMNS), "rows": rows}


'''
needle='class LocalLlamaProposer:'
assert text.count(needle)==1;text=text.replace(needle,helper+needle,1)
start=text.index('        # Preserve previous raw correctness/timing outcomes;')
end=text.index('        prompt=(',start)
text=text[:start]+'        table=compact_history(history)\n'+text[end:]
assert text.count('Lower median samples_ms is better')==1
text=text.replace('Lower median samples_ms is better','Lower median_ms is better',1)
assert text.count('json.dumps(rows,allow_nan=False)')==1
text=text.replace('json.dumps(rows,allow_nan=False)','json.dumps(table,allow_nan=False,separators=(",",":"))',1)
candidate=BASE/'candidate/autotune/proposer.py';candidate.parent.mkdir(parents=True,exist_ok=True)
candidate.write_bytes(text.replace('\n',newline).encode())
transport_source=REPO/'autotune/tests/test_proposer_transport.py'
t=transport_source.read_bytes();tn='\r\n' if b'\r\n' in t else '\n';s=t.decode().replace('\r\n','\n')
old='''assert len(catalog) == 12 and history[0]['status'] == 'initial_baseline'
assert history[0]['proposal'] == catalog[0]
assert history[0]['samples_ms'] == [1.0] * 7'''
new='''assert len(catalog) == 12
assert history['columns'] == ['attempt', 'config_id', 'status', 'median_ms', 'correctness_pass', 'error_type']
assert history['rows'] == [[-1, 0, 'initial_baseline', 1.0, None, None]]'''
assert s.count(old)==1;s=s.replace(old,new,1)
transport_candidate=BASE/'candidate/autotune/tests/test_proposer_transport.py'
transport_candidate.parent.mkdir(parents=True,exist_ok=True);transport_candidate.write_bytes(s.replace('\n',tn).encode())
newtest=BASE/'candidate/autotune/tests/test_proposer_history.py';assert newtest.exists()
deps=['autotune/__init__.py','autotune/core.py','bench/v2/kernels/candidate_build.py','bench/v2/kernels/protocol.py',
      'bench/v2/kernels/run.py','llm/__init__.py','llm/native_ops.py','server/__init__.py','server/metadata.py']
for stage,code in [('preimage',before),('postimage',candidate.read_bytes())]:
    root=BASE/stage;(root/'autotune/tests').mkdir(parents=True,exist_ok=True)
    (root/'autotune/proposer.py').write_bytes(code)
    (root/'autotune/tests/test_proposer_history.py').write_bytes(newtest.read_bytes())
    (root/'autotune/tests/test_proposer_transport.py').write_bytes(transport_candidate.read_bytes() if stage=='postimage' else t)
    for rel in deps:
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/rel).read_bytes())
manifest={'state':'prepared_not_integrated','scope':'Only prompt history-table capacity; no model/seed/sampling/context/output/timeout/budget/gate/strict-schema/transport changes',
          'files':[{'destination':'autotune/proposer.py','preimage':rec(source),'candidate':rec(candidate)},
                   {'destination':'autotune/tests/test_proposer_transport.py','preimage':rec(transport_source),'candidate':rec(transport_candidate)},
                   {'destination':'autotune/tests/test_proposer_history.py','preimage':None,'preimage_state':'absent','candidate':rec(newtest)}],
          'unchanged_dependencies':{rel:rec(REPO/rel) for rel in deps},
          'protocol_note':'Capacity-only interface repair. Preserve old interrupted runs; both methods and all three seeds must start fresh. Normal valid duplicates/incomplete unique budgets remain actual negative outcomes, not grounds for further prompt changes.',
          'pending':['CPU regression execution after root stop release','Same-GGUF tokenizer capacity verification after separate coordination']}
(BASE/'CANDIDATE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':manifest['files']},indent=2))
