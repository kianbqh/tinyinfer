"""CPU subprocess fixtures only; writes captured prompts for tokenizer review."""
from pathlib import Path
import argparse, hashlib, importlib, json, os, sys, unittest

parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args()
base=Path(__file__).resolve().parent;stage=base/args.stage
sys.path.insert(0,str(stage));sys.path.insert(0,str(stage/'autotune/tests'));os.chdir(stage)
suite=unittest.defaultTestLoader.discover(str(stage/'autotune/tests'),pattern='test_proposer*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
fixture=importlib.import_module('test_proposer_history')
out=base/'evidence';out.mkdir(exist_ok=True)
for label,history in [('long-history',fixture.capacity_history()),('wide-history',fixture.wide_capacity_history())]:
    prompt=fixture.capture_prompt(history)
    (out/(args.stage+'-'+label+'-prompt.txt')).write_bytes(prompt.encode('utf-8'))
    (out/(args.stage+'-'+label+'-history.json')).write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
forbidden=[name for name in sys.modules if name=='torch' or name.startswith(('torch.','transformers','llama_cpp'))]
assert not forbidden,forbidden
print(json.dumps({'stage':args.stage,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                  'skipped':len(result.skipped),'forbidden_ml_imports':forbidden}))
raise SystemExit(0 if result.wasSuccessful() else 1)
