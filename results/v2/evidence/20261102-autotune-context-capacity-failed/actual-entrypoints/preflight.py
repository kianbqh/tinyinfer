"""Actual released autotuner preflight; no model inference or compilation."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
root = Path('/mnt/g/ResumeProject/platforms/tinyinfer')
expected_head = '91ae0821a1a1e1129beeabbed0aa6ff1d2e4e947'
git = '/opt/codex-resume/all-modules-v2/m1a/tools/git-native/usr/bin/git'
def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
head = subprocess.check_output([git, '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
status = subprocess.check_output([git, '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
if head != expected_head or status.strip():
    raise ValueError('Released checkpoint/full clean tree required')
gpu = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                      '--format=csv,noheader'], text=True, capture_output=True, check=True)
if gpu.stdout.strip():
    raise ValueError('Other GPU compute process present')
pause = Path('/mnt/g/ResumeProject/.runtime/all-modules-v2/PAUSE_DATA_DOWNLOADS')
if not pause.is_file():
    raise ValueError('Root-coordinated data pause flag required before formal search')
config_path = here / 'autotune.candidate.json'
if digest(config_path) != '79451016aed0861c1248fbd6ad02bb74c512729e46383565ef2189e6adadac24':
    raise ValueError('Frozen candidate config changed')
config = json.loads(config_path.read_text())
from autotune.run import validate
validate(config)
python_argument = config['toolchain']['python']
if python_argument != '/opt/codex-resume/all-modules-v2/m2/venv/bin/python' or not Path(python_argument).is_file():
    raise ValueError('Private venv interpreter required')
builder = root / 'bench/v2/kernels/candidate_build.py'
text = builder.read_text()
if 'f"-DPython3_EXECUTABLE={Path(args.python).absolute()}"' not in text or 'Path(args.python).resolve()' in text:
    raise ValueError('Candidate builder must preserve passed absolute venv path')
ast.parse(text)
expected_cpath = '/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include/python3.12:/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include'
expected_cxxflags = '-I/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include'
if os.environ.get('CPATH') != expected_cpath or os.environ.get('CXXFLAGS') != expected_cxxflags:
    raise ValueError('Explicit approved existing headers required')
for directory in expected_cpath.split(':'):
    if not Path(directory).is_dir():
        raise FileNotFoundError(directory)
for kind in ('executable', 'model'):
    if digest(config['llama'][kind]) != config['llama'][kind + '_sha256']:
        raise ValueError('Pinned proposer ' + kind + ' bytes changed')
if Path(config['llama']['executable']).name != 'llama-completion':
    raise ValueError('Previously parser-verified completion entry point required')
run = Path('/opt/codex-resume/all-modules-v2/m2/runs/20261102T230756Z-m2-autotune-transportfix')
destination = run / 'autotune.json'
if Path(config['scratch_base']).exists() or (root / 'results/v2/20261102T230756Z-m2-autotune-transportfix').exists():
    raise FileExistsError('Fresh scratch/output required')
with destination.open('xb') as stream:
    stream.write(config_path.read_bytes())
record = {'status': 'passed', 'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'checkpoint': head, 'full_git_status': status, 'gpu_compute_apps_stdout': gpu.stdout,
    'pause_flag': str(pause), 'pause_flag_present': True,
    'config': str(destination), 'config_sha256': digest(destination),
    'candidate_builder_sha256': digest(builder), 'private_python_passed_absolute': python_argument,
    'private_python_resolved_target_for_inventory_only': str(Path(python_argument).resolve()),
    'headers': {'CPATH': expected_cpath, 'CXXFLAGS': expected_cxxflags},
    'proposer': config['llama'], 'model_and_executable_actual_hash_verified': True,
    'cpu_proposer_policy': 'Existing fixed -ngl 0 argv; subprocess exits before candidate build/GPU evaluation',
    'build_parallelism': 2, 'search_started': False}
(run / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
