"""Stop only this exact released orchestration and its verified CPU proposer."""
import datetime
import json
import os
from pathlib import Path
import re
import signal
import time

parent_pid = 373810
native = Path('/opt/codex-resume/all-modules-v2/m2/runs/20261102T230756Z-m2-autotune-transportfix')
results = Path('/mnt/g/ResumeProject/platforms/tinyinfer/results/v2/20261102T230756Z-m2-autotune-transportfix')
cli = '/opt/codex-resume/llm-risk-classifier/llama/llama-b12024/llama-completion'
model = '/opt/codex-resume/llm-risk-classifier/weights/qwen2.5-1.5b-instruct-q4_k_m.gguf'
parent_argv = ['/opt/codex-resume/all-modules-v2/m2/venv/bin/python', '-B', '-m', 'autotune.run',
    '--config', str(native / 'autotune.json'), '--output', 'results/v2/' + results.name]
record = {'status': 'interrupting_interface_failure',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reason': 'Root STOP: accumulated full search history exceeds the fixed8192 context; interface capacity failure, no matched-budget completion claim',
    'checkpoint': '91ae0821a1a1e1129beeabbed0aa6ff1d2e4e947',
    'parent_pid': parent_pid, 'signals': [], 'snapshots': [], 'unknown': [],
    'original_search_files_modified': False}

def identity(pid):
    path = Path('/proc') / str(pid)
    before = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    row = {'pid': pid, 'state': before[0], 'ppid': int(before[1]),
           'pgrp': int(before[2]), 'session': int(before[3]), 'start_ticks': before[19]}
    if row['state'] != 'Z':
        row.update(exe=str((path / 'exe').resolve(strict=True)),
                   argv=[os.fsdecode(value) for value in (path / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])
    after = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    if before[1:4] != after[1:4] or before[19] != after[19]:
        raise RuntimeError('Process identity changed')
    return row

def role(row):
    if row['pgrp'] != parent_pid or row['session'] != parent_pid:
        return 'unknown'
    if row['state'] == 'Z':
        return 'zombie'
    if row['pid'] == parent_pid and row['argv'] == parent_argv and row['exe'] == str(Path(parent_argv[0]).resolve()):
        return 'orchestrator'
    argv = row['argv']
    if row['exe'] != str(Path(cli).resolve()) or not argv or argv[0] != cli:
        return 'unknown'
    try:
        prompt = Path(argv[-1])
        parts = prompt.relative_to(results).parts
        seed = int(re.fullmatch(r'seed-(17|41|73)', parts[0]).group(1))
        attempt = int(re.fullmatch(r'proposal-(\d{4})', parts[3]).group(1))
        if len(parts) != 5 or parts[1:3] != ('llm', 'proposals') or parts[4] != 'prompt.txt':
            return 'unknown'
        expected = [cli, '-m', model, '-ngl', '0', '--seed', str((seed + attempt) % (2**31)),
                    '-n', '384', '-c', '8192', '--temp', '0.7', '--no-display-prompt',
                    '--no-conversation', '-f', str(prompt)]
        return 'cpu_proposer' if argv == expected else 'unknown'
    except (ValueError, IndexError, AttributeError):
        return 'unknown'

def scan(label):
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[2]) != parent_pid and int(stat[3]) != parent_pid:
                continue
            row = identity(int(path.name))
            row['role'] = role(row)
            rows.append(row)
        except (FileNotFoundError, ProcessLookupError):
            continue
    record['snapshots'].append({'label': label, 'rows': rows})
    return rows

def terminate(row, signum):
    fd = os.pidfd_open(row['pid'])
    try:
        current = identity(row['pid'])
        if current['start_ticks'] != row['start_ticks'] or role(current) != row['role']:
            raise RuntimeError('Identity changed before signal')
        signal.pidfd_send_signal(fd, signum)
        record['signals'].append({'pid': row['pid'], 'role': row['role'], 'signal': signum,
                                  'start_ticks': row['start_ticks']})
    finally:
        os.close(fd)

try:
    initial = scan('before_stop')
    parents = [row for row in initial if row['role'] == 'orchestrator']
    if not parents:
        raise RuntimeError('Exact owned orchestrator not confirmed')
    terminate(parents[0], signal.SIGTERM)  # Prevent another phase/proposal launch.
    for row in initial:
        if row['role'] == 'cpu_proposer':
            try:
                terminate(row, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
    deadline = time.monotonic() + 15
    remaining = scan('after_initial_signals')
    while remaining and time.monotonic() < deadline:
        time.sleep(.1)
        remaining = scan('cleanup_poll')
    for row in remaining:
        if row['role'] in ('orchestrator', 'cpu_proposer'):
            terminate(row, signal.SIGKILL)
    if remaining:
        time.sleep(.2)
    final = scan('final')
    record['unknown'] = [row for row in final if row['role'] == 'unknown']
    record['remaining_scope_members'] = final
    record['status'] = 'interrupted_interface_failure_children_closed' if not final else 'interrupt_cleanup_incomplete'
finally:
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    root_path = results / 'search.json'
    if root_path.exists():
        value = json.loads(root_path.read_text())
        record['original_search_status'] = value.get('status')
    llm = results / 'seed-41/llm/search.json'
    if llm.exists():
        value = json.loads(llm.read_text())
        record['partial_llm'] = {key: value.get(key) for key in ('status', 'evaluations', 'invalid', 'proposal_errors')}
        record['partial_llm']['attempt_count'] = len(value.get('attempts', []))
        record['partial_llm']['last_attempt'] = value.get('attempts', [])[-1:]
    with (native / 'interface-interrupt.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record))
