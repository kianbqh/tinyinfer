"""Read-only supplementary audit. DO NOT EXECUTE until root confirms closure.

No subprocess, project imports, builds, model runtime, network or CUDA access.
Native receipt/library bytes are inspected only after --closed-run-confirmed.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import shlex
import statistics

HEAD = '6b1b9f665052e05715b15427f0a9a4777d0414fb'
TRAILER = ' [end of text]\n\n\n'
UPSTREAM = 'ddf48f85c933c1f438212a02eb21e7b001d2d37d'
READS, CHECKS = {}, []
AUDIT_OUTPUT = None


def need(value, message):
    if not value:
        raise AssertionError(message)
    CHECKS.append(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    READS[str(path)] = dict(bytes=len(raw), sha256=sha(raw))
    return raw


def load(path):
    return json.loads(read(path))


def moment(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def before(left, right, label):
    need(moment(left) <= moment(right), label)


def proposal(identifier):
    return dict(config_id=identifier, tile=list(((32, 32, 16, 4), (64, 32, 32, 8),
                                               (64, 64, 32, 8))[identifier % 3]),
                stages=2 if (identifier // 3) % 2 == 0 else 1,
                epilogue='fused' if identifier < 6 else 'separate')


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    if not isinstance(raw, str) or len(raw.encode()) > 16384:
        raise ValueError('proposal output limit')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict) or set(value) != {'config_id', 'tile', 'stages', 'epilogue'}:
        raise ValueError('proposal schema')
    identifier = value['config_id']
    if type(identifier) is not int or not 0 <= identifier < 12:
        raise ValueError('proposal ID')
    if (type(value['stages']) is not int or not isinstance(value['tile'], list) or
            any(type(x) is not int for x in value['tile']) or value != proposal(identifier)):
        raise ValueError('proposal tuple')
    return value


def prompt_for(baseline, previous):
    def measured_median(record):
        samples = record.get('samples_ms')
        if not isinstance(samples, list) or not samples:
            return None
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in samples):
            return None
        return statistics.median(samples)
    rows = [[-1, baseline['proposal']['config_id'], 'initial_baseline',
             measured_median(baseline), baseline.get('correctness_pass'), baseline.get('error_type')]]
    for row in previous:
        evaluation = row.get('evaluation', {})
        rows.append([row['attempt'], row.get('proposal', {}).get('config_id'), row['status'],
                     measured_median(evaluation), evaluation.get('correctness_pass'),
                     row.get('error_type', evaluation.get('error_type'))])
    table = {'columns': ['attempt', 'config_id', 'status', 'median_ms', 'correctness_pass', 'error_type'], 'rows': rows}
    return ('Choose one GEMM configuration using the previous measured outcomes. Return exactly one JSON object, no markdown, prose, code, commands or extra keys. '
            'Use config_id, tile, stages, epilogue exactly as one catalog entry. Avoid previously proposed IDs. '
            'Lower median_ms is better; correctness failures cannot be accepted. All candidates compute the same GEMM with a nonzero bias.\n'
            + 'CATALOG=' + json.dumps([proposal(i) for i in range(12)])
            + '\nPREVIOUS_RESULTS=' + json.dumps(table, allow_nan=False, separators=(',', ':')) + '\nJSON:\n')


def main():
    global AUDIT_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed-run-confirmed', action='store_true', required=True)
    for name in ('repo', 'run', 'frozen-config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--map-prefix', action='append', default=[])
    parser.add_argument('--lifecycle', action='append', default=[])
    args = parser.parse_args()
    # This guard precedes every run/evidence read.
    if not args.closed_run_confirmed:
        parser.error('Root must explicitly confirm that the run is closed')
    repo, run, output = Path(args.repo), Path(args.run), Path(args.output)
    if output.resolve().parent != Path(__file__).resolve().parent:
        raise ValueError('Audit output must stay in this private final-autotune directory')
    if output.exists():
        raise FileExistsError('new audit output required: ' + str(output))
    AUDIT_OUTPUT = output
    mappings = []
    for item in args.map_prefix:
        source, target = item.split('=', 1)
        mappings.append((source.rstrip('/') + '/', target.rstrip('/\\') + '/'))
    mappings.sort(key=lambda item: -len(item[0]))
    def path(value):
        value = str(value)
        for source, target in mappings:
            if value.startswith(source):
                return Path(target + value[len(source):])
        return Path(value)
    def same_path(a, b):
        return path(a).resolve() == path(b).resolve()
    def helper(directory, label):
        record = load(directory / (label + '.json'))
        need(sha(read(directory / (label + '.log'))) == record['log_sha256'],
             'helper raw-log hash: ' + str(directory / label))
        before(record['timestamp'], record['finished_timestamp'], 'helper interval: ' + str(directory / label))
        return record

    top = load(run / 'search.json')
    need(top['status'] in ('completed', 'incomplete_incomparable') and 'finished_timestamp' in top,
         'top-level run is successfully terminal before evidence inspection')
    config = load(path(args.frozen_config))
    need(top['config'] == config and top['config_sha256'] == sha(read(path(args.frozen_config))) and
         top['config_canonical_sha256'] == canonical(config), 'exact retained frozen configuration')
    need(config['seeds'] == [17, 41, 73] and config['proposal_budget'] == 64 and
         config['evaluation_budget'] == 8 and config['baseline_id'] == 0, 'declared seeds/budgets/baseline')
    need(top['environment']['git']['sha'] == HEAD, 'top source checkpoint')
    python = top['environment']['python_executable']
    evaluation_paths, scratch_paths, intervals = {}, set(), {}

    def candidate(row, directory, fixture, role):
        need(load(directory / 'driver.json') == row, 'driver/embedded record: ' + str(directory))
        expected_scratch = path(config['scratch_base']) / run.name / directory.relative_to(run)
        need(same_path(row['build_scratch'], expected_scratch), 'exact native scratch mapping: ' + str(directory))
        scratch = path(row['build_scratch'])
        need(str(scratch.resolve()) not in scratch_paths, 'fresh unique build scratch: ' + str(scratch))
        scratch_paths.add(str(scratch.resolve()))
        bh = helper(directory, 'build-helper')
        expected = [python, '-m', 'bench.v2.kernels.candidate_build', '--proposal', str(directory / 'proposal.json'),
                    '--scratch', str(expected_scratch), '--timeout-seconds', str(config['build_timeout_seconds'])]
        # Path mappings change local inspection paths, not original receipt argv.
        actual = bh['argv']
        need(actual[:3] == expected[:3], 'build-helper interpreter/module: ' + str(directory))
        flags = ['--proposal', '--scratch', '--timeout-seconds'] + ['--' + k.replace('_', '-') for k in config['toolchain']]
        need(len(actual) == 3 + 2 * len(flags) and actual[3::2] == flags,
             'exact build-helper flag count/order without extra flags')
        for flag, value in [('--proposal', directory / 'proposal.json'), ('--scratch', expected_scratch)]:
            need(same_path(actual[actual.index(flag)+1], value), 'build-helper ' + flag + ': ' + str(directory))
        need(actual[actual.index('--timeout-seconds')+1] == str(config['build_timeout_seconds']), 'build timeout argv')
        for key, value in config['toolchain'].items():
            flag = '--' + key.replace('_', '-')
            need(actual[actual.index(flag)+1] == value, 'fixed toolchain argv ' + flag)
        native = load(scratch / 'candidate.json')
        before(bh['timestamp'], native['timestamp'], 'native helper starts after wrapper receipt')
        before(native['finished_timestamp'], bh['finished_timestamp'], 'native helper finishes before wrapper exit')
        need(native['proposal'] == row['proposal'] and native['candidate_id'] == row['proposal']['config_id'], 'native candidate identity')
        need(native['git']['sha'] == HEAD, 'native build source HEAD')
        for phase in native['attempts']:
            need(sha(read(scratch / (phase['phase'] + '.log'))) == phase['log_sha256'], 'native phase log hash')
        if row['status'] == 'compile_failed':
            need(bh['status'] == 'failed' and bh.get('exit_code', 1) != 0 and native['status'] == 'failed', 'retained real compile failure')
            return bh['timestamp'], bh['finished_timestamp']
        need(bh['status'] == 'completed' and bh['exit_code'] == 0 and native['status'] == 'compiled_not_gpu_tested', 'actual successful native build')
        need([r['phase'] for r in native['attempts']] == ['configure', 'build'] and
             all(r['status'] == 'completed' and r['exit_code'] == 0 for r in native['attempts']), 'configure/build terminal commands')
        manifest_path = scratch / 'build-manifest.json'
        build = load(manifest_path)
        need(build['candidate'] == row['proposal'] and build['build_exit_code'] == 0, 'external build-manifest identity')
        need(build['source_sha256'] == native['source_sha256'], 'native source stability map')
        for rel, digest in build['source_sha256'].items():
            need(sha(read(repo / rel)) == digest, 'native current source hash: ' + rel)
        need(sha(read(scratch / 'build.log')) == build['build_log_sha256'], 'external raw build log')
        full_commands = load(scratch / 'build/compile_commands.json')
        need(sha(read(scratch / 'build/compile_commands.json')) == build['compile_commands_sha256'], 'full compile-command hash')
        selected = [r for r in full_commands if '/tc/' in r['file'].replace('\\', '/') or r['file'].endswith('runtime_v2.cu')]
        need(selected == build['compile_commands'] and len(selected) == 4, 'actual four v2 compilation records')
        define = '-DTI_V2_ONLY_GEMM_CONFIG=' + str(row['proposal']['config_id'])
        need(all(define in shlex.split(r['command']) for r in selected), 'single-config actual compilation flag')
        cache = read(scratch / 'build/CMakeCache.txt')
        need(sha(cache) == build['cmake_cache_sha256'], 'CMake cache hash')
        entries = {line.split(':', 1)[0]: line.split('=', 1)[1] for line in cache.decode().splitlines()
                   if '=' in line and ':' in line.split('=', 1)[0] and not line.startswith(('#', '//'))}
        need(entries['TINYC_V2_GEMM_CONFIG'] == str(row['proposal']['config_id']), 'single-config CMake cache')
        need(build['build_command'] == native['attempts'][1]['argv'], 'native build command identity')
        configure = native['attempts'][0]['argv']
        need('-DTINYC_V2_GEMM_CONFIG=' + str(row['proposal']['config_id']) in configure, 'single-config configure argv')
        need('-DPython3_EXECUTABLE=' + config['toolchain']['python'] in configure, 'preserved configured venv Python path')
        library = path(build['library_path'])
        need(library == scratch / 'build/libtinyinfer_backend.so' and sha(read(library)) == build['library_sha256'], 'actual native library bytes')
        gh = helper(directory, 'gpu-helper')
        before(bh['finished_timestamp'], gh['timestamp'], 'build exits before GPU helper starts')
        ga = gh['argv']
        need(ga[:3] == [python, '-m', 'autotune.evaluate'], 'GPU helper interpreter/module')
        need(len(ga) == 11 and ga[3::2] == ['--config', '--proposal', '--build-manifest', '--output'], 'exact GPU helper flags without extras')
        for flag, value in [('--config', directory / 'fixture.json'), ('--proposal', directory / 'proposal.json'),
                            ('--build-manifest', manifest_path), ('--output', directory / 'gpu-evaluation')]:
            need(same_path(ga[ga.index(flag)+1], value), 'GPU helper exact ' + flag)
        epath = directory / 'gpu-evaluation/evaluation.json'
        evaluation = load(epath)
        need(evaluation['status'] == row['status'] and evaluation['proposal'] == row['proposal'], 'GPU evaluation status/tuple')
        need(all(row[k] == value for k, value in evaluation.items()), 'all actual evaluation fields equal embedded row')
        need(evaluation['build_manifest_sha256'] == sha(read(manifest_path)), 'evaluation binds external build-manifest bytes')
        need(load(directory / 'fixture.json') == config['fixture'] and
             evaluation['config_sha256'] == sha(read(directory / 'fixture.json')), 'exact frozen fixture file')
        before(gh['timestamp'], evaluation['timestamp'], 'actual evaluation starts within helper')
        before(evaluation['finished_timestamp'], gh['finished_timestamp'], 'actual evaluation ends before helper exit')
        if row['status'] == 'passed':
            need(gh['exit_code'] == 0 and gh['status'] == 'completed', 'passed GPU helper exit')
            need(same_path(row['evaluation_record_path'], epath) and row['evaluation_record_sha256'] == sha(read(epath)), 'exact fresh evaluation path/hash')
            key = str(epath.resolve())
            need(key not in evaluation_paths, 'independent/search evaluation path is unique: ' + key)
            evaluation_paths[key] = role
            env = evaluation['environment']
            need(env['build_manifest'] == build and env['build_manifest_sha256'] == sha(read(manifest_path)), 'embedded/external build exact bytes')
            need(env['library_sha256'] == build['library_sha256'] and same_path(env['library_path'], library), 'evaluated actual library identity')
            need(evaluation['fixture'] == fixture, 'common input/bias fixture bytes')
            correctness = load(epath.parent / 'correctness.json')
            cases = correctness['cases']
            need(correctness['status'] == 'passed' and cases and all(c['status'] == 'passed' for c in cases), 'all recorded native correctness cases pass')
            need(len({c['name'] for c in cases}) == len(cases), 'native correctness names unique')
        else:
            need(gh['exit_code'] != 0 and gh['status'] == 'failed', 'retained actual GPU/test failure')
        return bh['timestamp'], gh['finished_timestamp']

    run_end = top['timestamp']
    for seed_index, seed in enumerate(top['seeds']):
        need(seed['seed'] == config['seeds'][seed_index], 'ordered seed identity')
        directory = run / ('seed-' + str(seed['seed']))
        baseline, fixture = seed['baseline'], seed['baseline']['fixture']
        need(baseline['proposal'] == proposal(config['baseline_id']), 'fixed shared initial baseline ID')
        first, previous_end = candidate(baseline, directory / 'candidate-initial-baseline', fixture, 'initial_baseline')
        before(run_end, first, 'seed baseline starts after prior seed/native phase')
        for method in (('random', 'llm') if seed_index % 2 == 0 else ('llm', 'random')):
            outcome = seed['methods'][method]
            method_dir = directory / method
            need(load(method_dir / 'search.json') == outcome, 'final method journal identity')
            need(outcome['baseline'] == baseline and outcome['proposal_budget'] == 64 and outcome['evaluation_budget'] == 8, 'method equal baseline and budgets')
            rng, seen, unique = random.Random(seed['seed']), {0}, 0
            method_start, last_end = None, previous_end
            for index, attempt in enumerate(outcome['attempts']):
                need(attempt['attempt'] == index and unique < 8, 'attempt order and immediate unique-budget stop')
                if method == 'llm':
                    pdir = method_dir / 'proposals' / ('proposal-%04d' % index)
                    receipt = load(pdir / 'proposal.json')
                    prompt = read(pdir / 'prompt.txt')
                    need(sha(prompt) == receipt['prompt_sha256'] and prompt.decode() == prompt_for(baseline, outcome['attempts'][:index]), 'exact frozen prompt/history bytes')
                    expected = [receipt['argv'][0], '-m', receipt['argv'][2], '-ngl', '0', '--seed', str((seed['seed']+index) % (2**31)),
                                '-n', str(config['llama']['max_output_tokens']), '-c', str(config['llama']['context_tokens']), '--temp', '0.7', '--no-display-prompt', '--no-conversation', '-f']
                    need(receipt['argv'][:-1] == expected and same_path(receipt['argv'][-1], pdir / 'prompt.txt') and
                         same_path(receipt['argv'][0], config['llama']['executable']) and same_path(receipt['argv'][2], config['llama']['model']), 'fixed LLM subprocess argv')
                    for stream in ('stdout.txt', 'stderr.txt'):
                        present = (pdir / stream).exists()
                        need(present == (stream+'_sha256' in receipt), 'proposer stream existence/receipt consistency: ' + stream)
                        if present:
                            need(sha(read(pdir / stream)) == receipt[stream+'_sha256'], 'retained proposer raw ' + stream)
                    before(last_end, receipt['timestamp'], 'previous native/helper exits before next proposer')
                    before(receipt['timestamp'], receipt['finished_timestamp'], 'proposer interval')
                    method_start = method_start or receipt['timestamp']
                    last_end = receipt['finished_timestamp']
                    need(receipt['model_sha256'] == config['llama']['model_sha256'] and receipt['executable_sha256'] == config['llama']['executable_sha256'], 'pinned proposer identities')
                    if receipt['status'] == 'failed':
                        need(attempt['status'] == 'proposal_error' and 'raw_proposal' not in attempt, 'actual proposer failure consumes proposal attempt')
                        continue
                    need(receipt['status'] == 'returned_unvalidated' and receipt['exit_code'] == 0, 'actual proposer successful return')
                    stdout = read(pdir / 'stdout.txt').decode('utf-8')
                    removed = stdout.endswith(TRAILER)
                    raw = (stdout[:-len(TRAILER)] if removed else stdout).strip()
                    need(receipt['stdout_transport'] == dict(contract='llama.cpp_completion_eog_stdout_v1', upstream_revision=UPSTREAM,
                         exact_terminal_suffix=TRAILER, maximum_suffix_removals=1, cli_framing_removed=removed), 'exact single terminal framing receipt')
                    need(receipt['returned_payload_sha256'] == sha(raw.encode()) and attempt['raw_proposal'] == raw, 'raw bytes to exact returned payload')
                else:
                    raw = json.dumps(proposal(rng.randrange(12)))
                    need(attempt['raw_proposal'] == raw, 'replayed declared random proposal stream')
                try:
                    value = strict(raw)
                except ValueError:
                    need(attempt['status'] == 'invalid' and 'proposal' not in attempt, 'strict raw rejection consumes proposal only')
                    continue
                need(attempt['proposal'] == value, 'raw strict object equals recorded proposal')
                identifier = value['config_id']
                if identifier in seen:
                    need(attempt['status'] == 'duplicate' and 'evaluation' not in attempt, 'duplicate consumes no unique evaluation')
                    continue
                need(attempt['status'] in ('passed', 'compile_failed', 'test_failed', 'evaluation_error'), 'unique candidate terminal attempt')
                seen.add(identifier); unique += 1
                row = attempt['evaluation']
                start, end = candidate(row, method_dir / ('candidate-' + str(index)), fixture, 'search')
                before(last_end, start, 'proposer/previous helper exits before candidate build')
                method_start = method_start or start
                last_end = end
            need(outcome['evaluations'] == unique and outcome['proposal_attempts'] == len(outcome['attempts']), 'replayed unique/proposal counts')
            need(unique == 8 or len(outcome['attempts']) == 64, 'unique budget or strict full proposal exhaustion')
            need(outcome['status'] == ('completed' if unique == 8 else 'incomplete_proposal_cap') and outcome['budget_exhausted'] == (unique == 8), 'method terminal status')
            for name in ('baseline', 'winner'):
                row = outcome['independent'][name]
                start, end = candidate(row, method_dir / ('candidate-remeasure-' + name), fixture, 'independent_' + name)
                before(last_end, start, 'fresh independent ' + name + ' follows selection/previous remeasurement')
                last_end = end
            before(previous_end, method_start, 'declared alternating method sequence')
            intervals[str(seed['seed']) + '/' + method] = dict(start=method_start, end=last_end)
            previous_end = last_end
        run_end = previous_end
    need(top['status'] == ('completed' if top['comparable'] else 'incomplete_incomparable'), 'top terminal/comparability status')
    before(previous_end, top['finished_timestamp'], 'all native helpers end before top-level closure')
    for receipt in args.lifecycle:
        load(path(receipt))  # Attachment hash only; no invented/unrecorded PID inference.
    result = dict(status='passed_bounded_supplementary_provenance', utc=datetime.now(timezone.utc).isoformat(),
                  source=HEAD, run_status=top['status'], checks=len(CHECKS), assertions=CHECKS,
                  method_intervals=intervals, unique_evaluation_paths=evaluation_paths,
                  scope='Fresh output/scratch paths, exact argv/times, actual native build bytes/logs, proposer raw transport, strict classifications and stop boundaries. Root raw-statistics acceptance remains required.',
                  process_identity_scope='Helper/proposer receipts do not record per-child PID/start ticks. Fresh procedure is bounded to actual paths/argv/times/builds and reviewed synchronous subprocess control; OS identity/cleanup is limited to owner outer lifecycle/phase evidence. Absent PID fields are not a defect or rerun gate.',
                  lifecycle_attachments=args.lifecycle, read_files=READS)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], checks=len(CHECKS), output=str(output))))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if AUDIT_OUTPUT is not None and not AUDIT_OUTPUT.exists():
            AUDIT_OUTPUT.write_text(json.dumps(dict(
                status='failed_supplementary_review', utc=datetime.now(timezone.utc).isoformat(),
                error_type=type(error).__name__, error=str(error),
                needed_receipt=getattr(error, 'filename', None),
                passed_assertions=CHECKS, read_files=READS), indent=2)+'\n', encoding='utf-8')
        raise
