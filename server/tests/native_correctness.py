"""Finite native socket/direct token gate; explicitly run only in root's window.

The direct native context is closed BEFORE the server subprocess starts. This
is a functional gate, never a latency benchmark. No Torch/HF import is needed.
Direct synchronous native C calls rely on the external recorded-command timeout;
this script's own deadlines cover readiness, asynchronous sockets and cleanup.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def parse_proc_stat(text):
    """Linux stat fields; comm may itself contain spaces and parentheses."""
    opening, closing = text.find('('), text.rfind(')')
    if opening < 1 or closing < opening:
        raise ValueError('malformed process stat')
    rest = text[closing+1:].split()
    if len(rest) < 20:
        raise ValueError('short process stat')
    return {'pid': int(text[:opening].strip()), 'state': rest[0],
            'ppid': int(rest[1]), 'pgrp': int(rest[2]), 'session': int(rest[3]),
            'start_ticks': int(rest[19])}


def classify_member(row, scope):
    """Pure identity decision; no journal PID participates in ownership."""
    if row.get('pgrp') != scope['pgrp'] and row.get('session') != scope['session']:
        return 'outside'
    if row.get('pgrp') != scope['pgrp'] or row.get('session') != scope['session']:
        return 'unknown'
    if row.get('state') == 'Z':
        known = scope.get('known_members', {}).get(row.get('pid'))
        return ('zombie_known' if known and known['start_ticks'] == row.get('start_ticks')
                else 'unknown')
    if row.get('identity_error'):
        return 'unknown'
    if (row.get('exe') == scope['native_exe'] and row.get('argv') == scope['native_argv']):
        return 'native'
    if (row.get('pid') == scope['launcher_pid'] and row.get('exe') == scope['launcher_exe']
            and row.get('argv') == scope['launcher_argv']
            and (scope.get('launcher_start_ticks') is None
                 or row.get('start_ticks') == scope['launcher_start_ticks'])):
        return 'launcher'
    return 'unknown'


def process_identity(pid, scope):
    """Re-read stat around exe/cmdline to reject exit/exec/PID-reuse races."""
    process = Path('/proc')/str(pid)
    before = parse_proc_stat((process/'stat').read_text())
    if before['pgrp'] != scope['pgrp'] and before['session'] != scope['session']:
        return before
    row = dict(before)
    try:
        row['exe'] = str((process/'exe').resolve(strict=True))
        row['argv'] = [os.fsdecode(v) for v in (process/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    except OSError as exc:
        row['identity_error'] = type(exc).__name__+': '+str(exc)
    after = parse_proc_stat((process/'stat').read_text())
    if any(before[key] != after[key] for key in ('pid', 'pgrp', 'session', 'start_ticks')):
        row['identity_error'] = 'identity changed while read'
    row['state'] = after['state']
    return row


def scan_scope(scope):
    """Independent whole-/proc session/group scan, with uncertainty explicit."""
    found = {'confirmed': [], 'zombies': [], 'unknown': [], 'scan_errors': [], 'scan_complete': True}
    try:
        pids = [int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal()]
    except OSError as exc:
        found.update(scan_complete=False, scan_errors=[str(exc)])
        return found
    for pid in pids:
        try:
            row = process_identity(pid, scope)
        except (FileNotFoundError, ProcessLookupError):
            continue  # This entry no longer exists; never infer any live identity.
        except (OSError, ValueError) as exc:
            found['scan_complete'] = False
            found['scan_errors'].append({'pid': pid, 'error': type(exc).__name__+': '+str(exc)})
            continue
        role = classify_member(row, scope)
        if role == 'outside':
            continue
        row['role'] = role
        bucket = 'zombies' if role == 'zombie_known' else ('unknown' if role == 'unknown' else 'confirmed')
        found[bucket].append(row)
        if bucket == 'confirmed':
            scope.setdefault('known_members', {})[pid] = {'role': role, 'start_ticks': row['start_ticks']}
    return found


def signal_confirmed(row, scope, signum):
    """Signal a verified identity through pidfd, never an entire process group."""
    fd = os.pidfd_open(row['pid'], 0)
    try:
        current = process_identity(row['pid'], scope)
        role = classify_member(current, scope)
        if role == 'zombie_known' and current.get('start_ticks') == row['start_ticks']:
            return 'already_exited'
        if role != row['role'] or current.get('start_ticks') != row['start_ticks']:
            raise RuntimeError('confirmed identity changed before signal')
        signal.pidfd_send_signal(fd, signum)
        return 'signalled'
    finally:
        os.close(fd)


def verify_server_build_sources(root, expected):
    if not expected:
        raise ValueError('nonempty server build source hashes required')
    for relative, digest in expected.items():
        path = (root/relative).resolve(strict=True)
        if (not path.is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError('server source differs from completed build: '+relative)


def server_python_snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'server').rglob('*.py')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'binary', 'build-receipt', 'model-build-receipt',
                 'model-config', 'model-gate', 'kernel-gate', 'frozen-tokens',
                 'output', 'server-output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--startup-timeout', type=int, default=180)
    parser.add_argument('--request-timeout', type=int, default=60)
    parser.add_argument('--execute', required=True, action='store_true')
    args = parser.parse_args()
    if os.name != 'posix' or min(args.startup_timeout, args.request_timeout) < 1:
        parser.error('Linux and positive finite timeouts required')
    # Imports remain behind explicit execution; importing this file is inert.
    from llm.build_receipt import command, source_snapshot, verify
    from llm.evaluate import actual_environment, artifact_inputs, validate_config, verify_kernel_gate
    from llm.protocol import canonical_hash, validate_frozen
    from llm.runtime import NativeModel
    from server import protocol
    from server.metadata import digest, metadata, now, output_directory, write
    from server.run import validate

    root = Path(__file__).resolve().parents[2]
    out = output_directory(args.output, root)
    server_out = Path(args.server_output).resolve()
    t0 = time.monotonic()
    report = {'schema_version': 1, 'status': 'started', 'started_at': now(),
              'argv': sys.argv, 'scope': 'finite exact greedy socket/direct parity, no performance result',
              'timeout_scope': 'readiness/socket/cleanup deadlines only; synchronous direct C calls require external m2_record total timeout',
              'server_output': str(server_out), 'events': [], 'server_process': {}, 'cleanup_snapshots': []}
    write(out/'startup.json', report)
    proc = None
    input_hashes = None
    source_before = None
    git_before = None
    scope = None
    cleanup_attempted = False
    cleanup_confirmed = False
    cleanup_uncertain = False
    python_before = None
    launch_in_progress = False
    pending_interrupt = None

    def event(kind, **detail):
        row = {'at': now(), 'event': kind, **detail}
        report['events'].append(row)
        with (out/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def snapshot(label, retain=True):
        nonlocal cleanup_uncertain
        result = scan_scope(scope)
        if result['unknown'] or not result['scan_complete']:
            cleanup_uncertain = True
        if retain:
            report['cleanup_snapshots'].append({'label': label, 'at': now(), **result})
        return result

    def shutdown():
        nonlocal cleanup_attempted, cleanup_confirmed, cleanup_uncertain
        if proc is None:
            return True
        if cleanup_attempted:
            return cleanup_confirmed and not cleanup_uncertain
        cleanup_attempted = True
        if scope is None:
            cleanup_uncertain = True
            report['server_process']['cleanup_error'] = 'no independently constructed process scope'
            return False
        # Only numeric IDs are learned from our Popen object; expected exe/argv
        # were constructed before launch, not read from a possibly absent journal.
        scope.update(launcher_pid=proc.pid, pgrp=proc.pid, session=proc.pid)

        def signal_rows(rows, signum):
            nonlocal cleanup_uncertain
            for row in rows:
                try:
                    outcome = signal_confirmed(row, scope, signum)
                    event('confirmed_process_signal', outcome=outcome, pid=row['pid'], role=row['role'], signal=signum,
                          start_ticks=row['start_ticks'])
                except (ProcessLookupError, FileNotFoundError):
                    continue
                except (OSError, ValueError, RuntimeError) as exc:
                    cleanup_uncertain = True
                    event('cleanup_identity_error', pid=row['pid'], error=type(exc).__name__+': '+str(exc))

        initial = snapshot('before_cleanup')
        # Native first covers the launcher's Popen→handler/PID-journal race.
        signal_rows(sorted(initial['confirmed'], key=lambda row: row['role'] != 'native'), signal.SIGTERM)
        deadline = time.monotonic()+30
        empty_passes = 0
        while time.monotonic() < deadline:
            proc.poll()  # Reap the launcher independently of its optional journal.
            current = snapshot('cleanup_poll', retain=False)
            empty = (current['scan_complete'] and not current['confirmed']
                     and not current['zombies'] and not current['unknown'])
            empty_passes = empty_passes+1 if empty and proc.returncode is not None else 0
            if empty_passes >= 2:
                cleanup_confirmed = True
                break
            time.sleep(0.05)
        if not cleanup_confirmed:
            remaining = snapshot('before_final_kill')
            if remaining['confirmed']:
                signal_rows(remaining['confirmed'], signal.SIGKILL)
                report['server_process']['forced_kill'] = True
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleanup_uncertain = True
            final = snapshot('after_cleanup')
            cleanup_confirmed = (final['scan_complete'] and not final['confirmed']
                                 and not final['zombies'] and not final['unknown']
                                 and proc.poll() is not None)
        else:
            snapshot('after_cleanup')
        report['server_process']['launcher_exit_code'] = proc.returncode
        report['server_process'].update(cleanup_confirmed=cleanup_confirmed,
                                       cleanup_uncertain=cleanup_uncertain)
        return cleanup_confirmed and not cleanup_uncertain

    def interrupted(signum, frame):
        nonlocal pending_interrupt
        if launch_in_progress:
            pending_interrupt = signum
            return
        raise KeyboardInterrupt('native socket gate interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    try:
        if (server_out.parent != (root/'results/v2').resolve() or server_out == out
                or server_out.exists() or server_out.name.startswith('.')):
            raise ValueError('server-output must be a separate fresh direct results/v2 child')
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('race-safe scoped cleanup requires Linux pidfd APIs')
        config = validate(json.loads(Path(args.config).read_text()))
        model_config = validate_config(json.loads(Path(args.model_config).read_text()))
        if (config['model'] != 'native' or config['max_new_tokens'] < 5 or config['capacity'] < 64
                or config['max_pending'] < 3 or config['per_connection'] < 3):
            raise ValueError('native mode and capacity/admission for finite mixed cases required')
        for key in ('capacity', 'gemm_config', 'attention_config'):
            if config[key] != model_config[key]:
                raise ValueError('server/model-gate options differ: '+key)
        receipt = json.loads(Path(args.model_build_receipt).read_text())
        verify(receipt, root, receipt['library_path'])
        kernel_run_path = verify_kernel_gate(args.kernel_gate, receipt['backend_sha256'])
        server_receipt = json.loads(Path(args.build_receipt).read_text())
        libraries = server_receipt.get('native_libraries', {})
        if (server_receipt.get('build_exit_code') != 0 or digest(args.binary) != server_receipt.get('binary_sha256')
                or set(libraries) != {'tinyinfer_llm', 'tinyinfer_backend'}):
            raise ValueError('completed matching native server build receipt required')
        for key, prefix in (('tinyinfer_llm', 'library'), ('tinyinfer_backend', 'backend')):
            if (libraries[key]['sha256'] != receipt[prefix+'_sha256']
                    or Path(libraries[key]['path']).resolve() != Path(receipt[prefix+'_path']).resolve()):
                raise ValueError('server and direct model resolve different libraries')
        verify_server_build_sources(root, server_receipt['source_sha256'])
        python_before = server_python_snapshot(root)
        manifest = Path(config['manifest']).resolve(strict=True)
        artifact = json.loads(manifest.read_text())
        precision = artifact['precision']
        if precision not in ('fp16', 'int8') or artifact.get('binary') != 'weights.bin':
            raise ValueError('pinned standard FP16/INT8 artifact required')
        frozen = json.loads(Path(args.frozen_tokens).read_text())
        validate_frozen(frozen)
        if len(frozen['tokens']) < 128:
            raise ValueError('at least128 frozen IDs required for finite cases')
        gate = json.loads(Path(args.model_gate).read_text())
        source_before = source_snapshot(root)
        env = metadata(root)
        git_before = env['git']['sha']
        status = command(['git', 'status', '--porcelain', '--untracked-files=no'], root)
        if status['exit_code'] or status['stdout'].strip():
            raise ValueError('clean tracked checkpoint required')
        if (gate.get('status') != 'passed' or gate.get('environment') != actual_environment()
                or gate.get('source_sha256') != source_before
                or gate.get('library_sha256') != receipt['library_sha256']
                or gate.get('build_receipt_sha256') != digest(args.model_build_receipt)
                or gate.get('kernel_gate_sha256') != digest(args.kernel_gate)
                or gate.get('config_sha256') != canonical_hash(model_config)
                or gate.get('frozen_token_ids_sha256') != frozen['token_ids_sha256']
                or gate.get('artifact_manifest_sha256', {}).get(precision) != digest(manifest)
                or gate.get('git', {}).get('stdout', '').strip() != git_before
                or server_receipt['environment']['git']['sha'] != git_before):
            raise ValueError('same-checkpoint kernel/model/artifact/config/token gate required')
        fixed = [args.config, args.model_config, args.binary, args.build_receipt,
                 args.model_build_receipt, args.model_gate, args.kernel_gate, kernel_run_path, args.frozen_tokens,
                 receipt['library_path'], receipt['backend_path'], manifest, manifest.parent/'weights.bin',
                 Path(__file__), *[root/relative for relative in python_before]]
        input_hashes = artifact_inputs(fixed)
        if input_hashes[str((manifest.parent/'weights.bin').resolve())] != artifact['binary_sha256']:
            raise ValueError('artifact binary differs from its manifest')
        report.update(environment=env, precision=precision, input_sha256=input_hashes,
                      source_sha256=source_before, server_python_sha256=python_before,
                      server_build_source_sha256=server_receipt['source_sha256'],
                      frozen_token_ids_sha256=frozen['token_ids_sha256'])
        # Readiness is only accepted from our own process logs. Bind first to
        # reject occupied endpoints; a later race remains a recorded startup failure.
        with socket.socket() as probe:
            probe.bind((config['host'], config['port']))
        specs = [(0, 8, 1), (19, 17, 3), (53, 33, 5)]
        cases = []
        with NativeModel(receipt['library_path'], manifest, device=config['device'],
                         max_batch=config['max_batch'], cache_slots=config['max_batch'],
                         capacity=config['capacity'], gemm_config=config['gemm_config'],
                         attention_config=config['attention_config']) as model:
            for index, (offset, length, count) in enumerate(specs):
                tokens = frozen['tokens'][offset:offset+length]
                model.reset_all()
                result = model.greedy([tokens], count, slots=[0])
                if result['output_lengths'] != [count] or len(result['tokens'][0]) != count:
                    raise AssertionError('direct model output length mismatch')
                cases.append({'case': index, 'offset': offset, 'length': length, 'new_tokens': count,
                              'prompt_sha256': canonical_hash(tokens), 'expected': result['tokens'][0]})
            report['direct_trace'] = model.trace()
        event('direct_context_closed_before_server_start')
        report['cases'] = cases
        write(out/'direct.json', {'cases': cases, 'trace': report['direct_trace']})
        argv = [sys.executable, '-B', '-m', 'server.run', '--config', str(Path(args.config).resolve()),
                '--binary', str(Path(args.binary).resolve()), '--build-receipt', str(Path(args.build_receipt).resolve()),
                '--output', str(server_out)]
        # Construct exactly what current server.run will launch, BEFORE it can
        # create a child or write any journal PID. Config insertion order is kept.
        native_argv = [str(Path(args.binary).resolve(strict=True))]
        for key, value in config.items():
            if key != 'schema_version':
                native_argv.extend(['--'+key.replace('_', '-'), str(value)])
        scope = {'launcher_pid': None, 'pgrp': None, 'session': None,
                 'launcher_exe': str(Path(sys.executable).resolve(strict=True)),
                 'launcher_argv': argv, 'launcher_start_ticks': None,
                 'native_exe': str(Path(args.binary).resolve(strict=True)), 'native_argv': native_argv,
                 'known_members': {},
                 'creation': 'Popen(start_new_session=True); identity independent of server.json'}
        with (out/'launcher-stdout.log').open('x') as stdout, (out/'launcher-stderr.log').open('x') as stderr:
            launch_in_progress = True
            try:
                proc = subprocess.Popen(argv, cwd=root, stdout=stdout, stderr=stderr, start_new_session=True)
                scope.update(launcher_pid=proc.pid, pgrp=proc.pid, session=proc.pid)
            finally:
                launch_in_progress = False
            if pending_interrupt is not None:
                raise KeyboardInterrupt('native socket gate interrupted during launcher creation')
            try:
                launcher = process_identity(proc.pid, scope)
                if classify_member(launcher, scope) == 'launcher':
                    scope['launcher_start_ticks'] = launcher['start_ticks']
                    scope['known_members'][proc.pid] = {'role': 'launcher', 'start_ticks': launcher['start_ticks']}
            except (OSError, ValueError) as exc:
                event('launcher_initial_identity_unavailable', error=type(exc).__name__+': '+str(exc))
            report['server_process'].update(launcher_pid=proc.pid, argv=argv, independent_scope=scope)
            event('server_launcher_started', pid=proc.pid)
            deadline = time.monotonic()+args.startup_timeout
            ready = None
            while time.monotonic() < deadline:
                log = server_out/'stdout.log'
                if log.exists():
                    for line in log.read_text(errors='replace').splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get('ready') is True:
                            ready = row
                            break
                if ready is not None:
                    break
                if proc.poll() is not None:
                    raise RuntimeError('server exited before native readiness; logs retained')
                time.sleep(0.05)
            if ready is None:
                raise TimeoutError('native server startup timeout; logs retained')
            if ready.get('model') != 'native' or ready.get('port') != config['port']:
                raise AssertionError('own server readiness identity mismatch')
            report['ready'] = ready
            event('native_server_ready', **ready)
            ready_scope = snapshot('native_readiness')
            if (not ready_scope['scan_complete'] or ready_scope['unknown'] or ready_scope['zombies']
                    or sum(row['role'] == 'native' for row in ready_scope['confirmed']) != 1
                    or sum(row['role'] == 'launcher' for row in ready_scope['confirmed']) != 1):
                raise AssertionError('ready server process ownership could not be independently confirmed')

            async def exchange(group, items):
                writer = None
                pending = {identifier: case for identifier, case in items}
                first = {}
                finals = {}
                try:
                    async with asyncio.timeout(args.request_timeout):
                        reader, writer = await asyncio.open_connection(config['host'], config['port'])
                        frames = []
                        for identifier, case in items:
                            prompt = frozen['tokens'][case['offset']:case['offset']+case['length']]
                            frames.append(protocol.request(identifier, prompt, case['new_tokens']))
                        writer.write(b''.join(frames))
                        await writer.drain()
                        event('requests_sent', group=group, ids=list(pending))
                        while len(finals) < len(pending):
                            response = await protocol.read_response(reader)
                            event('socket_response', group=group, response=response)
                            identifier = response['id']
                            if identifier not in pending or identifier in finals:
                                raise AssertionError('unknown/duplicate terminal request ID')
                            case = pending[identifier]
                            if response['status'] != 0 or not 1 <= response['batch_size'] <= config['max_batch']:
                                raise AssertionError('native response error or invalid batch size')
                            if response['kind'] == protocol.FIRST:
                                if identifier in first or response['tokens'] != case['expected'][:1]:
                                    raise AssertionError('FIRST token mismatch or duplicate')
                                first[identifier] = response
                            else:
                                if identifier not in first or response['tokens'] != case['expected']:
                                    raise AssertionError('FINAL greedy tokens mismatch or missing FIRST')
                                finals[identifier] = response
                        return {'group': group, 'first': first, 'final': finals}
                except BaseException as exc:
                    event('socket_error', group=group, type=type(exc).__name__, message=str(exc),
                          first_ids=list(first), final_ids=list(finals))
                    raise
                finally:
                    if writer is not None:
                        writer.close()
                        try:
                            await asyncio.wait_for(writer.wait_closed(), timeout=3)
                        except (OSError, TimeoutError):
                            pass

            async def socket_cases():
                results = [await exchange('mixed_coalesced', list(zip((1, 2, 3), cases))),
                           await exchange('slot_reuse_reversed', list(zip((4, 5, 6), reversed(cases))))]
                results.extend(await asyncio.gather(exchange('connection_id_reuse_a', [(1, cases[0])]),
                                                    exchange('connection_id_reuse_b', [(1, cases[2])])))
                return results

            report['socket_cases'] = asyncio.run(socket_cases())
            observed = [r['batch_size'] for group in report['socket_cases'] for r in group['final'].values()]
            report['max_observed_batch'] = max(observed)
            if config['max_batch'] > 1 and max(observed) < 2:
                raise AssertionError('batching-on gate observed no actual multi-request batch')
            if not shutdown():
                raise AssertionError('native cleanup uncertain/incomplete; no clean outcome accepted')
        server_record = json.loads((server_out/'server.json').read_text())
        report['server_process']['record'] = server_record
        if (proc.returncode or report['server_process'].get('forced_kill')
                or server_record.get('status') != 'stopped' or not cleanup_confirmed or cleanup_uncertain):
            raise AssertionError('native server did not shut down cleanly')
        traces = []
        for line in (server_out/'stderr.log').read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'native_trace_status' in row:
                traces.append(row)
        active = 'gemm_fp16' if precision == 'fp16' else 'gemm_int8'
        required = ('forward_calls', 'embedding', active, 'layernorm', 'gelu', 'attention',
                    'residual', 'kv_scatter_gather', 'lm_head', 'argmax')
        if len(traces) != 1 or traces[0]['native_trace_status'] != 0 or any(traces[0].get(k, 0) <= 0 for k in required):
            raise AssertionError('real native worker dispatch trace missing/incomplete')
        report['server_trace'] = traces[0]
        verify_server_build_sources(root, server_receipt['source_sha256'])
        end_status = command(['git', 'status', '--porcelain', '--untracked-files=no'], root)
        report['end_tracked_status'] = end_status
        if (source_snapshot(root) != source_before or artifact_inputs(fixed) != input_hashes
                or server_python_snapshot(root) != python_before
                or end_status['exit_code'] or end_status['stdout'].strip()
                or metadata(root)['git']['sha'] != git_before):
            raise AssertionError('checkpoint or fixed inputs changed during native socket gate')
        report.update(status='passed', finished_at=now(), wall_seconds=time.monotonic()-t0)
        write(out/'results.json', report)
        return 0
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), finished_at=now(), wall_seconds=time.monotonic()-t0)
        shutdown()
        write(out/'failure.json', report)
        return 1
    finally:
        shutdown()
        saved_path = server_out/'server.json'
        if saved_path.exists():
            try:
                report['server_process']['final_server_record'] = json.loads(saved_path.read_text())
            except (OSError, ValueError):
                pass
        summary = dict(report['server_process'], finished_at=now(),
                       no_active_launcher=proc is None or proc.poll() is not None,
                       independent_scope_empty=cleanup_confirmed if proc is not None else True,
                       cleanup_uncertain=cleanup_uncertain,
                       clean=proc is None or (cleanup_confirmed and not cleanup_uncertain))
        write(out/'process-closure.json', summary)


if __name__ == '__main__':
    raise SystemExit(main())
