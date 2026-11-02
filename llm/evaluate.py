"""Formal, non-overwriting native/HF parity and greedy benchmark entry point.

Requires root's M2 release, completed model+kernel GPU gates, local pinned bytes
and a clean source checkpoint. Importing this module alone does not use Torch.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import traceback
from .build_receipt import command, source_snapshot, verify
from .export_weights import sha256_file
from .protocol import MODEL_SHA256, canonical_hash, perplexity, validate_frozen, verify_tokenizer
from .runtime import NativeModel


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def percentile(values, p):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('invalid raw samples')
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summary(values):
    return {'count': len(values), 'mean': statistics.mean(values),
            'std_population': statistics.pstdev(values),
            'p50': percentile(values, 50), 'p95': percentile(values, 95)}


def actual_environment():
    return 'Docker' if Path('/.dockerenv').exists() else ('WSL2' if 'microsoft' in platform.release().lower() else platform.system())


def verify_kernel_gate(path, backend_sha):
    gate = json.loads(Path(path).read_text())
    run_path = Path(path).resolve().parent / 'run.json'
    run = json.loads(run_path.read_text())
    env = run.get('environment', {})
    if (gate.get('status') != 'passed' or not gate.get('cases')
            or any(row.get('status') != 'passed' for row in gate['cases'])
            or run.get('status') != 'completed' or env.get('environment') != actual_environment()
            or env.get('library_sha256') != backend_sha):
        raise ValueError('actual matching CUDA kernel gate receipt required')
    return run_path


def artifact_inputs(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve(strict=True)
        result[str(path)] = sha256_file(path)
    return result


def loaded_libraries():
    maps = Path('/proc/self/maps')
    if not maps.exists():
        return []
    return sorted({line.split()[-1] for line in maps.read_text().splitlines()
                   if any(name in line for name in ('libcudart', 'libtinyinfer_llm', 'libtinyinfer_backend'))})


def validate_config(c):
    keys = {'schema_version', 'seed', 'warmups', 'repeats', 'batches', 'prompt_tokens',
            'new_tokens', 'prompt_offset_step', 'capacity', 'gemm_config', 'attention_config', 'cached_atol',
            'cached_rtol', 'hf_atol', 'hf_rtol', 'fp16_ppl_relative_tolerance', 'compile_ppl_relative_tolerance',
            'attempt_compile', 'compile_backend', 'compile_dynamic', 'compile_fullgraph'}
    if set(c) != keys or type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('evaluation config schema mismatch')
    for k in ('seed', 'warmups', 'repeats', 'prompt_tokens', 'prompt_offset_step', 'new_tokens', 'capacity'):
        if type(c[k]) is not int or c[k] < (0 if k in ('seed', 'warmups') else 1):
            raise ValueError(k)
    if c['batches'] != [1, 8, 32] or any(type(b) is not int for b in c['batches']) or c['capacity'] > 1024 or c['prompt_tokens'] + c['new_tokens'] > c['capacity'] or c['seed'] > 2**32 - 1:
        raise ValueError('required batches/context budget')
    if type(c['gemm_config']) is not int or c['gemm_config'] not in range(12) or type(c['attention_config']) is not int or c['attention_config'] not in (0, 1):
        raise ValueError('kernel config')
    for k in ('cached_atol', 'cached_rtol', 'hf_atol', 'hf_rtol', 'fp16_ppl_relative_tolerance', 'compile_ppl_relative_tolerance'):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(k)
    for k in ('attempt_compile', 'compile_dynamic', 'compile_fullgraph'):
        if type(c[k]) is not bool:
            raise ValueError(k)
    if c['compile_backend'] != 'inductor':
        raise ValueError('explicit inductor backend required')
    return c


def hf_score(model, frozen, journal):
    import torch
    import torch.nn.functional as F
    rows, total, count = [], 0., 0
    with torch.inference_mode():
        for index, window in enumerate(frozen['windows']):
            ids = torch.tensor([frozen['tokens'][window['begin']:window['end']]], dtype=torch.long, device='cuda')
            mask = torch.tensor(window['mask'][1:], dtype=torch.bool, device='cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
            losses = F.cross_entropy(logits, ids[0, 1:], reduction='none')
            nll = losses[mask].double().sum().item()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            targets = sum(window['mask'])
            total += nll
            count += targets
            row = {'index': index, 'begin': window['begin'], 'end': window['end'],
                   'sum_nll': nll, 'target_count': targets, 'wall_ms': elapsed}
            rows.append(row)
            journal('hf_ppl_window', row)
    if count != frozen['target_count']:
        raise AssertionError('HF target count mismatch')
    return {'sum_nll': total, 'target_count': count, 'perplexity': perplexity(total, count), 'windows': rows}


def native_score(model, frozen, journal):
    total, count, rows = 0., 0, []
    for index, window in enumerate(frozen['windows']):
        start = time.perf_counter()
        score = model.score(frozen['tokens'][window['begin']:window['end']], window['mask'])
        elapsed = (time.perf_counter() - start) * 1000
        total += score['sum_nll']
        count += score['target_count']
        row = {'index': index, 'begin': window['begin'], 'end': window['end'], **score, 'wall_ms': elapsed}
        rows.append(row)
        journal('native_ppl_window', row)
    if count != frozen['target_count']:
        raise AssertionError('native target count mismatch')
    return {'sum_nll': total, 'target_count': count, 'perplexity': perplexity(total, count), 'windows': rows}


def hf_greedy(model, prompts, new_tokens):
    import torch
    if len(set(map(len, prompts))) != 1:
        raise ValueError('benchmark uses matched equal-length prompts')
    # Host packing outside TTFT, like ctypes host packing in NativeModel.greedy.
    host = torch.tensor(prompts, dtype=torch.long)
    batch, prompt_length = host.shape
    generated = [[] for _ in prompts]
    start = time.perf_counter()
    ids = host.to('cuda')
    with torch.inference_mode():
        result = model(input_ids=ids, use_cache=True)
        past = result.past_key_values
        token = result.logits[:, -1].argmax(-1, keepdim=True)
        first_ids = token[:, 0].tolist()  # synchronizes/copies exactly the token IDs
        torch.cuda.synchronize()
        first = (time.perf_counter() - start) * 1000
        for i, value in enumerate(first_ids):
            generated[i].append(value)
        for step in range(1, new_tokens):
            position = torch.full((batch, 1), prompt_length + step - 1, dtype=torch.long, device='cuda')
            result = model(input_ids=token, past_key_values=past, position_ids=position, use_cache=True)
            past = result.past_key_values
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            values = token[:, 0].tolist()
            for i, value in enumerate(values):
                generated[i].append(value)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    return {'tokens': generated, 'output_lengths': [new_tokens] * batch,
            'ttft_ms': [first] * batch, 'total_ms': [elapsed] * batch}


def benchmark(fn, prepare, prompts, new_tokens, config, journal, *, cold=False):
    batch = len(prompts)
    rows = []
    for phase, repeats in ([('cold_invocation', 1)] if cold else []) + [('warmup', config['warmups']), ('repeat', config['repeats'])]:
        for repeat in range(repeats):
            prepare()
            start = time.perf_counter()
            result = fn(prompts, new_tokens)
            host_ms = (time.perf_counter() - start) * 1000
            first, end = max(result['ttft_ms']), max(result['total_ms'])
            if end <= 0 or first <= 0 or end < first:
                raise AssertionError('invalid timer sample')
            row = {'phase': phase, 'repeat': repeat, 'host_invocation_ms': host_ms,
                   'ttft_ms': result['ttft_ms'], 'total_ms': result['total_ms'],
                   'batch_ttft_ms': first, 'batch_total_ms': end,
                   'including_prefill_tokens_s': batch * new_tokens * 1000 / end,
                   'decode_tokens_s': batch * (new_tokens - 1) * 1000 / (end - first) if new_tokens > 1 and end > first else None,
                   'output_sha256': canonical_hash(result['tokens']), 'output_lengths': result['output_lengths']}
            rows.append(row)
            journal('greedy_sample', {'batch': batch, **row})
    measured = [r for r in rows if r['phase'] == 'repeat']
    return {'prompt_lengths': list(map(len, prompts)), 'prompt_sha256': canonical_hash(prompts), 'new_tokens': new_tokens,
            'raw_samples': rows, 'ttft_ms': summary([r['batch_ttft_ms'] for r in measured]),
            'total_ms': summary([r['batch_total_ms'] for r in measured]),
            'host_invocation_ms': summary([r['host_invocation_ms'] for r in measured]),
            'including_prefill_tokens_s': summary([r['including_prefill_tokens_s'] for r in measured]),
            'decode_tokens_s': summary([r['decode_tokens_s'] for r in measured]) if new_tokens > 1 else None,
            'deterministic_outputs': len(set(r['output_sha256'] for r in rows)) == 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'library', 'build-receipt', 'fp16-manifest', 'int8-manifest',
                 'hf-directory', 'frozen-tokens', 'kernel-gate', 'model-gate', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.parent != (root / 'results/v2').resolve() or output.name.startswith('.'):
        raise ValueError('output must be a direct unique results/v2/<run-id> child')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    journal_path = output / 'journal.jsonl'
    def journal(event, value):
        with journal_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
                                     'event': event, 'value': value}, allow_nan=False) + '\n')
    startup = {'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
               'argv': sys.argv, 'environment': actual_environment(), 'status': 'started'}
    write(output / 'startup.json', startup)
    before = None
    try:
        config = validate_config(json.loads(Path(args.config).read_text()))
        frozen_path = Path(args.frozen_tokens).resolve(strict=True)
        frozen = json.loads(frozen_path.read_text())
        validate_frozen(frozen)
        if len(frozen['tokens']) < max(60, config['prompt_tokens'] + 32):
            raise ValueError('frozen slice too short for fixed gates/batches')
        if frozen['context'] > config['capacity']:
            raise ValueError('PPL window exceeds native cache context')
        hf_dir = Path(args.hf_directory).resolve(strict=True)
        if sha256_file(hf_dir / 'model.safetensors') != MODEL_SHA256:
            raise ValueError('HF model bytes differ from pinned custom source')
        if verify_tokenizer(hf_dir) != frozen['tokenizer_sha256']:
            raise ValueError('HF tokenizer files differ from frozen tokenization')
        receipt = json.loads(Path(args.build_receipt).read_text())
        verify(receipt, root, args.library)
        kernel_run_path = verify_kernel_gate(args.kernel_gate, receipt['backend_sha256'])
        model_gate = json.loads(Path(args.model_gate).read_text())
        if (model_gate.get('status') != 'passed' or model_gate.get('environment') != actual_environment()
                or model_gate.get('library_sha256') != receipt['library_sha256']
                or model_gate.get('source_sha256') != source_snapshot(root)
                or model_gate.get('config_sha256') != canonical_hash(config)
                or model_gate.get('frozen_token_ids_sha256') != frozen['token_ids_sha256']
                or model_gate.get('artifact_manifest_sha256') != {p: sha256_file(f) for p, f in (('fp16', args.fp16_manifest), ('int8', args.int8_manifest))}):
            raise ValueError('actual matching completed native model gate required before formal measurements')
        git = command(['git', 'rev-parse', 'HEAD'], root)
        status = command(['git', 'status', '--porcelain', '--untracked-files=no'], root)
        if git['exit_code'] or status['exit_code'] or status['stdout'].strip():
            raise ValueError('formal LLM run requires clean tracked source checkpoint')
        before = source_snapshot(root)
        fixed_paths = [args.library, receipt['backend_path'], args.config, args.build_receipt,
                       args.kernel_gate, kernel_run_path, args.model_gate, frozen_path,
                       args.fp16_manifest, args.int8_manifest,
                       Path(args.fp16_manifest).parent / 'weights.bin',
                       Path(args.int8_manifest).parent / 'weights.bin']
        fixed_inputs = artifact_inputs(fixed_paths)
        random.seed(config['seed'])
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(0)
        result = {'schema_version': 1, **startup, 'config': config, 'config_sha256': canonical_hash(config),
                  'git': git, 'source_sha256': before, 'library_sha256': sha256_file(args.library),
                  'input_sha256': fixed_inputs,
                  'build_receipt_sha256': sha256_file(args.build_receipt),
                  'kernel_gate_sha256': sha256_file(args.kernel_gate), 'frozen_tokens_sha256': sha256_file(frozen_path),
                  'kernel_run_sha256': sha256_file(kernel_run_path),
                  'model_gate_sha256': sha256_file(args.model_gate),
                  'frozen_token_ids_sha256': frozen['token_ids_sha256'], 'target_count': frozen['target_count'],
                  'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'safetensors')},
                  'hardware': {'cpu': command(['lscpu'], root), 'gpu': properties.name,
                               'gpu_bytes': properties.total_memory, 'torch_cuda': torch.version.cuda,
                               'driver': command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], root)},
                  'timer_scope': 'TTFT: prepacked CPU tokens through H2D/native-or-HF dispatch/sync/D2H greedy ID; total through last ID; host invocation separately includes Python packing/output conversion; load/tokenization/reset excluded; no EOS stopping',
                  'decode_rate': 'B*(new_tokens-1)/(last-token completion minus first-token completion)',
                  'math': {'tf32': False, 'allow_fp16_reduced_precision_reduction': False,
                           'native_lm_head_output': 'FP32 accumulation/output',
                           'hf_lm_head_output': 'FP16, converted to FP32 for scoring'},
                  'percentile': 'linear interpolation across independent batch invocation samples; population std',
                  'methods': {}}
        write(output / 'config.json', config)
        # Full reversible corpus token IDs stay in the private data directory.
        write(output / 'frozen_protocol.json', {k: v for k, v in frozen.items() if k != 'tokens'})
        hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float16, attn_implementation='eager', local_files_only=True).eval().to('cuda')
        prompts = {b: [frozen['tokens'][(i * config['prompt_offset_step']) % (len(frozen['tokens']) - config['prompt_tokens']):(i * config['prompt_offset_step']) % (len(frozen['tokens']) - config['prompt_tokens']) + config['prompt_tokens']] for i in range(b)] for b in config['batches']}
        eager = {'attention_implementation': str(hf.config._attn_implementation), 'dtype': 'float16',
                 'ppl': hf_score(hf, frozen, lambda e, v: journal(e, {'method': 'hf_eager', **v})), 'benchmarks': {}}
        for batch in config['batches']:
            eager['benchmarks'][str(batch)] = benchmark(lambda p, n: hf_greedy(hf, p, n), lambda: None,
                prompts[batch], config['new_tokens'], config, lambda e, v: journal(e, {'method': 'hf_eager', **v}))
        result['methods']['hf_eager'] = eager
        write(output / 'hf_eager.json', eager)
        for precision, manifest in (('fp16', args.fp16_manifest), ('int8', args.int8_manifest)):
            with NativeModel(args.library, manifest, max_batch=32, cache_slots=32, capacity=config['capacity'],
                             gemm_config=config['gemm_config'], attention_config=config['attention_config']) as model:
                if model.manifest['precision'] != precision:
                    raise ValueError('precision artifact mismatch')
                report = {'artifact_manifest_sha256': sha256_file(manifest), 'artifact_binary_bytes': model.manifest['binary_bytes'],
                          'artifact_binary_sha256': model.manifest['binary_sha256'],
                          'correctness': model_gate['models'][precision],
                          'ppl': native_score(model, frozen, lambda e, v: journal(e, {'method': precision, **v})), 'benchmarks': {}}
                report['loaded_native_libraries'] = loaded_libraries()
                delta = report['ppl']['perplexity'] - eager['ppl']['perplexity']
                report['ppl_delta_vs_hf'] = delta
                report['ppl_relative_delta_vs_hf'] = delta / eager['ppl']['perplexity']
                if precision == 'fp16' and abs(report['ppl_relative_delta_vs_hf']) > config['fp16_ppl_relative_tolerance']:
                    raise AssertionError('FP16 PPL parity exceeds frozen tolerance')
                for batch in config['batches']:
                    report['benchmarks'][str(batch)] = benchmark(model.greedy, model.reset_all, prompts[batch],
                        config['new_tokens'], config, lambda e, v: journal(e, {'method': precision, **v}))
                report['actual_call_trace'] = model.trace()
                result['methods'][precision] = report
                write(output / (precision + '.json'), report)
        if config['attempt_compile']:
            compile_start = time.perf_counter()
            try:
                compiled = torch.compile(hf, backend=config['compile_backend'], dynamic=config['compile_dynamic'], fullgraph=config['compile_fullgraph'])
                compiled_report = {'status': 'completed', 'backend': config['compile_backend'],
                    'dynamic': config['compile_dynamic'], 'fullgraph': config['compile_fullgraph'],
                    'construction_ms': (time.perf_counter() - compile_start) * 1000,
                    'ppl': hf_score(compiled, frozen, lambda e, v: journal(e, {'method': 'hf_compile', **v})),
                    'benchmarks': {}}
                compiled_report['ppl_note'] = 'per-window wall times retain first-shape compilation cost; not steady-state PPL speed'
                compiled_report['ppl_relative_delta_vs_eager'] = (compiled_report['ppl']['perplexity'] - eager['ppl']['perplexity']) / eager['ppl']['perplexity']
                if abs(compiled_report['ppl_relative_delta_vs_eager']) > config['compile_ppl_relative_tolerance']:
                    raise AssertionError('compiled HF PPL drift exceeds frozen tolerance')
                for batch in config['batches']:
                    compiled_report['benchmarks'][str(batch)] = benchmark(lambda p, n: hf_greedy(compiled, p, n), lambda: None,
                        prompts[batch], config['new_tokens'], config, lambda e, v: journal(e, {'method': 'hf_compile', **v}), cold=True)
                compiled_report['dynamo_counters'] = {str(k): {str(kk): str(vv) for kk, vv in v.items()} for k, v in torch._dynamo.utils.counters.items()}
                result['methods']['hf_compile'] = compiled_report
            except Exception as exc:
                failure = {'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                           'wall_ms_including_compile_attempt': (time.perf_counter() - compile_start) * 1000,
                           'traceback': traceback.format_exc()}
                result['methods']['hf_compile'] = failure
                journal('compile_failure', failure)
        else:
            result['methods']['hf_compile'] = {'status': 'not_run', 'reason': 'disabled in explicit config'}
        result['source_unchanged'] = before == source_snapshot(root)
        result['end_git'] = command(['git', 'rev-parse', 'HEAD'], root)
        result['git_unchanged'] = result['end_git']['stdout'] == git['stdout']
        result['inputs_unchanged'] = fixed_inputs == artifact_inputs(fixed_paths)
        if not result['source_unchanged'] or not result['git_unchanged'] or not result['inputs_unchanged']:
            raise RuntimeError('source changed during formal LLM measurement')
        result['int8_minus_fp16_ppl'] = result['methods']['int8']['ppl']['perplexity'] - result['methods']['fp16']['ppl']['perplexity']
        result['comparisons'] = {}
        for batch in map(str, config['batches']):
            fp = result['methods']['fp16']['benchmarks'][batch]
            iq = result['methods']['int8']['benchmarks'][batch]
            ref = eager['benchmarks'][batch]
            row = {'fp16_total_speedup_vs_hf_eager': ref['total_ms']['p50'] / fp['total_ms']['p50'],
                   'int8_total_speedup_vs_fp16': fp['total_ms']['p50'] / iq['total_ms']['p50'],
                   'int8_ttft_speedup_vs_fp16': fp['ttft_ms']['p50'] / iq['ttft_ms']['p50']}
            if config['new_tokens'] > 1:
                row['int8_decode_rate_ratio_vs_fp16'] = iq['decode_tokens_s']['p50'] / fp['decode_tokens_s']['p50']
                row['fp16_decode_rate_ratio_vs_hf_eager'] = fp['decode_tokens_s']['p50'] / ref['decode_tokens_s']['p50']
            result['comparisons'][batch] = row
        result['status'] = 'completed'
        result['wall_seconds'] = time.perf_counter() - start
        result['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(output / 'results.json', result)
        print(json.dumps({'output': str(output), 'status': 'completed'}))
    except BaseException as exc:
        failure = {'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc(), 'wall_seconds': time.perf_counter() - start,
                   'source_unchanged': before == source_snapshot(root) if before is not None else None}
        write(output / 'failure.json', failure)
        journal('failure', failure)
        raise


if __name__ == '__main__':
    main()
