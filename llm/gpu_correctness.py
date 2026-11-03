"""Actual GPU gates, called only after standalone build and kernel gates pass."""


def check_native(model, hf_model, token_ids, config, *, compare_hf=True):
    import torch
    def tensor(result):
        value = torch.frombuffer(result['logits'], dtype=torch.float32).reshape(result['shape']).clone()
        if not torch.isfinite(value).all():
            raise AssertionError('native logits contain NaN/Inf')
        if result['next_tokens'] != value.argmax(-1).tolist():
            raise AssertionError('GPU argmax disagrees with returned logits')
        return value
    prompts = [token_ids[0:33], token_ids[41:58]]
    if min(map(len, prompts)) < 4:
        raise ValueError('GPU gates need at least58 frozen tokens')
    model.reset_all()
    independent = []
    for prompt in prompts:
        model.reset(0)
        independent.append(tensor(model.forward([0], [prompt]))[0])
    full = torch.stack(independent)
    model.reset_all()
    batched = tensor(model.forward([0, 1], prompts))
    torch.testing.assert_close(batched, full, atol=config['cached_atol'], rtol=config['cached_rtol'])
    model.reset_all()
    model.forward([0, 1], [p[:3] for p in prompts], return_logits=False, return_next=False)
    cached = tensor(model.forward([0, 1], [p[3:] for p in prompts]))
    torch.testing.assert_close(cached, full, atol=config['cached_atol'], rtol=config['cached_rtol'])
    # Reset one request without disturbing another; then extend the surviving one.
    model.reset(0)
    survivor = tensor(model.forward([1], [[token_ids[59]]]))
    model.reset(2)
    expected = tensor(model.forward([2], [prompts[1] + [token_ids[59]]]))
    torch.testing.assert_close(survivor, expected, atol=config['cached_atol'], rtol=config['cached_rtol'])
    before = model.lengths()
    for slots, rows in (([1, 1], [[3], [4]]), ([1], [[3] * model.options.capacity])):
        rejected = False
        try:
            model.forward(slots, rows, return_logits=False, return_next=False)
        except RuntimeError:
            rejected = True
        if not rejected or model.lengths() != before:
            raise AssertionError('invalid batch changed cache lengths')
    # Internal teacher-forcing slot must not modify public request caches.
    score = model.score(token_ids[:33], [0] + [1] * 32, positions=True)
    if model.lengths() != before or score['target_count'] != 32:
        raise AssertionError('scoring mutated public cache or target count')
    if abs(sum(score['position_nll']) - score['sum_nll']) > 1e-7 * max(1., score['sum_nll']):
        raise AssertionError('NLL position/scalar mismatch')
    subset_mask = [0] + [int(i % 3 != 0) for i in range(1, 33)]
    subset = model.score(token_ids[:33], subset_mask, positions=True)
    expected_subset = sum(v for v, use in zip(score['position_nll'], subset_mask) if use)
    if (subset['target_count'] != sum(subset_mask) or model.lengths() != before
            or abs(subset['sum_nll'] - expected_subset) > 1e-7 * max(1., expected_subset)
            or any(v != 0 for v, use in zip(subset['position_nll'], subset_mask) if not use)):
        raise AssertionError('masked teacher forcing scored wrong targets')
    model.reset_all()
    generation_prompts, requested = [prompts[0][:7], prompts[1][:5]], [3, 1]
    mixed = model.greedy(generation_prompts, requested, slots=[0, 1])
    if mixed['output_lengths'] != requested or model.lengths()[:2] != [9, 5]:
        raise AssertionError('mixed greedy lengths/cache ownership incorrect')
    for row, count, generated in zip(generation_prompts, requested, mixed['tokens']):
        model.reset(2)
        single = model.greedy([row], count, slots=[2])
        if single['tokens'][0] != generated:
            raise AssertionError('mixed greedy generation differs from independent requests')
    report = {'batched_vs_single_max_abs': (batched - full).abs().max().item(),
              'cached_vs_prefix_max_abs': (cached - full).abs().max().item(),
              'reset_isolation_max_abs': (survivor - expected).abs().max().item(),
              'invalid_batches_transactional': True, 'scoring_public_slots_unchanged': True,
              'small_window_score': score, 'masked_window_score': subset,
              'mixed_greedy': mixed, 'mixed_greedy_matches_independent': True}
    if hf_model is not None:
        reference = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt], dtype=torch.long, device='cuda')
                reference.append(hf_model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu())
        reference = torch.stack(reference)
        report['hf_last_logits_max_abs'] = (full - reference).abs().max().item()
        report['hf_last_logits_mean_abs'] = (full - reference).abs().mean().item()
        report['hf_greedy_agreement'] = (full.argmax(-1) == reference.argmax(-1)).float().mean().item()
        if compare_hf:
            torch.testing.assert_close(full, reference, atol=config['hf_atol'], rtol=config['hf_rtol'])
    trace = model.trace()
    active_gemm = 'gemm_int8' if model.manifest['precision'] == 'int8' else 'gemm_fp16'
    if not all(trace[k] > 0 for k in ('embedding', active_gemm, 'layernorm', 'gelu', 'attention', 'residual', 'kv_scatter_gather', 'lm_head', 'argmax', 'nll')):
        raise AssertionError('custom path did not dispatch every required native operation')
    report['actual_call_trace'] = trace
    model.reset_all()
    return report


def main():
    import argparse
    import datetime as dt
    import importlib.metadata
    import json
    from pathlib import Path
    import sys
    import time
    import traceback
    from .build_receipt import command, source_snapshot, verify
    from .evaluate import actual_environment, artifact_inputs, loaded_libraries, validate_config, verify_kernel_gate, write
    from .export_weights import sha256_file
    from .protocol import MODEL_SHA256, canonical_hash, validate_frozen, verify_tokenizer
    from .runtime import NativeModel
    parser = argparse.ArgumentParser(description='Actual standalone native model GPU gates, no formal benchmark')
    for name in ('library', 'build-receipt', 'fp16-manifest', 'int8-manifest',
                 'hf-directory', 'config', 'frozen-tokens', 'kernel-gate', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.parent != (root / 'results/v2').resolve() or output.name.startswith('.'):
        raise ValueError('GPU gate output must be unique results/v2/<run-id>')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'status': 'started', 'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
              'argv': sys.argv, 'environment': actual_environment(), 'models': {}}
    write(output / 'startup.json', report)
    try:
        config = validate_config(json.loads(Path(args.config).read_text()))
        frozen = json.loads(Path(args.frozen_tokens).read_text())
        validate_frozen(frozen)
        if len(frozen['tokens']) < 60:
            raise ValueError('GPU gates need at least60 frozen tokens')
        receipt = verify(json.loads(Path(args.build_receipt).read_text()), root, args.library)
        verify_kernel_gate(args.kernel_gate, receipt['backend_sha256'])
        hf_dir = Path(args.hf_directory).resolve(strict=True)
        if sha256_file(hf_dir / 'model.safetensors') != MODEL_SHA256:
            raise ValueError('HF reference bytes do not match pinned custom source')
        if verify_tokenizer(hf_dir) != frozen['tokenizer_sha256']:
            raise ValueError('HF tokenizer differs from frozen official tokenization')
        before = source_snapshot(root)
        fixed_paths = [args.library, receipt['backend_path'], args.config, args.build_receipt,
                       args.kernel_gate, args.frozen_tokens, args.fp16_manifest, args.int8_manifest,
                       Path(args.fp16_manifest).parent / 'weights.bin',
                       Path(args.int8_manifest).parent / 'weights.bin']
        fixed_inputs = artifact_inputs(fixed_paths)
        git = command(['git', 'rev-parse', 'HEAD'], root)
        status = command(['git', 'status', '--porcelain', '--untracked-files=no'], root)
        if git['exit_code'] or status['exit_code'] or status['stdout'].strip():
            raise ValueError('GPU gates require clean tracked source checkpoint')
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False
        hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float16, attn_implementation='eager', local_files_only=True).eval().to('cuda')
        report.update({'git': git, 'source_sha256': before, 'library_sha256': sha256_file(args.library),
                       'input_sha256': fixed_inputs,
                       'config_sha256': canonical_hash(config),
                       'frozen_token_ids_sha256': frozen['token_ids_sha256'],
                       'artifact_manifest_sha256': {p: sha256_file(f) for p, f in (('fp16', args.fp16_manifest), ('int8', args.int8_manifest))},
                       'versions': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'transformers', 'safetensors')},
                       'gpu': torch.cuda.get_device_properties(0).name,
                       'torch_cuda': torch.version.cuda,
                       'driver': command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], root),
                       'cpu': command(['lscpu'], root),
                       'build_receipt_sha256': sha256_file(args.build_receipt),
                       'kernel_gate_sha256': sha256_file(args.kernel_gate)})
        for precision, manifest in (('fp16', args.fp16_manifest), ('int8', args.int8_manifest)):
            with NativeModel(args.library, manifest, max_batch=32, cache_slots=32,
                             capacity=config['capacity'], gemm_config=config['gemm_config'],
                             attention_config=config['attention_config']) as model:
                if model.manifest['precision'] != precision:
                    raise ValueError('artifact precision mismatch')
                report['models'][precision] = check_native(model, hf, frozen['tokens'], config, compare_hf=precision == 'fp16')
                report['models'][precision]['loaded_native_libraries'] = loaded_libraries()
                write(output / (precision + '.json'), report['models'][precision])
        report['source_unchanged'] = before == source_snapshot(root)
        report['end_git'] = command(['git', 'rev-parse', 'HEAD'], root)
        report['inputs_unchanged'] = fixed_inputs == artifact_inputs(fixed_paths)
        if not report['source_unchanged'] or report['end_git']['stdout'] != git['stdout'] or not report['inputs_unchanged']:
            raise RuntimeError('source changed during model GPU gate')
        report['status'] = 'passed'
        report['wall_seconds'] = time.perf_counter() - started
        report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(output / 'results.json', report)
    except BaseException as exc:
        report.update({'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                       'traceback': traceback.format_exc(), 'wall_seconds': time.perf_counter() - started})
        write(output / 'failure.json', report)
        raise


if __name__ == '__main__':
    main()
