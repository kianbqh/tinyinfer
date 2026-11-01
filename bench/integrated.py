"""Matched calibration, frozen dispatch, and eager/codegen/hybrid execution."""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import ikernels
from tinyc import compile_model
from tinyc.models import MLP, SmallCNN, randomize_bn

def host_time(fn, repeats, iterations):
    for _ in range(20): fn()
    values = []
    for _ in range(repeats):
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-start)/1000)
        values.append(statistics.median(samples))
    return summary(values)

def summary(values):
    assert all(np.isfinite(v) and v > 0 for v in values), values
    return {'median_us': statistics.median(values), 'min_us': min(values), 'max_us': max(values), 'repeats_us': values}

def custom_time(cm, x, device, repeats, iterations):
    if device == 'cpu':
        out = np.empty(cm.out_shape, np.float32)
        array = x.numpy()
        return host_time(lambda: cm.run_into(array, out), repeats, iterations)
    cm.lib.tinyc_bench.argtypes = [ctypes.c_int]
    cm.lib.tinyc_bench.restype = ctypes.c_float
    return summary([cm.lib.tinyc_bench(iterations)*1000 for _ in range(repeats)])

def eager_time(model, x, device, repeats, iterations):
    if device == 'cpu':
        with torch.inference_mode(): return host_time(lambda: model(x), repeats, iterations)
    with torch.no_grad(): reference = model(x).numpy()
    model, x = copy.deepcopy(model).cuda(), x.cuda()
    with torch.inference_mode(): actual = model(x).cpu().numpy()
    np.testing.assert_allclose(actual,reference,atol=1e-4,rtol=1e-4)
    values = []
    with torch.inference_mode():
        for _ in range(repeats):
            for _ in range(10): model(x)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations): model(x)
            end.record(); end.synchronize()
            values.append(start.elapsed_time(end)*1000/iterations)
    result = summary(values)
    result['max_abs_error'] = float(np.abs(actual-reference).max())
    return result

def host_eager(model, x, device, repeats, iterations):
    with torch.no_grad(): reference = model(x).numpy()
    if device == 'cuda':
        model = copy.deepcopy(model).cuda()
        fn = lambda: model(x.cuda()).cpu().numpy()
    else:
        fn = lambda: model(x).numpy()
    with torch.inference_mode():
        actual = fn()
        np.testing.assert_allclose(actual,reference,atol=1e-4,rtol=1e-4)
        result = host_time(fn,repeats,iterations)
    result['max_abs_error'] = float(np.abs(actual-reference).max())
    return result

def cases():
    torch.manual_seed(0)
    result = []
    for name, model, bs, shape in [('mlp_b1', MLP, 1, (784,)), ('mlp_b64', MLP, 64, (784,)),
                                  ('cnn_b1', SmallCNN, 1, (1,28,28)), ('cnn_b32', SmallCNN, 32, (1,28,28))]:
        result.append((name, randomize_bn(model()).eval(), torch.randn(bs, *shape)))
    for name in ('layernorm', 'softmax', 'norm_softmax'):
        norm = nn.LayerNorm(1024)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(.5, 1.5, 1024)); norm.bias.copy_(torch.randn(1024)*.1)
        modules = [norm] if name == 'layernorm' else [nn.Softmax(-1)] if name == 'softmax' else [norm, nn.Softmax(-1)]
        result.append((name, nn.Sequential(*modules).eval(), torch.randn(1024, 1024)))
    return result

def fingerprint(model, x):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.numpy().tobytes())
    return {'weights_sha256': h.hexdigest(), 'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('TINYC_THREADS', '8')))
    if args.device == 'cuda':
        assert torch.cuda.is_available() and ikernels.has_cuda
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
    backend = 'cuda' if args.device == 'cuda' else 'cpp'
    data = cases()
    policy_path = ROOT / 'results/dispatch.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    calibration = []
    # Standalone normalization cases measure each operator. Matmul cases toggle all GEMMs
    # in that exact graph: they select a graph-specific configuration, not isolated-kernel superiority.
    for name, model, x in data:
        if name == 'norm_softmax': continue
        with torch.no_grad(): ref = model(x).numpy()
        row = {'case': name, **fingerprint(model,x)}
        for implementation in ('codegen', 'hand'):
            cm = compile_model(model, x, backend=backend, implementation=implementation, tag=f'cal_{name}_{implementation}')
            actual = cm(x)
            np.testing.assert_allclose(actual, ref, atol=1e-4, rtol=1e-4)
            row[implementation] = custom_time(cm, x, args.device, args.repeats, args.iterations)
            row[implementation]['max_abs_error'] = float(np.abs(actual-ref).max())
            row['dispatch'] = cm.info['dispatch']
        winner = 'hand' if row['hand']['median_us'] < .97*row['codegen']['median_us'] else 'codegen'
        row['winner'] = winner
        for decision in row['dispatch']:
            policy[decision['key']] = {'winner': winner, 'source_case': name,
                'scope': 'single operator' if name in ('layernorm','softmax') else 'matched whole graph; GEMM configuration toggled together',
                'codegen_us': row['codegen']['median_us'], 'hand_us': row['hand']['median_us'], 'minimum_relative_win': .03}
        calibration.append(row)
        print('CALIBRATION', name, winner, row['codegen']['median_us'], row['hand']['median_us'], flush=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2)+'\n')
    rows = []
    for name, model, x in data:
        with torch.no_grad(): ref = model(x).numpy()
        row = {'case': name, 'shape': list(x.shape), **fingerprint(model,x)}
        row['host_e2e'] = {}
        row['eager'] = eager_time(model,x,args.device,args.repeats,args.iterations)
        row['host_e2e']['eager'] = host_eager(model,x,args.device,args.repeats,args.iterations)
        for implementation in ('codegen', 'auto'):
            cm = compile_model(model, x, backend=backend, implementation=implementation, tag=f'final_{name}_{implementation}')
            actual = cm(x)
            np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-4)
            row[implementation] = custom_time(cm,x,args.device,args.repeats,args.iterations)
            row[implementation].update(max_abs_error=float(np.abs(actual-ref).max()), dispatch=cm.info['dispatch'],
                                       source_sha256=hashlib.sha256(cm.info['source'].encode()).hexdigest())
            row['host_e2e'][implementation] = host_time(lambda:cm(x),args.repeats,args.iterations)
            row['host_e2e'][implementation]['max_abs_error'] = float(np.abs(actual-ref).max())
        rows.append(row)
        print('FINAL', name, {k:round(row[k]['median_us'],3) for k in ('eager','codegen','auto')}, flush=True)
    meta = {'device': args.device, 'torch': torch.__version__, 'threads': torch.get_num_threads(), 'seed': 0,
            'repeats': args.repeats, 'iterations': args.iterations, 'native_module': ikernels.__file__,
            'protocol': 'CPU host per-call timer; eager returns Tensor, custom reuses output. GPU device-resident inputs, CUDA events around C++ enqueue loops(custom) or Python eager loops; 10 warmups/repeat, includes inter-launch gaps, excludes transfers/compile. Custom nvcc fast math; Torch TF32 disabled.',
            'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            'policy_rule': 'hand median at least 3% faster in calibration; frozen before independent final timings'}
    meta['host_e2e_protocol'] = 'Same CPU float32 Tensor input and fresh CPU NumPy result for all three paths; GPU weights preloaded. Eager copies input to GPU, evaluates, copies output to CPU; custom cm(x) copies through its preallocated device buffers, synchronizes and returns a host output copy. Host per-call timer, 20 warmups then 7 repeats of 200 calls; each repeat reports median. CPU case uses the same CPU-input/NumPy-output interface without transfers.'
    meta['native_openmp_threads'] = ikernels.openmp_threads()
    assert meta['native_openmp_threads'] == meta['threads']
    report = {'meta': meta, 'calibration': calibration, 'policy': {k:v for k,v in policy.items() if k.startswith(backend+':')}, 'rows': rows}
    if args.device == 'cpu':
        # Same public NumPy/torch allocation-returning APIs and eight threads as source3.
        rng = np.random.default_rng(0); gemms = []
        for n in (256,512,1024):
            a = rng.standard_normal((n,n),dtype=np.float32); b = rng.standard_normal((n,n),dtype=np.float32)
            ta,tb = torch.from_numpy(a),torch.from_numpy(b)
            actual = ikernels.matmul(a,b)
            with torch.inference_mode(): ref = (ta@tb).numpy()
            np.testing.assert_allclose(actual,ref,atol=1e-3,rtol=1e-3)
            hand = host_time(lambda:ikernels.matmul(a,b),7,20)
            with torch.inference_mode(): eager = host_time(lambda:ta@tb,7,20)
            gemms.append({'n':n,'hand':hand,'torch_mkl':eager,'hand_over_torch':hand['median_us']/eager['median_us'], 'max_abs_error':float(np.abs(actual-ref).max())})
        report['standalone_gemm'] = gemms
        report['torch_config'] = torch.__config__.show()
    (ROOT/f'results/integrated_{args.device}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('INTEGRATED_BENCHMARK_COMPLETE', args.device, flush=True)

if __name__ == '__main__': main()
