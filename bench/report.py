"""Render recorded measurements; this script never runs a benchmark."""
from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = {device: json.loads((ROOT / f'results/integrated_{device}.json').read_text())
               for device in ('cpu', 'cuda')}
    timings = 0
    errors = []

    def check(value):
        nonlocal timings
        if isinstance(value, dict):
            if 'repeats_us' in value:
                samples = value['repeats_us']
                assert len(samples) == 7 and all(math.isfinite(x) and x > 0 for x in samples)
                assert value['median_us'] == statistics.median(samples)
                assert value['min_us'] == min(samples) and value['max_us'] == max(samples)
                timings += 1
            if 'max_abs_error' in value:
                assert math.isfinite(value['max_abs_error'])
                errors.append(value['max_abs_error'])
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    for device, report in reports.items():
        check(report)
        assert report['meta']['threads'] == report['meta']['native_openmp_threads'] == 8
        assert report['meta']['iterations'] == 200 and report['meta']['repeats'] == 7
        for row in report['calibration']:
            expected = 'hand' if row['hand']['median_us'] < .97 * row['codegen']['median_us'] else 'codegen'
            assert row['winner'] == expected
        for row in report['rows']:
            for choice in row['auto']['dispatch']:
                assert choice['implementation'] == report['policy'].get(choice['key'], {'winner': 'codegen'})['winner']
            if row['case'].startswith(('mlp', 'cnn')):
                assert row['codegen']['source_sha256'] == row['auto']['source_sha256']
    for cpu, gpu in zip(reports['cpu']['rows'], reports['cuda']['rows']):
        assert cpu['case'] == gpu['case']
        assert all(cpu[k] == gpu[k] for k in ('input_sha256', 'weights_sha256'))

    def table(rows, layer=None, ranges=False):
        text = '| Case | Eager | Codegen | Compiler + kernels (auto) |\n|---|---:|---:|---:|\n'
        for row in rows:
            source = row[layer] if layer else row
            values = []
            for key in ('eager', 'codegen', 'auto'):
                value = source[key]
                label = f"{value['median_us']:.2f}"
                if ranges:
                    label += f" [{value['min_us']:.2f}, {value['max_us']:.2f}]"
                values.append(label)
            text += '| ' + row['case'] + ' | ' + ' | '.join(values) + ' |\n'
        return text

    gpu = reports['cuda']; cpu = reports['cpu']
    section = ('Measured on an i7-12700KF and RTX 3080 12 GB, using eight CPU/OpenMP threads, '
               'PyTorch 2.15.0+cu132, GCC 13.3 and nvcc 12.4.131. All values below are microseconds; '
               'each is the median of seven recorded repeat values. [Environment](results/environment.txt) · '
               '[all raw repeats and calibration](results/table.md).\n\n'
               '**GPU host-call end to end** — CPU input → GPU evaluation → fresh CPU NumPy output.\n\n')
    section += table(gpu['rows'], 'host_e2e')
    section += '\n**GPU device-event window** — device-resident inputs; entries show median [minimum, maximum] across repeats.\n\n'
    section += table(gpu['rows'], ranges=True)
    section += '\n**CPU host-call end to end** — CPU input → fresh CPU NumPy output, eight threads.\n\n'
    section += table(cpu['rows'], 'host_e2e')
    section += ('\nFor all four MLP/CNN configurations, auto retains codegen: the generated source hashes are identical. '
                'Differences between these two columns reflect the observed sampling variation, rather than a kernel substitution. '
                'The hand-written GEMM plus separate epilogue was slower in every matched calibration graph. '
                'Auto does substitute the hand-written LayerNorm and Softmax on both devices. '
                'On GPU those substitutions improve the compiler-owned implementations, while PyTorch eager remains faster for these normalization cases.\n\n')
    section += '**Standalone CPU GEMM, eight threads** — both allocation-returning APIs.\n\n'
    section += '| Size | Hand-written (ms) | Torch/MKL (ms) | Hand / MKL time | Earlier source ratio |\n|---|---:|---:|---:|---:|\n'
    old = {256: 7.59, 512: 3.64, 1024: 3.93}
    for row in cpu['standalone_gemm']:
        section += f"| {row['n']}³ | {row['hand']['median_us']/1000:.3f} | {row['torch_mkl']['median_us']/1000:.3f} | {row['hand_over_torch']:.2f}× | {old[row['n']]:.2f}× |\n"
    section += ('\nThe earlier 1024³ result was 3.93×, rather than a fixed sixfold ratio. '
                '[Earlier raw comparison](results/before/kernels3/torch_compare.md) · '
                '[new CPU samples and MKL build configuration](results/integrated_cpu.json). '
                'The integration leaves this slower GEMM available for inspection and chooses generated GEMM for the measured models.\n')
    readme = ROOT / 'README.md'
    before, marker, after = readme.read_text().partition('<!-- FRESH_RESULTS -->')
    assert marker, 'README result marker is missing'
    readme.write_text(before + '<!-- FRESH_RESULTS -->\n\n' + section + '\n' + after[after.index('## Timing layers'):])

    details = '# Recorded three-path measurements\n\nUnits are microseconds. Entries show median [minimum, maximum] of seven repeats.\n\n'
    for device, report in reports.items():
        details += f'## {device}: host-call end to end\n\n' + table(report['rows'], 'host_e2e', True)
        details += f'\n## {device}: compute / event window\n\n' + table(report['rows'], ranges=True)
        details += '\n### Calibration and frozen choices\n\n| Case | Codegen | Forced hand | Frozen choice |\n|---|---:|---:|---|\n'
        for row in report['calibration']:
            details += f"| {row['case']} | {row['codegen']['median_us']:.2f} | {row['hand']['median_us']:.2f} | {row['winner']} |\n"
        details += '\nThe MLP/CNN rows toggle all GEMM calls together; normalization rows measure one operator. Final timings are separate from calibration.\n\n'
    (ROOT / 'results/table.md').write_text(details.rstrip() + '\n')
    model_errors = [value['max_abs_error'] for report in reports.values() for row in report['rows']
                    for name in ('codegen', 'auto') for value in (row[name],)]
    validation = {'status': 'passed', 'timing_series': timings, 'repeats_per_series': 7,
                  'same_cpu_gpu_inputs_and_weights': True, 'frozen_policy_matches_auto_dispatch': True,
                  'mlp_cnn_codegen_auto_sources_identical': True, 'max_recorded_abs_error': max(errors),
                  'max_final_compiled_model_abs_error': max(model_errors)}
    (ROOT / 'results/validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
