"""Record an ALREADY completed native model build, or verify its receipt."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from .export_weights import sha256_file
from bench.v2.kernels.protocol import build_source_paths


def command(argv, root):
    if argv[0] == 'git':
        executable = os.environ.get('TINYC_GIT') or shutil.which('git')
        if not executable:
            raise RuntimeError('set TINYC_GIT to the existing Git executable')
        directory = str(root)
        if executable.lower().endswith('.exe') and platform.system() == 'Linux':
            directory = subprocess.check_output(['wslpath', '-w', directory], text=True).strip()
        argv = [executable, '-C', directory] + argv[1:]
    result = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    return {'argv': argv, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def source_snapshot(root):
    root = Path(root)
    sources = build_source_paths(root)
    sources += [p for p in (root / 'llm').rglob('*') if p.is_file() and
                p.suffix in ('.py', '.cu', '.cuh', '.h', '.hpp', '.json')]
    sources += [p for p in (root / 'kernels/src/cuda/tc').glob('*') if p.is_file()]
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(set(sources))}


def check_linkage(root, library, backend):
    detail = command(['ldd', str(library)], root)
    text = detail['stdout']
    matches = [line for line in text.splitlines() if 'libtinyinfer_backend' in line and '=>' in line]
    if detail['exit_code'] or 'libtorch' in text or 'not found' in text or len(matches) != 1:
        raise ValueError('standalone runtime linkage validation failed')
    resolved = matches[0].split('=>', 1)[1].rsplit(' (', 1)[0].strip()
    if Path(resolved).resolve(strict=True) != Path(backend).resolve(strict=True):
        raise ValueError('model runtime resolves a different backend library')
    return detail


def verify(receipt, root, library):
    if receipt.get('schema_version') != 1 or receipt.get('build_exit_code') != 0:
        raise ValueError('completed native build receipt required')
    library = Path(library).resolve(strict=True)
    if sha256_file(library) != receipt['library_sha256'] or source_snapshot(root) != receipt['source_sha256']:
        raise ValueError('native model library/source differ from completed build receipt')
    for key in ('backend', 'build_log', 'compile_commands', 'cmake_cache'):
        if sha256_file(receipt[key + '_path']) != receipt[key + '_sha256']:
            raise ValueError(f'build receipt {key} changed')
    check_linkage(root, library, receipt['backend_path'])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'backend', 'build-directory', 'build-log', 'build-command', 'nvcc', 'cxx', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--build-exit-code', required=True, type=int)
    args = parser.parse_args()
    if args.build_exit_code != 0:
        raise ValueError('failed builds must retain their log, not an accepted receipt')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    build = Path(args.build_directory).resolve(strict=True)
    paths = {'library': Path(args.library).resolve(strict=True), 'backend': Path(args.backend).resolve(strict=True),
             'build_log': Path(args.build_log).resolve(strict=True),
             'compile_commands': build / 'compile_commands.json', 'cmake_cache': build / 'CMakeCache.txt'}
    records = json.loads(paths['compile_commands'].read_text())
    selected = [row for row in records if row['file'].replace('\\', '/').endswith('/llm/src/runtime.cu')]
    if len(selected) != 1:
        raise ValueError('expected one standalone native model CUDA compilation record')
    text = json.dumps(selected)
    if any(forbidden in text for forbidden in ('torch/include', 'libtorch', '--use_fast_math')):
        raise ValueError('model build unexpectedly uses Torch headers/linkage or fast math')
    if not any(flag in text for flag in ('sm_86', 'compute_86')):
        raise ValueError('actual model compile commands must identify sm86')
    receipt = {'schema_version': 1, 'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
               'build_exit_code': 0, 'build_command': args.build_command, 'model_compile_commands': selected,
               'git': command(['git', 'rev-parse', 'HEAD'], root),
               'git_status': command(['git', 'status', '--porcelain'], root),
               'nvcc': command([args.nvcc, '--version'], root), 'cxx': command([args.cxx, '--version'], root),
               'ldd': check_linkage(root, paths['library'], paths['backend']), 'source_sha256': source_snapshot(root)}
    for key, path in paths.items():
        receipt[key + '_path'] = str(path)
        receipt[key + '_sha256'] = sha256_file(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='ascii') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
