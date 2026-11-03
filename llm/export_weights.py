"""Export a *local*, pinned official GPT-2 safetensors file. No downloads.

Stdlib implementation keeps the exporter independent of Torch/CUDA. It uses
bounded per-tensor host storage, not a second framework model. Expect CPU work.
"""
import argparse
import array
import datetime as dt
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import sys
import time
from .protocol import MODEL_ID, MODEL_REVISION, MODEL_SHA256

ALIGNMENT = 256
ARCHITECTURE = {'layers': 12, 'heads': 12, 'hidden': 768, 'vocab': 50257,
                'capacity': 1024, 'gelu': 'gelu_new', 'layer_norm_eps': 1e-5}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def expected_tensors():
    shapes = {'transformer.wte.weight': [50257, 768],
              'transformer.wpe.weight': [1024, 768],
              'transformer.ln_f.weight': [768], 'transformer.ln_f.bias': [768]}
    for layer in range(12):
        p = f'transformer.h.{layer}.'
        for norm in ('ln_1', 'ln_2'):
            shapes[p + norm + '.weight'] = [768]
            shapes[p + norm + '.bias'] = [768]
        for name, shape in (('attn.c_attn', [768, 2304]), ('attn.c_proj', [768, 768]),
                            ('mlp.c_fc', [768, 3072]), ('mlp.c_proj', [3072, 768])):
            shapes[p + name + '.weight'] = shape
            shapes[p + name + '.bias'] = [shape[1]]
    return shapes


class SafeTensorFile:
    def __init__(self, path):
        if sys.byteorder != 'little':
            raise ValueError('exporter currently requires little endian host')
        self.stream = open(path, 'rb')
        self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.mapping) < 10:
            raise ValueError('short safetensors')
        header_size = struct.unpack_from('<Q', self.mapping)[0]
        if header_size > 8 * 1024 * 1024 or header_size + 8 > len(self.mapping):
            raise ValueError('invalid safetensors header size')
        self.header = json.loads(self.mapping[8:8 + header_size], object_pairs_hook=_unique)
        self.data_start = header_size + 8
        ranges = []
        for name, item in self.header.items():
            if name == '__metadata__':
                continue
            shape, offsets = item['shape'], item['data_offsets']
            source_sizes = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8,
                            'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
            if item['dtype'] not in source_sizes or not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError('invalid tensor dtype/shape')
            if len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
                raise ValueError('invalid tensor offsets')
            begin, end = offsets
            size = math.prod(shape) * source_sizes[item['dtype']]
            if end - begin != size or self.data_start + end > len(self.mapping):
                raise ValueError('invalid tensor extent')
            ranges.append((begin, end))
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('overlapping tensors')

    def read(self, name, shape):
        # The pinned upstream GPT2Model file uses bare h.N/wte keys; newer
        # LMHeadModel exports may use transformer.*. Never transpose Conv1D.
        candidates = [key for key in (name, name.removeprefix('transformer.')) if key in self.header]
        candidates = list(dict.fromkeys(candidates))
        item = self.header[candidates[0]] if len(candidates) == 1 else None
        if item is None or item['shape'] != shape or item['dtype'] not in ('F32', 'F16'):
            raise ValueError(f'missing/wrong official tensor {name}')
        begin, end = item['data_offsets']
        raw = self.mapping[self.data_start + begin:self.data_start + end]
        if item['dtype'] == 'F32':
            values = array.array('f')
            values.frombytes(raw)
        else:
            values = array.array('f', (v[0] for v in struct.iter_unpack('<e', raw)))
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f'nonfinite tensor {name}')
        return values

    def close(self):
        self.mapping.close()
        self.stream.close()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def half_bytes(values):
    result = bytearray()
    for begin in range(0, len(values), 4096):
        chunk = values[begin:begin + 4096]
        result.extend(struct.pack('<' + 'e' * len(chunk), *chunk))
    return result


def half_values(values):
    return array.array('f', (x[0] for x in struct.iter_unpack('<e', half_bytes(values))))


def quantized_bytes(values, columns):
    # values have already been rounded to the reference FP16 storage format.
    maxima = array.array('f', [0.0]) * columns
    for index, value in enumerate(values):
        channel = index % columns
        maxima[channel] = max(maxima[channel], abs(value))
    scales = array.array('f', (v / 127.0 if v else 1.0 for v in maxima))
    if any(s <= 0 or not math.isfinite(s) for s in scales):
        raise ValueError('invalid quantization scale')
    q = array.array('b', (max(-127, min(127, round(v / scales[i % columns])))
                         for i, v in enumerate(values)))
    return q.tobytes(), scales.tobytes()


def export(source, destination, precision, host_codec=None):
    started = time.perf_counter()
    if precision not in ('fp16', 'int8'):
        raise ValueError('precision must be fp16/int8')
    if precision == 'int8' and host_codec is None:
        raise ValueError('INT8 export requires --host-codec; no silent Python fallback')
    def encode(values, columns):
        actual = host_codec.quantized_bytes(values, columns)
        reference = quantized_bytes(values, columns)  # independent original implementation
        if actual != reference:
            raise ValueError('native/reference INT8 bytes differ; export rejected')
        return actual
    source = Path(source).resolve()
    actual = sha256_file(source)
    if actual != MODEL_SHA256:
        raise ValueError('source bytes do not match the pinned official GPT-2 safetensors SHA256')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    reader = SafeTensorFile(source)
    tensor_entries = []
    binary_path = destination / 'weights.bin'
    try:
        with binary_path.open('xb') as binary:
            def write(name, dtype, shape, raw):
                padding = (-binary.tell()) % ALIGNMENT
                binary.write(bytes(padding))
                offset = binary.tell()
                binary.write(raw)
                tensor_entries.append({'name': name, 'dtype': dtype, 'shape': shape,
                                       'offset': offset, 'nbytes': len(raw),
                                       'sha256': hashlib.sha256(raw).hexdigest()})

            for name, shape in expected_tensors().items():
                values = half_values(reader.read(name, shape))
                is_linear = len(shape) == 2 and '.h.' in name and name.endswith('.weight')
                if precision == 'int8' and is_linear:
                    q, scales = encode(values, shape[1])
                    write(name, 'int8', shape, q)
                    write(name + '.scales', 'float32', [shape[1]], scales)
                else:
                    write(name, 'float16', shape, half_bytes(values))
                if name == 'transformer.wte.weight':
                    # Keep exact FP16 embedding values for the derived tied head.
                    embedding = values
            head = array.array('f', (embedding[o * 768 + i]
                                    for i in range(768) for o in range(50257)))
            if precision == 'int8':
                q, scales = encode(head, 50257)
                write('lm_head.weight', 'int8', [768, 50257], q)
                write('lm_head.weight.scales', 'float32', [50257], scales)
            else:
                write('lm_head.weight', 'float16', [768, 50257], half_bytes(head))
        manifest = {'schema_version': 1, 'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                    'source_sha256': actual, 'precision': precision, 'architecture': ARCHITECTURE,
                    'binary': 'weights.bin', 'binary_bytes': binary_path.stat().st_size,
                    'binary_sha256': sha256_file(binary_path), 'alignment': ALIGNMENT,
                    'quantization': {'axis': 'output', 'range': [-127, 127], 'rounding': 'ties_to_even',
                                     'zero_channel_scale': 1, 'reference': 'fp16', 'dequantize': 'fp32_then_fp16'},
                    'tied_head': 'transpose_of_fp16_transformer.wte.weight_then_optional_quantization',
                    'tensors': tensor_entries, 'export_command': sys.argv,
                    'export_python': sys.version, 'export_timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'export_wall_seconds': time.perf_counter() - started,
                     'host_codec': ({'abi_version': 1, 'library_sha256': sha256_file(host_codec.path),
                                     'reference_parity': 'byte_exact_q_and_fp32_scales'}
                                    if precision == 'int8' else None)}
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='ascii')
    finally:
        reader.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--safetensors', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', required=True, choices=['fp16', 'int8'])
    parser.add_argument('--host-codec', help='explicit host-only codec shared library (required for INT8)')
    args = parser.parse_args()
    from .host_codec import HostCodec
    codec = HostCodec(args.host_codec) if args.host_codec else None
    manifest = export(args.safetensors, args.output, args.precision, codec)
    print(json.dumps({'manifest': str(Path(args.output).resolve() / 'manifest.json'),
                      'binary_sha256': manifest['binary_sha256']}))


if __name__ == '__main__':
    main()
