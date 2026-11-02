import array
import ctypes
from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from llm.cache_reference import KVCache, causal_keys
from llm.export_weights import SafeTensorFile, half_values, quantized_bytes
from llm.protocol import (MODEL_ID, MODEL_REVISION, DATA_ID, DATA_REVISION, DATA_SHA256,
                          canonical_hash, validate_frozen, windows)
from llm.quantization import dequantize, fp16, quantize
from llm.runtime import Options, Trace, _rows
from llm.evaluate import percentile, validate_config


class QuantizationTests(unittest.TestCase):
    def test_output_axis_zero_and_signed_channels(self):
        rows = [[0, 127, -254], [0, 0, 127], [0, -127, 0]]
        q, scales = quantize(rows)
        self.assertEqual(scales, [1.0, 1.0, 2.0])
        self.assertEqual(q, [[0, 127, -127], [0, 0, 64], [0, -127, 0]])
        self.assertEqual(dequantize(q, scales), [[0, 127, -254], [0, 0, 128], [0, -127, 0]])

    def test_ties_to_even_and_clipping_range(self):
        q, scale = quantize([[127], [.5], [1.5], [2.5], [-.5], [-1.5], [-2.5], [-127]])
        self.assertEqual(scale, [1.0])
        self.assertEqual(q, [[127], [0], [2], [2], [0], [-2], [-2], [-127]])
        self.assertTrue(all(-127 <= row[0] <= 127 for row in q))

    def test_exporter_matches_independent_reference(self):
        rows = [[.12, 0, -3.25], [1.5, 0, 7.75], [-2.0, 0, .125]]
        q, scales = quantize(rows)
        flat = half_values(array.array('f', (v for row in rows for v in row)))
        qb, sb = quantized_bytes(flat, 3)
        self.assertEqual(list(array.array('b', qb)), [v for row in q for v in row])
        self.assertEqual(list(array.array('f', sb)), scales)

    def test_dequantization_rounds_back_to_fp16(self):
        rows = [[.00123, -1.75], [-.00345, .3]]
        q, scales = quantize(rows)
        decoded = dequantize(q, scales)
        self.assertTrue(all(v == fp16(v) for row in decoded for v in row))
        self.assertLess(max(abs(decoded[i][j] - fp16(rows[i][j])) for i in range(2) for j in range(2)), .02)

    def test_invalid_shapes_dtypes_and_scales(self):
        for rows in ([], [[]], [[1], [1, 2]], [[float('nan')]], [[True]], [[float('inf')]]):
            with self.assertRaises((ValueError, OverflowError)):
                quantize(rows)
        for q, scales in (([[1]], []), ([[1]], [0]), ([[1]], [-1]), ([[1]], [math.nan]),
                          ([[1]], [True]), ([[-128]], [1]), ([[1.0]], [1]), ([[128]], [1])):
            with self.assertRaises(ValueError):
                dequantize(q, scales)


class CacheTests(unittest.TestCase):
    def test_append_variable_lengths_absolute_offsets_and_isolation(self):
        cache = KVCache(3, 2, 5)
        cache.append_batch([0, 2], [[[10, 11], [20, 21]], [[30], [40]]],
                           [[[110, 111], [120, 121]], [[130], [140]]])
        self.assertEqual(cache.lengths, [2, 0, 1])
        self.assertEqual(cache.read(0, 1), ([20, 21], [120, 121]))
        cache.append_batch([2], [[[31, 32], [41, 42]]], [[[131, 132], [141, 142]]])
        self.assertEqual(cache.lengths, [2, 0, 3])
        self.assertEqual(cache.read(2, 0), ([30, 31, 32], [130, 131, 132]))
        self.assertEqual(list(causal_keys(0, 2, 5)), [0, 1, 2])
        self.assertEqual(list(causal_keys(1, 2, 3)), [0, 1, 2])

    def test_owned_input_output_and_reset(self):
        cache = KVCache(2, 1, 3)
        keys, values = [[[{'x': 1}]]], [[[{'y': 2}]]]
        cache.append_batch([0], keys, values)
        keys[0][0][0]['x'] = 99
        observed = cache.read(0, 0)
        self.assertEqual(observed[0][0]['x'], 1)
        observed[0][0]['x'] = 42
        self.assertEqual(cache.read(0, 0)[0][0]['x'], 1)
        cache.reset(0)
        self.assertEqual(cache.lengths, [0, 0])
        self.assertEqual(cache.read(0, 0), ([], []))
        cache.append_batch([0], [[[7]]], [[[8]]])
        self.assertEqual(cache.read(0, 0), ([7], [8]))

    def test_validation_transactional(self):
        cache = KVCache(2, 1, 2)
        cache.append_batch([0], [[[1]]], [[[2]]])
        cases = [([0, 0], [[[3]], [[4]]], [[[5]], [[6]]]),
                 ([1, 0], [[[3]], [[4, 5]]], [[[6]], [[7, 8]]]),
                 ([1], [[[]]], [[[]]]), ([True], [[[3]]], [[[4]]]),
                 ([2], [[[3]]], [[[4]]]), ([1], [[[3]]], [[[4, 5]]])]
        for args in cases:
            with self.assertRaises(ValueError):
                cache.append_batch(*args)
            self.assertEqual(cache.lengths, [1, 0])
            self.assertEqual(cache.read(1, 0), ([], []))


class ProtocolTests(unittest.TestCase):
    def test_target_ownership_across_overlapping_windows(self):
        for size in (2, 5, 8, 9, 10, 15, 16, 17, 31):
            tokens = list(range(size))
            document = {'schema_version': 1, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                'dataset_id': DATA_ID, 'dataset_revision': DATA_REVISION, 'dataset_sha256': DATA_SHA256,
                'dataset_config': 'wikitext-2-raw-v1', 'split': 'test', 'join': '\n\n',
                'insert_eos': False, 'add_special_tokens': False, 'preserve_blank_rows': True,
                'tokens': tokens, 'context': 8, 'stride': 3, 'windows': windows(tokens, 8, 3),
                'token_ids_sha256': canonical_hash(tokens), 'target_count': size - 1}
            self.assertEqual(validate_frozen(document), size - 1)
            document['windows'][0]['mask'][1] = 0
            with self.assertRaises(ValueError):
                validate_frozen(document)

    def test_context_and_exact_ids(self):
        for args in (([1, 2], 1025, 1), ([1, 2], 8, 8), ([True, 2], 8, 4), ([1, 50257], 8, 4)):
            with self.assertRaises(ValueError):
                windows(*args)


class ReaderAndAbiTests(unittest.TestCase):
    def test_safetensors_bare_conv1d_orientation(self):
        header = {'h.0.attn.c_attn.weight': {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [0, 24]},
                  'h.0.attn.bias': {'dtype': 'BOOL', 'shape': [1], 'data_offsets': [24, 25]}}
        encoded = json.dumps(header).encode()
        # TMPDIR must be supplied by CI/preparation runner to a G-backed path locally.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'small.safetensors'
            path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + struct.pack('<6f', *range(6)) + b'\x01')
            reader = SafeTensorFile(path)
            try:
                self.assertEqual(list(reader.read('transformer.h.0.attn.c_attn.weight', [2, 3])), list(range(6)))
                with self.assertRaises(ValueError):
                    reader.read('transformer.h.0.attn.c_attn.weight', [3, 2])
            finally:
                reader.close()

    def test_host_abi_metadata_and_no_heavy_import(self):
        self.assertEqual(ctypes.sizeof(Options), 28)
        self.assertEqual(ctypes.sizeof(Trace), 96)
        ids, lengths, stride = _rows([[10, 11], [12]])
        self.assertEqual((list(ids), list(lengths), stride), ([10, 11, 12, 0], [2, 1], 2))
        for rows in ([], [[]], [[True]], [[-1]], [[50257]]):
            with self.assertRaises(ValueError):
                _rows(rows)
        for name in ('torch', 'numpy', 'transformers', 'datasets'):
            self.assertNotIn(name, sys.modules)


class EvaluationContractTests(unittest.TestCase):
    def test_fixed_protocol_and_strict_config(self):
        config = json.loads((Path(__file__).parents[1] / 'configs/evaluation.json').read_text())
        self.assertEqual(validate_config(config), config)
        for key, value in (('schema_version', True), ('attention_config', True), ('gemm_config', True),
                           ('batches', [True, 8, 32]), ('batches', [1, 8]), ('capacity', 1025),
                           ('prompt_tokens', 1000), ('repeats', 0), ('attempt_compile', 1),
                           ('seed', 2**32), ('compile_backend', 'eager'), ('hf_atol', math.nan)):
            candidate = deepcopy(config)
            candidate[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_config(candidate)

    def test_percentile_definition_and_invalid_samples(self):
        self.assertEqual(percentile([10, 0, 20], 50), 10)
        self.assertEqual(percentile([0, 10], 95), 9.5)
        for values in ([], [math.nan], [-1], [math.inf]):
            with self.assertRaises(ValueError):
                percentile(values, 50)


if __name__ == '__main__':
    unittest.main()
