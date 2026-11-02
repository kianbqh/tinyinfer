"""Thin stdlib ctypes wrapper of the native C++ server's exact model ABI."""
import ctypes as C
import json
from pathlib import Path
from .cache_reference import exact_int

VOCAB = 50257


class Options(C.Structure):
    _fields_ = [('struct_size', C.c_uint32)] + [(n, C.c_int32) for n in
        ('device', 'max_batch', 'cache_slots', 'capacity', 'gemm_config', 'attention_config')]


class Trace(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('forward_calls', 'embedding', 'gemm_fp16',
        'gemm_int8', 'layernorm', 'gelu', 'attention', 'residual', 'kv_scatter_gather',
        'lm_head', 'argmax', 'nll')]


def _i32(values):
    return (C.c_int32 * len(values))(*values)


def _rows(rows):
    if not rows:
        raise ValueError('nonempty token batch required')
    lengths = [len(row) for row in rows]
    if not all(lengths):
        raise ValueError('empty token row')
    for row in rows:
        for token in row:
            exact_int(token, 'token', 0, VOCAB - 1)
    stride = max(lengths)
    flat = [t for row in rows for t in list(row) + [0] * (stride - len(row))]
    return _i32(flat), _i32(lengths), stride


class NativeModel:
    """Owns one synchronous native model/context; serialize all calls.

    Import/constructor never imports Torch. Native code owns device selection,
    stream, GPU buffers and synchronization. Retain this object until calls end.
    Use close()/with to release resources explicitly (including poisoned models).
    """
    def __init__(self, library, manifest, *, device=0, max_batch=32, cache_slots=32,
                 capacity=1024, gemm_config=0, attention_config=0):
        self.options = Options(C.sizeof(Options), exact_int(device, 'device', 0, 1024),
            exact_int(max_batch, 'max_batch', 1, 32),
            exact_int(cache_slots, 'cache_slots', max_batch, 1024),
            exact_int(capacity, 'capacity', 2, 1024),
            exact_int(gemm_config, 'gemm_config', 0, 11),
            exact_int(attention_config, 'attention_config', 0, 1))
        self.library_path = str(Path(library).resolve())
        self.manifest_path = str(Path(manifest).resolve())
        self.manifest = json.loads(Path(self.manifest_path).read_text(encoding='ascii'))
        self.lib = C.CDLL(self.library_path)
        self._model = C.c_void_p()
        p = C.c_void_p
        i = C.c_int32
        signatures = {
            'ti_llm_abi_version': ([], C.c_int),
            'ti_llm_last_error': ([], C.c_char_p),
            'ti_llm_create': ([C.c_char_p, C.POINTER(Options), C.POINTER(p)], C.c_int),
            'ti_llm_destroy': ([p], C.c_int), 'ti_llm_reset': ([p, i], C.c_int),
            'ti_llm_lengths': ([p, p, i], C.c_int),
            'ti_llm_get_trace': ([p, C.POINTER(Trace)], C.c_int),
            'ti_llm_forward_batch': ([p, p, p, p, i, i, p, p], C.c_int),
            'ti_llm_greedy_batch': ([p, p, p, p, p, i, i, p, i, p, p, p], C.c_int),
            'ti_llm_score_window': ([p, p, p, i, p, p, p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        if self.lib.ti_llm_abi_version() != 1:
            raise RuntimeError('unsupported native model ABI')
        self._check(self.lib.ti_llm_create(self.manifest_path.encode(), C.byref(self.options), C.byref(self._model)))

    def _check(self, code):
        if code:
            raise RuntimeError(f'ti_llm status {code}: {self.lib.ti_llm_last_error().decode("utf-8", errors="replace")}')

    def _live(self):
        if not self._model:
            raise RuntimeError('model is closed')

    def close(self):
        if self._model:
            self._check(self.lib.ti_llm_destroy(self._model))
            self._model = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def reset(self, slot):
        self._live()
        self._check(self.lib.ti_llm_reset(self._model, exact_int(slot, 'slot', 0, self.options.cache_slots - 1)))

    def reset_all(self):
        for slot in range(self.options.cache_slots):
            self.reset(slot)

    def lengths(self):
        self._live()
        result = _i32([0] * self.options.cache_slots)
        self._check(self.lib.ti_llm_lengths(self._model, result, len(result)))
        return list(result)

    def trace(self):
        self._live()
        result = Trace()
        self._check(self.lib.ti_llm_get_trace(self._model, C.byref(result)))
        return {name: getattr(result, name) for name, _ in Trace._fields_}

    def forward(self, slots, rows, *, return_logits=True, return_next=True):
        self._live()
        if len(slots) != len(rows) or len(rows) > self.options.max_batch:
            raise ValueError('batch/slot extent mismatch')
        ss = _i32([exact_int(s, 'slot', 0, self.options.cache_slots - 1) for s in slots])
        tokens, lengths, stride = _rows(rows)
        logits = (C.c_float * (len(rows) * VOCAB))() if return_logits else None
        next_ids = _i32([0] * len(rows)) if return_next else None
        self._check(self.lib.ti_llm_forward_batch(self._model, ss, tokens, lengths, len(rows), stride, logits, next_ids))
        # Return contiguous ctypes storage rather than millions of Python floats.
        # torch.frombuffer(logits,dtype=torch.float32).reshape(B,V) is zero copy.
        return {'logits': logits, 'shape': [len(rows), VOCAB],
                'next_tokens': list(next_ids) if next_ids is not None else None}

    def greedy(self, prompts, max_new_tokens, *, slots=None):
        self._live()
        batch = len(prompts)
        if not batch or batch > self.options.max_batch:
            raise ValueError('invalid batch size')
        slots = list(range(batch)) if slots is None else list(slots)
        if len(slots) != batch:
            raise ValueError('slot extent mismatch')
        requested = [max_new_tokens] * batch if type(max_new_tokens) is int else list(max_new_tokens)
        if len(requested) != batch:
            raise ValueError('output-length extent mismatch')
        requested = [exact_int(n, 'max_new_tokens', 1, self.options.capacity) for n in requested]
        ss = _i32([exact_int(s, 'slot', 0, self.options.cache_slots - 1) for s in slots])
        tokens, lengths, stride = _rows(prompts)
        out_stride = max(requested)
        output = _i32([-1] * (batch * out_stride))
        counts = _i32([0] * batch)
        ttft, total = (C.c_double * batch)(), (C.c_double * batch)()
        self._check(self.lib.ti_llm_greedy_batch(self._model, ss, tokens, lengths, _i32(requested),
                    batch, stride, output, out_stride, counts, ttft, total))
        return {'tokens': [list(output[i * out_stride:i * out_stride + counts[i]]) for i in range(batch)],
                'output_lengths': list(counts), 'ttft_ms': list(ttft), 'total_ms': list(total)}

    def score(self, tokens, mask, *, positions=False):
        self._live()
        if len(tokens) != len(mask) or not 2 <= len(tokens) <= self.options.capacity:
            raise ValueError('invalid score extent')
        ids = _i32([exact_int(t, 'token', 0, VOCAB - 1) for t in tokens])
        mm = (C.c_uint8 * len(mask))(*[exact_int(m, 'mask', 0, 1) for m in mask])
        nll, count = C.c_double(), C.c_uint64()
        losses = (C.c_double * len(tokens))() if positions else None
        self._check(self.lib.ti_llm_score_window(self._model, ids, mm, len(tokens), C.byref(nll), C.byref(count), losses))
        return {'sum_nll': nll.value, 'target_count': count.value,
                'position_nll': list(losses) if losses is not None else None}
