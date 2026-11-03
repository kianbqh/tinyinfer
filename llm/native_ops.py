"""Inference-only device tensor wrappers for the standalone v2 C ABI.

No Torch headers, libtorch, cpp_extension or silent CPU copies. See the v2 kernel
interface contract for layouts, stream ownership and deferred CUDA errors.
"""
from __future__ import annotations

import ctypes
import math
import json
from pathlib import Path

_I32_MAX = 2**31 - 1
GEMM_CATALOG = tuple({"id":i,"tile":((32,32,16,4),(64,32,32,8),(64,64,32,8))[i%3],
                     "stages":1 if 3<=i<6 or i>=9 else 2,
                     "epilogue":"separate" if i>=6 else "fused"} for i in range(12))


def _dim(value, name):
    if type(value) is not int or not 0 < value <= _I32_MAX:
        raise ValueError(f"{name} must be an integer in [1, INT32_MAX]")
    return value


def _metadata(value, batch, default, maximum, name):
    """CPU-checkable validation; intentionally reject device tensor metadata."""
    values = [default] * batch if value is None else ([value] * batch if type(value) is int else list(value))
    if len(values) != batch or any(type(x) is not int or x < 0 or x > maximum for x in values):
        raise ValueError(f"{name} must contain {batch} integers in [0,{maximum}]")
    return values


class NativeOps:
    def __init__(self, library_path):
        import torch
        self.torch = torch
        self.library_path = str(Path(library_path).expanduser().resolve(strict=True))
        self.lib = ctypes.CDLL(self.library_path)
        p, i, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        signatures = {
            "ti_v2_gemm_f16": [p,p,p,p,i,i,i,i,i,u,i],
            "ti_v2_gemm_i8": [p,p,p,p,p,i,i,i,i,i,u,i],
            "ti_v2_gemm_ex_f16": [p,p,p,p,p,i,i,i,i,i,u,i],
            "ti_v2_gemm_ex_i8": [p,p,p,p,p,p,i,i,i,i,i,u,i],
            "ti_v2_legacy_gemm_f32": [p,p,p,i,i,i,u,i],
            "ti_v2_attention_f16": [p,p,p,p,p,p,p,i,i,i,i,i,i,i,u,i],
            "ti_v2_layernorm_f16": [p,p,p,p,i,i,ctypes.c_float,u,i],
            "ti_v2_gelu_new_f16": [p,p,ctypes.c_int64,u,i],
            "ti_v2_softmax_f16": [p,p,i,i,u,i],
        }
        self.lib.ti_v2_abi_version.argtypes = []
        self.lib.ti_v2_abi_version.restype = i
        self.lib.ti_v2_last_error.argtypes = []
        self.lib.ti_v2_last_error.restype = ctypes.c_char_p
        if self.lib.ti_v2_abi_version() != 2:
            raise RuntimeError("native v2 ABI version mismatch")
        self.lib.ti_v2_gemm_catalog_json.argtypes = []
        self.lib.ti_v2_gemm_catalog_json.restype = ctypes.c_char_p
        self.catalog = json.loads(self.lib.ti_v2_gemm_catalog_json())
        normalized = [{**entry,"tile":tuple(entry["tile"])} for entry in self.catalog["configs"]]
        if normalized != list(GEMM_CATALOG):
            raise RuntimeError("native GEMM catalog does not match wrapper allow-list")
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, i

    def _tensor(self, tensor, dtype, name, ndim=None, device=None):
        t = self.torch
        if not isinstance(tensor, t.Tensor) or tensor.device.type != "cuda":
            raise TypeError(f"{name} must be a CUDA Tensor")
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise TypeError(f"{name} must have dtype={dtype} and contiguous row-major storage")
        if tensor.requires_grad:
            raise ValueError(f"{name} requires_grad is unsupported by inference-only native ops")
        if ndim is not None and tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if tensor.ndim == 0 or tensor.numel() == 0:
            raise ValueError(f"{name} must have at least one nonempty dimension")
        for index, size in enumerate(tensor.shape):
            _dim(size, f"{name}.shape[{index}]")
        if tensor.numel() > (2**63 - 1) // tensor.element_size():
            raise ValueError(f"{name} byte extent exceeds signed 64-bit indexing")
        if device is not None and tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
        return tensor

    @staticmethod
    def _ptr(tensor):
        return None if tensor is None else tensor.data_ptr()

    def _call(self, name, tensors, args):
        device = tensors[0].device
        with self.torch.cuda.device(device):
            stream = self.torch.cuda.current_stream(device)
            # Allocation tracking precedes native dispatch, including error paths.
            # The local tuple retains Python objects for the entire ctypes call.
            for tensor in tensors:
                if tensor is not None:
                    tensor.record_stream(stream)
            status = getattr(self.lib, name)(*args, stream.cuda_stream, device.index)
            if status:
                reason = self.lib.ti_v2_last_error().decode("utf-8", errors="replace")
                raise RuntimeError(f"{name} failed ({status}): {reason}")

    def _gemm_args(self, a, b, bias, config, output_dtype, weight_dtype):
        t = self.torch
        self._tensor(a, t.float16, "a", 2)
        self._tensor(b, weight_dtype, "b", 2, a.device)
        m, k = a.shape
        if b.shape[0] != k:
            raise ValueError("GEMM a[M,K] and b[K,N] inner dimensions differ")
        n = b.shape[1]
        if type(config) is not int or not 0 <= config < len(GEMM_CATALOG):
            raise ValueError("GEMM config must be allow-listed ID0..11")
        only = getattr(self,"catalog",{}).get("only_config",-1)
        if only != -1 and config != only:
            raise ValueError("GEMM config is unavailable in this single-candidate build")
        bm = GEMM_CATALOG[config]["tile"][0]
        if (m - 1) // bm + 1 > 65535:
            raise ValueError("GEMM M exceeds CUDA grid-y limit")
        if bias is not None:
            self._tensor(bias, t.float16, "bias", 1, a.device)
            if bias.shape != (n,):
                raise ValueError("bias must have shape [N]")
        dtype = t.float16 if output_dtype is None else output_dtype
        if dtype not in (t.float16, t.float32):
            raise TypeError("output_dtype must be torch.float16 or torch.float32")
        output = t.empty((m, n), dtype=dtype, device=a.device)
        return output, (m, n, k, int(dtype == t.float32), config)

    def gemm(self, a, b, *, bias=None, config=0, output_dtype=None):
        out, dims = self._gemm_args(a, b, bias, config, output_dtype, self.torch.float16)
        workspace = self.torch.empty(out.shape,dtype=self.torch.float32,device=out.device) if config>=6 else None
        tensors = (a, b, bias, out, workspace)
        self._call("ti_v2_gemm_ex_f16", tensors, tuple(map(self._ptr, tensors)) + dims)
        return out

    def gemm_int8(self, a, b, scales, *, bias=None, config=0, output_dtype=None):
        # Scale numerical validity must be checked by the quantizer before upload.
        self._tensor(a, self.torch.float16, "a", 2)
        self._tensor(b, self.torch.int8, "b", 2, a.device)
        self._tensor(scales, self.torch.float32, "scales", 1, a.device)
        if scales.shape != (b.shape[1],):
            raise ValueError("scales must have shape [N], one scale per output channel")
        out, dims = self._gemm_args(a, b, bias, config, output_dtype, self.torch.int8)
        workspace = self.torch.empty(out.shape,dtype=self.torch.float32,device=out.device) if config>=6 else None
        tensors = (a, b, scales, bias, out, workspace)
        self._call("ti_v2_gemm_ex_i8", tensors, tuple(map(self._ptr, tensors)) + dims)
        return out

    def legacy_gemm(self, a, b):
        self._tensor(a, self.torch.float32, "a", 2)
        self._tensor(b, self.torch.float32, "b", 2, a.device)
        if a.shape[1] != b.shape[0] or (a.shape[0] - 1) // 64 + 1 > 65535:
            raise ValueError("invalid FP32 GEMM dimensions")
        out = self.torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
        tensors = (a, b, out)
        self._call("ti_v2_legacy_gemm_f32", tensors, tuple(map(self._ptr, tensors)) + (a.shape[0], b.shape[1], a.shape[1]))
        return out

    def layernorm(self, x, gamma, beta, *, eps=1e-5):
        t = self.torch
        self._tensor(x, t.float16, "x")
        self._tensor(gamma, t.float16, "gamma", 1, x.device)
        self._tensor(beta, t.float16, "beta", 1, x.device)
        cols, rows = x.shape[-1], x.numel() // x.shape[-1]
        _dim(rows, "rows")
        if gamma.shape != (cols,) or beta.shape != (cols,) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("LayerNorm gamma/beta shape or eps invalid")
        out = t.empty_like(x)
        tensors = (x, gamma, beta, out)
        self._call("ti_v2_layernorm_f16", tensors, tuple(map(self._ptr, tensors)) + (rows, cols, eps))
        return out

    def gelu_new(self, x):
        self._tensor(x, self.torch.float16, "x")
        out = self.torch.empty_like(x)
        self._call("ti_v2_gelu_new_f16", (x, out), (x.data_ptr(), out.data_ptr(), x.numel()))
        return out

    def softmax(self, x):
        self._tensor(x, self.torch.float16, "x")
        cols, rows = x.shape[-1], x.numel() // x.shape[-1]
        _dim(rows, "rows")
        out = self.torch.empty_like(x)
        self._call("ti_v2_softmax_f16", (x, out), (x.data_ptr(), out.data_ptr(), rows, cols))
        return out

    def attention(self, q, k, v, *, q_lengths=None, kv_lengths=None, query_offsets=None,
                  causal=True, config=0):
        t = self.torch
        self._tensor(q, t.float16, "q", 4)
        self._tensor(k, t.float16, "k", 4, q.device)
        self._tensor(v, t.float16, "v", 4, q.device)
        batch, heads, nq, d = q.shape
        nk = k.shape[2]
        if k.shape != v.shape or k.shape[:2] != q.shape[:2] or k.shape[3] != d or d not in (64, 128):
            raise ValueError("attention expects q[B,H,Q,D], k/v[B,H,K,D], D64/128")
        if batch > 65535 or heads > 65535 or type(causal) is not bool or type(config) is not int or config not in (0, 1):
            raise ValueError("invalid attention grid, causal flag or config")
        ql = _metadata(q_lengths, batch, nq, nq, "q_lengths")
        kl = _metadata(kv_lengths, batch, nk, nk, "kv_lengths")
        offsets = _metadata(query_offsets, batch, 0, _I32_MAX - nq, "query_offsets")
        with t.cuda.device(q.device):
            metadata = tuple(t.tensor(values, dtype=t.int32, device=q.device) for values in (ql, kl, offsets))
            out = t.empty_like(q)
            tensors = (q, k, v, out) + metadata
            self._call("ti_v2_attention_f16", tensors, tuple(map(self._ptr, tensors)) +
                       (batch, heads, nq, nk, d, int(causal), config))
        return out
