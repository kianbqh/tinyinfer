"""tinyc: a tiny ahead-of-time compiler for small PyTorch models.

torch.fx capture -> tinyc IR -> optimization passes -> C++ (OpenMP/SIMD) or CUDA source -> shared library.
"""
from .capture import capture
from .passes import optimize, PASSES
from .compile import compile_model

__all__ = ["capture", "optimize", "PASSES", "compile_model"]
