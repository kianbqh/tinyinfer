"""Minimal SSA graph IR. Every value is a float32 tensor with a static shape."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Elementwise ops and their scalar C expression templates ({0}, {1} = operands).
UNARY = {
    "relu": "fmaxf({0}, 0.f)",
    "gelu": "(0.5f * {0} * (1.f + erff({0} * 0.70710678118654752f)))",
    "sigmoid": "(1.f / (1.f + expf(-{0})))",
    "tanh": "tanhf({0})",
    "neg": "(-{0})",
}
BINARY = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})"}
ELEMENTWISE = set(UNARY) | set(BINARY)
NP_UNARY = {
    "relu": lambda x: np.maximum(x, 0),
    "gelu": lambda x: 0.5 * x * (1 + _erf(x / np.sqrt(2))),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "neg": np.negative,
}
NP_BINARY = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}


def _erf(x):
    from scipy.special import erf
    return erf(x)


@dataclass
class Node:
    name: str
    op: str                      # input | const | linear | conv2d | batchnorm | maxpool2d | reshape | <elementwise> | fused_ew
    inputs: list[str]
    shape: tuple
    attrs: dict = field(default_factory=dict)

    def __repr__(self):
        a = {k: v for k, v in self.attrs.items() if k not in ("expr",)}
        return f"%{self.name} = {self.op}({', '.join('%' + i for i in self.inputs)}) {list(self.shape)} {a if a else ''}"


@dataclass
class Graph:
    nodes: list[Node]
    consts: dict[str, np.ndarray]
    inputs: list[str]
    outputs: list[str]

    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def users(self) -> dict[str, list[Node]]:
        u: dict[str, list[Node]] = {n.name: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                u[i].append(n)
        return u

    def is_const(self, name: str) -> bool:
        return name in self.consts

    def op_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for n in self.nodes:
            if n.op not in ("input", "const"):
                c[n.op] = c.get(n.op, 0) + 1
        return c

    def num_kernels(self) -> int:
        return sum(1 for n in self.nodes if n.op not in ("input", "const", "reshape"))

    def __str__(self):
        return "\n".join(repr(n) for n in self.nodes if n.op != "const") + f"\nreturn {self.outputs}"
