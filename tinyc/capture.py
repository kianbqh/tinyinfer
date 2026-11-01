"""torch.fx graph capture -> tinyc IR (with static shapes from ShapeProp)."""
from __future__ import annotations

import operator

import numpy as np
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from .ir import Graph, Node

_FN = {
    torch.relu: "relu", F.relu: "relu", F.gelu: "gelu", torch.sigmoid: "sigmoid", torch.tanh: "tanh",
    operator.add: "add", torch.add: "add", operator.sub: "sub", operator.mul: "mul", torch.mul: "mul",
    operator.truediv: "div", operator.neg: "neg",
}
_MOD = {nn.ReLU: "relu", nn.GELU: "gelu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh"}


def capture(model: nn.Module, example: torch.Tensor) -> Graph:
    model = model.eval()
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(example)
    mods = dict(gm.named_modules())
    nodes: list[Node] = []
    consts: dict[str, np.ndarray] = {}
    inputs, outputs = [], []
    n_const = [0]

    def const(arr, hint="c") -> str:
        name = f"{hint}.{n_const[0]}"   # "." cannot clash with fx node names
        n_const[0] += 1
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        consts[name] = arr
        nodes.append(Node(name, "const", [], tuple(arr.shape)))
        return name

    def arg(a) -> str:
        if isinstance(a, torch.fx.Node):
            return a.name
        if isinstance(a, (int, float)):
            return const(np.array(a, dtype=np.float32), "s")
        raise NotImplementedError(f"argument {a!r}")

    for fx in gm.graph.nodes:
        shape = tuple(fx.meta["tensor_meta"].shape) if "tensor_meta" in fx.meta else ()
        if fx.op == "placeholder":
            nodes.append(Node(fx.name, "input", [], shape)); inputs.append(fx.name)
        elif fx.op == "output":
            outputs.append(fx.args[0].name)
        elif fx.op == "get_attr":
            t = gm
            for p in fx.target.split("."):
                t = getattr(t, p)
            consts[fx.name] = t.detach().numpy().astype(np.float32)
            nodes.append(Node(fx.name, "const", [], tuple(t.shape)))
        elif fx.op == "call_module":
            m = mods[fx.target]
            x = fx.args[0].name
            if isinstance(m, nn.Linear):
                b = m.bias if m.bias is not None else torch.zeros(m.out_features)
                nodes.append(Node(fx.name, "linear", [x, const(m.weight.detach(), "w"), const(b.detach(), "b")], shape))
            elif isinstance(m, nn.Conv2d):
                assert m.groups == 1 and m.dilation == (1, 1) and isinstance(m.padding, tuple)
                b = m.bias if m.bias is not None else torch.zeros(m.out_channels)
                nodes.append(Node(fx.name, "conv2d", [x, const(m.weight.detach(), "w"), const(b.detach(), "b")], shape,
                                  {"stride": m.stride[0], "pad": m.padding[0], "k": m.kernel_size[0]}))
            elif isinstance(m, nn.BatchNorm2d):
                scale = (m.weight / torch.sqrt(m.running_var + m.eps)).detach()
                shift = (m.bias - m.running_mean * scale).detach()
                nodes.append(Node(fx.name, "batchnorm", [x, const(scale, "bn_s"), const(shift, "bn_t")], shape))
            elif isinstance(m, nn.MaxPool2d):
                k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                s = m.stride if isinstance(m.stride, int) else m.stride[0]
                nodes.append(Node(fx.name, "maxpool2d", [x], shape, {"k": k, "stride": s}))
            elif isinstance(m, nn.Flatten):
                nodes.append(Node(fx.name, "reshape", [x], shape))
            elif isinstance(m, (nn.Dropout, nn.Identity)):
                nodes.append(Node(fx.name, "reshape", [x], shape))  # eval: identity
            elif type(m) in _MOD:
                if isinstance(m, nn.GELU):
                    assert m.approximate == "none"
                nodes.append(Node(fx.name, _MOD[type(m)], [x], shape))
            else:
                raise NotImplementedError(f"module {type(m).__name__}")
        elif fx.op == "call_function":
            if fx.target in _FN:
                nodes.append(Node(fx.name, _FN[fx.target], [arg(a) for a in fx.args], shape))
            elif fx.target is torch.flatten:
                nodes.append(Node(fx.name, "reshape", [fx.args[0].name], shape))
            else:
                raise NotImplementedError(f"function {fx.target}")
        elif fx.op == "call_method":
            if fx.target in ("view", "reshape", "flatten"):
                nodes.append(Node(fx.name, "reshape", [fx.args[0].name], shape))
            elif fx.target in ("relu", "sigmoid", "tanh"):
                nodes.append(Node(fx.name, fx.target, [fx.args[0].name], shape))
            else:
                raise NotImplementedError(f"method {fx.target}")
    g = Graph(nodes, consts, inputs, outputs)
    _check_broadcast(g)
    return g


def _check_broadcast(g: Graph):
    """Binary elementwise ops support: same shape, scalar, or per-channel [C,1,1] / [N] bias-like vectors."""
    shapes = {n.name: n.shape for n in g.nodes}
    for n in g.nodes:
        if n.op in ("add", "sub", "mul", "div"):
            a, b = (shapes[i] for i in n.inputs)
            for s in (a, b):
                if s != n.shape and int(np.prod(s)) != 1:
                    raise NotImplementedError(f"broadcast {a} x {b} in {n.name}")
