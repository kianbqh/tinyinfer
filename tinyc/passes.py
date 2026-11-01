"""Optimization passes over the tinyc IR + a NumPy reference interpreter used to verify them.

Pipeline (each pass is Graph -> Graph and semantics-preserving):
  fold_batchnorm      conv2d -> batchnorm  ==>  conv2d with rescaled weights/bias
  constant_fold       nodes whose inputs are all constants are evaluated at compile time
  dce                 drop nodes that do not reach an output
  fuse_epilogue       linear/conv2d followed by an elementwise chain -> one GEMM with a fused epilogue
  fuse_elementwise    remaining elementwise chains -> one fused loop (expression tree)
  pretranspose        linear weights [N,K] -> [K,N] so the GEMM inner loop is contiguous (layout)
Memory planning (liveness-based buffer reuse) runs at codegen time: see plan_memory().
"""
from __future__ import annotations

import copy

import numpy as np

from .ir import BINARY, ELEMENTWISE, NP_BINARY, NP_UNARY, UNARY, Graph, Node


# ---------------------------------------------------------------- reference interpreter
def interpret(g: Graph, *xs: np.ndarray) -> np.ndarray:
    env: dict[str, np.ndarray] = dict(g.consts)
    for name, x in zip(g.inputs, xs):
        env[name] = np.asarray(x, np.float32)
    for n in g.nodes:
        if n.op in ("input", "const"):
            continue
        a = [env[i] for i in n.inputs]
        env[n.name] = _eval(n, a).astype(np.float32).reshape(n.shape)
    return env[g.outputs[0]]


def _eval(n: Node, a):
    import torch
    import torch.nn.functional as F
    if n.op in UNARY:
        return NP_UNARY[n.op](a[0])
    if n.op in BINARY:
        return NP_BINARY[n.op](a[0], a[1])
    if n.op == "linear":
        w = a[1].T if n.attrs.get("pretransposed") else a[1]
        y = a[0] @ w.T + a[2]
        return _apply_epilogue(y, n, a, channel_axis=-1)
    if n.op == "conv2d":
        y = F.conv2d(torch.from_numpy(a[0]), torch.from_numpy(a[1]), torch.from_numpy(a[2]),
                     stride=n.attrs["stride"], padding=n.attrs["pad"]).numpy()
        return _apply_epilogue(y, n, a, channel_axis=1)
    if n.op == "batchnorm":
        return a[0] * a[1][None, :, None, None] + a[2][None, :, None, None]
    if n.op == 'layernorm':
        x = a[0].astype(np.float64)
        return ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + n.attrs['eps'])) * a[1] + a[2]
    if n.op == 'softmax':
        e = np.exp(a[0] - a[0].max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    if n.op == "maxpool2d":
        return F.max_pool2d(torch.from_numpy(a[0]), n.attrs["k"], n.attrs["stride"]).numpy()
    if n.op == "reshape":
        return a[0].reshape(n.shape)
    if n.op == "fused_ew":
        return _eval_expr(n.attrs["tree"], a)
    raise NotImplementedError(n.op)


def _apply_epilogue(y, n: Node, a, channel_axis):
    for step in n.attrs.get("epilogue", []):
        if step[0] == "unary":
            y = NP_UNARY[step[1]](y)
        else:
            _, op, idx, lhs = step
            o = a[idx]
            y = NP_BINARY[op](o, y) if lhs else NP_BINARY[op](y, o)
    return y


def _eval_expr(t, a):
    if t[0] == "in":
        return a[t[1]]
    if t[0] == "lit":
        return np.float32(t[1])
    if t[0] in UNARY:
        return NP_UNARY[t[0]](_eval_expr(t[1], a))
    return NP_BINARY[t[0]](_eval_expr(t[1], a), _eval_expr(t[2], a))


def _prune_consts(g: Graph) -> Graph:
    used = {i for n in g.nodes for i in n.inputs} | set(g.outputs)
    g.nodes = [n for n in g.nodes if n.op != "const" or n.name in used]
    g.consts = {k: v for k, v in g.consts.items() if k in used}
    return g


def _replace_uses(g: Graph, old: str, new: str):
    for n in g.nodes:
        n.inputs = [new if i == old else i for i in n.inputs]
    g.outputs = [new if o == old else o for o in g.outputs]


# ---------------------------------------------------------------- passes
def fold_batchnorm(g: Graph) -> Graph:
    g = copy.deepcopy(g)
    users = g.users()
    for bn in [n for n in g.nodes if n.op == "batchnorm"]:
        src = g.node(bn.inputs[0])
        if src.op != "conv2d" or len(users[src.name]) != 1 or src.attrs.get("epilogue") or src.name in g.outputs:
            continue
        s, t = g.consts[bn.inputs[1]], g.consts[bn.inputs[2]]
        w, b = g.consts[src.inputs[1]], g.consts[src.inputs[2]]
        g.consts[src.inputs[1]] = (w * s[:, None, None, None]).astype(np.float32)
        g.consts[src.inputs[2]] = (b * s + t).astype(np.float32)
        _replace_uses(g, bn.name, src.name)
        g.nodes.remove(bn)
        users = g.users()
    return _prune_consts(g)


def constant_fold(g: Graph) -> Graph:
    g = copy.deepcopy(g)
    for n in list(g.nodes):
        if n.op in ("input", "const") or not n.inputs:
            continue
        if all(g.is_const(i) for i in n.inputs):
            val = _eval(n, [g.consts[i] for i in n.inputs]).astype(np.float32).reshape(n.shape)
            g.consts[n.name] = val
            n.op, n.inputs, n.attrs = "const", [], {}
    return _prune_consts(g)


def dce(g: Graph) -> Graph:
    g = copy.deepcopy(g)
    live = set(g.outputs)
    for n in reversed(g.nodes):
        if n.name in live:
            live.update(n.inputs)
    g.nodes = [n for n in g.nodes if n.name in live or n.op == "input"]
    return _prune_consts(g)


def _scalar(g: Graph, name: str):
    return g.is_const(name) and g.consts[name].size == 1


def fuse_epilogue(g: Graph) -> Graph:
    g = copy.deepcopy(g)
    pos = {n.name: i for i, n in enumerate(g.nodes)}
    changed = True
    while changed:
        changed = False
        users = g.users()
        for n in g.nodes:
            if n.op not in ("linear", "conv2d") or n.name in g.outputs:
                continue
            us = users[n.name]
            if len(us) != 1 or us[0].op not in ELEMENTWISE or us[0].shape != n.shape:
                continue
            u = us[0]
            if u.op in UNARY:
                step = ("unary", u.op)
            else:
                other_i = 1 if u.inputs[0] == n.name else 0
                other = u.inputs[other_i]
                if other == n.name:
                    continue                     # x op x
                if _scalar(g, other):
                    step = ("binary", u.op, other, other_i == 0)
                elif g.node(other).shape == n.shape and pos[other] < pos[n.name]:
                    step = ("binary", u.op, other, other_i == 0)   # e.g. residual add
                else:
                    continue
            epi = list(n.attrs.get("epilogue", []))
            if step[0] == "binary":
                n.inputs.append(step[2])
                step = ("binary", step[1], len(n.inputs) - 1, step[3])
            epi.append(step)
            n.attrs["epilogue"] = epi
            _replace_uses(g, u.name, n.name)
            g.nodes.remove(u)
            # keep the user's name so outputs/consumers stay readable
            changed = True
            break
    return g


def fuse_elementwise(g: Graph) -> Graph:
    g = copy.deepcopy(g)
    # 1) wrap every elementwise node as a single-op fused_ew
    for n in g.nodes:
        if n.op in ELEMENTWISE:
            ins, args = [], []
            for i in n.inputs:
                if _scalar(g, i):
                    args.append(("lit", float(g.consts[i].reshape(()))))
                else:
                    args.append(("in", len(ins))); ins.append(i)
            n.attrs = {"tree": (n.op, *args)}
            n.op, n.inputs = "fused_ew", ins
    # 2) merge producer into its single consumer
    changed = True
    while changed:
        changed = False
        users = g.users()
        for p in g.nodes:
            if p.op != "fused_ew" or p.name in g.outputs or len(users[p.name]) != 1:
                continue
            c = users[p.name][0]
            if c.op != "fused_ew" or c.shape != p.shape:
                continue
            # new input list: c's inputs minus p, plus p's inputs (dedup)
            new_inputs = [i for i in c.inputs if i != p.name]
            for i in p.inputs:
                if i not in new_inputs:
                    new_inputs.append(i)
            idx = {name: k for k, name in enumerate(new_inputs)}
            p_tree = _remap(p.attrs["tree"], {k: idx[name] for k, name in enumerate(p.inputs)})
            c_map = {k: idx[name] for k, name in enumerate(c.inputs) if name != p.name}
            sub_at = {k for k, name in enumerate(c.inputs) if name == p.name}
            c.attrs["tree"] = _substitute(c.attrs["tree"], c_map, sub_at, p_tree)
            c.inputs = new_inputs
            g.nodes.remove(p)
            changed = True
            break
    return g


def _remap(t, m):
    if t[0] == "in":
        return ("in", m[t[1]])
    if t[0] == "lit":
        return t
    return (t[0], *[_remap(x, m) for x in t[1:]])


def _substitute(t, m, sub_at, p_tree):
    if t[0] == "in":
        return p_tree if t[1] in sub_at else ("in", m[t[1]])
    if t[0] == "lit":
        return t
    return (t[0], *[_substitute(x, m, sub_at, p_tree) for x in t[1:]])


def pretranspose(g: Graph) -> Graph:
    g = copy.deepcopy(g)
    for n in g.nodes:
        if n.op == "linear" and not n.attrs.get("pretransposed"):
            w = n.inputs[1]
            g.consts[w] = np.ascontiguousarray(g.consts[w].T)
            g.node(w).shape = g.consts[w].shape
            n.attrs["pretransposed"] = True
    return g


def lower_required(g: Graph) -> Graph:
    """Minimum lowering needed by codegen (used for the unoptimized -O0 build)."""
    g = fuse_elementwise_single(g)
    return pretranspose(g)


def fuse_elementwise_single(g: Graph) -> Graph:
    """Turn each elementwise op into its own fused_ew kernel without merging (O0)."""
    g = copy.deepcopy(g)
    for n in g.nodes:
        if n.op in ELEMENTWISE:
            ins, args = [], []
            for i in n.inputs:
                if _scalar(g, i) and not (g.is_const(n.inputs[0]) and g.is_const(n.inputs[-1])):
                    args.append(("lit", float(g.consts[i].reshape(()))))
                else:
                    args.append(("in", len(ins))); ins.append(i)
            n.attrs = {"tree": (n.op, *args)}
            n.op, n.inputs = "fused_ew", ins
    return g


PASSES = [fold_batchnorm, constant_fold, dce, fuse_epilogue, fuse_elementwise, pretranspose]


def optimize(g: Graph, passes=None, log: list | None = None) -> Graph:
    for p in (PASSES if passes is None else passes):
        before = g.num_kernels()
        g = p(g)
        if log is not None:
            log.append((p.__name__, before, g.num_kernels()))
    return g


# ---------------------------------------------------------------- memory planning
def plan_memory(g: Graph, reuse: bool = True):
    """Assign each computed tensor an offset in one arena. reshape aliases its input.
    Returns (offsets: name -> (buffer_id or 'out'/'in'), buffers: list of sizes in floats)."""
    alias: dict[str, str] = {}
    for n in g.nodes:
        if n.op == "reshape":
            alias[n.name] = alias.get(n.inputs[0], n.inputs[0])

    def root(x):
        return alias.get(x, x)

    out_root = root(g.outputs[0])
    last_use: dict[str, int] = {}
    for k, n in enumerate(g.nodes):
        for i in n.inputs:
            last_use[root(i)] = k
    buf_of: dict[str, int] = {}
    sizes: list[int] = []
    free: list[int] = []
    for k, n in enumerate(g.nodes):
        if n.op in ("input", "const", "reshape"):
            continue
        if n.name == out_root:
            buf_of[n.name] = -1                      # written straight into the caller's output
        else:
            need = int(np.prod(n.shape))
            cands = [b for b in free if sizes[b] >= need]
            if reuse and cands:
                b = min(cands, key=lambda b: sizes[b]); free.remove(b)
            elif reuse and free:                       # grow the largest free buffer
                b = max(free, key=lambda b: sizes[b]); free.remove(b); sizes[b] = need
            else:
                b = len(sizes); sizes.append(need)
            buf_of[n.name] = b
        if reuse:
            for i in set(root(x) for x in n.inputs):
                if last_use.get(i) == k and i in buf_of and buf_of[i] >= 0:
                    free.append(buf_of[i])
    return buf_of, sizes, root
