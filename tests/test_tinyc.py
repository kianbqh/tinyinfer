import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from tinyc import PASSES, capture, compile_model, optimize
from tinyc.compile import lower
from tinyc.models import MLP, SmallCNN, randomize_bn
from tinyc.passes import interpret, plan_memory

TOL = 1e-4


def _case(name, bs=4):
    torch.manual_seed(1)
    if name == "mlp":
        m, x = MLP(), torch.randn(bs, 784)
    else:
        m, x = randomize_bn(SmallCNN()), torch.randn(bs, 1, 28, 28)
    m.eval()
    with torch.no_grad():
        ref = m(x).numpy()
    return m, x, ref


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_capture_matches_torch(name):
    m, x, ref = _case(name)
    g = capture(m, x)
    assert np.abs(interpret(g, x.numpy()) - ref).max() < TOL


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_each_pass_preserves_numerics(name):
    m, x, ref = _case(name)
    g = capture(m, x)
    for p in PASSES:
        g = p(g)
        g.validate() if hasattr(g, "validate") else None
        err = np.abs(interpret(g, x.numpy()) - ref).max()
        assert err < TOL, (p.__name__, err)


def test_dce_removes_dead_branch():
    m, x, _ = _case("mlp")
    g = capture(m, x)
    assert "aux" in [n.name for n in g.nodes]
    g2 = optimize(g)
    assert "aux" not in [n.name for n in g2.nodes]


def test_constant_fold_removes_const_subexpr():
    m, x, _ = _case("mlp")
    g = capture(m, x)
    assert g.op_counts().get("sigmoid", 0) == 1           # sigmoid(alpha) only depends on a parameter
    from tinyc.passes import constant_fold
    assert constant_fold(g).op_counts().get("sigmoid", 0) == 0


def test_batchnorm_folded_into_conv():
    m, x, _ = _case("cnn")
    g2 = optimize(capture(m, x))
    assert "batchnorm" not in g2.op_counts()


@pytest.mark.parametrize("name,o0,o2", [("mlp", 10, 3), ("cnn", 16, 7)])
def test_kernel_count_drops(name, o0, o2):
    m, x, _ = _case(name)
    assert lower(m, x, opt=False).num_kernels() == o0
    assert lower(m, x, opt=True).num_kernels() == o2


def test_memory_plan_reuses_buffers():
    m, x, _ = _case("cnn")
    g = lower(m, x, opt=True)
    _, s_noreuse, _ = plan_memory(g, reuse=False)
    _, s_reuse, _ = plan_memory(g, reuse=True)
    assert sum(s_reuse) < sum(s_noreuse)


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_arena_shrinks_O0_to_O2(name):
    m, x, _ = _case(name)
    a0 = compile_model(m, x, opt=False, tag=f"t_{name}").info["arena_bytes"]
    a2 = compile_model(m, x, opt=True, tag=f"t_{name}").info["arena_bytes"]
    assert a2 < a0 / 2


@pytest.mark.parametrize("name", ["mlp", "cnn"])
@pytest.mark.parametrize("opt", [False, True])
@pytest.mark.parametrize("bs", [1, 5])
def test_compiled_cpp_matches_torch(name, opt, bs):
    m, x, ref = _case(name, bs)
    cm = compile_model(m, x, opt=opt, tag=f"t_{name}")
    assert np.abs(cm(x) - ref).max() < TOL
    x2 = torch.randn_like(x)                               # weights baked in, inputs are not
    with torch.no_grad():
        assert np.abs(cm(x2) - m(x2).numpy()).max() < TOL


@pytest.mark.parametrize("drop", ["fuse_epilogue", "fuse_elementwise", "fold_batchnorm", "constant_fold"])
def test_pass_ablation_still_correct(drop):
    m, x, ref = _case("cnn")
    cm = compile_model(m, x, passes=[p for p in PASSES if p.__name__ != drop], tag="abl")
    assert np.abs(cm(x) - ref).max() < TOL


def test_gemm_ragged_shapes():
    torch.manual_seed(3)
    for (i, o) in [(7, 3), (65, 129), (130, 1)]:
        m = nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, 5)).eval()
        x = torch.randn(13, i)
        cm = compile_model(m, x, tag="ragged")
        with torch.no_grad():
            assert np.abs(cm(x) - m(x).numpy()).max() < TOL


@pytest.mark.parametrize("backend", ["cpp", "cuda"])
def test_compiled_instances_keep_their_own_weights(backend):
    if backend == "cuda" and (shutil.which("nvcc") is None or not torch.cuda.is_available()):
        pytest.skip("CUDA execution requires nvcc and a GPU")
    torch.manual_seed(17)
    a = nn.Sequential(nn.Linear(7, 5), nn.ReLU()).eval()
    b = nn.Sequential(nn.Linear(7, 5), nn.ReLU()).eval()
    x = torch.randn(3, 7)
    ca = compile_model(a, x, backend=backend)
    with torch.no_grad():
        np.testing.assert_allclose(ca(x), a(x).numpy(), rtol=0, atol=TOL)
    cb = compile_model(b, x, backend=backend)
    for inp in (x, torch.randn_like(x)):
        with torch.no_grad():
            np.testing.assert_allclose(ca(inp), a(inp).numpy(), rtol=0, atol=TOL)
            np.testing.assert_allclose(cb(inp), b(inp).numpy(), rtol=0, atol=TOL)
    del cb
    with torch.no_grad():
        np.testing.assert_allclose(ca(x), a(x).numpy(), rtol=0, atol=TOL)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not on PATH")
@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_cuda_codegen_compiles(name):
    m, x, _ = _case(name)
    obj, info = compile_model(m, x, opt=True, tag=f"t_{name}", backend="cuda", compile_only=True)
    assert obj.exists() and obj.stat().st_size > 0
    assert "tc::gemm" in info["source"] and "struct Epi_" in info["source"]
