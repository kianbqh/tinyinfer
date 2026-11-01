"""pytest: pybind11 module output must match PyTorch reference ops."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.environ.get("IK_BUILD_DIR", os.path.join(os.path.dirname(__file__), "..", "build")))
ik = pytest.importorskip("ikernels")


@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 33, 65), (128, 256, 300), (257, 129, 64)])
@pytest.mark.parametrize("blocked", [False, True])
def test_matmul_matches_torch(shape, blocked):
    m, n, k = shape
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    ref = (torch.from_numpy(a).double() @ torch.from_numpy(b).double()).float().numpy()
    np.testing.assert_allclose(ik.matmul(a, b, blocked=blocked), ref, atol=1e-4 * k, rtol=1e-4)


@pytest.mark.parametrize("cols", [1, 100, 1024, 5000])
def test_softmax_matches_torch(cols):
    x = np.random.default_rng(1).standard_normal((33, cols), dtype=np.float32) * 6
    ref = torch.softmax(torch.from_numpy(x), -1).numpy()
    for fused in (False, True):
        np.testing.assert_allclose(ik.softmax(x, fused=fused), ref, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("cols", [8, 768, 4096])
def test_layernorm_matches_torch(cols):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((21, cols), dtype=np.float32) * 3 + 1
    g = rng.uniform(0.5, 1.5, cols).astype(np.float32)
    b = rng.standard_normal(cols, dtype=np.float32)
    ref = torch.nn.functional.layer_norm(torch.from_numpy(x), (cols,), torch.from_numpy(g), torch.from_numpy(b)).numpy()
    for fused in (False, True):
        np.testing.assert_allclose(ik.layernorm(x, g, b, fused=fused), ref, atol=1e-4, rtol=1e-4)


def test_shape_errors():
    with pytest.raises(ValueError):
        ik.matmul(np.ones((2, 3), np.float32), np.ones((4, 2), np.float32))
    with pytest.raises(ValueError):
        ik.softmax(np.ones(5, np.float32))
