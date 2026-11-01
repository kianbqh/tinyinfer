import shutil
import numpy as np
import pytest
import torch
from torch import nn
from tinyc import compile_model
from tinyc.models import MLP, SmallCNN, randomize_bn
from tinyc.passes import interpret
from tinyc import capture, optimize

@pytest.mark.parametrize('backend', ['cpp', 'cuda'])
@pytest.mark.parametrize('implementation', ['codegen', 'hand'])
@pytest.mark.parametrize('name', ['mlp', 'cnn', 'norm'])
def test_shared_backend_graphs(backend, implementation, name):
    if backend == 'cuda' and (not shutil.which('nvcc') or not torch.cuda.is_available()):
        pytest.skip('CUDA hardware/toolkit required')
    torch.manual_seed(37)
    if name == 'norm':
        model, x = nn.Sequential(nn.LayerNorm(129), nn.Softmax(-1)).eval(), torch.randn(5, 129)
        with torch.no_grad():
            np.testing.assert_allclose(interpret(optimize(capture(model, x)), x.numpy()), model(x).numpy(), atol=1e-5, rtol=1e-5)
    else:
        model = randomize_bn((MLP if name == 'mlp' else SmallCNN)()).eval()
        x = torch.randn(2, 784) if name == 'mlp' else torch.randn(2, 1, 28, 28)
    compiled = compile_model(model, x, backend=backend, implementation=implementation, tag=f'integration_{name}')
    for value in (x, torch.randn_like(x)):
        with torch.no_grad():
            np.testing.assert_allclose(compiled(value), model(value).numpy(), atol=1e-4, rtol=1e-4)
    assert all(d['implementation'] == implementation for d in compiled.info['dispatch'])
