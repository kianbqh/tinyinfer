import json
import shutil
import numpy as np
import pytest
import torch
from torch import nn
from tinyc import compile_model
from tinyc.compile import lower
from tinyc.dispatch import apply_dispatch

@pytest.mark.parametrize('backend', ['cpp', 'cuda'])
def test_frozen_shape_policy_selects_real_calls(tmp_path, monkeypatch, backend):
    if backend == 'cuda' and (not shutil.which('nvcc') or not torch.cuda.is_available()):
        pytest.skip('CUDA hardware/toolkit required')
    model = nn.Sequential(nn.LayerNorm(33), nn.Softmax(-1)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(.2, 1.7, 33)); model[0].bias.copy_(torch.linspace(-.3,.3,33))
    x = torch.linspace(-2,2,99).reshape(3,33)
    decisions = apply_dispatch(lower(model,x),backend,'codegen')
    policy = {d['key']:{'winner':'hand' if d['op']=='layernorm' else 'codegen'} for d in decisions}
    path = tmp_path/'dispatch.json'; path.write_text(json.dumps(policy))
    monkeypatch.setenv('TINYC_POLICY',str(path))
    cm = compile_model(model,x,backend=backend,implementation='auto',tag='frozen')
    assert [d['implementation'] for d in cm.info['dispatch']] == ['hand','codegen']
    assert f'ti_{"cuda" if backend=="cuda" else "cpu"}_layernorm(' in cm.info['source']
    path.write_text('{}')  # compiled choices are frozen, not read during inference
    with torch.no_grad():
        np.testing.assert_allclose(cm(x),model(x).numpy(),atol=1e-4,rtol=1e-4)
