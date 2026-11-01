"""Only measured shape-specific wins select an external hand-written kernel."""
import json
import hashlib
import os
from pathlib import Path

def key(node, backend):
    signature = hashlib.sha256(node.attrs['dispatch_signature'].encode()).hexdigest()[:12]
    return f"{backend}:{node.op}:{','.join(map(str, node.shape))}:{signature}"

def apply_dispatch(graph, backend, implementation):
    if implementation not in ('codegen', 'hand', 'auto'):
        raise ValueError(implementation)
    path = Path(os.environ.get('TINYC_POLICY', Path(__file__).resolve().parents[1] / 'results/dispatch.json'))
    policy = json.loads(path.read_text()) if implementation == 'auto' and path.exists() else {}
    decisions = []
    for node in graph.nodes:
        if node.op in ('linear', 'conv2d', 'layernorm', 'softmax'):
            node.attrs['dispatch_signature'] = json.dumps({'inputs': [graph.node(i).shape for i in node.inputs],
                'epilogue': node.attrs.get('epilogue'), 'eps': node.attrs.get('eps'),
                'k': node.attrs.get('k'), 'stride': node.attrs.get('stride'), 'pad': node.attrs.get('pad')}, sort_keys=True)
            chosen = implementation == 'hand' or (implementation == 'auto' and policy.get(key(node, backend), {}).get('winner') == 'hand')
            node.attrs['hand'] = chosen
            decisions.append({'node': node.name, 'op': node.op, 'key': key(node, backend), 'implementation': 'hand' if chosen else 'codegen'})
    return decisions
