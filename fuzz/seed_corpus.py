"""Deterministic tiny seeds for LLVM libFuzzer testing of our own code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

TARGETS = ('json','artifact','host_validation','cpu_ops','codec')


def artifact(dtype, payload, shape):
    manifest = {'schema_version':1, 'alignment':256, 'binary':'weights.bin',
        'model_id':'openai-community/gpt2', 'revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e',
        'source_sha256':'248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
        'architecture':{'layers':12,'heads':12,'hidden':768,'vocab':50257,'capacity':1024,
                        'gelu':'gelu_new','layer_norm_eps':1e-5},
        'precision':'int8' if dtype=='int8' else 'fp16', 'binary_bytes':len(payload),
        'binary_sha256':hashlib.sha256(payload).hexdigest(),
        'tensors':[{'name':'small-parser-fixture','dtype':dtype,'shape':shape,'offset':0,
                    'nbytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode('ascii')
    return struct.pack('<I',len(raw))+raw+payload


def seeds():
    result={'json':[b'',b'{}',b'[]',b'{"a":[1,2.5,1e-5],"b":"hello"}',b'18446744073709551615'],
            'artifact':[artifact('float16',struct.pack('<e',1.0),[1]),
                        artifact('float32',struct.pack('<f',1.0),[1]),
                        artifact('int8',b'\x00\x7f\x81',[3])],
            'cpu_ops':[bytes([op,0,0,0])+bytes(range(32)) for op in range(3)],
            'codec':[bytes([0,0])+struct.pack('<e',v) for v in (0.0,-0.0,1.0,-1.0,65504.0,2**-24)]}
    # Eight real-buffer pointer selectors, then GEMM and attention metadata.
    result['host_validation']=[bytes(8)+struct.pack('<'+'i'*12,1,1,1,0,c,1,1,1,1,64,1,0)+struct.pack('<fq',1e-5,1)
                               for c in (0,5,6,11)]
    return result


def write(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[]
    for target,items in seeds().items():
        directory=output/target;directory.mkdir(exist_ok=True)
        for data in items:
            digest=hashlib.sha256(data).hexdigest();path=directory/digest
            if path.exists() and path.read_bytes()!=data:raise ValueError('corpus collision')
            path.write_bytes(data)
            records.append({'target':target,'path':path.relative_to(output).as_posix(),'bytes':len(data),'sha256':digest})
    (output/'MANIFEST.json').write_text(json.dumps({'schema_version':1,'generated_by':'fuzz/seed_corpus.py',
        'purpose':'deterministic seed fixtures, not observed failures or full GPT-2 weights','files':records},indent=2)+'\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();write(a.output)
