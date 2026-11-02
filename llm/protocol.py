"""Frozen shifted-target PPL protocol shared by native and HF evaluation."""
import hashlib
import json
import math
from pathlib import Path

MODEL_ID = 'openai-community/gpt2'
MODEL_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
MODEL_SHA256 = '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
DATA_ID = 'Salesforce/wikitext'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
DATA_SHA256 = '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
TOKENIZER_BLOBS = {
    'config.json': '10c66461e4c109db5a2196bff4bb59be30396ed8',
    'merges.txt': '226b0752cac7789c48f0cb3ec53eda48b7be36cc',
    'tokenizer.json': '4b988bccc9dc5adacd403c00b4704976196548f8',
    'tokenizer_config.json': 'be4d21d94f3b4687e5a54d84bf6ab46ed0f8defd',
    'vocab.json': '1f1d9aaca301414e7f6c9396df506798ff4eb9a6',
}


def verify_tokenizer(directory):
    hashes = {}
    for name, expected in TOKENIZER_BLOBS.items():
        raw = (Path(directory) / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
        if blob != expected:
            raise ValueError(f'local {name} does not match pinned official Git blob')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('ascii')).hexdigest()


def windows(tokens, context=1024, stride=512):
    if type(context) is not int or type(stride) is not int or not 2 <= context <= 1024 or not 1 <= stride < context:
        raise ValueError('require 2<=context<=1024 and 1<=stride<context')
    if len(tokens) < 2 or any(type(t) is not int or not 0 <= t < 50257 for t in tokens):
        raise ValueError('at least two GPT-2 token IDs required')
    previous_end = 1
    result = []
    for begin in range(0, len(tokens) - 1, stride):
        end = min(begin + context, len(tokens))
        mask = [int(i >= previous_end and i > begin) for i in range(begin, end)]
        if any(mask):
            result.append({'begin': begin, 'end': end, 'mask': mask})
        previous_end = end
        if end == len(tokens):
            break
    return result


def validate_frozen(document):
    if (document.get('schema_version') != 1 or document.get('model_revision') != MODEL_REVISION
            or document.get('model_id') != MODEL_ID or document.get('dataset_id') != DATA_ID
            or document.get('dataset_revision') != DATA_REVISION or document.get('dataset_sha256') != DATA_SHA256
            or document.get('dataset_config') != 'wikitext-2-raw-v1' or document.get('split') != 'test'
            or document.get('join') != '\n\n' or document.get('insert_eos') is not False
            or document.get('add_special_tokens') is not False or document.get('preserve_blank_rows') is not True):
        raise ValueError('wrong frozen schema/model revision')
    tokens = document['tokens']
    expected = windows(tokens, document['context'], document['stride'])
    if document['windows'] != expected or document['token_ids_sha256'] != canonical_hash(tokens):
        raise ValueError('frozen token/window hash mismatch')
    count = sum(sum(w['mask']) for w in expected)
    if count != len(tokens) - 1 or type(document['target_count']) is not int or document['target_count'] != count:
        raise ValueError('each shifted target must be scored exactly once')
    # Check global ownership independently of the window constructor.
    ownership = [0] * len(tokens)
    for w in document['windows']:
        for j, use in enumerate(w['mask']):
            if type(use) is not int or use not in (0, 1) or (j == 0 and use):
                raise ValueError('invalid shifted mask')
            ownership[w['begin'] + j] += use
    if ownership != [0] + [1] * (len(tokens) - 1):
        raise ValueError('target omission/duplication')
    return count


def perplexity(nll, count):
    if not math.isfinite(nll) or nll < 0 or type(count) is not int or count <= 0:
        raise ValueError('invalid NLL/count')
    return math.exp(nll / count)
