"""Freeze local WikiText-2 tokens BEFORE comparison; never fetch remote data."""
import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from .export_weights import sha256_file
from .protocol import (DATA_ID, DATA_REVISION, DATA_SHA256, MODEL_ID, MODEL_REVISION,
                       canonical_hash, validate_frozen, verify_tokenizer, windows)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--tokenizer', required=True, help='local pinned GPT-2 tokenizer directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rows', type=int, default=4358)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.rows < 1 or args.max_tokens < 2:
        raise ValueError('positive row/token slice required')
    parquet = Path(args.parquet).resolve(strict=True)
    if sha256_file(parquet) != DATA_SHA256:
        raise ValueError('local dataset bytes do not match pinned WikiText-2 raw test parquet')
    tokenizer_path = Path(args.tokenizer).resolve(strict=True)
    tokenizer_hashes = verify_tokenizer(tokenizer_path)
    # Resolve/pin these packages only after root releases M2. Imports are local.
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    rows = pq.read_table(parquet, columns=['text']).column('text').to_pylist()
    if len(rows) != 4358 or args.rows > len(rows) or any(not isinstance(x, str) for x in rows):
        raise ValueError('unexpected WT2 raw test schema/count')
    text = '\n\n'.join(rows[:args.rows])
    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)['input_ids'][:args.max_tokens]
    frozen = {'schema_version': 1, 'frozen_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
              'dataset_id': DATA_ID, 'dataset_revision': DATA_REVISION,
              'dataset_config': 'wikitext-2-raw-v1', 'split': 'test', 'dataset_sha256': DATA_SHA256,
              'row_slice': [0, args.rows], 'join': '\n\n', 'preserve_blank_rows': True,
              'insert_eos': False, 'add_special_tokens': False, 'token_slice': [0, len(tokens)],
              'tokenizer_sha256': tokenizer_hashes,
              'versions': {p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers', 'pyarrow')},
              'context': args.context, 'stride': args.stride, 'tokens': tokens,
              'token_ids_sha256': canonical_hash(tokens),
              'windows': windows(tokens, args.context, args.stride), 'target_count': len(tokens) - 1,
              'command': sys.argv, 'wall_seconds': time.perf_counter() - started}
    validate_frozen(frozen)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='ascii') as stream:
        json.dump(frozen, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(output), 'sha256': sha256_file(output), 'target_count': frozen['target_count']}))


if __name__ == '__main__':
    main()
