# Native GPT-2 path

I use this path to run GPT-2 124M through the standalone CUDA backend, then compare it with the same local weights in Transformers. Both the Python wrapper and the C++ serving worker call `tinyinfer_llm`; model execution has no Python subprocess or libtorch dependency. The existing FP32 compiler path is separate.

This source is a preparation candidate. **CUDA compilation, model correctness, perplexity and performance have not run yet.** No table here claims a new measured GPU result.

## Model and data

The model is [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, MIT. The exporter accepts only the pinned official `model.safetensors` SHA256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. It reads the local file directly, preserving GPT-2 Conv1D `[input,output]` weights, learned positions, tied token embedding/head, pre-LayerNorm residual ordering, epsilon `1e-5` and `gelu_new`. Dropout is disabled for inference. The context limit is1024.

The PPL source is the test split of [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The local test parquet must have SHA256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`. Upstream API tags name CC-BY-SA3.0/GFDL while card prose names CC-BY-SA4.0; I preserve that discrepancy. Raw text, weights and caches stay outside Git.

`freeze_tokens` joins the selected test rows with exactly two newlines, preserves blank rows, inserts no EOS/special token, and freezes a token-ID slice before either runtime is compared. It verifies the official tokenizer/config Git blobs and records their local SHA256 values. Overlapping context/stride windows assign every shifted target exactly once; the first global token is excluded. The same token IDs, masks, window positions and target count feed both runtimes.

## Setup and commands

Use the M2 private WSL2 environment and G-backed paths. CUDA12.4 builds this standalone C ABI; Torch's installed CUDA major need not match the compiler at build time because no Torch header/library is linked. Actual cross-runtime context/stream correctness must pass the separate kernel GPU gate. Pin and record the chosen Transformers/tokenizers/safetensors/pyarrow versions in the private environment; no script here downloads or installs packages implicitly.

Root CMake must add `tinyinfer_llm` after the additive v2 kernel target. Configure with `TINYC_CUDA=ON`, `TINYC_V2_CUDA=ON`, `TINYC_LLM=ON`, actual sm86 flags and `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. Keep the complete build log and record it with `python -m llm.build_receipt --help`. The receipt verifies the actual model compilation record and standalone linkage, and binds the library/backend/source hashes.

After the local official model/tokenizer files and test parquet are present:

```bash
M2_ROOT=/opt/codex-resume/all-modules-v2/m2
python -m llm.export_weights --safetensors "$M2_ROOT/hf-gpt2/model.safetensors" --precision fp16 --output "$M2_ROOT/weights/gpt2-fp16"
python -m llm.export_weights --safetensors "$M2_ROOT/hf-gpt2/model.safetensors" --precision int8 --output "$M2_ROOT/weights/gpt2-int8"
python -m llm.freeze_tokens --parquet "$M2_ROOT/data/wikitext-2-raw-test.parquet" --tokenizer "$M2_ROOT/hf-gpt2" --output "$M2_ROOT/data/frozen-wt2-16384.json" --rows 4358 --max-tokens 16384 --context 1024 --stride 512
python -B -m unittest discover -s llm/tests -p 'test_*.py' -v
```

Artifact output directories and the frozen token file must not already exist. Native output is `manifest.json` plus256-byte-aligned `weights.bin`; the loader verifies the full binary, every tensor hash, exact shapes/dtypes, extents and non-overlap. The head is a physically transposed copy of the FP16 embedding values for row-major GEMM. The model/data bytes and exported artifacts remain outside the repository.

First run actual backend correctness using `bench.v2.kernels.run --correctness-only`. Then run model gates; both commands require a unique new output path. For example, after assigning actual completed receipt/gate paths:

```bash
python -m llm.gpu_correctness --library "$M2_ROOT/build/libtinyinfer_llm.so" --build-receipt "$M2_ROOT/build/model-build-receipt.json" --fp16-manifest "$M2_ROOT/weights/gpt2-fp16/manifest.json" --int8-manifest "$M2_ROOT/weights/gpt2-int8/manifest.json" --hf-directory "$M2_ROOT/hf-gpt2" --config llm/configs/evaluation.json --frozen-tokens "$M2_ROOT/data/frozen-wt2-16384.json" --kernel-gate results/v2/ACTUAL-KERNEL-GATE-ID/correctness.json --output results/v2/NEW-MODEL-GATE-ID
python -m llm.evaluate --library "$M2_ROOT/build/libtinyinfer_llm.so" --build-receipt "$M2_ROOT/build/model-build-receipt.json" --fp16-manifest "$M2_ROOT/weights/gpt2-fp16/manifest.json" --int8-manifest "$M2_ROOT/weights/gpt2-int8/manifest.json" --hf-directory "$M2_ROOT/hf-gpt2" --config llm/configs/evaluation.json --frozen-tokens "$M2_ROOT/data/frozen-wt2-16384.json" --kernel-gate results/v2/ACTUAL-KERNEL-GATE-ID/correctness.json --model-gate results/v2/ACTUAL-MODEL-GATE-ID/results.json --output results/v2/NEW-LLM-RUN-ID
```

Replace the uppercase run IDs with the actual gate IDs and fresh output IDs. Formal measurements require clean tracked source, matching completed build/gate receipts, matching artifact/config/token hashes and unchanged source/commit at completion. Startup and failure files remain available if any step fails. Never use the old benchmark/report scripts for this path.

## Native API and cache

`include/tinyinfer_llm.h` defines the C++ worker's host API. `runtime.NativeModel` is a thin ctypes wrapper of that exact API and imports no Torch. A context owns its CUDA device/stream, weights, scratch and isolated contiguous K/V slots. One worker serializes calls; different contexts share no mutable storage. Callers retain host buffers until the synchronous call returns. Destroy synchronizes and releases resources.

`forward_batch` appends variable-length CPU token rows to distinct slot IDs, optionally returns last-position FP32 logits, and/or returns greedy IDs selected on GPU. The cache keeps `[slot,layer,head,position,64]` K and V. Host-owned valid lengths set absolute learned positions and causal query offsets. Each layer packs selected slots into the tightly packed attention layout. `reset` changes only one valid length; stale bytes cannot become visible through a valid request. Validation errors enqueue no work and preserve lengths. A CUDA/execution failure poisons the context; destroy/recreate it.

`greedy_batch` requires empty slots, accepts individual prompt/output lengths, and generates the requested fixed count. It does not stop on EOS. The last returned token has not yet been appended to cache. Cancellation can be observed between synchronous worker calls; the server may use `forward_batch` to advance one decode step at a time. This is a contiguous cache; paged KV is not implemented.

`score_window` uses an additional internal slot without modifying public slots. It scores `token[t]` from hidden position`t-1`, applies the exact frozen mask and computes stable NLL on GPU in at most16 vocabulary rows at a time. Only position losses/scalars move to host; no sequence×vocabulary host allocation is needed.

## Quantization and measurement

All activations, embeddings, biases and norms are FP16. Linear projections and the head support FP16 or symmetric INT8 weights `[input,output]`, with FP32 scales per output channel, ties-to-even rounding and clipping to[-127,127]. An all-zero channel uses q0/scale1. Quantization starts from the exact FP16 reference values. The kernel converts `q*scale` through FP32 to FP16 before WMMA; this is **INT8 weight-only**, with FP32 accumulation, not INT8 tensor-core compute. The FP16 embedding table remains available in both modes.

The config fixes batch1/8/32, seed, prompt/output lengths, warmups, repeats and tolerances. Raw warmups and repeated batch samples are retained. TTFT covers prepacked CPU tokens through H2D, runtime dispatch, synchronization and the first greedy ID copied to host. Total time ends at the last copied ID; queueing, socket transfer, load, tokenization and cache reset are excluded. Separate host-invocation samples also include Python packing/output conversion. Decode rate is `B*(new_tokens-1)/(last completion-first completion)`; throughput including prefill is reported separately. Percentiles use linear interpolation across batch invocations; std is population std.

HF is explicit FP16/eager attention with TF32 and reduced-precision FP16 GEMM reduction disabled. Native head logits have FP32 accumulation/output; HF head logits are FP16 before FP32 scoring. A `torch.compile`/Inductor attempt is separate: first-shape/cold invocation cost and per-window wall times remain visible, errors are retained, and graph-break counters are reported. Compile failure does not become an eager success. Actual native operation counters identify GEMM precision, normalization/GELU/attention, cache stages and head/NLL dispatch; `kv_scatter_gather` counts one scatter+gather pair per layer, not a hardware kernel count.

## Limitations

Single-GPU GPT-2-scale inference,1024 positions, serialized context and fixed-count greedy generation. The cache is contiguous and packing selected slots adds work. The custom tiled kernels may lose to library kernels. PPL/latency/storage changes, including INT8 slowdowns or quality loss, require the actual result files. No paged KV, speculative decoding, multi-GPU, distributed serving or production workload claim is made. CPU reference checks do not establish native compilation, GPU correctness or performance.

Primary interfaces: [GPT-2 architecture/cache](https://huggingface.co/docs/transformers/model_doc/gpt2), [Safetensors metadata format](https://huggingface.co/docs/safetensors/en/metadata_parsing), [Transformers attention selection](https://github.com/huggingface/transformers/blob/main/docs/source/en/attention_interface.md), [compile/caching considerations](https://huggingface.co/docs/transformers/main/en/llm_optims).
