# CUDA v2 kernel path

These operators add a device-resident FP16 path alongside the existing FP32 backend. The WMMA GEMM uses FP32 accumulation and zero-padded shared tiles. The bounded catalog combines tiles (32,32,16,4warps), (64,32,32,8warps), (64,64,32,8warps) with one/two shared staging buffers and fused/separate bias epilogues: 12 choices. IDs0..2 are two-stage/fused,3..5 one-stage/fused,6..8 two-stage/separate,9..11 one-stage/separate. Copies are synchronous; this is not a cp.async pipeline. The separate epilogue uses a caller-owned FP32 intermediate so both modes add bias before the final output cast. INT8 weights are dequantized to FP16 on load using one FP32 scale per output channel, then use the same FP16 tensor-core GEMM.

Attention shares key/value tiles across four query warps, maintains the running maximum/normalizer/output in FP32, and writes FP16. It accepts D64/128 and separate Q/K lengths plus absolute query offsets for cached decoding. It never materializes Q×K scores. Dot products use SIMT instructions; this implementation does not claim tensor-core attention or the performance of the FlashAttention library.

LayerNorm uses FP32 two-pass population variance; GELU uses GPT-2's tanh approximation. The Python interface is `llm.native_ops.NativeOps`. It validates CUDA tensor metadata and honors the current stream. The standalone C ABI is `backend/include/tinyinfer_backend_v2.h`; callers of that raw API own dtype/layout/allocation extent and storage lifetime. Success means queued, so synchronize explicitly to observe deferred errors. None of these operations provides autograd.

Build through the existing root CMake after enabling additive `TINYC_V2_CUDA` and `TINYC_CUDA`; the new object target avoids changing old compilation flags and is linked into the same backend library. No Torch C++ extension is needed. The CUDA12.4 standalone library / Torch CUDA13.2 stream and allocator interaction requires the actual non-default-stream/lifetime gate; compilation alone does not prove it.

`TINYC_V2_GEMM_CONFIG=-1` compiles the full catalog; an ID0..11 compiles one GEMM candidate and rejects others. `ti_v2_gemm_catalog_json()` identifies the catalog and compiled availability. `bench.v2.kernels.candidate_build` accepts a strict JSON proposal with `config_id,tile,stages,epilogue`, checks the exact allow-list tuple, then configures/builds into unique G-backed scratch using explicit existing CMake/Ninja/nvcc/compiler/Python/headers/googletest inputs. It retains configure/build commands, logs, failures, source stability and a build receipt; it does not launch GPU correctness or accept a speed claim. The autotuner must separately run the full selected candidate correctness gate, then matched timing and independent winner remeasurement. Random/LLM candidate budgets are counted by the autotuner, not this compiler helper.

Benchmark command after the coordinated execution gate:

```sh
python -m bench.v2.kernels.run --config bench/v2/configs/kernels.json \
  --library /absolute/private/build/libtinyinfer_backend.so \
  --build-manifest /absolute/private/build/build-manifest.json \
  --output results/v2/kernels-UNIQUE-ID
```

`--correctness-only` runs edge dimensions, INT8 dequantization, mixed-length/masked/decode attention, norms, invalid inputs and stream/lifetime checks before any timing. Formal groups can be split with `--group gemm|attention|softmax`; config and environment are retained in every unique output. All data is seeded synthetic. Before measurement, capture the completed build using `python -m bench.v2.kernels.build_receipt --library ... --build-directory ... --build-log ... --build-command 'the actual command' --build-exit-code 0 --nvcc /actual/nvcc --cxx /actual/g++ --output /private/build/build-manifest.json`; CMake must export compile commands. The receipt keeps actual compilation records, toolkit/compiler versions and hashes of the exact library/build log. It does not build or certify GPU correctness.

The runner retains individual raw event and synchronized host-call samples, explicit SDPA backend failures and peak additional Torch allocator memory. Its event interval encloses one dispatched Python call, including any GPU idle enqueue gap; it is not a hardware-only instruction latency. Attention's safe wrapper currently creates validated length/offset tensors per call; that overhead is included.

For hardware-only profiling, `profile_v2` is a standalone CUDA target; e.g. `profile_v2 gemm 1024 1 64 20` (CLI configs are 1-based). It loads deterministic synthetic data before the repeated launches. Use Nsight launch filters and keep raw reports outside Git. A real permission or tool failure must remain in the evidence; no profile results are available during code preparation.

## Limitations

No measured speed, correctness or memory result has been established for this candidate. The backend supports contiguous finite inputs, dense GEMM, FP16 activations, one CUDA device per call and inference only. Scales are checked numerically by the quantizer before upload, not by a host-synchronizing kernel wrapper. The old FP32 benchmark comparison uses a distinct explicit-stream adapter with the same numerical algorithm. Triton remains a separately identified implementation in `kernels/triton_v2/`.
