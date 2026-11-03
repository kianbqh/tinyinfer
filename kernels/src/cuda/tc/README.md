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

For hardware-only profiling, `profile_v2` is a standalone CUDA target; e.g. `profile_v2 gemm 1024 1 64 20` (CLI configs are 1-based). It loads deterministic synthetic data before the repeated launches. Use Nsight launch filters and keep raw reports outside Git. A real permission or tool failure must remain in the evidence; the measured capture protocol and results are in the [profiling README](../../../../bench/v2/profiling/README.md#measured-profiles).

## Measured results

Measured in WSL2 on an i7-12700KF and RTX 3080 12 GB, with PyTorch 2.15.0+cu132, Triton 3.9.0 and a standalone CUDA 12.4.131 build at source `3edcfeb`. Displayed decimals are rounded; linked JSON retains full precision. [The actual kernel gate](../../../../results/v2/20261102T132951Z-m2-native-headers-kernel-gate/correctness.json) passed all 115 cases, including the non-default-stream/lifetime checks. Completed performance groups contain 56 GEMM, 12 softmax and 72 attention cases; all numerical checks passed the frozen `atol=rtol=0.02`.

Seed `20261101`; concurrency 1; 10 warmups, 50 CUDA-event samples and 20 separately synchronized host samples per case. Events surround one device-resident Python wrapper call and include enqueue gaps. Host samples include wrapper dispatch, allocation, stream handling and synchronization. Attention also uploads validated lengths/offsets on each call. These are separate timing windows, not hardware-only kernel latency.

### GEMM

These tables hold c0 fixed: FP16 A/B, FP32 accumulation and FP16 output for WMMA/Triton. The INT8 path converts per-channel weight values to FP16 before WMMA. The explicit-stream legacy adapter takes FP32 inputs/outputs. TFLOP/s uses `2*M*N*K / event_p50`; the percentage compares that rate with the FP16 cuBLAS row.

| Size | Path | Event p50 (ms) | Host p50 (ms) | TFLOP/s | % of cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 512³ | cuBLAS via torch.matmul | 0.053248 | 0.095328 | 5.0412 | 100.0000 |
| 512³ | Legacy FP32 adapter | 1.350656 | 0.677374 | 0.1987 | 3.9424 |
| 512³ | WMMA FP16, c0 | 0.044032 | 0.121206 | 6.0964 | 120.9302 |
| 512³ | Triton FP16, c0 | 0.062464 | 0.124016 | 4.2974 | 85.2459 |
| 512³ | INT8 weights, c0 | 0.048128 | 0.096732 | 5.5775 | 110.6383 |
| 1024³ | cuBLAS via torch.matmul | 0.044032 | 0.076325 | 48.7710 | 100.0000 |
| 1024³ | Legacy FP32 adapter | 0.250880 | 0.314568 | 8.5598 | 17.5510 |
| 1024³ | WMMA FP16, c0 | 0.230400 | 0.338682 | 9.3207 | 19.1111 |
| 1024³ | Triton FP16, c0 | 0.063488 | 0.118137 | 33.8250 | 69.3548 |
| 1024³ | INT8 weights, c0 | 0.260096 | 0.319873 | 8.2565 | 16.9291 |
| 2048³ | cuBLAS via torch.matmul | 0.305152 | 0.339769 | 56.2994 | 100.0000 |
| 2048³ | Legacy FP32 adapter | 1.706496 | 1.812325 | 10.0673 | 17.8818 |
| 2048³ | WMMA FP16, c0 | 1.602560 | 1.742960 | 10.7203 | 19.0415 |
| 2048³ | Triton FP16, c0 | 0.420864 | 0.520067 | 40.8205 | 72.5061 |
| 2048³ | INT8 weights, c0 | 1.677824 | 1.946356 | 10.2394 | 18.1874 |
| 4096³ | cuBLAS via torch.matmul | 2.171392 | 2.251240 | 63.2953 | 100.0000 |
| 4096³ | Legacy FP32 adapter | 14.621696 | 14.696854 | 9.3997 | 14.8505 |
| 4096³ | WMMA FP16, c0 | 13.902848 | 14.116478 | 9.8857 | 15.6183 |
| 4096³ | Triton FP16, c0 | 5.444096 | 5.593244 | 25.2455 | 39.8853 |
| 4096³ | INT8 weights, c0 | 15.497216 | 15.926268 | 8.8686 | 14.0115 |

[All 56 cases, c0/c1/c2, FP32-output variants and raw samples](../../../../results/v2/20261102T132951Z-m2-native-headers-gemm-cpath/run.json) retain the complete comparison. At 4096³, c0 WMMA reaches 15.6183% of cuBLAS throughput; the weight-only INT8 path is slower than FP16. The small 512³ event sample favors c0 WMMA, while its synchronized host latency is higher than cuBLAS; the two timing layers should be read separately.

The [first GEMM attempt](../../../../results/v2/20261102T132951Z-m2-native-headers-gemm/run.json) is preserved as `partial_failure`: 44 cases completed and 12 Triton cases failed to find `Python.h`. The completed retry supplied only existing Python header paths through child-process `CPATH`; source, tolerances and the native library were unchanged.

### Attention

The fixed endpoint slice below is causal self-attention with B=1, H=4 and Q=K=4096. SDPA backends were explicitly forced, rather than inferred from automatic selection. Both custom configurations are shown; the [complete sweep](../../../../results/v2/20261102T132951Z-m2-native-headers-attention-cpath/run.json) covers sequence lengths 128, 256, 512, 1024, 2048 and 4096 and D64/128.

| Head dim | Path | Event p50 (ms) | Host p50 (ms) | Peak additional Torch allocator bytes |
| --- | --- | --- | --- | --- |
| 64 | SDPA math | 5.413888 | 5.503594 | 687882752 |
| 64 | SDPA flash | 0.276480 | 0.363272 | 2163712 |
| 64 | SDPA efficient | 0.326656 | 0.376000 | 2098176 |
| 64 | Naive materialized | 1.816624 | 1.981749 | 279052288 |
| 64 | Custom online, c0 | 8.594944 | 8.740782 | 2098688 |
| 64 | Custom online, c1 | 9.831936 | 9.877690 | 2098688 |
| 128 | SDPA math | 5.895680 | 6.028267 | 704659968 |
| 128 | SDPA flash | 0.424448 | 0.509700 | 4260864 |
| 128 | SDPA efficient | 0.545792 | 0.632924 | 4195328 |
| 128 | Naive materialized | 2.040832 | 2.217450 | 281149440 |
| 128 | Custom online, c0 | 13.239296 | 13.328756 | 4195840 |
| 128 | Custom online, c1 | 17.053696 | 17.183085 | 4195840 |

Custom attention is slower than forced-flash SDPA in every tested case. It keeps additional allocator use below the naive materialized path in the endpoint rows. Memory is the peak **additional Torch allocator bytes after inputs exist**; it excludes the CUDA context, code, shared/register storage and non-Torch allocations. This is not total device memory. The dot products remain SIMT, and the synchronous staging buffers do not implement asynchronous copy/compute overlap.

## Limitations

The backend supports contiguous finite inputs, dense GEMM, FP16 activations, one CUDA device per call and inference only. Scales are checked numerically by the quantizer before upload, not by a host-synchronizing kernel wrapper. The old FP32 benchmark comparison uses a distinct explicit-stream adapter with the same numerical algorithm. Triton remains a separately identified implementation in `kernels/triton_v2/`.
