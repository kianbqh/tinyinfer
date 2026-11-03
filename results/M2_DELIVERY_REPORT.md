# M2 delivery report — native kernels, GPT-2 and serving

**Experiments complete.** I extended the standalone backend with native GPT-2 and serving, then measured the kernels, model quality, generation, batching and hardware counters. Those experiments and the predefined three-seed search are complete. The LLM did not fill the matched evaluation budget, so the search has no matched aggregate. The closed results passed independent raw-sample and native/proposer provenance checks.

## 1. Links and source checkpoints

| Item | Recorded state |
| --- | --- |
| Repository | [kianbqh/tinyinfer](https://github.com/kianbqh/tinyinfer) |
| Published core-result commit | [961eee603a27f4af3faf827eeaad84b091295fd2](https://github.com/kianbqh/tinyinfer/commit/961eee603a27f4af3faf827eeaad84b091295fd2) |
| Published core-result CI | [41540160425](https://github.com/kianbqh/tinyinfer/actions/runs/41540160425): all five CPU jobs successful; conditional manual CUDA job skipped |
| Current source checkpoint | [6b1b9f665052e05715b15427f0a9a4777d0414fb](https://github.com/kianbqh/tinyinfer/commit/6b1b9f665052e05715b15427f0a9a4777d0414fb); three-seed search finished as incomplete_incomparable |
| Current source CI | [41548410550](https://github.com/kianbqh/tinyinfer/actions/runs/41548410550): all five CPU jobs successful; conditional manual CUDA job skipped |
| Native build/kernel/model/server measurement source | `3edcfebd24d5efbf5da81cda13f2bf684031b35e` |
| Nsight capture source | `0faa5148d1694a00c201846ac96fd2b98309f0da` |
| Readable-label profile renderer | `b7fbeb201ab66a36d46dcbd5bdd1b68977b64229` |
| Historical host robustness source | `793b62ecf694e6eae6c7f590bf886cd139fb65c8` |
| PR | [#1](https://github.com/kianbqh/tinyinfer/pull/1) |
| Final module commit / CI / merge | [Publication history](https://github.com/kianbqh/tinyinfer/commits/main/results/M2_DELIVERY_REPORT.md) · [PR checks and merge record](https://github.com/kianbqh/tinyinfer/pull/1/checks) |
| Release assets | No new model-weight or full Nsight-report release was published for this closed block. Large model binaries and full `.ncu-rep` files remain local; committed JSON/CSV/log/PNG evidence identifies them by hash. Final release decision: the pinned export command reproduces the GPT-2 weight files; no duplicate weight release is needed |

The [result index](v2/README.md), [closed command receipts](v2/evidence/20261102-local/) and [independent checks](v2/evidence/20261102-local/acceptance/) preserve commands, source/config/input hashes and failed attempts. Build, capture and renderer records identify their own checkpoints. The PR checks track the final documentation/result publication separately from each measurement source.

## 2. Environment, inputs and implementation

All new runtime measurements here ran locally in WSL2 on an i7-12700KF and NVIDIA GeForce RTX 3080 12 GB. The closed LLM environment records driver `621.01`, PyTorch `2.15.0+cu132`, Transformers `5.22.1`, tokenizers `0.23.3` and safetensors `0.8.0`; the kernel environment records Triton `3.9.0` and NumPy `2.5.4`. The standalone native build used CUDA `12.4.131`, GCC `13.3` and SM86 without libtorch/Torch headers. Nsight Compute was `2024.1`. [Kernel environment/build](v2/20261102T132951Z-m2-native-headers-gemm-cpath/run.json) · [LLM environment and precision settings](v2/20261102T132951Z-m2-native-headers-llm-cpath/results.json) · [profile provenance](v2/20261102T221317Z-m2-profile-enabled-profile-labels/results.json).

Kernel benchmarks use seeded synthetic tensors. Serving uses seeded synthetic token IDs and Poisson arrivals, with real native GPT-2 execution. The language-model evaluation uses the real pretrained [GPT-2 124M](https://huggingface.co/openai-community/gpt2), revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (MIT), and a frozen test slice of [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The dataset API license tags and card prose differ; the [model/data README](../llm/README.md#model-and-data) records that discrepancy. Raw text, weights and caches are outside Git. No model training was performed for this block.

The WMMA catalog combines three tile choices, one/two synchronous shared staging buffers and fused/separate bias epilogues. Copies are synchronous. INT8 is symmetric per-output-channel **weight-only storage**, dequantized through FP32 to FP16 for FP32-accumulating WMMA. The attention kernel uses SIMT dot products, online softmax and a causal mask without materializing the score matrix. The native decoder uses GEMM, attention, LayerNorm and GELU through the same C ABI, with per-request contiguous KV slots. [Kernel implementation/protocol](../kernels/src/cuda/tc/README.md) · [decoder API/cache](../llm/README.md#native-api-and-cache).

| Requested M2 item | Closed result or remaining scope |
| --- | --- |
| FP16 WMMA GEMM vs cuBLAS/legacy FP32 | Completed, fixed sizes 512/1024/2048/4096; full c0/c1/c2 records retained |
| Triton GEMM and fused softmax | Completed; original missing-header failure retained before the child-CPATH retry |
| Online causal attention D64/128 | Completed across lengths 128–4096; forced SDPA math/flash/efficient and naive comparisons retained |
| Real GPT-2, perplexity and contiguous KV cache | Completed model gate, fixed-slice PPL and batch 1/8/32 generation |
| FP16 / INT8 weight-only quality, storage and speed | Completed; quality loss and slowdown reported |
| Paged KV / speculative decoding | Not implemented; optional stretch work |
| Nsight occupancy, throughput and roofline | Completed three standalone profile points with separate counting models |
| epoll server, MPSC queue and batching | Implemented; CPU/TSAN CI and closed native off/on socket/load evidence |
| Test-gated local-LLM vs random autotuner | Predefined three seeds finished; LLM exhausted proposal caps before matching eight unique evaluations, so aggregate remains empty |
| Own-code libFuzzer + ASan/UBSan PR job | Completed finite historical CI run; zero confirmed/fixed random-input issues |

## 3. Measured results

Displayed decimals are rounded; the linked JSON retains full precision. The earlier compiler and FP32 results remain their original experiment and are not regenerated here.

### 3.1 GEMM and Triton

The tables hold configuration c0 fixed rather than selecting the fastest configuration after observation. GEMM/Triton inputs are FP16 with FP32 accumulation and FP16 output; the legacy explicit-stream adapter has FP32 inputs/outputs. TFLOP/s uses the algorithmic `2*M*N*K` numerator divided by event p50. The cuBLAS percentage is the ratio of that throughput to the FP16 `torch.matmul` row.

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

[All 56 GEMM cases and raw samples](v2/20261102T132951Z-m2-native-headers-gemm-cpath/run.json) also retain c1/c2 and FP32-output variants. At 4096³, c0 WMMA is 15.6183% of cuBLAS throughput. At 512³ it has a lower event p50 but a higher synchronized host p50 than cuBLAS; those are separate timing layers.

Protocol: seed `20261101`, concurrency 1, ten warmups, 50 CUDA-event samples and 20 separately synchronized host samples per case. Events enclose one device-resident Python wrapper call and include enqueue gaps. Host samples include wrapper dispatch, output/scratch allocation, stream handling and synchronization. Neither timer is a hardware-only instruction latency. JIT warmup is outside the steady-state samples. Fixed tolerances are `atol=rtol=0.02`.

The [first GEMM attempt](v2/20261102T132951Z-m2-native-headers-gemm/run.json) is a preserved partial failure: 44 cases completed and 12 Triton cases failed to find `Python.h`. Only existing Python header directories in child-process `CPATH` changed for the successful fresh-output retry; source, native library and tolerances did not change.

### 3.2 Softmax

| Shape | Softmax path | Event p50 (ms) | Host p50 (ms) |
| --- | --- | --- | --- |
| 128×128 | PyTorch | 0.013360 | 0.029576 |
| 128×128 | Custom CUDA | 0.069632 | 0.071710 |
| 128×128 | Triton | 0.072656 | 0.048528 |
| 128×512 | PyTorch | 0.020480 | 0.030346 |
| 128×512 | Custom CUDA | 0.028672 | 0.043818 |
| 128×512 | Triton | 0.040960 | 0.047194 |
| 128×2048 | PyTorch | 0.017408 | 0.038586 |
| 128×2048 | Custom CUDA | 0.028512 | 0.045108 |
| 128×2048 | Triton | 0.031744 | 0.050560 |
| 128×4096 | PyTorch | 0.016224 | 0.035743 |
| 128×4096 | Custom CUDA | 0.029696 | 0.045192 |
| 128×4096 | Triton | 0.037888 | 0.048824 |

[All twelve softmax cases](v2/20261102T132951Z-m2-native-headers-softmax-cpath/run.json) use the same kernel benchmark timing protocol. Both custom CUDA and Triton lose to PyTorch at every tested shape in both event and host p50.

### 3.3 Attention and memory

The endpoint table is causal self-attention with B=1, H=4, Q=K=4096. SDPA backends were explicitly forced; both custom configurations are shown.

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

[The complete 72-case sweep](v2/20261102T132951Z-m2-native-headers-attention-cpath/run.json) covers lengths 128, 256, 512, 1024, 2048 and 4096 with head dimensions 64/128. Custom attention loses to forced-flash SDPA in every measured case. The endpoint rows use less additional allocator memory than naive materialized attention.

The wrapper uploads validated length/offset metadata per custom call; that work remains inside the measured window. Memory is **peak additional Torch allocator bytes after inputs exist**, excluding CUDA context/code, shared/register storage and non-Torch allocations. It is not total device memory. Attention remains SIMT and the shared staging copies remain synchronous.

### 3.4 GPT-2 quality, contiguous KV and quantization

The slice was frozen before comparison: 16,384 token IDs, 16,383 uniquely scored shifted targets, context 1024 and stride 512. Every runtime receives the same token IDs, window positions and target mask. This is fixed-slice perplexity, not full-corpus perplexity. Native head logits use FP32 accumulation/output; HF produces FP16 logits before FP32 scoring.

| Runtime | WikiText-2 slice perplexity | Scored targets |
| --- | --- | --- |
| HF eager | 26.01968506 | 16383 |
| HF compile / Inductor | 26.01571659 | 16383 |
| Native FP16 | 26.01567248 | 16383 |
| Native INT8 weights | 26.58238995 | 16383 |

[Four-method PPL and raw generation records](v2/20261102T132951Z-m2-native-headers-llm-cpath/results.json). Native FP16 is within the declared 2% relative PPL tolerance against HF eager; the arithmetic is not identical. INT8 increases PPL by 0.56671747 over native FP16. Exported binary storage is 326,074,368 bytes for FP16 and 203,075,140 bytes for INT8, including the physical transposed head and unchanged FP16 embedding/norm/bias tensors. Those bytes are storage, not GPU peak memory.

The [model gate](v2/20261102T132951Z-m2-native-headers-model-gate/results.json) exercises batch versus independent requests, cached versus full-prefix logits, reset/isolation, masked NLL and mixed-length greedy generation. Cache slots are contiguous; packing selected slots adds work. Generation does not stop on EOS and returns the requested fixed count.

### 3.5 TTFT and decode rate

| Batch | Runtime | TTFT p50 (ms) | Total p50 (ms) | Decode tokens/s p50 | Host invocation p50 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | HF eager | 7.9992 | 204.3914 | 157.9543 | 204.4858 |
| 1 | HF compile / Inductor | 2.7999 | 79.3666 | 403.9396 | 79.4295 |
| 1 | Native FP16 | 5.9634 | 169.4185 | 189.6474 | 169.4987 |
| 1 | Native INT8 weights | 7.5972 | 236.2165 | 135.6140 | 236.2916 |
| 8 | HF eager | 12.6888 | 228.4519 | 1147.8863 | 228.5512 |
| 8 | HF compile / Inductor | 8.5432 | 101.4574 | 2665.8449 | 101.5614 |
| 8 | Native FP16 | 21.5241 | 196.2503 | 1419.7580 | 196.4578 |
| 8 | Native INT8 weights | 24.4484 | 266.4578 | 1024.4578 | 266.6613 |
| 32 | HF eager | 34.5194 | 252.5337 | 4544.5842 | 252.7660 |
| 32 | HF compile / Inductor | 25.4504 | 129.7190 | 9505.0475 | 129.9503 |
| 32 | Native FP16 | 81.8142 | 280.1815 | 4999.0442 | 280.7953 |
| 32 | Native INT8 weights | 89.0197 | 352.3949 | 3767.5408 | 353.0147 |

Seed 42; 128-token prompts; 32 new tokens; concurrency 1; three warmups and ten recorded repeats per batch/runtime. Rows are medians across batch invocations. Aggregate decode rate is `B*31/(last-token completion - first-token completion)` in seconds. TTFT/total cover prepacked CPU tokens through H2D, dispatch, synchronization and copied greedy IDs. Host invocation adds Python packing/output conversion. Loading, tokenization, cache reset, sockets and queueing are excluded.

Native FP16 has lower total p50 than HF eager at batches 1/8, but higher total p50 at batch 32. Its batch-8/32 TTFT is higher than HF eager. HF compile/Inductor has lower total latency and higher decode rate than native FP16 at all three batches. Native INT8 is slower than native FP16 throughout.

The compile attempt completed with `dynamic=True` and `fullgraph=False`. Cold costs are separate from the table: constructing the compiled wrapper took 3658.3594 ms; the first recorded batch-1 and batch-8 cold host invocations took 29972.4356 and 46281.7399 ms. PPL window wall times and Dynamo/Inductor counters remain in JSON; they are not steady-state generation samples.

### 3.6 Nsight profiles

| Profile point | Count model | Occupancy (%) | SM throughput (%) | Memory throughput (%) | Counted roofline (%) |
| --- | --- | --- | --- | --- | --- |
| FP16 GEMM | SM86 Tensor x512 | 59.95 | 68.40 | 45.09 | 3.171549 |
| Weight-only INT8 | SM86 Tensor x512 | 49.73 | 58.04 | 38.42 | 2.268871 |
| Online attention | FP32 FADD+FMUL+2×FFMA | 17.95 | 31.31 | 31.31 | 5.986942 |

[Profile JSON](v2/20261102T221317Z-m2-profile-enabled-profile-labels/results.json) · [summary CSV](v2/20261102T221317Z-m2-profile-enabled-profile-labels/summary.csv) · [roofline](v2/20261102T221317Z-m2-profile-enabled-profile-labels/roofline.png) · [occupancy/throughput plot](v2/20261102T221317Z-m2-profile-enabled-profile-labels/occupancy.png).

GEMM and weight-only INT8 use 1024³/config0; attention uses B=H=1, Q=K=1024, D64, causal mask and config0. Each point selects one launch with `--launch-skip 5 --launch-count 1 --cache-control all --clock-control none`; the profiler may replay it for counter collection. Ceilings use measured clocks and DRAM traffic. These points are not averages across independent trials, event-wrapper timings or end-to-end inference latency.

The SM86 Tensor x512 convention and predicated FP32 FADD+FMUL+2×FFMA convention are distinct instruction-count models. INT8 weights still execute FP16 WMMA. Attention's model excludes special functions, conversions, integer and control work. The counted roofline percentage is throughput divided by `min(measured-clock compute ceiling, arithmetic intensity × measured-clock DRAM ceiling)`. It is separate from algorithmic GEMM TFLOP/s above. Occupancy uses active-cycle percentages; SM/memory throughput use elapsed-cycle denominators. The roofline uses DRAM traffic.

The first long-label rendering and earlier failed attempts remain preserved; the readable-label version changes display labels. Supported raw export is `ncu --import "$REPORT" --page raw --csv --print-units base`. Full binary reports remain outside Git, with hashes and capture/export commands in the result JSON.

### 3.7 Native server batching off/on

| Arrival rate parameter (req/s) | Batching | Completed/offered | Errors/drops | Completed/full-drain s | p50 (ms) | p99 (ms) | p99.9 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | off | 24/24 | 0/0 | 0.7964 | 85.4616 | 385.0251 | 409.3120 |
| 1 | on | 24/24 | 0/0 | 0.7955 | 81.8387 | 302.1562 | 303.0348 |
| 5 | off | 159/159 | 0/0 | 5.2733 | 79.2576 | 239.2550 | 271.1832 |
| 5 | on | 159/159 | 0/0 | 5.2905 | 79.5078 | 151.5516 | 162.1429 |
| 10 | off | 314/314 | 0/0 | 10.3841 | 86.9200 | 547.7797 | 580.1385 |
| 10 | on | 314/314 | 0/0 | 10.4660 | 80.7457 | 155.4473 | 158.2046 |
| 20 | off | 587/587 | 0/0 | 17.1223 | 2219.0913 | 4302.3986 | 4372.4957 |
| 20 | on | 587/587 | 0/0 | 19.4860 | 82.6331 | 157.2028 | 160.6538 |

[Matched comparison](v2/20261102T132951Z-m2-native-headers-serving-comparison/comparison.json) · [off summary](v2/20261102T132951Z-m2-native-headers-load-off/load.json) · [on summary](v2/20261102T132951Z-m2-native-headers-load-on/load.json) · [off server](v2/20261102T132951Z-m2-native-headers-server-off/server.json) · [on server](v2/20261102T132951Z-m2-native-headers-server-on/server.json).

Off uses max batch 1/wait 0 µs; on uses max batch 8/wait 1,000 µs. Seed 41 creates byte-identical exponential arrivals and synthetic token-ID payloads: prompt lengths 8/16/32 and output lengths 8/16. There is one 30-second offered window per rate, no warmup, an outstanding cap of 128 and full drain. Each request opens its own localhost TCP connection. The same binary, FP16 model and GEMM/attention config0 are used, with one serialized GPU worker and isolated KV slots. Both server processes exited normally.

Each mode completes all 1,084 offers with zero errors/drops. At the 20 req/s arrival parameter there are 587 actual offers in 30 seconds; the row's throughput divides completed requests by the complete offered-plus-drain window. JSON also retains the offered-window denominator, exact drain durations and realized arrival rates. p50/p99/p99.9 interpolate due-time-to-final-socket observations within that single trial, including scheduling, TCP/framing, queue/batch wait, native compute and the return path. Requests share queues/batches; they are not independent repeated trials. [Example off raw records](v2/20261102T132951Z-m2-native-headers-load-off/rate-3/raw.json) · [on raw records](v2/20261102T132951Z-m2-native-headers-load-on/rate-3/raw.json).

Batching-on has lower p99 at every tested rate. At 20 req/s p99 is 157.2028 ms on versus 4302.3986 ms off. At 5 req/s the on median is slightly higher: 79.5078 versus 79.2576 ms. Socket/client/server timers retain their distinct boundaries; worker service is not CUDA-event kernel latency.

### 3.8 Test-gated local-LLM versus random autotuner

The predefined seeds 17, 41, 73 finished at their configured stop boundaries. The run status is **`incomplete_incomparable`**, `comparable=false`, and the recorded aggregate is exactly **`summary={}`**. Random filled the eight-unique-evaluation budget in every seed; the local LLM reached its 64-proposal limit with only five, five and four unique evaluations. The experiment has finished, but it did not produce a matched-budget aggregate.

[Full search record](v2/20261102T234500Z-m2-autotune-compact-history/search.json) · [raw-sample/budget/gate checks](v2/evidence/20261102-autotune-compact-history/acceptance/raw-samples-and-gates.json) · [native-build/proposer provenance checks](v2/evidence/20261102-autotune-compact-history/acceptance/native-and-proposer-provenance.json). Both independent checks passed. The latter covers fresh output paths, exact native build artifacts/commands/logs, raw proposer transport and stop boundaries within its recorded process-identity scope. Publication checks and merge history are linked in section 1.

Source is `6b1b9f665052e05715b15427f0a9a4777d0414fb`. Recorded start/end are `2026-11-02T23:54:59.102789+00:00` / `2026-11-03T00:33:08.235873+00:00`, with total command wall time 2289.233779 seconds. The local proposer is Qwen2.5-1.5B-Instruct Q4_K_M through llama.cpp, with CPU-only `-ngl 0`; each proposal process exits before compilation/correctness/timing. Both methods share proposal cap 64 and unique nonbaseline evaluation cap 8; invalid/duplicate/error attempts consume proposal budget. Initial baseline and fresh independent remeasurements are outside those caps.

The fixed fixture is 512x512x512, seed 193, with an actual nonzero FP16 bias, FP32 accumulation and FP16 output. Each timing uses 20 warmups, 101 event samples and 21 separate synchronized host samples, concurrency 1 and `atol=rtol=0.02`. Event windows retain Python-wrapper enqueue gaps. Proposal/build overhead and total wall time are separate from event latency. The exact fixture/config/tool/model hashes remain in JSON.

#### Search accounting

| Seed | Method | Proposals | Unique evals / cap | Invalid | Duplicate | Valid compiles | Test passes | Accepted updates | Search status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | llm | 64 | 5/8 | 47 | 12 | 5 | 5 | 1 | incomplete_proposal_cap |
| 17 | random | 12 | 8/8 | 0 | 4 | 8 | 8 | 1 | completed |
| 41 | llm | 64 | 5/8 | 44 | 15 | 5 | 5 | 0 | incomplete_proposal_cap |
| 41 | random | 9 | 8/8 | 0 | 1 | 8 | 8 | 1 | completed |
| 73 | llm | 64 | 4/8 | 43 | 17 | 4 | 4 | 0 | incomplete_proposal_cap |
| 73 | random | 15 | 8/8 | 0 | 7 | 8 | 8 | 1 | completed |

#### Selection and fresh independent measurements

| Seed | Method | Selected config | Selection initial / selected event p50 (ms) | Selection speedup | Fresh baseline / selected event p50 (ms) | Independent speedup | Independent 3% gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | llm | 10 | 0.044032 / 0.041984 | 1.048781x | 0.045056 / 0.053248 | 0.846154x | false |
| 17 | random | 10 | 0.044032 / 0.041984 | 1.048781x | 0.044032 / 0.045056 | 0.977273x | false |
| 41 | llm | 0 | 0.044032 / 0.044032 | 1.000000x | 0.044032 / 0.044032 | 1.000000x | false |
| 41 | random | 4 | 0.044032 / 0.037888 | 1.162162x | 0.044032 / 0.038912 | 1.131579x | true |
| 73 | llm | 0 | 0.044032 / 0.044032 | 1.000000x | 0.044032 / 0.044032 | 1.000000x | false |
| 73 | random | 4 | 0.044032 / 0.037888 | 1.162162x | 0.044032 / 0.037888 | 1.162162x | true |

| Selected config | Tile (BM,BN,BK,warps) | Stages | Epilogue |
| --- | --- | --- | --- |
| 0 | 32,32,16,4 | 2 | fused |
| 4 | 64,32,32,8 | 1 | fused |
| 10 | 64,32,32,8 | 1 | separate |

Accepted updates count selection-time incumbent replacements, requiring correctness and selected median <=0.97 times the incumbent median. The fresh baseline/selected pair is compiled and measured separately after selection; its gate column is read directly from `passes_3_percent_gate`. `independent.status=passed` means the remeasurement records passed their correctness/measurement checks, not that the performance gate passed.

LLM seed17 selected config10 at 1.048781x in search, then measured 0.846154x independently. Random seed17 also lost the gate independently at 0.977273x. Only random seeds41/73 pass the independent 3% gate, at 1.131579x/1.162162x; LLM seeds41/73 retained the initial config0 at 1.000000x. These are descriptive outcomes against each method's own fresh baseline, not evidence of statistical superiority under unequal used evaluation budgets.

All six method/seed groups record zero proposal-process errors, compile failures, correctness-test failures and evaluation errors. Strict invalid proposals and duplicates are recorded separately. This normal cap termination is distinct from the preserved earlier missing-header, CLI-output and history-capacity failures. No matched mean +/- std is computed: the program's summary remains `{}`. The recorded three-seed experiment is reported as-is, without supplemental budget or prompt tuning.

### 3.9 Robustness and correctness

The [historical PR host run](robustness/ci-793b62e/README.md) uses libFuzzer with ASan/UBSan on tinyinfer's production parser, shared GEMM/attention shape-entry validators, CPU operators and host quantization codec. Seed `20261102`; every target reached its 10,000-execution budget, including seed initialization.

| Target | Executions | Sanitizer diagnostics |
| --- | --- | --- |
| JSON parser | 10,000 | 0 |
| Model/weight artifact parser | 10,000 | 0 |
| Shape and operator entry validation | 10,000 | 0 |
| CPU GEMM, softmax and LayerNorm | 10,000 | 0 |
| Quantization/dequantization codec | 10,000 | 0 |

All 21 fixed seed replays passed. There were **0 crash artifacts, 0 confirmed random-input issues and 0 fixes**; no crash-derived regressions are registered. The evidence preserves 530 corpus files and original logs. The separate native/Python codec gate includes exact q-byte/FP32-scale/dequantized-half comparisons, including all 63,488 finite FP16 bit patterns. These are host checks, not CUDA device sanitizer coverage; code-review/setup fixes are excluded from random-input issue counts.

The [native GPU gate](v2/20261102T132951Z-m2-native-headers-kernel-gate/correctness.json) passed 115 cases, including non-default-stream/input-lifetime checks. The [model gate](v2/20261102T132951Z-m2-native-headers-model-gate/results.json) is separate from kernel arithmetic and host fuzzing. [Current source CI](https://github.com/kianbqh/tinyinfer/actions/runs/41548410550) passed `cpu`, `llm-cpu-contracts`, `server-autotune-cpu`, `robustness` and `profiling-contracts`; its conditional manual CUDA job was skipped. GPU measurements above are local. CPU checks cover parser/SHA, KV/cache and quantization contracts, protocol/load-generator logic, MPSC contention/transport/socket tests, TSAN builds, autotuner fake-proposer gates and profiler parsing/formulas/provenance. Counts from overlapping runners remain separate.

## 4. Negative and surprising results

- At 4096³, c0 WMMA reaches only 15.6183% of cuBLAS throughput; Triton c0 reaches 39.8853%.
- At 512³, c0 WMMA event p50 is lower than cuBLAS (0.044032/0.053248 ms), while its host p50 is higher (0.121206/0.095328 ms).
- Both custom softmax implementations lose to PyTorch at every tested shape.
- At S4096/D64, online attention c0 event p50 is 8.594944 ms versus forced-flash SDPA's 0.276480 ms, despite lower additional Torch allocator use than naive materialized attention.
- INT8 saves exported storage but adds 0.56671747 PPL versus native FP16 and is slower at all three batches.
- At batch 32, native FP16 total p50 is 280.1815 ms versus HF eager's 252.5337 ms. HF compile wins over native FP16 at every measured batch.
- Batching-on improves the measured p99, but its 5 req/s median is slightly higher: 79.5078 versus 79.2576 ms.
- The first Triton benchmark failed on existing Python header resolution; the failed 44-completed/12-error attempt remains. The first autotuner baseline failure is also retained.
- The closed autotuner experiment has no matched aggregate: LLM used only 5/5/4 unique evaluations versus random 8/8/8. Only random seeds41/73 retained the independent 3% improvement gate; unequal used budgets prevent a matched statistical ranking.

All numerical comparisons above refer to the result tables and evidence links in section 3, with their stated timing layer and trial scope.

## 5. Not done, cut and limits

- Paged KV and speculative decoding were optional stretch items and were not implemented. KV storage is contiguous.
- The predefined compact-history search finished with `incomplete_incomparable`: LLM reached the proposal cap without filling the matched unique-evaluation budget. The earlier interface failures remain history; the final result is normal cap termination.
- The experiment runs on one consumer GPU with GPT-2, context 1024, a serialized native context and fixed-count greedy generation.
- Benchmarks use synthetic tensors; server payloads are synthetic token IDs. GPT-2 PPL is a fixed WikiText-2 test slice. Real pretrained weights do not make these a real user-workload study.
- Staging buffers are synchronous, attention dot products are SIMT, and INT8 is FP16-dequantized weight-only compute.
- Each serving rate has one offered-load trial. Each profile point is one selected/replayed launch with measured clocks. Neither is an independent multi-trial estimate.

## 6. Counter access

Counter access was available for the completed captures. Counter-setting/restoration records are kept separately from the measured tables. Remaining user steps: none; GPU performance-counter access has been restored to administrators only.

## 7. Supported descriptions and scope

| Result-backed description | Scope of this experiment |
| --- | --- |
| Implemented FP16 WMMA GEMM and compared it with cuBLAS/Triton | Fixed synthetic shapes; cuBLAS remains faster at the reported large size; synchronous staging |
| Implemented online causal attention and measured latency/memory tradeoffs | SIMT dot products; the custom kernel loses to forced-flash SDPA |
| Ran native GPT-2 with per-request KV cache and perplexity agreement | Contiguous cache and a fixed WikiText-2 slice; paged KV and speculative decoding remain future work |
| Measured weight-only INT8 storage, quality and speed | FP16-dequantized WMMA; smaller exported weights, higher perplexity and latency |
| Profiled three kernels with Nsight counter-derived DRAM rooflines | Separate Tensor/FP32 instruction models and one selected/replayed launch per point |
| Implemented epoll/MPSC serving with dynamic batching | One local GPU worker and synthetic token requests, one load trial per rate |
| Ran own-code libFuzzer with ASan/UBSan | Production host paths; zero confirmed findings in the retained finite campaign; CUDA execution checked separately |
| Ran a test-gated local-LLM/random search over three predefined seeds | LLM proposal caps prevented a matched aggregate; selection and independent performance are reported separately |

## 8. Existing-result preservation and final acceptance

The module-start baseline is `d4971ec389fe96907abbe5f8d79429757848e494`. The pre-import preservation check at `b7fbeb201ab66a36d46dcbd5bdd1b68977b64229` recorded all 27 old result files unchanged, unchanged original `cpu`/`cuda` job bodies, and empty stdout for the scoped command below. The final documentation/result working tree also passed the same 27-file and original-job preservation check.

```bash
git diff --stat d4971ec389fe96907abbe5f8d79429757848e494 -- results/before/bench_cpu.json results/before/bench_cpu.md results/before/bench_gpu.json results/before/bench_gpu.md results/before/environment.txt results/before/kernels3/bench_cpu.csv results/before/kernels3/bench_cpu.md results/before/kernels3/bench_cuda.csv results/before/kernels3/bench_cuda.md results/before/kernels3/environment.json results/before/kernels3/torch_compare.md results/before/kernels3/torch_compare_gpu.md results/before/kernels3/validation.md results/before/run_all.log results/before/run_gpu.log results/before/test_final.log results/dispatch.json results/environment.txt results/integrated_cpu.json results/integrated_cuda.json results/run.json results/run_cpu.log results/run_gpu.log results/table.md results/test-policy.log results/tests.log results/validation.json
```

Recorded stdout at both checkpoints: empty. Final publication working-tree command/output and byte-preservation evidence:

[Preservation record](v2/evidence/20261102-autotune-compact-history/acceptance/frozen-original-results.json): 27 original result files match their recorded bytes; original `cpu` and `cuda` workflow job bodies are unchanged; the scoped diff stdout is empty.

The closed search passed [raw-sample/budget/gate verification](v2/evidence/20261102-autotune-compact-history/acceptance/raw-samples-and-gates.json) and [bounded native-build/proposer provenance verification](v2/evidence/20261102-autotune-compact-history/acceptance/native-and-proposer-provenance.json). Fresh artifact paths, native binaries/manifests/commands/logs and proposer prompt/history/stdout/stderr/transport were checked within the records' stated process-identity scope. Final publication commit, exact-head CI and merge verification: [PR #1 checks and merge record](https://github.com/kianbqh/tinyinfer/pull/1/checks). Measured-source CI is pinned above; publication commits retain the measured source identities rather than relabeling the measurements.
