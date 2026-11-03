# Test-gated local GEMM search

This module compares a local llama.cpp proposer with seeded random search over the same12 GEMM choices. Proposals contain data only: exact catalog ID, tile, stages and epilogue. Candidate text never becomes code, shell or compiler flags.

The predefined three-seed experiment is complete. The LLM reached its proposal cap before filling the evaluation budget in every seed, so the run has no matched-budget aggregate. The selected configurations were rebuilt and remeasured; none of the LLM's three independent measurements retained the 3% improvement gate.

## Measured search

The [closed run](../results/v2/20261102T234500Z-m2-autotune-compact-history/search.json) used source `6b1b9f6`, seeds 17/41/73, 64 proposals and up to eight unique nonbaseline evaluations per method. Each method shares its seed's initial baseline. The fixture is 512³, seed 193, with nonzero FP16 bias, FP32 accumulation and FP16 output. Timings use 20 warmups, 101 CUDA-event samples and 21 separate synchronized host samples. Model proposal inference runs on the CPU and exits before GPU evaluation.

| Seed | Method | Proposals | Unique evaluations | Invalid | Duplicate | Valid compiles / test passes | Accepted updates | Selected config |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | random | 12 | 8 | 0 | 4 | 8 / 8 | 1 | 10 |
| 17 | LLM | 64 | 5 | 47 | 12 | 5 / 5 | 1 | 10 |
| 41 | random | 9 | 8 | 0 | 1 | 8 / 8 | 1 | 4 |
| 41 | LLM | 64 | 5 | 44 | 15 | 5 / 5 | 0 | 0 |
| 73 | random | 15 | 8 | 0 | 7 | 8 / 8 | 1 | 4 |
| 73 | LLM | 64 | 4 | 43 | 17 | 4 / 4 | 0 | 0 |

| Seed | Method | Selection speedup | Fresh independent speedup | Independent 3% gate |
| --- | --- | ---: | ---: | --- |
| 17 | random | 1.048781× | 0.977273× | fail |
| 17 | LLM | 1.048781× | 0.846154× | fail |
| 41 | random | 1.162162× | 1.131579× | pass |
| 41 | LLM | 1.000000× | 1.000000× | fail |
| 73 | random | 1.162162× | 1.162162× | pass |
| 73 | LLM | 1.000000× | 1.000000× | fail |

The six independent baseline/selected pairs passed correctness checks. Only random seeds 41 and 73 kept a performance gain above the gate. Seed 17 selected the same config in both methods, but its gain disappeared in fresh measurements. Selection and independent speedups use their own measured baselines; [the complete table](../results/M2_DELIVERY_REPORT.md#38-test-gated-local-llm-versus-random-autotuner) includes those raw medians.

Proposal-process, compile, correctness-test and evaluation errors were all zero. Invalid JSON/schema outputs and duplicates consumed proposal budget normally. The final status is `incomplete_incomparable`, with `summary={}`, because the actual unique evaluation counts differ. The bounded run took 2,289.234 seconds; proposal and build overhead are outside the kernel timing samples. [Independent checks and build/proposer receipts](../results/v2/evidence/20261102-autotune-compact-history/) retain all 53 evaluations, including fresh remeasurements.

The earlier [output-transport failure](../results/v2/20261102T224446Z-m2-autotune-venvfix/interrupted.json) and [context-capacity failure](../results/v2/20261102T230756Z-m2-autotune-transportfix/interrupted.json) retain their original outputs. The latter reached 8,206 input tokens against the CLI's 8,188-token input limit. The final compact-table run restarted both methods and all three seeds from scratch; it added no proposals after the preset caps.

## Setup and execution

I run this in an isolated WSL2 Python environment with the v2 native build dependencies. `config.example.json` contains placeholder hashes and paths; replace them with the local executable, model and toolchain identities. At least three unique search seeds are required.

The recorded runs use the local llama.cpp **completion CLI** and Qwen2.5-1.5B-Instruct Q4_K_M GGUF. The [preflight and command receipts](../results/v2/evidence/20261102-autotune-context-capacity-failed/) retain the exact executable/model hashes and supported interface. The runner checks both hashes before every proposal.

The fixed command uses `-ngl 0`, one prompt file, bounded output/context and process timeout. Every proposal process fully exits before the kernel build/evaluator starts. This explicitly uses CPU proposal inference; its overhead is reported separately and no GPU proposal speed is claimed. At execution, verify process flags and GPU telemetry to confirm no persistent LLM/context remains in the benchmark window. Do not point this path at an unrelated external server.

The pinned `llama-completion` CLI sends generated tokens through its logger, so the command keeps logging enabled and captures diagnostics separately on stderr. Its program-generated terminal ` [end of text]` plus three line feeds is decoded once, only at the exact end of stdout. Raw stdout stays unchanged on disk, and the receipt records framing removal and the returned payload hash. Model prose, extra JSON fields and malformed output still fail the strict schema. [The transport probe and regression logs](../results/v2/evidence/20261102-autotune-interface-failed/) retain both the original failure and the fix.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s autotune/tests -p 'test_*.py'
python -m autotune.run --config "$ACTUAL_AUTOTUNE_CONFIG" --output "results/v2/$UNIQUE_SEARCH_RUN"
```

Only the first command is a CPU gate. The second performs actual compilation and GPU tests and must wait for the coordinated runtime window. Each output directory is new and cannot overwrite older evidence. Candidate build scratch is restricted to the coordinated G-backed `/opt/codex-resume/all-modules-v2/m2/…` or `/mnt/g/ResumeProject/.runtime/all-modules-v2/m2/…` roots.

## Exact build/evaluation seam

`core.parse_json` rejects oversized/non-JSON output, duplicate keys, NaN/Infinity, extra fields, booleans-as-integers, and catalog tuples that disagree with their IDs. It delegates the exact contract to **`bench.v2.kernels.candidate_build.validate_proposal`**. All12 choices come from `llm.native_ops.GEMM_CATALOG` (three tile tuples ×stage1/2 ×fused/separate bias epilogues).

`run.RealEvaluator` invokes **`python -m bench.v2.kernels.candidate_build`** with the operator-owned toolchain paths and the validated proposal file. That existing helper configures `TINYC_V2_GEMM_CONFIG`, compiles one allow-listed instantiation, and retains configure/build logs, source hashes, flags and `build-manifest.json`. There is no proposal-controlled command adapter.

After a successful build it launches a separate **`python -m autotune.evaluate`** process. That evaluator loads the exact `NativeOps` library, checks native `only_config`, calls the existing full **`run_correctness`** suite, and then checks the actual performance fixture against FP32 GEMM+bias. The same fixed seed creates FP16 A[M,K], B[K,N] and a real nonzero FP16 bias[N] for every candidate/method/seed/remeasurement; actual tensor-byte SHA256 values must match the initial baseline. Bias is passed to both fused and separate paths, and the numerical output/accumulator types match. Only after all checks pass does the existing **`bench.v2.kernels.run.measure`** collect per-call CUDA events and separately labeled synchronized host timings. Actual compiler/toolkit/library/source/Git/environment data come from **`run.environment`** and the build receipt. GPU processes exit between candidates; compilation and proposal overhead do not enter kernel samples.

## Gates, budgets and reporting

A candidate can replace the current incumbent only when all correctness tests pass and its median is **≤0.97 × incumbent median**. Equality at3% is accepted. There must be at least seven positive finite raw samples; zero/NaN/negative/boolean/missing timings cannot win. Each accepted candidate becomes the next incumbent, so a later1% improvement cannot pass by comparing against the original slower baseline.

Each seed compiles/measures one initial baseline; both methods use that exact baseline record. Both get the same fixed proposal cap and the same cap on unique nonbaseline candidate evaluation attempts. Invalid/duplicate/proposal-error attempts consume proposal budget; duplicates are not silently evaluated again. Compile failures and correctness failures consume unique evaluation attempts and remain in the ledger. The random proposer samples the identical catalog with replacement. Search stops at the evaluation cap or proposal cap. A method that runs out of proposals before filling the evaluation cap is labeled **incomplete**, and the pair is not reported as a matched-budget comparison.

The LLM receives the unchanged catalog and a table containing the baseline and every previous attempt. Columns are attempt, config ID, status, measured median, correctness and error type. Raw timing samples and full error text stay in the journals. This replaces the oversized full-sample prompt; it does not filter duplicates, remove failed attempts, repair model output or add hints. The model, sampling, context, proposal/evaluation caps and acceptance gate stay fixed. [Capacity checks](../results/v2/evidence/20261102-autotune-history-capacity-fix/) use the same GGUF tokenizer, separately from model generation and GPU timing.

Every attempt retains prompt, raw stdout/stderr, command/exit/timeout, tuple validation, build logs, correctness records, all samples, acceptance and errors. Counts separate proposals, duplicates/invalids, unique evaluations, valid compiles, test passes, compile/test/evaluation/proposal failures and accepted updates. No parse/build/test failure is converted to a speedup.

After selection, the selected winner and initial baseline are recompiled and remeasured in fresh processes on the identical fixture. Those two measurements are outside the search budget. The report preserves selection speedup and independent speedup separately, including a regression or loss of the3% gate. Summary means and population standard deviations use the three-or-more fixed search seeds only when both methods completed matched budgets and independent measurements passed. That matching condition was not met in this run.

Tests using a fake proposer establish parsing, failure accounting and gate logic. They are not local-LLM execution or learning-quality evidence. Missing CLI/model/GPU/toolchain access is a real unmet gate, never replaced with synthetic performance data.
