# Test-gated local GEMM search

This module compares a local llama.cpp proposer with seeded random search over the same12 GEMM choices. Proposals contain data only: exact catalog ID, tile, stages and epilogue. Candidate text never becomes code, shell or compiler flags.

**Status:** the three-seed comparison is pending a fresh run after a completion-output fix. An earlier attempt reached real CUDA candidate tests and timing, then stopped when the CLI returned empty stdout. Its [interruption record](../results/v2/20261102T224446Z-m2-autotune-venvfix/interrupted.json) and [raw command evidence](../results/v2/evidence/20261102-autotune-interface-failed/) are retained. That interrupted run does not establish a comparison between proposers.

## Setup and execution

Root first integrates/tests the v2 kernel and native decoder changes after opening M2. Use an isolated G-backed M2 interpreter with the already coordinated matching dependencies. Do not mutate the shared read-only environment. `config.example.json` deliberately contains invalid placeholder hashes/paths; fill them from current actual evidence. At least three unique search seeds are mandatory.

The proposer requires a locally installed llama.cpp **completion CLI** and local GGUF. The previously verified M1-A artifact is a candidate; the CLI executable and its supported flags still require checking with that exact build's `--help/--version`. Do not assume `llama-server` is interchangeable with a completion CLI. If the local build lacks this CLI, record that fact and coordinate a bounded build before attempting the experiment. No download, service startup or paid API is implicit.

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

Each seed compiles/measures one initial baseline; both methods use that exact baseline record. Both get the same fixed proposal cap and the same cap on unique nonbaseline candidate evaluation attempts. Invalid/duplicate/proposal-error attempts consume proposal budget; duplicates are not silently evaluated again. Compile failures and correctness failures consume unique evaluation attempts and remain in the ledger. The random proposer samples the identical catalog with replacement, making duplicate handling explicit. The LLM receives previous raw outcome/timing rows and the catalog. Search stops at the evaluation cap or proposal cap. A method that runs out of proposals before filling the evaluation cap is labeled **incomplete**, and the pair is not reported as a matched-budget comparison.

Every attempt retains prompt, raw stdout/stderr, command/exit/timeout, tuple validation, build logs, correctness records, all samples, acceptance and errors. Counts separate proposals, duplicates/invalids, unique evaluations, valid compiles, test passes, compile/test/evaluation/proposal failures and accepted updates. No parse/build/test failure is converted to a speedup.

After selection, the selected winner and initial baseline are recompiled and remeasured in fresh processes on the identical fixture. Those two measurements are explicitly outside the search budget. The report preserves selection speedup and independent speedup separately, including a regression or loss of the3% gate. Summary means and population standard deviations use the three-or-more fixed search seeds only when both methods completed matched budgets and independent measurements passed. If random is equal/better, report that directly; no real result exists yet.

Tests using a fake proposer establish parsing, failure accounting and gate logic. They are not local-LLM execution or learning-quality evidence. Missing CLI/model/GPU/toolchain access is a real unmet gate, never replaced with synthetic performance data.
