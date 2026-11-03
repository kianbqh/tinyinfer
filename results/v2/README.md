# Native decoder measurements

These runs use an RTX 3080 12 GB and i7-12700KF in WSL2. The native CUDA build is pinned to `3edcfebd24d5efbf5da81cda13f2bf684031b35e`. Later profiling captures and rendering have their own recorded source checkpoints. Raw samples retain their original bytes; the tables in the module READMEs round them for display.

| Experiment | Recorded output |
| --- | --- |
| Native CUDA correctness | [115-case gate](20261102T132951Z-m2-native-headers-kernel-gate/correctness.json) |
| GEMM, including cuBLAS and Triton | [56 cases](20261102T132951Z-m2-native-headers-gemm-cpath/run.json) |
| Softmax | [12 cases](20261102T132951Z-m2-native-headers-softmax-cpath/run.json) |
| Attention | [72 cases](20261102T132951Z-m2-native-headers-attention-cpath/run.json) |
| GPT-2 numerical agreement and cache behavior | [Model gate](20261102T132951Z-m2-native-headers-model-gate/results.json) |
| WikiText-2 slice and generation | [Four runtimes](20261102T132951Z-m2-native-headers-llm-cpath/results.json) |
| Native server batching off/on | [Matched Poisson load](20261102T132951Z-m2-native-headers-serving-comparison/comparison.json) |
| Nsight counters and plots | [Readable-label result](20261102T221317Z-m2-profile-enabled-profile-labels/results.json) |
| Autotuner first attempt | [Initial baseline failure](20261102T132951Z-m2-native-headers-autotune/search.json) |
| Autotuner output transport | [Interrupted run](20261102T224446Z-m2-autotune-venvfix/interrupted.json) |
| Autotuner history capacity | [Interrupted run](20261102T230756Z-m2-autotune-transportfix/interrupted.json) |
| Autotuner, all three preset searches | [Closed run](20261102T234500Z-m2-autotune-compact-history/search.json), actual evaluation budgets unmatched |

The [first GEMM attempt](20261102T132951Z-m2-native-headers-gemm/run.json) retains the missing-header errors before the successful retry. All three preset autotuner searches finished: random filled 8 unique evaluations per seed, while the LLM filled 5/5/4 before its proposal caps. The terminal status is `incomplete_incomparable`, with no matched-budget aggregate. [Native and proposer evidence](evidence/20261102-autotune-compact-history/) includes the independent audits. The first profiling rendering is also retained; the readable-label version changes only display labels.

[Closed local command receipts](evidence/20261102-local/) include build, correctness, benchmark and profiling logs, with failed attempts preserved. [Independent checks](evidence/20261102-local/acceptance/) recompute sample statistics, compare request counts and verify source/artifact hashes. Large model binaries and full Nsight reports stay in the local run archive. The separate [host robustness results](../robustness/ci-793b62e/README.md) contain the sanitizer logs and retained libFuzzer corpus.
