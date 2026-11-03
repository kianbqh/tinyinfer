# Closed compact-history search

The [original run](../../20261102T234500Z-m2-autotune-compact-history/search.json) finished all three preset seeds at source `6b1b9f665052e05715b15427f0a9a4777d0414fb`. Random search filled 8 unique nonbaseline evaluations per seed; the local LLM filled 5/5/4 within its 64-proposal caps. The terminal state remains `incomplete_incomparable`, and the outer recorder's exit 1 is the runner's prescribed outcome for unmatched evaluation budgets. No additional proposals or reruns were added.

This directory retains 53 native candidate builds, exact configure/build commands and logs, compiler inputs, CMake caches, manifests, run configuration and actual process closure. Native libraries and model weights stay outside Git; their hashes remain in the records.

Two independent checks cover the closed run:

- [Raw samples and gates](acceptance/raw-samples-and-gates.json): all 53 evaluation records, timing summaries, fixture/source identities, proposal counts, incumbent updates and independent remeasurement ratios.
- [Native and proposer provenance](acceptance/native-and-proposer-provenance.json): 7,231 assertions covering actual library/build bytes, fresh paths and ordering, complete prompt histories, original stdout/stderr, strict parsing, random draws and exact stopping rules.

The retained audit scripts describe their original local paths. Individual helper receipts record commands and timestamps; OS process-identity checks are limited to the separately recorded outer lifecycle and phase observations. [Source CI](acceptance/source-ci.json) is separate from these local measurements.
