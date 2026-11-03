# Interrupted history-capacity run

The search at `91ae0821a1a1e1129beeabbed0aa6ff1d2e4e947` used a full-sample history prompt. Seed 17's proposal 26 exceeded the fixed context: the CLI reported 8,206 prompt tokens against a limit of 8,188. Subsequent over-limit proposals returned process errors. These are input-capacity failures, separate from JSON validity or configuration quality.

The run stopped during seed 41. [The interruption record](../../20261102T230756Z-m2-autotune-transportfix/interrupted.json) describes the actual stop and partial state; the original `search.json` and raw files are unchanged. This directory retains closed command, build, context-error and process-closure evidence. Native binaries and model weights remain outside Git.

The replacement history table keeps the baseline and every previous attempt, using actual medians instead of full timing arrays. Its new source checkpoint and fresh results are reported separately. The interrupted run supplies no three-seed matched-budget comparison.
