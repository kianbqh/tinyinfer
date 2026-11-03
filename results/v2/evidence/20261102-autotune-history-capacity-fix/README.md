# History-table capacity checks

The previous completion input grew beyond its fixed context. I replaced per-attempt sample arrays with a compact table of actual medians, keeping the baseline and every previous attempt. The complete catalog is unchanged; raw arrays and full errors remain in the search journals. Model, sampling, context, budgets, strict proposal validation and the acceptance gate are unchanged.

[CPU regression records](CPU_RESULT.json) show the original source failing three new history checks, with all eight transport checks passing. The repaired source passes all eleven. The [integrated suite](integrated-regression/check.json) passes 23 tests. These are CPU fixtures, separate from actual model proposals and GPU measurements.

The same GGUF tokenizer counted these retained prompt bytes with default BOS and special-token handling:

| Input | Bytes | Tokens |
| --- | ---: | ---: |
| Actual interrupted proposal 26 | 10,798 | 8,206 |
| Original full-history fixture | 205,130 | 104,639 |
| Compact mixed-history fixture | 5,274 | 2,263 |
| Compact full-width numeric fixture | 7,384 | 3,163 |
| Compact high-entropy error-type fixture | 7,387 | 4,518 |

[Exact commands, hashes and counts](TOKEN_CAPACITY_RESULT.json) · [original stdout/stderr](tokenizer-capacity/) · [synthetic history and prompt fixtures](evidence/).

Each capacity fixture includes the baseline plus 64 prior rows, one more prior row than the final proposal needs. The largest tested fixture leaves 3,670 tokens below the CLI's 8,188-token prompt limit; including all 384 output tokens totals 4,902 against an 8,192-token context. The envelope uses finite full-width numbers and error-type strings up to 32 characters. These are tokenizer measurements, without generation or GPU inference.

The retained scripts and source manifest document the original check environment. The three history regressions also run in the regular CPU CI suite. Both search methods and all three seeds run fresh after this repair; interrupted search values remain separate.
