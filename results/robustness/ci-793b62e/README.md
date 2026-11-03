# Host robustness CI evidence

This directory retains the measured host robustness run for source `793b62e`, checked out by CI as merge `6db431a`. [The CI job](https://github.com/kianbqh/tinyinfer/actions/runs/41444025668/job/125272576550) completed all five libFuzzer targets, 10,000 runs each, with ASan and UBSan. All 21 deterministic seed replays passed. No sanitizer diagnostic or crash artifact was observed; confirmed/fixed random-input issues were 0/0. These finite host checks do not execute CUDA device instructions.

`acceptance.json` records the scope, limits, counts and provenance. `source-parity.json` matches every one of the 61 CI source hashes against exact committed Git blob bytes. Original receipts, raw build/fuzzer/replay logs and the native codec parity result are retained here. `preserved-corpus.tar.gz` preserves all 530 corpus files byte for byte, including all 21 initial fixtures; `corpus-manifest.json` lists their hashes. libFuzzer's final active corpus contained 501 inputs; retained disk files also include redundant/original seeds.

The original Actions ZIP was 4,911,627 bytes (SHA256 `6c23b4a23deb84a3c4b4dbd2e647e02894da34e3cf18d82e5a13c83bebc711ea`). Build objects and executable binaries are excluded from this compact evidence copy.
