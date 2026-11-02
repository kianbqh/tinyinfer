# Robustness checks

These tests feed random inputs to tinyinfer's own host code. Five libFuzzer
targets cover the JSON/model and weight parser, tensor metadata checks, CPU
GEMM/softmax/LayerNorm, and the quantization codec used by the weight exporter.
The GEMM and attention targets call the same metadata validators as the CUDA
entry points. CUDA device instructions are checked separately by the GPU suite.

The build enables ASan and UBSan. The PR job uses a fixed initial seed corpus
and a finite per-target budget; it retains raw logs, generated corpus inputs,
crash artifacts and source/compiler/binary hashes as an Actions artifact.
`regressions/manifest.json` lists confirmed regressions. An empty list means no
regressions have been registered, not that a run has passed.

```bash
python -B fuzz/build.py --cxx "$(command -v clang++)" --output-parent /path/to/results
python -B fuzz/replay_regressions.py --build-receipt /path/to/build.json \
  --include-seeds --output-parent /path/to/results
python -B fuzz/run_campaign.py --config fuzz/configs/local.json \
  --build-receipt /path/to/build.json --output-parent /path/to/results
```

On the local workstation, use a G-backed results directory. CI uses its runner
temporary directory. The build script installs no compiler or dependencies.

`check_codec.py` also compares the actual host library with the retained Python
quantizer byte for byte, including every finite FP16 bit pattern, and checks
that rejected input leaves output buffers unchanged. This is a separate numeric
gate, not a substitute for the sanitizer run.

Actual run summaries go under `results/robustness/`. A failed run retains its
inputs and diagnostics before any fix is made. Issue counts are written only
after the run and triage; code-review fixes are recorded separately.

## Recorded PR run

The [PR workflow](https://github.com/kianbqh/tinyinfer/actions/runs/41444025668/job/125272576550) ran five libFuzzer targets with ASan and UBSan on source `793b62e`, using seed `20261102`. They exercise the production model/weight parser, shared tensor-shape and GEMM/attention entry validation, CPU operators, and the FP16/INT8 host codec used by the exporter. Each target reached 10,000 executions; counters include seed initialization.

All 21 fixed seed replays passed. The campaign recorded **0 sanitizer diagnostics, 0 crash artifacts, 0 confirmed issues and 0 fixes**. No crash-derived regressions were registered. The [receipts, logs and 530 retained corpus files](../results/robustness/ci-793b62e/README.md) preserve the run. These are host-code checks; CUDA execution has separate [numerical and stream/lifetime results](../results/v2/20261102T132951Z-m2-native-headers-kernel-gate/correctness.json).
