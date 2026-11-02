# Nsight capture postprocessing

`python -B -m bench.v2.profiling.postprocess --manifest <actual-captures.json> --output <new-result-directory>` processes retained standalone CUDA Nsight Compute reports after real build and correctness acceptance. It does not launch a profiler or access CUDA.

The manifest identifies a clean checkpoint, actual build/command receipts, profile executable/source hashes, the installed reviewed Nsight section files and exact launch IDs/kernel names. Each capture must retain its `.ncu-rep` and successful raw CSV export command. Use `--page raw --csv --print-units base --print-metric-name name`. The supported raw wide table has exact metric-name columns and a dedicated unit row; ambiguous, missing, N/A or differently scaled data fails explicitly. Check a real export before relying on the adapter.

The result package contains raw CSV copies, `summary.csv`, `roofline.png`, `occupancy.png` and `results.json` with commands, formulas, counter values/units and source/input/output hashes. Only `status=completed` accepts all outputs; preserve failed attempts separately. The plotting dependency is imported only after input validation.

Profiling command receipts must bind the actual ncu executable before/after version, capture and export, identify CLI version2024.1 from real version stdout, and share the approved recorder-source hash. The version receipt is an immutable input to capture/export. Each capture report must be absent before capture and its recorded output hash must equal both the export's input hash and current report bytes. A same path or a report hash computed only during postprocessing does not establish that chain.

WMMA and weight-only INT8 use the reviewed SM86 Tensor instruction x512 convention. The INT8 path dequantizes into FP16 WMMA, so it does not measure INT8 tensor-core arithmetic. SIMT attention uses predicated-on FADD + FMUL + 2 FFMA accounting, excluding its special-function and other instruction costs. Both rooflines use measured DRAM traffic and measured-clock ceilings. Tensor and FP32 count models are plotted separately. Occupancy uses active-cycle percentages; SM and memory throughput use elapsed-cycle percentages. A profiler-replayed single launch is not an end-to-end latency or an averaged timing result.

Retain the private full reports, completed build evidence and profiler companion-library inventory alongside the public CSV/PNG package. Source/fixture checks are preparation evidence; actual runtime captures and independent calculation/plot review are required for measured claims.
