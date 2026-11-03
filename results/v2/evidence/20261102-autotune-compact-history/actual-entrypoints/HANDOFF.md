# Compact-history fresh-run candidate — not executed

Requested/accepted agent configuration continues as gpt-6.1-sol / ultra. Only private files were prepared; no model, tokenizer, build, search or GPU command ran.

Run ID: `20261102T234500Z-m2-autotune-compact-history`. Native run/config: `/opt/codex-resume/all-modules-v2/m2/runs/20261102T234500Z-m2-autotune-compact-history` / `/opt/codex-resume/all-modules-v2/m2/runs/20261102T234500Z-m2-autotune-compact-history/autotune.json`. Candidate scratch base: `/opt/codex-resume/all-modules-v2/m2/autotune/20261102T234500Z-compact-history`. Actual per-candidate scratch will append the run name and seed/method/label under that base. Proposed output: `results/v2/20261102T234500Z-m2-autotune-compact-history`.

Config SHA256: `1af32f7d69b598ee92ccd4484031eccb07aa1f1f2a24dc1b0b57658be3aea31e`. Relative to frozen79451016..., only scratch_base changed. Both methods and all seeds17/41/73 restart; no old baseline, random result or partial LLM history is reused. Model/executable hashes, context8192, output384, timeout180s, proposal64/unique8, fixture512³/nonzero bias/seed193, warm20/event101/host21, tolerances and >=3% incumbent gate remain the original values. Independent baseline/winner remeasurements remain outside search budgets.

Root must first accept the actual same-GGUF tokenizer capacity evidence, commit/push the final source, replace `ROOT_FINAL_COMMIT_REQUIRED` in BOTH preflight.py and record_autotune.sh, and explicitly release execution. The pending Python and Bash guards reject an unpinned checkpoint. No queue, daemon, polling launcher or automatic next stage exists.

Preserve root's data-pause flag and the still-live OS-suspended KITTI downloader; this task must not resume it. The inherited exact private LD path and CPATH/CXXFLAGS headers remain unchanged. Preflight will actually check clean source/resources/pinned input bytes only after release.

Manual preflight, after release:

```text
wsl -d CodexResearchATTAlf01 -- /bin/bash /mnt/g/ResumeProject/.runtime/all-modules-v2/m2/runtime/autotune-next-20261102T234500Z/record_autotune.sh autotune-preflight 600 /usr/bin/env CPATH=/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include/python3.12:/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include CXXFLAGS=-I/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include /opt/codex-resume/all-modules-v2/m2/venv/bin/python -B /mnt/g/ResumeProject/.runtime/all-modules-v2/m2/runtime/autotune-next-20261102T234500Z/preflight.py
```

Only after inspecting the passed preflight, separately run:

```text
wsl -d CodexResearchATTAlf01 -- /bin/bash /mnt/g/ResumeProject/.runtime/all-modules-v2/m2/runtime/autotune-next-20261102T234500Z/record_autotune.sh autotune-formal 28800 /usr/bin/env CPATH=/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include/python3.12:/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include CXXFLAGS=-I/opt/codex-resume/cpp-cuda-inference-kernels/python-dev/usr/include /opt/codex-resume/all-modules-v2/m2/venv/bin/python -B -m autotune.run --config /opt/codex-resume/all-modules-v2/m2/runs/20261102T234500Z-m2-autotune-compact-history/autotune.json --output results/v2/20261102T234500Z-m2-autotune-compact-history
```

No experiment outcome is claimed by this candidate. READINESS.json preserves exact paths/hashes, pending HEAD and unchanged protocol fields.
