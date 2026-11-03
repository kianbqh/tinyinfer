"""Derive per-launch DRAM rooflines from retained Nsight Compute raw CSV.

Only the explicit raw-wide-with-unit-row adapter is supported. Its compatibility
with the first real local capture must still be checked; synthetic tests are not
capture evidence. No profiler, subprocess, CUDA or ML package is invoked here.
Matplotlib is imported only by render(), after all inputs validate.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import sys


# Independently read in full from the installed Nsight Compute 2024.1.0 bundle.
# Changing these requires review of the actual section formulas, not a bypass.
SECTIONS = {
    "SpeedOfLight_RooflineChart.section": "aa2e015a035fcaf400b71fc7a1a95be7c7ec471e278aeef1dc056ea4016c41bf",
    "SpeedOfLight_HierarchicalTensorRooflineChart.section": "aa625ad01e9d20d79633c43b55c445455eab9a15da8e97f02a3579381eaa8c45",
    "Occupancy.section": "60923f104984b71d7b1b94344ebe78e00951c7fcb9096106f53f3b7f44fe46e9",
    "SpeedOfLight.section": "fd72e6755dbe9a0ad2a3064bcef84dfd90713a05e84fba5c96cb935f79c84b68",
}
OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
MEM = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
DRAM = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
SM_HZ = "sm__cycles_elapsed.avg.per_second"
SMSP_HZ = "smsp__cycles_elapsed.avg.per_second"
DRAM_HZ = "dram__cycles_elapsed.avg.per_second"
DRAM_PEAK = "dram__bytes.sum.peak_sustained"
DRAM_RATE = "dram__bytes.sum.per_second"
TENSOR_PEAK = "sm__inst_executed_pipe_tensor.sum.peak_sustained"
TENSOR_RATE = "smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed"
FP32_PEAK = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"
FP32_ADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
FP32_MUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"
FP32_FMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"
UNITS = {
    OCC: "%", SM: "%", MEM: "%", DRAM: "%",
    SM_HZ: "cycle/second", SMSP_HZ: "cycle/second", DRAM_HZ: "cycle/second",
    DRAM_PEAK: "byte/cycle", DRAM_RATE: "byte/second",
    TENSOR_PEAK: "inst/cycle", TENSOR_RATE: "inst/cycle",
    FP32_PEAK: "inst/cycle", FP32_ADD: "inst/cycle",
    FP32_MUL: "inst/cycle", FP32_FMA: "inst/cycle",
}
COMMON = (OCC, SM, MEM, DRAM, SM_HZ, SMSP_HZ, DRAM_HZ, DRAM_PEAK, DRAM_RATE)
FORMULAS = {
    "tensor_sm86_x512": {
        "achieved_flop_per_second": f"512 * {TENSOR_RATE} * {SMSP_HZ}",
        "peak_compute_flop_per_second": f"512 * {TENSOR_PEAK} * {SM_HZ}",
        "section": "SpeedOfLight_HierarchicalTensorRooflineChart.section",
    },
    "fp32_fadd_fmul_ffma": {
        "achieved_flop_per_second": f"({FP32_ADD} + {FP32_MUL} + 2 * {FP32_FMA}) * {SMSP_HZ}",
        "peak_compute_flop_per_second": f"2 * {FP32_PEAK} * {SM_HZ}",
        "section": "SpeedOfLight_RooflineChart.section",
    },
    "common": {
        "achieved_dram_byte_per_second": DRAM_RATE,
        "peak_dram_byte_per_second": f"{DRAM_PEAK} * {DRAM_HZ}",
        "arithmetic_intensity_flop_per_byte": "achieved_flop_per_second / achieved_dram_byte_per_second",
        "ridge_flop_per_byte": "peak_compute_flop_per_second / peak_dram_byte_per_second",
        "ceiling_at_point_flop_per_second": "min(peak_compute_flop_per_second, arithmetic_intensity_flop_per_byte * peak_dram_byte_per_second)",
        "pct_of_roofline": "100 * achieved_flop_per_second / ceiling_at_point_flop_per_second",
        "achieved_occupancy_pct": OCC + " [%], Occupancy.section",
        "sm_throughput_pct": SM + " [%], SpeedOfLight.section",
        "memory_throughput_pct": MEM + " [%], SpeedOfLight.section",
        "dram_throughput_pct": DRAM + " [%], SpeedOfLight.section",
    },
}
METADATA = {
    "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
    "Stream", "Block Size", "Grid Size", "Device", "CC", "Section Name",
}
METRIC_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*__[A-Za-z0-9_.]+\Z")
NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SUMMARY_FIELDS = (
    "label", "operation", "kernel_id", "kernel_name", "roofline_model",
    "achieved_occupancy_pct", "sm_throughput_pct", "memory_throughput_pct", "dram_throughput_pct",
    "achieved_flop_per_second", "peak_compute_flop_per_second",
    "achieved_dram_byte_per_second", "peak_dram_byte_per_second",
    "arithmetic_intensity_flop_per_byte", "ridge_flop_per_byte",
    "ceiling_at_point_flop_per_second", "pct_of_roofline",
    "raw_csv_relative_path", "raw_csv_sha256", "report_sha256",
)


@dataclass(frozen=True)
class Launch:
    kernel_id: str
    kernel_name: str
    metrics: dict[str, tuple[str, str]]  # exact metric -> (base-unit text, raw value text)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected a file: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_raw_wide(text):
    """Strict explicit adapter; no metric-label or long-table guessing.

    Accepted preamble consists only of empty lines and benign ==PROF== lines.
    A header contains exact metric names; the next row contains their units.
    Every remaining row represents one unique launch ID and exact kernel name.
    """
    lines = text.lstrip("\ufeff").splitlines()
    while lines and (not lines[0].strip() or lines[0].startswith("==PROF==")):
        line = lines.pop(0)
        require(not re.search(r"error|ERR_", line, re.I), "profiler error in CSV preamble")
    rows = list(csv.reader(io.StringIO("\n".join(lines)), strict=True))
    require(len(rows) >= 3, "raw CSV needs header, unit row and launch data")
    header, units, *data = rows
    require(len(header) == len(set(header)), "duplicate CSV columns")
    require("ID" in header and "Kernel Name" in header, "expected exact ID and Kernel Name columns")
    require("Metric Name" not in header, "long/details CSV unsupported; export --page raw")
    require(all(name in METADATA or METRIC_NAME.fullmatch(name) for name in header),
            "unknown CSV column/schema; do not guess metric labels")
    require(any(METRIC_NAME.fullmatch(name) for name in header), "no exact metric-name columns")
    require(len(units) == len(header), "unit-row width mismatch")
    id_col, name_col = header.index("ID"), header.index("Kernel Name")
    require(not units[id_col] and not units[name_col], "missing/ambiguous dedicated unit row")
    launches, seen = [], set()
    for row in data:
        require(len(row) == len(header), "launch-row width mismatch (including trailing output)")
        kid, name = row[id_col], row[name_col]
        require(re.fullmatch(r"\d+", kid) and bool(name), "invalid launch ID or empty kernel name")
        require(kid not in seen, "duplicate launch ID; no averaging or row merging")
        seen.add(kid)
        metrics = {key: (unit, value) for key, unit, value in zip(header, units, row)
                   if METRIC_NAME.fullmatch(key)}
        launches.append(Launch(kid, name, metrics))
    require(bool(launches), "no launch rows")
    return launches


def metric_value(launch, name):
    require(name in launch.metrics, f"missing metric {name}")
    unit, raw = launch.metrics[name]
    require(unit == UNITS[name], f"{name}: unit {unit!r}; expected base {UNITS[name]!r}")
    require(NUMBER.fullmatch(raw.strip()) is not None, f"{name}: missing/N/A/non-numeric {raw!r}")
    try:
        value = float(Decimal(raw.strip().replace(",", "")))
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError(f"{name}: invalid number") from exc
    require(math.isfinite(value) and value >= 0, f"{name}: nonfinite/negative value")
    return value


def derive(launch, operation, label):
    require(operation in ("gemm", "int8", "attention"), "unsupported operation")
    expected = "online_attention" if operation == "attention" else "wmma_gemm"
    require(expected in launch.kernel_name, "kernel identity does not match the requested operation")
    extra = (FP32_PEAK, FP32_ADD, FP32_MUL, FP32_FMA) if operation == "attention" else (TENSOR_PEAK, TENSOR_RATE)
    values = {name: metric_value(launch, name) for name in COMMON + extra}
    if operation == "attention":
        achieved = (values[FP32_ADD] + values[FP32_MUL] + 2 * values[FP32_FMA]) * values[SMSP_HZ]
        peak = 2 * values[FP32_PEAK] * values[SM_HZ]
        model = "fp32_fadd_fmul_ffma"
    else:
        achieved = 512 * values[TENSOR_RATE] * values[SMSP_HZ]
        peak = 512 * values[TENSOR_PEAK] * values[SM_HZ]
        model = "tensor_sm86_x512"
    bandwidth = values[DRAM_RATE]
    peak_bandwidth = values[DRAM_PEAK] * values[DRAM_HZ]
    require(all(math.isfinite(x) and x > 0 for x in (achieved, peak, bandwidth, peak_bandwidth)),
            "positive finite work/traffic/ceilings required for a DRAM log-roofline")
    intensity = achieved / bandwidth
    ceiling = min(peak, intensity * peak_bandwidth)
    ridge = peak / peak_bandwidth
    require(all(math.isfinite(x) and x > 0 for x in (intensity, ceiling, ridge)),
            "nonfinite/underflowed derived roofline ratios")
    percent = 100 * (achieved / ceiling)
    require(math.isfinite(percent) and percent > 0, "nonfinite/underflowed derived roofline ratios")
    row = {"label": label, "operation": operation, "kernel_id": launch.kernel_id,
           "kernel_name": launch.kernel_name, "roofline_model": model,
           "achieved_occupancy_pct": values[OCC], "sm_throughput_pct": values[SM],
           "memory_throughput_pct": values[MEM], "dram_throughput_pct": values[DRAM],
           "achieved_flop_per_second": achieved, "peak_compute_flop_per_second": peak,
           "achieved_dram_byte_per_second": bandwidth, "peak_dram_byte_per_second": peak_bandwidth,
           "arithmetic_intensity_flop_per_byte": intensity, "ridge_flop_per_byte": ridge,
           "ceiling_at_point_flop_per_second": ceiling, "pct_of_roofline": percent}
    return row, {name: {"unit": launch.metrics[name][0], "raw_value": launch.metrics[name][1], "value": value}
                 for name, value in values.items()}


def command_receipt(path):
    record = load_json(path)
    require(record.get("schema_version") == 1 and record.get("status") == "completed"
            and record.get("exit_code") == 0 and not record.get("forced_kill"),
            f"not a completed successful recorded command: {path}")
    argv = record.get("argv")
    require(isinstance(argv, list) and argv and all(isinstance(x, str) and x for x in argv), "invalid argv")
    parent = Path(path).resolve().parent
    require(load_json(parent / "argv.json") == argv, "command/argv.json mismatch")
    evidence = {"receipt": artifact(path), "argv_file": artifact(parent / "argv.json"), "record": record}
    for name in ("stdout.log", "stderr.log"):
        item = artifact(parent / name)
        require(item["sha256"] == record.get(name + "_sha256"), f"retained {name} hash mismatch")
        evidence[name] = item
    if "artifact_bindings" in record:
        verified_bindings(record)
    return record, evidence


def current_snapshot(item):
    require(isinstance(item, dict) and Path(item.get("requested_path", "")).is_absolute(),
            "invalid bound file snapshot")
    actual = artifact(item["requested_path"])
    require(all(actual[key] == item.get(key) for key in ("path", "sha256", "bytes")),
            "command-time bound file differs from current bytes")
    return item


def verified_bindings(record):
    bindings = record.get("artifact_bindings", {})
    require(bindings.get("schema_version") == 1 and bindings.get("status") == "verified",
            "successful command-time artifact bindings required")
    for prefix in ("specification", "recorder_source"):
        before, after = bindings.get(prefix + "_before"), bindings.get(prefix + "_after")
        require(before == after, prefix + " changed during command")
        current_snapshot(before)
    before, after = bindings.get("inputs_before"), bindings.get("inputs_after")
    require(isinstance(before, dict) and before == after, "bound inputs changed during command")
    for item in before.values():
        current_snapshot(item)
    absent, outputs = bindings.get("outputs_before_absent"), bindings.get("outputs_after")
    require(isinstance(absent, dict) and isinstance(outputs, dict) and set(absent) == set(outputs),
            "bound output inventory mismatch")
    for name, item in outputs.items():
        require(absent[name].get("absent") is True and absent[name].get("path") == item.get("path")
                and absent[name].get("requested_path") == item.get("requested_path"), "output was not a fresh bound file")
        current_snapshot(item)
    return bindings


def bound_file(record, name, expected_path, output=False):
    bindings = verified_bindings(record)
    key = "outputs_after" if output else "inputs_before"
    require(name in bindings[key], f"missing named command-time binding: {name}")
    item = bindings[key][name]
    require(Path(item["path"]) == Path(expected_path).resolve(strict=True), f"bound {name} path differs")
    return item


def option(argv, flag):
    require(argv.count(flag) == 1, f"expected exactly one {flag}")
    index = argv.index(flag)
    require(index + 1 < len(argv), f"missing value for {flag}")
    return argv[index + 1]


def verify_inputs(manifest_path):
    config = load_json(manifest_path)
    require(config.get("schema_version") == 1 and config.get("evidence_kind") == "measured",
            "only explicit measured capture manifests can produce published results")
    checkpoint = config.get("checkpoint", "")
    require(re.fullmatch(r"[0-9a-f]{40}", checkpoint), "exact actual checkpoint required")
    require(config.get("compute_capability") == [8, 6], "reviewed Tensor x512 adapter is SM86 only")
    root = Path(config["source_root"]).resolve(strict=True)
    build = load_json(config["kernel_build_receipt"])
    require(build.get("build_exit_code") == 0 and build.get("git", {}).get("sha") == checkpoint
            and not build.get("git", {}).get("status_porcelain"), "build receipt is not the clean checkpoint")
    build_command, build_evidence = command_receipt(config["build_command"])
    require(shlex.split(build["build_command"]) == build_command["argv"], "build argv differs from receipt")
    require(sha256(build["build_log_path"]) == build["build_log_sha256"]
            == build_evidence["stdout.log"]["sha256"], "build stdout differs from receipt")
    require(sha256(build["library_path"]) == build["library_sha256"], "backend library changed")
    source_hashes = build.get("source_sha256", {})
    require(bool(source_hashes), "build receipt has no source hashes")
    for relative, expected in source_hashes.items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and sha256(path) == expected, f"build source changed: {relative}")
    profile_source = root / "bench/v2/kernels/profile.cu"
    require(SHA.fullmatch(config.get("profile_source_sha256", ""))
            and sha256(profile_source) == config["profile_source_sha256"], "profile source differs from pre-capture manifest")
    require(source_hashes.get("bench/v2/kernels/profile.cu") == config["profile_source_sha256"],
            "kernel build receipt must directly bind profile.cu")
    compile_path = Path(build["build_directory"]) / "compile_commands.json"
    require(sha256(compile_path) == build["compile_commands_sha256"], "compile commands changed")
    commands = load_json(compile_path)
    profile_commands = [row for row in commands if Path(row["file"]).resolve() == profile_source.resolve()]
    require(len(profile_commands) == 1, "need exactly one profile.cu compilation record")
    profile = artifact(config["profile_executable"])
    require(profile["sha256"] == config.get("profile_executable_sha256"), "profile executable differs from pre-capture manifest")
    version, version_evidence = command_receipt(config["ncu_version_command"])
    require(version["argv"][1:] == ["--version"], "expected recorded actual ncu --version")
    ncu = artifact(version["argv"][0])
    version_tool = bound_file(version, "ncu", version["argv"][0])
    approved_recorder = config.get("recorder_source_sha256", "")
    require(SHA.fullmatch(approved_recorder) and verified_bindings(version)["recorder_source_before"]["sha256"] == approved_recorder,
            "unreviewed recorder source hash")
    version_text = Path(version_evidence["stdout.log"]["path"]).read_text(encoding="utf-8")
    require("Nsight Compute" in version_text and "Command Line Profiler" in version_text
            and re.search(r"(?im)^\s*Version\s+2024\.1(?:\.\d+)*(?:\s|$)", version_text),
            "actual ncu --version stdout must identify Nsight Compute CLI 2024.1")
    sections = {}
    section_dir = Path(config["section_directory"])
    for name, expected in SECTIONS.items():
        item = artifact(section_dir / name)
        require(item["sha256"] == expected, f"unreviewed Nsight section formulas: {name}")
        sections[name] = item
    captures = config.get("captures")
    require(isinstance(captures, list) and captures, "no captures supplied")
    rows, capture_evidence, seen = [], [], set()
    for capture in captures:
        label = capture["label"]
        require(isinstance(label, str) and label.strip() and label not in seen, "empty or duplicate capture label")
        seen.add(label)
        recorded, recorded_evidence = command_receipt(capture["capture_command"])
        exported, exported_evidence = command_receipt(capture["export_command"])
        require(recorded["argv"][0] == exported["argv"][0] == version["argv"][0], "ncu executable differs")
        raw = artifact(capture["raw_csv"])
        report = artifact(capture["report"])
        require(report["bytes"] > 0, "empty ncu report")
        for command in (recorded, exported):
            tool = bound_file(command, "ncu", version["argv"][0])
            require(tool["sha256"] == version_tool["sha256"] == ncu["sha256"], "ncu changed between version/capture/export")
            bound_file(command, "version_receipt", config["ncu_version_command"])
            require(verified_bindings(command)["recorder_source_before"]["sha256"] == approved_recorder,
                    "recorder changed between version/capture/export")
        captured_report = bound_file(recorded, "report", capture["report"], output=True)
        exported_report = bound_file(exported, "report", capture["report"])
        require(captured_report["sha256"] == exported_report["sha256"] == report["sha256"],
                "capture output report / export input report / current report hash differs")
        require(raw["sha256"] == exported_evidence["stdout.log"]["sha256"], "CSV differs from recorded export stdout")
        argv = recorded["argv"]
        require(option(argv, "--launch-count") == "1", "each capture must select exactly one launch")
        export_path = option(argv, "--export")
        expected_report = export_path if export_path.endswith(".ncu-rep") else export_path + ".ncu-rep"
        require(Path(expected_report).resolve() == Path(report["path"]), "capture report path mismatch")
        require(argv.count(profile["path"]) == 1, "capture must execute this exact standalone profile binary")
        target = argv.index(profile["path"])
        require(argv[target + 1] == capture["operation"], "profile operation differs from manifest")
        require(option(exported["argv"], "--import") == capture["report"]
                and option(exported["argv"], "--page") == "raw" and "--csv" in exported["argv"]
                and option(exported["argv"], "--print-units") == "base", "export raw/base command required")
        launches = parse_raw_wide(Path(raw["path"]).read_text(encoding="utf-8-sig"))
        require(len(launches) == 1, "one launch per capture; no silent omission of extra launches")
        launch = launches[0]
        require(launch.kernel_id == str(capture["kernel_id"]) and launch.kernel_name == capture["kernel_name"],
                "actual launch ID/Kernel Name differs from exact selection")
        row, metrics = derive(launch, capture["operation"], label)
        row.update(raw_csv_sha256=raw["sha256"], report_sha256=report["sha256"])
        rows.append(row)
        capture_evidence.append({"label": label, "raw_csv": raw, "report": report,
                                 "capture_command": recorded_evidence, "export_command": exported_evidence,
                                 "selected_kernel_id": launch.kernel_id, "selected_kernel_name": launch.kernel_name,
                                 "metrics_used": metrics})
    provenance = {"manifest": artifact(manifest_path), "checkpoint": checkpoint, "compute_capability": [8, 6],
                  "kernel_build_receipt": artifact(config["kernel_build_receipt"]), "build_command": build_evidence,
                  "backend_library": artifact(build["library_path"]), "build_source_sha256": source_hashes,
                  "profile_source": artifact(profile_source), "profile_executable": profile,
                  "profile_compile_record": profile_commands[0], "compile_commands": artifact(compile_path),
                  "ncu_executable": ncu, "ncu_version_command": version_evidence,
                  "sections": sections, "captures": capture_evidence,
                  "processor_sources": {p.name: artifact(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    return rows, provenance


def render(rows, output):
    # Deliberate late import: stdlib fixtures never import plotting/ML libraries.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({row["roofline_model"] for row in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        selected = [row for row in rows if row["roofline_model"] == model]
        bounds = [row[key] for row in selected for key in ("arithmetic_intensity_flop_per_byte", "ridge_flop_per_byte")]
        lo, hi = math.log10(min(bounds)) - 1, math.log10(max(bounds)) + 1
        xs = [10 ** (lo + (hi - lo) * i / 100) for i in range(101)]
        for row in selected:
            line, = ax.loglog(xs, [min(row["peak_compute_flop_per_second"], x * row["peak_dram_byte_per_second"]) / 1e12
                                  for x in xs], label=row["label"] + " measured-clock ceiling")
            ax.scatter([row["arithmetic_intensity_flop_per_byte"]], [row["achieved_flop_per_second"] / 1e12],
                       color=line.get_color(), zorder=5)
            ax.annotate(row["label"] + " / ID " + row["kernel_id"],
                        (row["arithmetic_intensity_flop_per_byte"], row["achieved_flop_per_second"] / 1e12),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Arithmetic intensity [counted FLOP/DRAM byte]", ylabel="Counted performance [TFLOP/s]",
               title="SM86 Tensor x512" if model.startswith("tensor") else "FP32 FADD + FMUL + 2 FFMA")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Nsight DRAM roofline — one actual launch per point")
    fig.tight_layout()
    fig.savefig(output / "roofline.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 2.5), 5))
    fields = (("achieved_occupancy_pct", "Achieved occupancy (active cycles)"),
              ("sm_throughput_pct", "Compute SM throughput (elapsed)"),
              ("memory_throughput_pct", "Memory throughput (elapsed)"),
              ("dram_throughput_pct", "DRAM throughput (elapsed)"))
    for offset, (field, label) in enumerate(fields):
        ax.bar([i + (offset - 1.5) * .19 for i in range(len(rows))], [row[field] for row in rows], .19, label=label)
    ax.set_xticks(range(len(rows)), [row["label"] + "\nID " + row["kernel_id"] for row in rows])
    ax.set(ylabel="Nsight reported [%]", title="Occupancy and SOL throughput (different counter denominators)")
    ax.set_ylim(0, max(105, max(row[field] for row in rows for field, _ in fields) * 1.15))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "occupancy.png", dpi=180)
    plt.close(fig)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    require(not output.exists(), "output already exists; preserve prior attempts")
    rows, provenance = verify_inputs(Path(args.manifest).resolve(strict=True))
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "started", "evidence_kind": "measured",
              "adapter": "ncu_raw_wide_unit_row_v1", "provenance": provenance, "formulas": FORMULAS,
              "counting_limits": [
                  "Tensor: section x512 instruction accounting for SM86; INT8 is weight-only FP16 WMMA, not INT8 compute.",
                  "FP32: predicated-on FADD + FMUL + 2 FFMA only; excludes special functions, FP16/conversion, integer and control work.",
                  "DRAM traffic only; no L1/L2 hierarchical roofline is claimed.",
                  "Occupancy uses active-cycle denominator; SOL throughput uses elapsed-cycle denominators.",
                  "Each point is one profiler replayed launch, not the event benchmark or an averaged end-to-end operator.",
                  "A retained successful receipt is provenance; root independently accepts correctness, build/linkage and capture identity.",
              ]}
    write_json(output / "results.json", result)
    try:
        (output / "raw").mkdir()
        raw_outputs = {}
        for index, (row, capture) in enumerate(zip(rows, provenance["captures"])):
            relative = f"raw/capture-{index:03d}.csv"
            shutil.copyfile(capture["raw_csv"]["path"], output / relative)
            item = artifact(output / relative)
            require(item["sha256"] == row["raw_csv_sha256"], "copied raw CSV hash mismatch")
            row["raw_csv_relative_path"] = relative
            raw_outputs[relative] = item
        with (output / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        render(rows, output)
        # Re-hash every retained input after rendering. No changed evidence is accepted.
        _, after = verify_inputs(Path(args.manifest).resolve(strict=True))
        require(provenance == after, "capture/build/source evidence changed during postprocessing")
        result.update(status="completed", launches=rows,
                      outputs={**raw_outputs, **{name: artifact(output / name) for name in ("summary.csv", "roofline.png", "occupancy.png")}})
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "results.json", result)
        raise
    write_json(output / "results.json", result)
    print(json.dumps({"status": "completed", "output": str(output), "launches": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
