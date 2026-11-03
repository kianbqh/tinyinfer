"""Stdlib parser/provenance tests; never invoke profiler, main(), or render().

Synthetic values/receipts exercise rejection rules. One unchanged real Nsight
CSV checks the captured raw schema and units, without a performance rerun.
"""
from __future__ import annotations
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
MODULE = HERE / "postprocess.py"
spec = importlib.util.spec_from_file_location("profile_postprocess_candidate", MODULE)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


def synthetic_metrics():
    return {p.OCC: 50, p.SM: 25, p.MEM: 10, p.DRAM: 20,
            p.SM_HZ: 1000, p.SMSP_HZ: 2000, p.DRAM_HZ: 3000,
            p.DRAM_PEAK: 4, p.DRAM_RATE: 6000, p.TENSOR_PEAK: 10, p.TENSOR_RATE: 2,
            p.FP32_PEAK: 20, p.FP32_ADD: 3, p.FP32_MUL: 5, p.FP32_FMA: 7}


def wide(values=None, units=None, kid="0", name="void wmma_gemm<32, 32, 16, 4, false, 2>()"):
    values = synthetic_metrics() if values is None else values
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["ID", "Process ID", "Kernel Name"] + list(values))
    writer.writerow(["", "", ""] + [(units or {}).get(key, p.UNITS[key]) for key in values])
    writer.writerow([kid, "1234", name] + list(values.values()))
    return text.getvalue()


class ParserTests(unittest.TestCase):
    def test_actual_ncu_2024_1_raw_gemm_schema_and_units(self):
        fixture = HERE / "fixtures/ncu-2024.1-sm86-gemm-raw.csv"
        self.assertEqual(p.sha256(fixture),
                         "7ffa40988bba05aed987c5bca4900db1ed314f32a9fb472b5213db6ee00055bd")
        text = fixture.read_text(encoding="utf-8")
        self.assertEqual(len(next(csv.reader(io.StringIO(text)))), 518)
        launches = p.parse_raw_wide(text)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].kernel_id, "0")
        self.assertEqual(launches[0].metrics[p.TENSOR_RATE], ("inst/cycle", "1.28"))
        row, _ = p.derive(launches[0], "gemm", "actual-schema-fixture")
        self.assertEqual(row["achieved_occupancy_pct"], 59.95)
        self.assertEqual(row["roofline_model"], "tensor_sm86_x512")

    def test_tensor_section_arithmetic(self):
        launch = p.parse_raw_wide(wide())[0]
        result, metrics = p.derive(launch, "gemm", "synthetic-only")
        self.assertEqual(result["achieved_flop_per_second"], 2048000)
        self.assertEqual(result["peak_compute_flop_per_second"], 5120000)
        self.assertEqual(result["peak_dram_byte_per_second"], 12000)
        self.assertEqual(result["pct_of_roofline"], 50)
        self.assertEqual(result["achieved_occupancy_pct"], 50)
        self.assertEqual(metrics[p.OCC]["unit"], "%")

    def test_weight_only_int8_uses_same_fp16_tensor_accounting(self):
        row, _ = p.derive(p.parse_raw_wide(wide())[0], "int8", "synthetic-only")
        self.assertEqual(row["roofline_model"], "tensor_sm86_x512")

    def test_fp32_add_mul_and_double_fma_no_clipping(self):
        launch = p.parse_raw_wide(wide(name="void online_attention<64, 32>()"))[0]
        row, _ = p.derive(launch, "attention", "synthetic-only")
        self.assertEqual(row["achieved_flop_per_second"], 44000)
        self.assertEqual(row["peak_compute_flop_per_second"], 40000)
        self.assertAlmostEqual(row["pct_of_roofline"], 110)

    def test_actual_zero_occupancy_stays_zero(self):
        values = synthetic_metrics(); values[p.OCC] = 0
        row, _ = p.derive(p.parse_raw_wide(wide(values))[0], "gemm", "synthetic-only")
        self.assertEqual(row["achieved_occupancy_pct"], 0)

    def test_na_empty_nan_infinite_negative_and_locale_rejected(self):
        for raw in ("N/A", "", "nan", "inf", "-1", "1.234,56", "1,00", "1e9999"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                values = synthetic_metrics(); values[p.OCC] = raw
                p.derive(p.parse_raw_wide(wide(values))[0], "gemm", "synthetic-only")

    def test_thousands_and_scientific_notation(self):
        values = synthetic_metrics(); values[p.SMSP_HZ] = "2,000.0"; values[p.SM_HZ] = "1e3"
        row, _ = p.derive(p.parse_raw_wide(wide(values))[0], "gemm", "synthetic-only")
        self.assertEqual(row["achieved_flop_per_second"], 2048000)

    def test_scaled_or_wrong_units_rejected(self):
        for key, unit in ((p.DRAM_RATE, "Gbyte/second"), (p.OCC, "ratio"), (p.SM_HZ, "cycle/nsecond")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.derive(p.parse_raw_wide(wide(units={key: unit}))[0], "gemm", "synthetic-only")

    def test_missing_metric_not_zero(self):
        values = synthetic_metrics(); del values[p.TENSOR_RATE]
        with self.assertRaisesRegex(ValueError, "missing metric"):
            p.derive(p.parse_raw_wide(wide(values))[0], "gemm", "synthetic-only")

    def test_ambiguous_missing_unit_row_rejected(self):
        rows = wide().splitlines()
        with self.assertRaises(ValueError):
            p.parse_raw_wide("\n".join((rows[0], rows[2], rows[2])))

    def test_duplicate_id_rejected_even_with_different_kernel_name(self):
        rows = wide().splitlines()
        with self.assertRaisesRegex(ValueError, "duplicate launch ID"):
            p.parse_raw_wide("\n".join(rows + [rows[2].replace("wmma_gemm", "other_kernel")]))

    def test_duplicate_columns_rejected(self):
        rows = list(csv.reader(io.StringIO(wide())))
        for row in rows:
            row.append(row[-1])
        text = io.StringIO(); csv.writer(text).writerows(rows)
        with self.assertRaisesRegex(ValueError, "duplicate CSV columns"):
            p.parse_raw_wide(text.getvalue())

    def test_long_schema_and_labels_rejected(self):
        with self.assertRaises(ValueError):
            p.parse_raw_wide('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"","","","",""\n"0","foo","Achieved Occupancy","%","50"\n')
        with self.assertRaises(ValueError):
            p.parse_raw_wide(wide().replace(p.OCC, "Achieved Occupancy"))

    def test_bad_preamble_and_trailing_output_rejected(self):
        for text in ("==PROF== ERR_NVGPUCTRPERM\n" + wide(), "Warning: no data\n" + wide(), wide() + "tail\n"):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                p.parse_raw_wide(text)

    def test_benign_preamble_and_bom(self):
        self.assertEqual(p.parse_raw_wide("\ufeff==PROF== Importing report\n" + wide())[0].kernel_id, "0")

    def test_wrong_kernel_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "kernel identity"):
            p.derive(p.parse_raw_wide(wide())[0], "attention", "synthetic-only")

    def test_zero_dram_cannot_be_invented_intensity(self):
        values = synthetic_metrics(); values[p.DRAM_RATE] = 0
        with self.assertRaisesRegex(ValueError, "positive finite"):
            p.derive(p.parse_raw_wide(wide(values))[0], "gemm", "synthetic-only")

    def test_nonfinite_derived_ratios_rejected(self):
        values = synthetic_metrics(); values[p.DRAM_RATE] = "1e-320"
        with self.assertRaisesRegex(ValueError, "derived roofline ratios"):
            p.derive(p.parse_raw_wide(wide(values))[0], "gemm", "synthetic-only")


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        scratch = HERE / ".tmp-profile-unit-fixtures"; scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-unit-", dir=scratch)
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        try:
            self.folder.parent.rmdir()
        except OSError:
            pass

    def fixture(self, status="completed", exit_code=0):
        (self.folder / "argv.json").write_text(json.dumps(["synthetic-ncu", "--version"]))
        (self.folder / "stdout.log").write_text("SYNTHETIC UNIT FIXTURE, NOT NCU OUTPUT\n")
        (self.folder / "stderr.log").write_text("")
        value = {"schema_version": 1, "status": status, "exit_code": exit_code,
                 "argv": ["synthetic-ncu", "--version"]}
        for name in ("stdout.log", "stderr.log"):
            value[name + "_sha256"] = p.sha256(self.folder / name)
        path = self.folder / "command.json"; path.write_text(json.dumps(value))
        return path

    def test_completed_receipt_hashes(self):
        record, evidence = p.command_receipt(self.fixture())
        self.assertEqual(record["exit_code"], 0)
        self.assertIn("stderr.log", evidence)

    def test_failed_partial_or_nonzero_receipts_rejected(self):
        for status, code in (("started", None), ("failed", 1), ("completed", 2)):
            with self.subTest(status=status, code=code), self.assertRaises(ValueError):
                p.command_receipt(self.fixture(status, code))

    def test_argv_or_log_tamper_rejected(self):
        path = self.fixture()
        (self.folder / "stdout.log").write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            p.command_receipt(path)
        path = self.fixture(); (self.folder / "argv.json").write_text('["other"]')
        with self.assertRaisesRegex(ValueError, "argv.json mismatch"):
            p.command_receipt(path)

    def test_incomplete_and_synthetic_manifest_rejected_before_any_plot_import(self):
        for value in ({}, {"schema_version": 1, "evidence_kind": "synthetic"}):
            path = self.folder / "captures.json"; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "measured capture"):
                p.verify_inputs(path)
        self.assertNotIn("matplotlib", sys.modules)
        self.assertNotIn("torch", sys.modules)


class ManifestTests(unittest.TestCase):
    """Entirely synthetic tiny receipts: exercises validation, never main/render."""

    setUp = ReceiptTests.setUp
    tearDown = ReceiptTests.tearDown

    def recorded(self, name, argv, stdout="SYNTHETIC UNIT FIXTURE\n", inputs=None, outputs=None):
        folder = self.folder / name; folder.mkdir()
        bindings = None
        if inputs is not None:
            outputs = outputs or {}
            spec_path = folder / "artifacts.json"
            spec_path.write_text(json.dumps({"schema_version": 1, "inputs": inputs, "outputs": outputs}))
            def snapshot(path):
                return {"requested_path": str(Path(path)), **p.artifact(path)}
            bindings = {"schema_version": 1, "status": "verified",
                        "specification_before": snapshot(spec_path), "specification_after": snapshot(spec_path),
                        "recorder_source_before": snapshot(self.folder / "synthetic-recorder.py"),
                        "recorder_source_after": snapshot(self.folder / "synthetic-recorder.py"),
                        "inputs_before": {key: snapshot(value) for key, value in inputs.items()},
                        "inputs_after": {key: snapshot(value) for key, value in inputs.items()},
                        "outputs_before_absent": {}, "outputs_after": {}}
            for key, value in outputs.items():
                self.assertFalse(Path(value).exists())
                bindings["outputs_before_absent"][key] = {"requested_path": value, "path": str(Path(value).resolve()), "absent": True}
                Path(value).write_text("SYNTHETIC, NOT AN NCU REPORT\n")
                bindings["outputs_after"][key] = snapshot(value)
        (folder / "argv.json").write_text(json.dumps(argv))
        with (folder / "stdout.log").open("w", newline="") as handle:
            handle.write(stdout)
        (folder / "stderr.log").write_text("")
        record = {"schema_version": 1, "status": "completed", "exit_code": 0, "argv": argv}
        if bindings is not None:
            record["artifact_bindings"] = bindings
        for log in ("stdout.log", "stderr.log"):
            record[log + "_sha256"] = p.sha256(folder / log)
        path = folder / "command.json"; path.write_text(json.dumps(record))
        return path

    def make_manifest(self):
        root = self.folder / "source"; (root / "bench/v2/kernels").mkdir(parents=True)
        source = root / "bench/v2/kernels/profile.cu"; source.write_text("// SYNTHETIC, NEVER COMPILED\n")
        backend = self.folder / "backend.so"; backend.write_text("SYNTHETIC, NOT A LIBRARY\n")
        exe = self.folder / "profile_v2"; exe.write_text("SYNTHETIC, NOT EXECUTABLE\n")
        ncu = self.folder / "ncu"; ncu.write_text("SYNTHETIC, NOT EXECUTABLE\n")
        recorder = self.folder / "synthetic-recorder.py"; recorder.write_text("# SYNTHETIC RECEIPT FIXTURE, NEVER EXECUTED\n")
        compile_path = self.folder / "compile_commands.json"
        compile_path.write_text(json.dumps([{"file": str(source), "command": "SYNTHETIC, NOT RUN"}]))
        build_argv = ["synthetic-cmake", "--build", str(self.folder)]
        command = self.recorded("build", build_argv)
        build = {"build_exit_code": 0, "git": {"sha": "a" * 40, "status_porcelain": ""},
                 "build_command": p.shlex.join(build_argv), "build_log_path": str(command.parent / "stdout.log"),
                 "build_log_sha256": p.sha256(command.parent / "stdout.log"), "library_path": str(backend),
                 "library_sha256": p.sha256(backend), "source_sha256": {"bench/v2/kernels/profile.cu": p.sha256(source)},
                 "build_directory": str(self.folder), "compile_commands_sha256": p.sha256(compile_path)}
        build_path = self.folder / "build.json"; build_path.write_text(json.dumps(build))
        version = self.recorded("version", [str(ncu), "--version"],
                                "SYNTHETIC VERSION FIXTURE, NOT NCU OUTPUT\nNVIDIA (R) Nsight Compute Command Line Profiler\nVersion 2024.1.0.0\n",
                                inputs={"ncu": str(ncu)})
        section = self.folder / "Synthetic.section"; section.write_text("SYNTHETIC SECTION, NOT NVIDIA\n")
        report = self.folder / "synthetic.ncu-rep"
        tool_inputs = {"ncu": str(ncu), "version_receipt": str(version)}
        cap = self.recorded("capture", [str(ncu), "--launch-count", "1", "--export", str(report), str(exe), "gemm", "1024", "1", "64", "20"],
                            inputs=tool_inputs, outputs={"report": str(report)})
        export = self.recorded("export", [str(ncu), "--import", str(report), "--page", "raw", "--csv",
                                            "--print-units", "base"], wide(),
                               inputs={**tool_inputs, "report": str(report)})
        value = {"schema_version": 1, "evidence_kind": "measured", "checkpoint": "a" * 40,
                 "compute_capability": [8, 6], "source_root": str(root), "kernel_build_receipt": str(build_path),
                 "build_command": str(command), "profile_executable": str(exe), "profile_executable_sha256": p.sha256(exe),
                 "profile_source_sha256": p.sha256(source), "ncu_version_command": str(version),
                 "recorder_source_sha256": p.sha256(recorder),
                 "section_directory": str(self.folder), "captures": [{"label": "synthetic-test-only", "operation": "gemm",
                 "kernel_id": "0", "kernel_name": p.parse_raw_wide(wide())[0].kernel_name,
                 "raw_csv": str(export.parent / "stdout.log"), "report": str(report),
                 "capture_command": str(cap), "export_command": str(export)}]}
        path = self.folder / "captures.json"; path.write_text(json.dumps(value))
        return path, {section.name: p.sha256(section)}, value

    def test_synthetic_manifest_validates_without_plotting(self):
        path, sections, value = self.make_manifest()
        export = p.load_json(value["captures"][0]["export_command"])
        self.assertNotIn("--print-metric-name", export["argv"])
        with mock.patch.dict(p.SECTIONS, sections, clear=True):
            rows, evidence = p.verify_inputs(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pct_of_roofline"], 50)
        self.assertIn("profile_compile_record", evidence)
        self.assertNotIn("matplotlib", sys.modules)

    def test_exact_id_and_kernel_filter_required(self):
        path, sections, value = self.make_manifest()
        for field, replacement in (("kernel_id", "1"), ("kernel_name", "other")):
            original = value["captures"][0][field]
            value["captures"][0][field] = replacement; path.write_text(json.dumps(value))
            with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "exact selection"):
                p.verify_inputs(path)
            value["captures"][0][field] = original

    def test_profile_and_section_hashes_must_match(self):
        path, sections, value = self.make_manifest()
        value["profile_executable_sha256"] = "0" * 64; path.write_text(json.dumps(value))
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "profile executable"):
            p.verify_inputs(path)
        value["profile_executable_sha256"] = p.sha256(value["profile_executable"]); path.write_text(json.dumps(value))
        with mock.patch.dict(p.SECTIONS, {"Synthetic.section": "0" * 64}, clear=True), self.assertRaisesRegex(ValueError, "unreviewed Nsight section"):
            p.verify_inputs(path)

    def test_no_silent_omission_of_extra_launch(self):
        path, sections, value = self.make_manifest()
        export_path = Path(value["captures"][0]["export_command"])
        stdout = export_path.parent / "stdout.log"
        data = list(csv.reader(io.StringIO(wide())))
        row = data[-1].copy(); row[0] = "1"; data.append(row)
        text = io.StringIO(); csv.writer(text).writerows(data)
        with stdout.open("w", newline="") as handle:
            handle.write(text.getvalue())
        receipt = json.loads(export_path.read_text()); receipt["stdout.log_sha256"] = p.sha256(stdout)
        export_path.write_text(json.dumps(receipt))
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "one launch per capture"):
            p.verify_inputs(path)

    def test_report_overwrite_after_export_is_rejected(self):
        path, sections, value = self.make_manifest()
        Path(value["captures"][0]["report"]).write_text("DIFFERENT SYNTHETIC REPORT BYTES\n")
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "bound file differs"):
            p.verify_inputs(path)

    def test_ncu_overwrite_after_version_is_rejected(self):
        path, sections, _ = self.make_manifest()
        (self.folder / "ncu").write_text("DIFFERENT SYNTHETIC TOOL BYTES\n")
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "bound file differs"):
            p.verify_inputs(path)

    def test_forged_current_report_does_not_replace_command_time_chain(self):
        path, sections, value = self.make_manifest()
        export_path = Path(value["captures"][0]["export_command"])
        receipt = json.loads(export_path.read_text())
        receipt["artifact_bindings"]["inputs_before"]["report"]["sha256"] = "0" * 64
        receipt["artifact_bindings"]["inputs_after"]["report"]["sha256"] = "0" * 64
        export_path.write_text(json.dumps(receipt))
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "bound file differs"):
            p.verify_inputs(path)

    def test_missing_command_time_bindings_rejected(self):
        path, sections, value = self.make_manifest()
        capture = Path(value["captures"][0]["capture_command"])
        receipt = json.loads(capture.read_text()); del receipt["artifact_bindings"]
        capture.write_text(json.dumps(receipt))
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "command-time artifact"):
            p.verify_inputs(path)

    def test_version_text_must_identify_exact_2024_1(self):
        path, sections, value = self.make_manifest()
        version_path = Path(value["ncu_version_command"])
        stdout = version_path.parent / "stdout.log"
        stdout.write_text("NVIDIA (R) Nsight Compute Command Line Profiler\nVersion 2024.10.0\n")
        record = json.loads(version_path.read_text()); record["stdout.log_sha256"] = p.sha256(stdout)
        version_path.write_text(json.dumps(record))
        with mock.patch.dict(p.SECTIONS, sections, clear=True), self.assertRaisesRegex(ValueError, "CLI 2024.1"):
            p.verify_inputs(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
