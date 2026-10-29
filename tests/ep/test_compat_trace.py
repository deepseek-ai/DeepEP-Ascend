"""Host-only tests. Symbol fixtures were read from the EP16 final-run reports.

Source: compare_20261019/results/compare_20261019T041704Z_c041a13b13/
        ep16/{auto,official}.json, BS 2 and 64, rank 0.
These symbols are real device-profile observations; synthetic surrounding CSV/
Chrome documents test the parser without asserting a new NPU validation.
"""

import builtins
from contextlib import nullcontext
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import compat_trace as trace


DISPATCH_MTE = "_ZN7deep_ep13dispatch_implILi16ELi128ELi8ELi1ELb0ELb0ELi64ELi14336ELi0ELi2ELi221184ELl100000000000ELb1ELb1ELi1ELi64ELi6EEEvPvPsPlPfPhPiS6_S6_S6_S6_S6_S1_S1_S1_S1_iiii"
DISPATCH_URMA = "_ZN7deep_ep13dispatch_implILi16ELi128ELi8ELi1ELb0ELb0ELi64ELi14336ELi0ELi64ELi221184ELl100000000000ELb1ELb0ELi1ELi64ELi6EEEvPvPsPlPfPhPiS6_S6_S6_S6_S6_S1_S1_S1_S1_iiii"
COMBINE_MTE = "_ZN7deep_ep12combine_implILi16ELi8ELi7168ELi2ELi64ELl100000000000ELi221184ELb1ELb1ELb1EEEvPu6__bf16PfPiS3_PhS4_PvS5_i"
COMBINE_URMA = "_ZN7deep_ep12combine_implILi16ELi8ELi7168ELi64ELi64ELl100000000000ELi221184ELb1ELb1ELb0EEEvPu6__bf16PfPiS3_PhS4_PvS5_i"
OFFICIAL_DISPATCH = "_ZN7deep_ep13dispatch_implILi16ELi128ELi8ELi1ELb0ELb0ELi64ELi14336ELi0ELi2ELi221184ELl100000000000ELb1ELi1ELi64ELi6EEEvPvPsPlPfPhPiS6_S6_S6_S6_S6_S1_S1_S1_S1_iiii"
OFFICIAL_COMBINE = "_ZN7deep_ep12combine_implILi16ELi8ELi7168ELi2ELi64ELl100000000000ELi221184ELb1ELb1EEEvPu6__bf16PfPiS3_PhS4_PvS5_i"
DISPATCH_EPILOGUE = "_ZN7deep_ep27dispatch_copy_epilogue_implILi16ELi128ELi8ELi1ELb0ELb0ELi64ELi14336ELi0ELi2ELi221184ELl100000000000ELb0EEEvPvS1_S1_PiS2_S1_PsPfS2_iii"
COMBINE_EPILOGUE = "_ZN7deep_ep28combine_reduce_epilogue_implILi16ELi128ELi8ELi7168ELi2ELi221184ELi64ELl100000000000ELb0ELi14336EEEvPu6__bf16PfPlS1_S1_S1_PvS4_i"


class KernelParserTests(unittest.TestCase):
    def test_real_candidate_symbols_and_parameters(self):
        for operation, mte, urma in (("dispatch", DISPATCH_MTE, DISPATCH_URMA),
                                     ("combine", COMBINE_MTE, COMBINE_URMA)):
            self.assertEqual(trace.identify_transport(mte, operation), "mte")
            self.assertEqual(trace.identify_transport(urma, operation), "urma")
            parameters = trace.parse_kernel(mte, operation)
            self.assertEqual(parameters["num_vec_cores"], 64)
            self.assertEqual(parameters["num_ub_bytes"], 221184)
            self.assertTrue(parameters["barrier_in_prologue"])
            self.assertTrue(parameters["low_latency_transport"])

    def test_real_official_symbols_and_backend_mismatch(self):
        for operation, official, candidate in (
                ("dispatch", OFFICIAL_DISPATCH, DISPATCH_MTE),
                ("combine", OFFICIAL_COMBINE, COMBINE_MTE)):
            self.assertEqual(trace.identify_transport(official, operation, "official"), "urma")
            with self.assertRaises(trace.TraceVerificationError):
                trace.identify_transport(candidate, operation, "official")
            with self.assertRaises(trace.TraceVerificationError):
                trace.identify_transport(official, operation, "candidate")

    def test_demangled_symbol(self):
        name = "void deep_ep::combine_impl<16, 8, 7168, 2, 64, 100000000000L, 221184, false, true, true>(...)"
        parameters = trace.parse_kernel(name, "combine")
        self.assertEqual(parameters["transport"], "mte")
        self.assertFalse(parameters["barrier_in_prologue"])
        self.assertTrue(parameters["has_topk_weights"])

    def test_unknown_or_epilogue_is_not_main_evidence(self):
        for operation, name in (("dispatch", DISPATCH_EPILOGUE), ("combine", COMBINE_EPILOGUE),
                                ("dispatch", "dispatch"), ("combine", "ep_compat::combine")):
            with self.assertRaises(trace.TraceVerificationError):
                trace.parse_kernel(name, operation)

    def test_missing_or_invalid_candidate_bool_is_rejected(self):
        malformed = COMBINE_MTE.replace("ELb1ELb1ELb1E", "ELb1ELb1ELb2E")
        with self.assertRaises(trace.TraceVerificationError):
            trace.parse_kernel(malformed, "combine")

    def test_absent_mixed_or_unexpected_transports_fail(self):
        for rows, expected in (([], None),
                               ([{"name": DISPATCH_MTE}, {"name": DISPATCH_URMA}], None),
                               ([{"name": DISPATCH_URMA}], "mte")):
            with self.assertRaises(trace.TraceVerificationError):
                trace.verify_kernel_records(rows, "dispatch", expected_transport=expected)

    def test_cli_candidate_backend_aliases(self):
        for backend in ("auto", "candidate-auto", "candidate-disabled"):
            self.assertEqual(trace.TraceCollector(backend=backend).backend, "candidate")
            self.assertEqual(trace.identify_transport(DISPATCH_URMA, "dispatch", backend), "urma")

    def test_case_template_metadata_checks_and_unobservable_fields(self):
        parameters = {DISPATCH_MTE: trace.parse_kernel(DISPATCH_MTE, "dispatch")}
        metadata = dict(M=2, B=2, cached=False, cpu_sync=False, barrier_in_prologue=True,
                        dtype="bf16", hidden=7168, num_ranks=16, num_topk=8, num_experts=128,
                        alignment=1, num_vec_cores=64, min_num_ub_bytes=190 * 1024,
                        deferred=False, weight_side_channel=True, scale_layout="row")
        checks = trace.verify_case_metadata(parameters, "dispatch", metadata)
        self.assertTrue({"M", "cached", "cpu_sync", "dtype", "hidden", "barrier_in_prologue"}
                        <= set(checks["checked_fields"]))
        self.assertEqual(checks["not_observable_from_symbol"],
                         ["B", "deferred", "scale_layout", "weight_side_channel"])
        for key, bad_value in (("M", 256), ("cached", True), ("cpu_sync", None),
                               ("barrier_in_prologue", False), ("dtype", "fp8"),
                               ("hidden", 4096), ("min_num_ub_bytes", 240 * 1024)):
            with self.subTest(key=key), self.assertRaises(trace.TraceVerificationError):
                trace.verify_case_metadata(parameters, "dispatch", dict(metadata, **{key: bad_value}))

    def test_combine_weight_specialization_is_checked(self):
        parameters = {COMBINE_MTE: trace.parse_kernel(COMBINE_MTE, "combine")}
        trace.verify_case_metadata(parameters, "combine", {"weight_side_channel": True, "M": 2})
        with self.assertRaises(trace.TraceVerificationError):
            trace.verify_case_metadata(parameters, "combine", {"weight_side_channel": False})

    def test_cached_default_cpu_sync_and_fp8_are_checked(self):
        # Derived synthetic specializations exercise branches absent from the
        # historical BF16 fresh-dispatch profile fixtures.
        cached = DISPATCH_MTE.replace("ELi1ELb0ELb0ELi64", "ELi1ELb1ELb0ELi64")
        fp8 = DISPATCH_URMA.replace("ELi14336ELi0E", "ELi7168ELi112E")
        trace.verify_case_metadata(
            {cached: trace.parse_kernel(cached, "dispatch")}, "dispatch",
            {"cached": True, "cpu_sync": None})
        trace.verify_case_metadata(
            {fp8: trace.parse_kernel(fp8, "dispatch")}, "dispatch",
            {"dtype": "fp8", "hidden": 7168, "expected_num_sf_packs": 112})
        with self.assertRaises(trace.TraceVerificationError):
            trace.verify_case_metadata(
                {fp8: trace.parse_kernel(fp8, "dispatch")}, "dispatch", {"dtype": "bf16"})


class ProfileFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, names):
        path = self.root / "kernel_details.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Name", "Device Id", "Stream Id",
                                                       "Start Time(us)", "Duration(us)"])
            writer.writeheader()
            for index, name in enumerate(names):
                writer.writerow({"Name": name, "Device Id": 0, "Stream Id": 3,
                                 "Start Time(us)": index * 10, "Duration(us)": 2})
        return path

    def write_trace(self, events):
        path = self.root / "chrome_trace.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return path

    def test_device_csv_ignores_epilogues_and_cpu_labels(self):
        self.write_csv([DISPATCH_MTE, DISPATCH_EPILOGUE, "ep_compat::dispatch"])
        result = trace.parse_profile_directory(self.root, "dispatch", expected_transport="mte")
        self.assertEqual(result["actual_transport"], "mte")
        self.assertEqual(result["device_kernel_count"], 1)
        self.assertEqual(result["device_kernels"][0]["device_id"], "0")
        self.assertEqual(len(result["file_sha256"]), 1)

    def test_chrome_device_kernel_events(self):
        event = dict(name=COMBINE_MTE, cat="kernel", ph="X", pid=12, tid=3, ts=1, dur=2)
        self.write_trace([event, dict(event, cat="cpu_op"), dict(event, ph="M")])
        result = trace.parse_profile_directory(self.root, "combine", expected_transport="mte")
        self.assertEqual(result["device_kernel_count"], 1)
        self.assertEqual(result["evidence_format"], "chrome_kernel_events")

    def test_cpu_symbol_events_are_not_device_evidence(self):
        self.write_trace([dict(name=COMBINE_MTE, cat="cpu_op", ph="X", pid=12, tid=3, ts=1, dur=2)])
        with self.assertRaises(trace.TraceVerificationError):
            trace.parse_profile_directory(self.root, "combine")

    def test_epilogue_only_csv_fails_instead_of_falling_back(self):
        self.write_csv([DISPATCH_EPILOGUE])
        self.write_trace([dict(name=DISPATCH_MTE, cat="kernel", ph="X", pid=12, tid=3, ts=1, dur=2)])
        with self.assertRaises(trace.TraceVerificationError):
            trace.parse_profile_directory(self.root, "dispatch")

    def test_missing_files_and_invalid_duration_fail(self):
        with self.assertRaises(trace.TraceVerificationError):
            trace.parse_profile_directory(self.root, "dispatch")
        self.write_trace([dict(name=DISPATCH_MTE, cat="kernel", ph="X", pid=12, tid=3, ts=1, dur=0)])
        with self.assertRaises(trace.TraceVerificationError):
            trace.parse_profile_directory(self.root, "dispatch")

    def test_mixed_device_transports_fail(self):
        self.write_csv([DISPATCH_MTE, DISPATCH_URMA])
        with self.assertRaises(trace.TraceVerificationError):
            trace.parse_profile_directory(self.root, "dispatch")


class DisabledCollectorTests(unittest.TestCase):
    def test_import_and_disabled_capture_never_import_torch(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torch_npu"):
                raise AssertionError("Host-only code imported Torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            spec = importlib.util.spec_from_file_location("fresh_compat_trace", trace.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            collector = module.TraceCollector()
            with collector.capture("fresh", "dispatch", "mte", {"bs": 2}) as result:
                pass
            self.assertEqual(result["status"], "unverified")
            self.assertIsNone(result["actual_transport"])
            self.assertEqual(collector.records, [result])

    def test_enabled_requires_output_and_accepts_auto_alias(self):
        with self.assertRaises(trace.TraceVerificationError):
            trace.TraceCollector(enabled=True)
        self.assertEqual(trace.TraceCollector(backend="auto").backend, "candidate")

    def test_disabled_manifest_remains_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = trace.TraceCollector(temporary)
            with collector.capture("case", "combine", "urma"):
                pass
            document = json.loads(collector.write_manifest().read_text(encoding="utf-8"))
            self.assertFalse(document["profiling_enabled"])
            self.assertEqual(document["records"][0]["status"], "unverified")
            self.assertIsNone(document["records"][0]["actual_transport"])


class EnabledCollectorLifecycleTests(unittest.TestCase):
    """Mock the profiler boundary, without treating it as real device validation."""

    def modules(self, symbol):
        def profile(**kwargs):
            directory = Path(kwargs["on_trace_ready"])

            class Profile:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    if symbol is not None:
                        with (directory / "kernel_details.csv").open("w", newline="", encoding="utf-8") as stream:
                            writer = csv.DictWriter(stream, fieldnames=[
                                "Name", "Device Id", "Stream Id", "Start Time(us)", "Duration(us)"])
                            writer.writeheader()
                            writer.writerow({"Name": symbol, "Device Id": 0, "Stream Id": 3,
                                             "Start Time(us)": 1, "Duration(us)": 2})

            return Profile()

        fake_torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=Mock()),
            profiler=SimpleNamespace(record_function=lambda label: nullcontext()))
        fake_npu = SimpleNamespace(profiler=SimpleNamespace(
            _ExperimentalConfig=lambda **kwargs: kwargs,
            ExportType=SimpleNamespace(Text="text"), ProfilerLevel=SimpleNamespace(Level1=1),
            ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            tensorboard_trace_handler=lambda directory, **kwargs: directory,
            profile=profile))
        return {"torch": fake_torch, "torch_npu": fake_npu}

    def test_enabled_capture_records_verified_parameters_and_synchronizes(self):
        modules = self.modules(DISPATCH_MTE)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, modules):
            collector = trace.TraceCollector(temporary, enabled=True)
            with collector.capture("fresh", "dispatch", "mte", {"bs": 2}) as evidence:
                self.assertEqual(evidence["status"], "unverified")
            modules["torch"].npu.synchronize.assert_called_once()
            self.assertEqual(evidence["status"], "verified")
            self.assertEqual(evidence["actual_transport"], "mte")
            saved = json.loads((Path(evidence["profile_directory"]) / "evidence.json").read_text())
            self.assertEqual(saved, evidence)

    def test_enabled_missing_or_unexpected_device_evidence_is_saved_as_failed(self):
        for symbol in (None, DISPATCH_EPILOGUE, DISPATCH_URMA):
            with self.subTest(symbol=symbol), tempfile.TemporaryDirectory() as temporary:
                with patch.dict(sys.modules, self.modules(symbol)):
                    collector = trace.TraceCollector(temporary, enabled=True)
                    with self.assertRaises(trace.TraceVerificationError):
                        with collector.capture("fresh", "dispatch", "mte"):
                            pass
                evidence = collector.records[0]
                self.assertEqual(evidence["status"], "failed")
                self.assertTrue((Path(evidence["profile_directory"]) / "evidence.json").is_file())

    def test_api_failure_is_propagated_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, self.modules(DISPATCH_MTE)):
            collector = trace.TraceCollector(temporary, enabled=True)
            with self.assertRaisesRegex(ValueError, "API launch failed"):
                with collector.capture("fresh", "dispatch", "mte"):
                    raise ValueError("API launch failed")
            self.assertEqual(collector.records[0]["status"], "failed")
            self.assertIn("API launch failed", collector.records[0]["reason"])


if __name__ == "__main__":
    unittest.main()
