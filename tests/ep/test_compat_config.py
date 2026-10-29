"""Host-only checks for the compatibility planner and durable run records."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import compat_config as config


class PlannerTests(unittest.TestCase):
    def args(self, *extra):
        return config.parse_args(list(extra))

    def plan(self, *extra, world=8):
        return config.plan_cases(self.args(*extra), world)

    def test_defaults_cover_cutoffs_and_keep_cpu_wait_out_of_default(self):
        cases = self.plan()
        ordinary = [case for case in cases if not case["edge_routes"]]
        self.assertEqual({case["bs"] for case in ordinary}, {2, 32, 33, 64, 65, 128, 256})
        self.assertTrue(all(case["M"] == 256 and case["cpu_sync"] is False for case in cases))
        self.assertEqual({case["deferred"] for case in cases}, {False, True})
        self.assertEqual({case["deterministic"] for case in cases}, {False, True})
        self.assertTrue(all(case["barrier_in_prologue"] is False for case in cases))
        self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
        json.dumps(cases, allow_nan=False)

    def test_cpu_filter_preserves_other_cases_and_ids(self):
        unsynchronized = self.plan("--cpu-sync", "false")
        all_modes = self.plan("--cpu-sync", "all")
        self.assertEqual(unsynchronized, [case for case in all_modes if case["cpu_sync"] is False])
        self.assertEqual(len(all_modes), 3 * len(unsynchronized))
        self.assertEqual({type(case["cpu_sync"]) for case in all_modes}, {bool, type(None)})

    def test_capacity_changes_are_explicit_and_group_wide(self):
        cases = self.plan("--bs", "2", "32", "--capacity-mode", "both", "--edge-routes", "off")
        for case in cases:
            self.assertEqual(case["M"], 256 if case["capacity_mode"] == "fixed" else case["bs"])
            self.assertEqual(case["rank_bs"], [case["bs"]] * 8)
        self.assertEqual({case["M"] for case in cases}, {2, 32, 256})

    def test_route_seed_is_independent_of_transport_and_feature_configuration(self):
        cases = self.plan("--capacity-mode", "both", "--cpu-sync", "all",
                          "--dtype", "bf16", "fp8", "--scale-layout", "row", "col")
        for case in cases:
            self.assertEqual(case["seed"], 23 if case["edge_routes"] else case["bs"])
        self.assertTrue(self.args("--no-huge1g-page").no_huge1g_page)

    def test_edge_pattern_uses_one_shared_capacity(self):
        cases = self.plan("--edge-routes", "only", "--capacity-mode", "both", world=16)
        for case in cases:
            self.assertEqual(case["rank_bs"], [0, 2, 128, 32] * 4)
            self.assertEqual(case["bs"], 128)
            self.assertEqual(case["M"], 256 if case["capacity_mode"] == "fixed" else 128)
        self.assertTrue(all(case["edge_routes"] for case in cases))

    def test_edge_auto_can_be_disabled_and_does_not_expand_small_batch_selection(self):
        self.assertFalse(any(case["edge_routes"] for case in self.plan("--edge-routes", "off")))
        self.assertFalse(any(case["edge_routes"] for case in self.plan("--bs", "2", "32")))

    def test_fp8_layouts_expand_without_repeating_bf16(self):
        cases = self.plan("--bs", "2", "--dtype", "bf16", "fp8", "--scale-layout", "row", "col")
        bf16 = [case for case in cases if case["dtype"] == "bf16"]
        fp8 = [case for case in cases if case["dtype"] == "fp8"]
        self.assertEqual(len(fp8), 2 * len(bf16))
        self.assertEqual({case["scale_layout"] for case in bf16}, {"row"})
        self.assertEqual({case["scale_layout"] for case in fp8}, {"row", "col"})

    def test_exact_case_selection_and_repeated_batches(self):
        cases = self.plan("--bs", "2", "2", "--edge-routes", "off")
        first, second = cases[:2]
        self.assertNotEqual(first["case_id"], second["case_id"])
        self.assertEqual(
            self.plan("--bs", "2", "2", "--edge-routes", "off", "--case-id", second["case_id"]),
            [second],
        )
        with self.assertRaisesRegex(ValueError, "Unknown --case-id"):
            self.plan("--case-id", "missing")

    def test_reject_invalid_capacity_world_and_irrelevant_scale_request(self):
        invalid = (
            ("--capacity", "0"),
            ("--bs", "0"),
            ("--bs", "257"),
            ("--capacity-mode", "batch", "--bs", "257"),
            ("--edge-routes", "only", "--capacity", "64"),
            ("--num-processes", "0"),
            ("--scale-layout", "col"),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.plan(*arguments)
        for world in (0, -1, 3, 256):
            with self.subTest(world=world), self.assertRaises(ValueError):
                self.plan(world=world)

    def test_node_count_world_convention(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            cases = config.plan_cases(self.args("--bs", "2"))
        self.assertEqual(len(cases[0]["rank_bs"]), 16)

    def test_small_worlds_are_fallback_diagnostics_without_truncated_edge_pattern(self):
        for world in (1, 2):
            with self.subTest(world=world):
                cases = self.plan(world=world)
                self.assertFalse(any(case["edge_routes"] for case in cases))
                with self.assertRaisesRegex(ValueError, "at least four ranks"):
                    self.plan("--edge-routes", "only", world=world)
        edge = self.plan("--edge-routes", "only", "--capacity-mode", "batch", world=4)[0]
        self.assertEqual(edge["M"], 128)

    def test_payload_identity_limits_are_about_actual_b_not_arena_capacity(self):
        with self.assertRaisesRegex(ValueError, "test-data restriction, not a library limit"):
            self.plan("--dtype", "fp8", "--bs", "257", "--capacity", "8192")
        with self.assertRaisesRegex(ValueError, "test-data restriction, not a library limit"):
            self.plan("--dtype", "bf16", "--bs", "4097", "--capacity", "8192")
        fp8 = self.plan("--dtype", "fp8", "--bs", "256", "--capacity", "8192")
        self.assertTrue(all(case["M"] == 8192 for case in fp8))
        bf16 = self.plan("--dtype", "bf16", "--bs", "4096", "--capacity", "8192")
        self.assertTrue(all(case["M"] == 8192 for case in bf16))
        edge_only = self.plan("--edge-routes", "only", "--bs", "99999",
                              "--capacity", "128", "--dtype", "bf16", "fp8")
        self.assertTrue(all(case["bs"] == 128 and case["M"] == 128 for case in edge_only))

    def test_help_and_dry_run_do_not_import_torch(self):
        helper_dir = str(Path(config.__file__).resolve().parent)
        script = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def no_device_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torch_npu', 'deep_ep'):
        raise AssertionError('unexpected device dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = no_device_import
import compat_config
raise SystemExit(compat_config.main(sys.argv[2:]))
"""
        env = dict(os.environ, WORLD_SIZE="1", PYTHONDONTWRITEBYTECODE="1")
        for arguments in (["--help"], ["--dry-run", "--bs", "2", "--edge-routes", "off"]):
            result = subprocess.run([sys.executable, "-c", script, helper_dir, *arguments],
                                    text=True, capture_output=True, env=env, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            if "--help" in arguments:
                self.assertIn("--profile-paths", result.stdout)
            else:
                self.assertGreater(json.loads(result.stdout)["num_cases"], 0)


class ReportingTests(unittest.TestCase):
    def test_backend_configuration_preserves_official_identity(self):
        env = {config.TRANSPORT_ENV: "1", "KEEP": "value"}
        official = config.configure_backend("official", env)
        self.assertTrue(official["requires_official_build"])
        self.assertEqual(env[config.TRANSPORT_ENV], "1")
        config.configure_backend("candidate-auto", env)
        self.assertNotIn(config.TRANSPORT_ENV, env)
        config.configure_backend("candidate-disabled", env)
        self.assertEqual(env[config.TRANSPORT_ENV], "1")
        self.assertEqual(env["KEEP"], "value")

    def test_records_flush_and_existing_rank_file_is_never_replaced(self):
        case = config.plan_cases(config.parse_args(["--bs", "2"]), 8)[0]
        with tempfile.TemporaryDirectory() as directory:
            with config.JsonlReport(directory, 3, {"commit": "test"}) as report:
                report.record("started", case)
                self.assertEqual(len(report.path.read_text(encoding="utf-8").splitlines()), 1)
                report.record("pass", case, duration_seconds=0.1)
                report.record("fail", case, error="diagnostic")
                before = report.path.read_bytes()
                with self.assertRaises(FileExistsError):
                    config.JsonlReport(directory, 3)
                self.assertEqual(report.path.read_bytes(), before)
                records = [json.loads(line) for line in before.decode("utf-8").splitlines()]
                self.assertEqual([entry["status"] for entry in records], ["started", "pass", "fail"])
                self.assertTrue(all(entry["rank"] == 3 and entry["case_id"] == case["case_id"]
                                    for entry in records))
            with config.JsonlReport(directory, 4) as other_rank:
                other_rank.record("started", case)

    def test_bad_records_cannot_corrupt_identity_or_leave_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with config.JsonlReport(directory, 0) as report:
                with self.assertRaises(ValueError):
                    report.record("unknown")
                with self.assertRaises(ValueError):
                    report.record("pass", rank=99)
                with self.assertRaises(TypeError):
                    report.record("pass", result=object())
                self.assertEqual(report.path.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
