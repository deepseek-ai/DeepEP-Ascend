"""Host-only tests for complete, partial, malformed and inconsistent reports."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from summarize_compat import main, summarize


def metadata(world=2):
    return dict(backend="candidate-auto", world=world,
                runtime=dict(arch=3510, num_vec_cores=64, local_rank=0),
                build=dict(source_commit="8d41d2f", build_label="candidate",
                           package_sha256="a" * 64, extension_sha256="b" * 64,
                           harness_sha256="c" * 64, tracked_changes=[],
                           harness_files_sha256={"compat_config.py": "d" * 64},
                           runtime_files_sha256={"buffers/ep.py": "e" * 64}),
                arguments=dict(profile_paths=False), transport_override=None)


def operation(rank, case_id, kind):
    name = f"{kind}_impl<device kernel>"
    return dict(label="fresh" if kind == "dispatch" else "return", operation=kind,
                backend="candidate", status="verified", profiling_enabled=True,
                expected_transport="mte", actual_transport="mte",
                metadata=dict(rank=rank, case_id=case_id),
                kernel_names=[name], kernel_parameters={name: dict(transport="mte")},
                device_kernel_count=1,
                device_kernels=[dict(name=name, device_id=str(rank), stream_id="1")],
                evidence_format="kernel_details_csv", file_sha256={"remote/kernel_details.csv": "d" * 64})


def records(rank, *, world=2, plan=("case-a", "case-b"), verified=False):
    identity = metadata(world)
    identity["runtime"]["local_rank"] = rank
    result = []

    def add(status, case=None, **extra):
        result.append(dict(rank=rank, metadata=deepcopy(identity), status=status,
                           case_id=None if case is None else case["case_id"], case=case, **extra))

    add("started", scope="run", planned_case_ids=list(plan))
    for case_id in plan:
        case = dict(case_id=case_id, bs=16)
        add("started", case, correctness="pending", path_status="pending")
        add("pass", case, correctness="passed", path_status="verified" if verified else "unverified",
            operations=[operation(rank, case_id, kind) for kind in ("dispatch", "combine")]
            if verified else [])
    add("pass", scope="run", completed_case_count=len(plan))
    return result


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, rank, rows=None):
        path = self.root / f"rank-{rank:05d}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in (rows if rows is not None else records(rank))),
                        encoding="utf-8")
        return path

    def full(self, *, verified=False):
        for rank in range(2):
            self.write(rank, records(rank, verified=verified))

    def test_complete_correctness_without_paths(self):
        self.full()
        result = summarize(self.root)
        self.assertTrue(result["accepted"])
        self.assertEqual((result["status"], result["correctness"], result["path_status"]),
                         ("pass", "passed", "unverified"))
        self.assertFalse(summarize(self.root, require_paths=True)["accepted"])

    def test_complete_verified_paths(self):
        self.full(verified=True)
        result = summarize(self.root, require_paths=True)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["path_status"], "verified")

    def test_missing_rank_is_incomplete(self):
        self.write(0)
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_ranks"], [1])
        self.assertFalse(result["accepted"])

    def test_missing_terminal_run_pass(self):
        self.full()
        self.write(1, records(1)[:-1])
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["complete"])

    def test_failed_case(self):
        self.full()
        rows = records(1)[:3]
        rows[-1].update(status="fail", correctness="failed")
        self.write(1, rows)
        result = summarize(self.root)
        self.assertEqual(result["correctness"], "failed")
        self.assertEqual(result["status"], "fail")

    def test_verified_label_without_trace_is_unverified(self):
        self.full(verified=True)
        rows = records(1, verified=True)
        rows[2]["operations"] = []
        self.write(1, rows)
        result = summarize(self.root, require_paths=True)
        self.assertEqual(result["correctness"], "passed")
        self.assertEqual(result["path_status"], "unverified")
        self.assertFalse(result["accepted"])

    def test_incomplete_embedded_trace_is_unverified(self):
        self.full(verified=True)
        rows = records(1, verified=True)
        del rows[2]["operations"][0]["file_sha256"]
        self.write(1, rows)
        result = summarize(self.root, require_paths=True)
        self.assertEqual(result["path_status"], "unverified")
        self.assertFalse(result["accepted"])

    def test_conflicting_plan(self):
        self.write(0)
        self.write(1, records(1, plan=("case-a", "case-c")))
        result = summarize(self.root)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("conflicting planned_case_ids across ranks", result["errors"])

    def test_conflicting_case_definition(self):
        self.full()
        rows = records(1)
        rows[1]["case"]["bs"] = 32
        rows[2]["case"]["bs"] = 32
        self.write(1, rows)
        result = summarize(self.root)
        self.assertFalse(result["accepted"])
        self.assertIn("conflicting case definition for case-a", result["errors"])

    def test_build_hash_conflict_is_reported_and_rejected(self):
        self.full()
        rows = records(1)
        for row in rows:
            row["metadata"]["build"]["extension_sha256"] = "e" * 64
        self.write(1, rows)
        result = summarize(self.root)
        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["build_variants"]["extension_sha256"]), 2)
        self.assertIn("conflicting build.extension_sha256 across ranks", result["errors"])

    def test_source_commit_and_build_label_conflicts(self):
        for field, value in (("source_commit", "different-commit"), ("build_label", "other-build")):
            with self.subTest(field=field):
                self.full()
                rows = records(1)
                for row in rows:
                    row["metadata"]["build"][field] = value
                self.write(1, rows)
                result = summarize(self.root)
                self.assertFalse(result["accepted"])
                self.assertIn(f"conflicting build.{field} across ranks", result["errors"])

    def test_helper_and_runtime_file_hash_conflicts(self):
        for field in ("harness_files_sha256", "runtime_files_sha256"):
            with self.subTest(field=field):
                self.full()
                rows = records(1)
                for row in rows:
                    hashes = row["metadata"]["build"][field]
                    hashes[next(iter(hashes))] = "f" * 64
                self.write(1, rows)
                result = summarize(self.root)
                self.assertFalse(result["accepted"])
                self.assertIn(f"conflicting build.{field} across ranks", result["errors"])
                self.assertEqual(len(result["build_variants"][field]), 2)

    def test_file_hash_mappings_must_be_present_nonempty_and_valid(self):
        for field in ("harness_files_sha256", "runtime_files_sha256"):
            for invalid in (None, {}, [], {"helper.py": "bad"}, {"": "f" * 64}):
                with self.subTest(field=field, invalid=invalid):
                    self.full()
                    rows = records(1)
                    for row in rows:
                        if invalid is None:
                            del row["metadata"]["build"][field]
                        else:
                            row["metadata"]["build"][field] = invalid
                    self.write(1, rows)
                    result = summarize(self.root)
                    self.assertEqual(result["status"], "invalid")
                    self.assertTrue(any(field in error for error in result["errors"]))

    def test_candidate_transport_override_conflict(self):
        self.full()
        rows = records(1)
        for row in rows:
            row["metadata"]["transport_override"] = "1"
        self.write(1, rows)
        result = summarize(self.root)
        self.assertFalse(result["accepted"])
        self.assertIn("conflicting candidate transport_override across ranks", result["errors"])
        self.assertEqual(len(result["transport_override_variants"]), 2)

    def test_candidate_transport_override_presence_and_consistent_disabled(self):
        for invalid in ("missing", 1):
            with self.subTest(invalid=invalid):
                self.full()
                rows = records(1)
                for row in rows:
                    if invalid == "missing":
                        del row["metadata"]["transport_override"]
                    else:
                        row["metadata"]["transport_override"] = invalid
                self.write(1, rows)
                self.assertEqual(summarize(self.root)["status"], "invalid")
        for rank in range(2):
            rows = records(rank)
            for row in rows:
                row["metadata"].update(backend="candidate-disabled", transport_override="1")
            self.write(rank, rows)
        self.assertTrue(summarize(self.root)["accepted"])

    def test_metadata_conflicts(self):
        for field, value in (("world", 3), ("backend", "official")):
            with self.subTest(field=field):
                self.full()
                rows = records(1)
                for row in rows:
                    row["metadata"][field] = value
                self.write(1, rows)
                self.assertEqual(summarize(self.root)["status"], "invalid")

    def test_truncated_json_and_missing_newline(self):
        for suffix in ('{"rank":', '{"rank": 0}'):
            with self.subTest(suffix=suffix):
                self.full()
                with (self.root / "rank-00001.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(suffix)
                self.assertEqual(summarize(self.root)["status"], "invalid")

    def test_malformed_json_records(self):
        for contents in ('{"rank":\n', '[]\n', '{"rank": 1, "rank": 0}\n', '{"rank": NaN}\n'):
            with self.subTest(contents=contents):
                self.full()
                (self.root / "rank-00001.jsonl").write_text(contents, encoding="utf-8")
                self.assertEqual(summarize(self.root)["status"], "invalid")

    def test_invalid_state_sequences(self):
        variants = {}
        rows = records(1)
        variants["terminal_without_start"] = rows[:1] + rows[2:]
        variants["case_missing"] = rows[:3] + rows[-1:]
        variants["terminal_count"] = deepcopy(rows)
        variants["terminal_count"][-1]["completed_case_count"] = 1
        variants["unknown_status"] = deepcopy(rows)
        variants["unknown_status"][2]["status"] = "ok"
        variants["incorrect_correctness"] = deepcopy(rows)
        variants["incorrect_correctness"][2]["correctness"] = "pending"
        variants["extra_terminal"] = rows + rows[-1:]
        for name, bad_rows in variants.items():
            with self.subTest(name=name):
                self.full()
                self.write(1, bad_rows)
                self.assertEqual(summarize(self.root)["status"], "invalid")

    def test_inflight_case_and_empty_directory_are_incomplete(self):
        self.assertEqual(summarize(self.root)["status"], "incomplete")
        self.write(0, records(0)[:2])
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("case case-a has no terminal record", result["ranks"][0]["incomplete_reasons"])

    def test_file_rank_mismatch(self):
        self.full()
        self.write(1, records(0))
        self.assertEqual(summarize(self.root)["status"], "invalid")

    def test_cli_json_and_exit_codes(self):
        self.full()
        for require_paths, expected in ((False, 0), (True, 1)):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--input-dir", str(self.root)] + (["--require-paths"] if require_paths else []))
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(output.getvalue())["accepted"], expected == 0)


if __name__ == "__main__":
    unittest.main()
