"""Fail-closed aggregation of the compatibility runner's per-rank JSONL files.

Copy every node's rank-00000.jsonl-style reports into one directory. This tool
uses only the standard library; it does not import Torch or access an NPU.
Path verification uses the device evidence embedded by TraceCollector. Original
profiler files are identified by their recorded hashes, not reopened through
the originating node's absolute paths.
"""

import argparse
import json
from pathlib import Path
import re
import sys


BACKENDS = {"official", "candidate-disabled", "candidate-auto"}
BUILD_FIELDS = (
    "source_commit", "build_label", "package_sha256", "extension_sha256",
    "harness_sha256", "tracked_changes", "expected_source_commit", "torch", "torch_npu",
    "harness_files_sha256", "runtime_files_sha256",
)
HASH_FIELDS = ("package_sha256", "extension_sha256", "harness_sha256")
HASH_MAP_FIELDS = ("harness_files_sha256", "runtime_files_sha256")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metadata(metadata):
    require(isinstance(metadata, dict), "metadata must be an object")
    require(metadata.get("backend") in BACKENDS, "invalid or missing metadata.backend")
    require(type(metadata.get("world")) is int and 0 < metadata["world"] <= 100000,
            "metadata.world must fit the five-digit rank filename space")
    for key in ("runtime", "build", "arguments"):
        require(isinstance(metadata.get(key), dict), f"metadata.{key} must be an object")
    build = metadata["build"]
    for key in ("source_commit", "build_label"):
        require(key in build and (build[key] is None or isinstance(build[key], str)),
                f"build.{key} must be a string or null")
    for key in HASH_FIELDS:
        require(is_hash(build.get(key)), f"invalid or missing build.{key}")
    for key in HASH_MAP_FIELDS:
        values = build.get(key)
        require(isinstance(values, dict) and bool(values)
                and all(isinstance(path, str) and bool(path.strip()) and is_hash(digest)
                        for path, digest in values.items()),
                f"build.{key} must be a nonempty file-to-SHA256 mapping")
    if metadata["backend"] != "official":
        require("transport_override" in metadata
                and (metadata["transport_override"] is None
                     or isinstance(metadata["transport_override"], str)),
                "candidate metadata.transport_override must be a string or null")


def check_paths(record, rank, case_id, backend):
    """A status label alone is insufficient device-path evidence."""
    claimed = record.get("path_status")
    require(claimed in ("verified", "unverified"), "invalid case path_status")
    operations = record.get("operations")
    if operations is None or operations == []:
        return "unverified", ["no operation device evidence"], 0
    require(isinstance(operations, list), "operations must be a list")
    reasons, observed, labels = [], set(), set()
    normalized_backend = "official" if backend == "official" else "candidate"
    for index, operation in enumerate(operations):
        require(isinstance(operation, dict), f"operation {index} must be an object")
        kind, label = operation.get("operation"), operation.get("label")
        require(kind in ("dispatch", "combine"), f"invalid operation {index} kind")
        require(isinstance(label, str) and label, f"missing operation {index} label")
        require((kind, label) not in labels, f"duplicate operation label {kind}/{label}")
        labels.add((kind, label))
        require(operation.get("status") in ("verified", "unverified", "failed"),
                f"invalid operation {index} status")
        require(operation["status"] != "failed", "passed case contains a failed operation")
        if operation["status"] != "verified":
            reasons.append(f"{kind}/{label}: device evidence unverified")
            continue
        evidence = operation.get("metadata", {})
        kernels = operation.get("device_kernels")
        names = operation.get("kernel_names")
        parameters = operation.get("kernel_parameters")
        hashes = operation.get("file_sha256")
        complete = (
            operation.get("profiling_enabled") is True
            and operation.get("backend") == normalized_backend
            and operation.get("actual_transport") in ("mte", "urma")
            and operation.get("expected_transport") in (None, "mte", "urma")
            and (operation.get("expected_transport") is None
                 or operation["actual_transport"] == operation["expected_transport"])
            and isinstance(evidence, dict)
            and evidence.get("rank") == rank and evidence.get("case_id") == case_id
            and isinstance(names, list) and bool(names)
            and all(isinstance(name, str) and name for name in names)
            and isinstance(parameters, dict) and set(parameters) == set(names)
            and all(isinstance(value, dict)
                    and value.get("transport") == operation["actual_transport"]
                    for value in parameters.values())
            and isinstance(kernels, list) and bool(kernels)
            and type(operation.get("device_kernel_count")) is int
            and operation["device_kernel_count"] == len(kernels)
            and all(isinstance(kernel, dict) and kernel.get("name") in names
                    and kernel.get("device_id") is not None
                    and kernel.get("stream_id") is not None for kernel in kernels)
            and operation.get("evidence_format") in ("kernel_details_csv", "chrome_kernel_events")
            and isinstance(hashes, dict) and bool(hashes)
            and all(isinstance(path, str) and path and is_hash(value)
                    for path, value in hashes.items())
        )
        if complete:
            observed.add(kind)
        else:
            reasons.append(f"{kind}/{label}: incomplete or inconsistent embedded device evidence")
    if observed != {"dispatch", "combine"}:
        reasons.append("both dispatch and combine device evidence are required")
    if claimed != "verified":
        reasons.append("case path_status is unverified")
    return ("unverified" if reasons else "verified"), reasons, len(operations)


def read_rank(path, rank):
    result = dict(rank=rank, file=str(path), metadata=None, plan=None, cases=[],
                  complete=False, errors=[], incomplete_reasons=[])
    active, ended = None, False
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    require(line.endswith("\n"), "unterminated JSONL line; possible truncation")
                    record = json.loads(line, parse_constant=reject_constant,
                                        object_pairs_hook=unique_object)
                    require(isinstance(record, dict), "record must be an object")
                    require(not ended, "record after terminal run record")
                    require(type(record.get("rank")) is int and record["rank"] == rank,
                            "record rank does not match filename")
                    check_metadata(record.get("metadata"))
                    if result["metadata"] is None:
                        result["metadata"] = record["metadata"]
                    require(record["metadata"] == result["metadata"],
                            "metadata changes within one rank report")
                    status, case_id, case = record.get("status"), record.get("case_id"), record.get("case")
                    require(status in ("started", "pass", "fail"), "invalid record status")
                    if case is None:
                        require(case_id is None and record.get("scope") == "run",
                                "case=None requires scope=run and case_id=None")
                        if status == "started":
                            require(line_number == 1 and result["plan"] is None,
                                    "run started must be the first and only start record")
                            plan = record.get("planned_case_ids")
                            require(isinstance(plan, list) and bool(plan)
                                    and all(isinstance(value, str) and value for value in plan)
                                    and len(set(plan)) == len(plan),
                                    "planned_case_ids must be nonempty unique strings")
                            result["plan"] = plan
                        else:
                            require(result["plan"] is not None, "terminal run record before start")
                            require(active is None, "terminal run record while a case is active")
                            require(status == "pass", "run failed")
                            require(type(record.get("completed_case_count")) is int
                                    and record["completed_case_count"] == len(result["plan"]),
                                    "completed_case_count does not match the plan")
                            require(len(result["cases"]) == len(result["plan"])
                                    and all(case["status"] == "pass" for case in result["cases"]),
                                    "run pass without every planned case passing")
                            ended = True
                            result["complete"] = True
                        continue
                    require(result["plan"] is not None, "case record before run started")
                    require(record.get("scope") in (None, "case"), "invalid case scope")
                    require(isinstance(case, dict) and isinstance(case_id, str)
                            and case.get("case_id") == case_id, "case identity mismatch")
                    if status == "started":
                        require(active is None, "case started before previous case finished")
                        position = len(result["cases"])
                        require(position < len(result["plan"])
                                and case_id == result["plan"][position], "case start differs from planned order")
                        require(record.get("correctness") == "pending"
                                and record.get("path_status") == "pending",
                                "case start requires pending correctness/path status")
                        active = record
                    else:
                        require(active is not None and active["case_id"] == case_id,
                                "case terminal record without matching start")
                        require(case == active["case"], "case definition changes between start and terminal")
                        require(record.get("correctness") == ("passed" if status == "pass" else "failed"),
                                "case correctness disagrees with status")
                        if status == "pass":
                            path_status, reasons, operation_count = check_paths(
                                record, rank, case_id, result["metadata"]["backend"])
                        else:
                            path_status, reasons, operation_count = "unverified", ["case failed"], 0
                        result["cases"].append(dict(case_id=case_id, case=case, status=status,
                            path_status=path_status, path_reasons=reasons, operation_count=operation_count))
                        active = None
                except (ValueError, TypeError, KeyError) as error:
                    result["errors"].append(f"{path.name}:{line_number}: {error}")
                    result["complete"] = False
                    break
    except (OSError, UnicodeError) as error:
        result["errors"].append(f"{path.name}: {error}")
        result["complete"] = False
    if result["plan"] is None:
        result["incomplete_reasons"].append("missing run started record")
    if active is not None:
        result["incomplete_reasons"].append(f"case {active['case_id']} has no terminal record")
    if not ended:
        result["incomplete_reasons"].append("missing terminal run pass after successful teardown")
    return result


def variants(reports, section, key):
    groups = {}
    for report in reports:
        metadata = report["metadata"]
        if metadata is None:
            continue
        value = metadata.get(key) if section is None else metadata.get(section, {}).get(key)
        group = groups.setdefault(canonical(value), dict(value=value, ranks=[]))
        group["ranks"].append(report["rank"])
    return list(groups.values())


def summarize(input_dir, require_paths=False):
    directory = Path(input_dir)
    reports, errors = [], []
    if not directory.is_dir():
        errors.append(f"input directory does not exist: {directory}")
    for path in sorted(directory.glob("rank-*.jsonl")):
        match = re.fullmatch(r"rank-(\d{5})\.jsonl", path.name)
        if match is None:
            errors.append(f"invalid rank report filename: {path.name}")
            continue
        reports.append(read_rank(path, int(match.group(1))))
    valid_metadata = [report for report in reports if report["metadata"] is not None]
    worlds = sorted({report["metadata"]["world"] for report in valid_metadata})
    backends = sorted({report["metadata"]["backend"] for report in valid_metadata})
    if len(worlds) > 1:
        errors.append(f"conflicting world sizes: {worlds}")
    if len(backends) > 1:
        errors.append(f"conflicting backends: {backends}")
    transport_variants = variants(reports, None, "transport_override")
    if len(backends) == 1 and backends[0] != "official" and len(transport_variants) > 1:
        errors.append("conflicting candidate transport_override across ranks")
    world = worlds[0] if len(worlds) == 1 else None
    observed = [report["rank"] for report in reports]
    expected = set(range(world)) if world is not None else set()
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected) if world is not None else []
    if unexpected:
        errors.append(f"rank files outside declared world: {unexpected}")
    plans = {canonical(report["plan"]) for report in reports if report["plan"] is not None}
    if len(plans) > 1:
        errors.append("conflicting planned_case_ids across ranks")
    plan = json.loads(next(iter(plans))) if len(plans) == 1 else []
    build_variants = {key: variants(reports, "build", key) for key in BUILD_FIELDS}
    for key, values in build_variants.items():
        if len(values) > 1:
            errors.append(f"conflicting build.{key} across ranks")
    runtime_keys = sorted({key for report in valid_metadata for key in report["metadata"]["runtime"]
                           if key != "local_rank"})
    runtime_variants = {key: variants(reports, "runtime", key) for key in runtime_keys}
    definitions, case_results = {}, {}
    for report in reports:
        errors.extend(report["errors"])
        for case in report["cases"]:
            case_id, definition = case["case_id"], canonical(case["case"])
            definitions.setdefault(case_id, set()).add(definition)
            case_results.setdefault(case_id, []).append((report["rank"], case))
    for case_id, values in definitions.items():
        if len(values) > 1:
            errors.append(f"conflicting case definition for {case_id}")
    complete = (world is not None and bool(reports) and not missing and not unexpected
                and all(report["complete"] for report in reports) and not errors)
    failed = any(case["status"] == "fail" for report in reports for case in report["cases"])
    correctness = "invalid" if errors else "failed" if failed else "passed" if complete else "incomplete"
    paths_verified = complete and all(case["path_status"] == "verified"
                                      for report in reports for case in report["cases"])
    path_status = "verified" if paths_verified else "unverified"
    accepted = correctness == "passed" and (not require_paths or paths_verified)
    status = ("pass" if accepted else "invalid" if errors else "fail" if failed
              else "unverified" if complete else "incomplete")
    cases = []
    for case_id in plan:
        results = case_results.get(case_id, [])
        cases.append(dict(case_id=case_id,
                          passed_ranks=[rank for rank, case in results if case["status"] == "pass"],
                          failed_ranks=[rank for rank, case in results if case["status"] == "fail"],
                          path_verified_ranks=[rank for rank, case in results
                                               if case["path_status"] == "verified"]))
    ranks = [dict(rank=report["rank"], file=report["file"], complete=report["complete"],
                  completed_case_count=len(report["cases"]),
                  failed_case_ids=[case["case_id"] for case in report["cases"] if case["status"] == "fail"],
                  unverified_paths=[dict(case_id=case["case_id"], reasons=case["path_reasons"])
                                    for case in report["cases"] if case["path_status"] != "verified"],
                  errors=report["errors"], incomplete_reasons=report["incomplete_reasons"])
             for report in reports]
    return dict(schema_version=1, input_dir=str(directory), status=status, accepted=accepted,
                complete=complete, correctness=correctness, path_status=path_status,
                require_paths=require_paths, backend=backends[0] if len(backends) == 1 else None,
                expected_world=world, observed_ranks=observed, missing_ranks=missing,
                unexpected_ranks=unexpected, planned_case_ids=plan, cases=cases, ranks=ranks,
                build_variants=build_variants, runtime_variants=runtime_variants, errors=errors,
                transport_override_variants=transport_variants,
                path_evidence_source="embedded TraceCollector records; original trace files not rehashed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing the rank reports collected from every node")
    parser.add_argument("--require-paths", action="store_true",
                        help="Require verified embedded device evidence in addition to correctness")
    args = parser.parse_args(argv)
    result = summarize(args.input_dir, args.require_paths)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
