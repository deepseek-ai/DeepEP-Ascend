"""Standard-library-only planning and result records for EP compatibility runs.

Importing this module, asking for help, and generating a dry-run plan never
imports Torch or initializes a device. WORLD_SIZE follows the existing launcher
convention: number of nodes, multiplied by --num-processes.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys


DEFAULT_BS = (2, 32, 33, 64, 65, 128, 256, 2)
BACKENDS = ("official", "candidate-disabled", "candidate-auto")
PRODUCTION_WORLDS = (8, 16, 32, 64, 128)
DIAGNOSTIC_WORLDS = (1, 2, 4)
TRANSPORT_ENV = "EP_DISABLE_LOW_LATENCY_TRANSPORT"


def create_parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=BACKENDS, default="candidate-auto")
    result.add_argument("--bs", nargs="+", type=int, default=list(DEFAULT_BS))
    result.add_argument("--capacity", type=int, default=256)
    result.add_argument("--capacity-mode", choices=("fixed", "batch", "both"), default="fixed")
    result.add_argument("--cpu-sync", choices=("false", "true", "default", "all"), default="false")
    result.add_argument("--barrier-in-prologue", choices=("false", "true", "both"), default="false")
    result.add_argument("--alignment", nargs="+", type=int, choices=(1, 16, 128), default=[1, 16])
    result.add_argument("--deferred", choices=("false", "true", "both"), default="both")
    result.add_argument("--deterministic", choices=("false", "true", "both"), default="both")
    result.add_argument("--route", nargs="+", choices=("random", "hotspot"), default=["random", "hotspot"])
    result.add_argument("--dtype", nargs="+", choices=("bf16", "fp8"), default=["bf16"])
    result.add_argument("--scale-layout", nargs="+", choices=("row", "col"), default=["row"])
    result.add_argument("--edge-routes", choices=("auto", "off", "only"), default="auto")
    result.add_argument("--num-processes", type=int, default=8)
    result.add_argument("--no-huge1g-page", action="store_true",
                        help="Preserve the existing launcher's ordinary-page allocation option.")
    result.add_argument("--output-dir", default="compat-results",
                        help="Results directory; an existing rank result is never overwritten.")
    result.add_argument("--expected-source-commit",
                        help="Expected source revision, independently checked by the device runner.")
    result.add_argument("--build-label",
                        help="Human-readable build identity saved alongside actual package evidence.")
    result.add_argument("--case-id", nargs="+", action="extend", default=[],
                        help="Run only these exact IDs from a dry-run plan; repeat the option if needed.")
    result.add_argument("--dry-run", action="store_true",
                        help="Print the complete JSON plan without importing Torch.")
    result.add_argument("--profile-paths", action="store_true",
                        help="Collect actual kernel-path evidence (implemented by the device runner).")
    return result


# The original test launchers use parser(); retain that spelling for callers.
parser = create_parser


def parse_args(argv=None):
    return create_parser().parse_args(argv)


def _unique(values):
    return list(dict.fromkeys(values))


def _booleans(value):
    return [False, True] if value == "both" else [value == "true"]


def _validate(args, world):
    if type(world) is not int or world <= 0 or 128 % world:
        raise ValueError("EP world size must be positive and divide the fixed 128 experts")
    if args.num_processes <= 0:
        raise ValueError("--num-processes must be positive")
    if args.capacity <= 0:
        raise ValueError("--capacity must be positive")
    if not args.bs or any(value <= 0 for value in args.bs):
        raise ValueError("--bs must contain positive token counts; empty ranks use --edge-routes")
    if args.edge_routes != "only" and max(args.bs) > args.capacity:
        raise ValueError("Every selected B must fit --capacity, including in batch capacity mode")
    if args.edge_routes == "only" and args.capacity < 128:
        raise ValueError("Edge cases require --capacity >= 128")
    if args.edge_routes == "only" and world < 4:
        raise ValueError("The edge pattern (0,2,128,32) requires at least four ranks")
    actual_max_b = 128 if args.edge_routes == "only" else max(args.bs)
    if "fp8" in args.dtype and actual_max_b > 256:
        raise ValueError(
            "FP8 B > 256 exceeds this test payload's exact row-identity encoding; "
            "this is a test-data restriction, not a library limit")
    if "bf16" in args.dtype and actual_max_b > 4096:
        raise ValueError(
            "BF16 B > 4096 exceeds this test payload's exact row-identity encoding; "
            "this is a test-data restriction, not a library limit")
    if set(args.dtype) == {"bf16"} and "row" not in args.scale_layout:
        raise ValueError("Column-major scales require --dtype fp8 (BF16 cases have no scales)")


def plan_cases(args, world=None):
    """Return reproducible JSON-safe case dictionaries; reject invalid inputs.

    ``bs`` is the maximum rank B used to construct reference routes. The device
    runner must use ``rank_bs[rank]`` for local inputs. M is one group-wide value.
    IDs depend on the case contents and repeated-case occurrence, not its index
    among unrelated cases, so selecting CPU-sync modes does not renumber cases.
    """
    if world is None:
        try:
            world = int(os.environ.get("WORLD_SIZE", "1")) * args.num_processes
        except ValueError as error:
            raise ValueError("WORLD_SIZE must be an integer node count") from error
    _validate(args, world)
    capacities = ["fixed", "batch"] if args.capacity_mode == "both" else [args.capacity_mode]
    sync_modes = {
        "false": [False], "true": [True], "default": [None], "all": [False, True, None],
    }[args.cpu_sync]
    batches = []
    if args.edge_routes != "only":
        batches.extend((bs, [bs] * world, False) for bs in args.bs)
    if args.edge_routes == "only" or (
        args.edge_routes == "auto" and world >= 4 and args.capacity >= 128 and max(args.bs) >= 128
    ):
        edge_bs = [(0, 2, 128, 32)[rank % 4] for rank in range(world)]
        batches.append((max(edge_bs), edge_bs, True))

    cases = []
    occurrences = defaultdict(int)
    common = itertools.product(
        capacities, sync_modes, _booleans(args.barrier_in_prologue),
        _unique(args.alignment), _booleans(args.deferred),
        _booleans(args.deterministic), _unique(args.route), _unique(args.dtype),
    )
    for mode, sync, barrier, alignment, deferred, deterministic, route, dtype in common:
        scale_layouts = _unique(args.scale_layout) if dtype == "fp8" else ["row"]
        for scale_layout, (bs, rank_bs, edge) in itertools.product(scale_layouts, batches):
            maximum = max(rank_bs)
            capacity = args.capacity if mode == "fixed" else maximum
            case = {
                "bs": bs,
                "rank_bs": list(rank_bs),
                "M": capacity,
                "capacity_mode": mode,
                "alignment": alignment,
                "deferred": deferred,
                "cpu_sync": sync,
                "deterministic": deterministic,
                "barrier_in_prologue": barrier,
                "route": route,
                "dtype": dtype,
                "scale_layout": scale_layout,
                "edge_routes": edge,
                "seed": 23 if edge else bs,
            }
            canonical = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
            repeat = occurrences[canonical]
            occurrences[canonical] += 1
            case["case_id"] = f"c-{digest}-{repeat}"
            cases.append(case)
    wanted = set(args.case_id)
    if wanted:
        known = {case["case_id"] for case in cases}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError("Unknown --case-id for this plan: " + ", ".join(unknown))
        cases = [case for case in cases if case["case_id"] in wanted]
    return cases


def configure_backend(backend, environ=None):
    """Configure only candidate transport selection; never impersonate a build.

    The runner must independently record/verify the loaded package and commit.
    For ``official`` this helper does not modify the diagnostic environment and
    does not claim that a candidate build has become an official build.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    target = os.environ if environ is None else environ
    if backend == "candidate-disabled":
        target[TRANSPORT_ENV] = "1"
    elif backend == "candidate-auto":
        target.pop(TRANSPORT_ENV, None)
    return {
        "backend": backend,
        "transport_override": target.get(TRANSPORT_ENV),
        "requires_official_build": backend == "official",
    }


class JsonlReport:
    """Exclusive per-rank JSONL output, flushed after every state transition."""

    def __init__(self, output_dir, rank, metadata=None):
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.rank = rank
        self.metadata = dict(metadata or {})
        json.dumps(self.metadata, allow_nan=False)
        self.path = Path(output_dir) / f"rank-{rank:05d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")

    def record(self, status, case=None, **fields):
        if status not in ("started", "pass", "fail"):
            raise ValueError("status must be started, pass, or fail")
        reserved = {"status", "rank", "case", "case_id", "timestamp_utc", "metadata"}
        if reserved.intersection(fields):
            raise ValueError("Result fields cannot replace reserved record identity fields")
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rank": self.rank,
            "status": status,
            "case_id": None if case is None else case["case_id"],
            "case": case,
            "metadata": self.metadata,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        return record

    write = record

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main(argv=None):
    argument_parser = create_parser()
    args = argument_parser.parse_args(argv)
    try:
        cases = plan_cases(args)
    except ValueError as error:
        argument_parser.error(str(error))
    if not args.dry_run:
        argument_parser.error("This helper only plans cases; use the device runner or --dry-run")
    json.dump({"backend": args.backend, "num_cases": len(cases),
               "expected_source_commit": args.expected_source_commit, "build_label": args.build_label,
               "world_size": len(cases[0]["rank_bs"]),
               "diagnostic_world": len(cases[0]["rank_bs"]) in DIAGNOSTIC_WORLDS,
               "cases": cases},
              sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
