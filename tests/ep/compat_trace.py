"""Optional, fail-closed device-kernel evidence for the native EP compatibility suite.

Importing this module needs only the standard library. ``enabled=False`` records
an explicitly unverified operation and never imports Torch or touches an NPU.
Template parsing follows compare_20261019/compare_native.py, with additional
validation of the official and candidate template layouts.
"""

from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import re


class TraceVerificationError(RuntimeError):
    """The captured device evidence does not prove the requested transport."""


def _require(condition, message):
    if not condition:
        raise TraceVerificationError(message)


def _backend(value):
    if value in ("auto", "candidate-auto", "candidate-disabled"):
        value = "candidate"
    _require(value in ("candidate", "official"), f"Unknown backend: {value}")
    return value


def _operation(value):
    _require(value in ("dispatch", "combine"), f"Unknown operation: {value}")
    return value


def _is_main(name, operation):
    # An epilogue or a generic Python label is never a main-kernel symbol.
    return re.search(r"(?<![A-Za-z_])" + operation + r"_impl(?:I|<)", name) is not None


def template_values(name, operation):
    """Parse integral Itanium template arguments or a demangled specialization."""
    _operation(operation)
    symbol = operation + "_impl"
    if not _is_main(name, operation):
        return []
    if symbol + "I" in name:
        tail = name.split(symbol + "I", 1)[1]
        found = re.match(r"((?:L[bilmjy][n0-9]+E)+)", tail)
        if found:
            return [(kind, int(value.replace("n", "-")))
                    for kind, value in re.findall(r"L([bilmjy])([n0-9]+)E", found.group(1))]
    found = re.search(re.escape(symbol) + r"<([^<>]+)>", name)
    if found:
        result = []
        for token in found.group(1).split(","):
            token = token.strip()
            if token in ("true", "false"):
                result.append(("b", int(token == "true")))
            elif re.fullmatch(r"-?\d+[uUlL]*", token):
                result.append(("i", int(re.sub(r"[uUlL]+$", "", token))))
            else:
                return []
        return result
    return []


def parse_kernel(name, operation, backend="candidate"):
    """Return actual native-kernel parameters; unknown layouts are errors."""
    operation, backend = _operation(operation), _backend(backend)
    values = template_values(name, operation)
    minimum = 13 if operation == "dispatch" else 9
    _require(len(values) >= minimum, f"Unrecognized native {operation} kernel: {name}")

    def integer(index, label, *, positive=False):
        kind, value = values[index]
        _require(kind != "b", f"Unexpected bool for {label}: {name}")
        _require(not positive or value > 0, f"Invalid {label}: {name}")
        return value

    def boolean(index, label):
        kind, value = values[index]
        _require(kind == "b" and value in (0, 1), f"Invalid bool {label}: {name}")
        return bool(value)

    if operation == "dispatch":
        result = dict(num_ranks=integer(0, "num_ranks", positive=True),
                      num_experts=integer(1, "num_experts", positive=True),
                      num_topk=integer(2, "num_topk", positive=True),
                      expert_alignment=integer(3, "expert_alignment", positive=True),
                      cached_mode=boolean(4, "cached_mode"),
                      do_cpu_sync=boolean(5, "do_cpu_sync"),
                      num_vec_cores=integer(6, "num_vec_cores", positive=True),
                      num_hidden_bytes=integer(7, "num_hidden_bytes", positive=True),
                      num_sf_packs=integer(8, "num_sf_packs"),
                      num_max_tokens_per_rank=integer(9, "capacity", positive=True),
                      num_ub_bytes=integer(10, "num_ub_bytes", positive=True),
                      barrier_in_prologue=boolean(12, "barrier_in_prologue"))
        transport_index = 13
    else:
        result = dict(num_ranks=integer(0, "num_ranks", positive=True),
                      num_topk=integer(1, "num_topk", positive=True),
                      hidden=integer(2, "hidden", positive=True),
                      num_max_tokens_per_rank=integer(3, "capacity", positive=True),
                      num_vec_cores=integer(4, "num_vec_cores", positive=True),
                      num_ub_bytes=integer(6, "num_ub_bytes", positive=True),
                      barrier_in_prologue=boolean(7, "barrier_in_prologue"),
                      has_topk_weights=boolean(8, "has_topk_weights"))
        transport_index = 9

    if backend == "candidate":
        _require(len(values) > transport_index,
                 f"Missing candidate MTE template argument: {name}")
        low_latency = boolean(transport_index, "low_latency_transport")
    else:
        # Official dispatch has three integral derived defaults after arg 12;
        # candidate dispatch inserted a bool at arg 13. Official combine has 9
        # arguments. Reject a candidate symbol labelled as an official run.
        if operation == "dispatch":
            _require(len(values) == 16 and all(kind != "b" for kind, _ in values[13:]),
                     f"Kernel does not match the pinned official dispatch template: {name}")
        else:
            _require(len(values) == 9,
                     f"Kernel does not match the pinned official combine template: {name}")
        low_latency = False
    result.update(low_latency_transport=low_latency,
                  transport="mte" if low_latency else "urma")
    return result


def identify_transport(name, operation, backend="candidate"):
    return parse_kernel(name, operation, backend)["transport"]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value, label, *, positive=False):
    try:
        value = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as error:
        raise TraceVerificationError(f"Invalid device {label}: {value!r}") from error
    _require(math.isfinite(value) and (not positive or value > 0),
             f"Invalid device {label}: {value!r}")
    return value


def _normalized(row):
    return {re.sub(r"[\s_]+", "", str(key)).lower(): value
            for key, value in row.items() if key is not None}


def read_kernel_csv(path, operation):
    """Read only main device kernels from a profiler kernel_details CSV."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, raw in enumerate(csv.DictReader(stream), 2):
            row = _normalized(raw)
            name = row.get("name") or row.get("kernelname") or ""
            if not _is_main(name, operation):
                continue
            _require(str(row.get("deviceid", "")) != "" and str(row.get("streamid", "")) != "",
                     f"Device/stream ID missing in {path}:{line}")
            rows.append(dict(name=name, source=str(path), csv_line=line,
                             device_id=str(row["deviceid"]), stream_id=str(row["streamid"]),
                             start_us=_number(row.get("starttime(us)"), "start"),
                             duration_us=_number(row.get("duration(us)"), "duration", positive=True)))
    return rows


def read_chrome_trace(path, operation):
    """Accept Chrome X events explicitly categorized as device kernels.

    CPU record_function labels and launch calls are not kernel evidence, even if
    their names contain the full native symbol.
    """
    with Path(path).open(encoding="utf-8-sig") as stream:
        trace = json.load(stream)
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    _require(isinstance(events, list), f"Invalid Chrome trace: {path}")
    rows = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        categories = {item.strip().lower() for item in str(event.get("cat", "")).split(",")}
        if not categories.intersection({"kernel", "kernels", "npu_kernel", "aicore_kernel"}):
            continue
        name = event.get("name", "")
        if not isinstance(name, str) or not _is_main(name, operation):
            continue
        arguments = _normalized(event.get("args", {}))
        device_id = arguments.get("deviceid", event.get("pid"))
        stream_id = arguments.get("streamid", event.get("tid"))
        _require(device_id is not None and stream_id is not None,
                 f"Device/stream identity missing in {path} event {index}")
        rows.append(dict(name=name, source=str(path), trace_event=index,
                         device_id=str(device_id), stream_id=str(stream_id),
                         start_us=_number(event.get("ts"), "start"),
                         duration_us=_number(event.get("dur"), "duration", positive=True)))
    return rows


def verify_kernel_records(records, operation, backend="candidate", expected_transport=None):
    """Verify actual symbols and fail on absent, mixed or unexpected transports."""
    operation, backend = _operation(operation), _backend(backend)
    _require(expected_transport in (None, "mte", "urma"),
             f"Invalid expected transport: {expected_transport}")
    _require(bool(records), f"No {operation} main device kernel was captured")
    names = sorted({record["name"] for record in records})
    parameters = {name: parse_kernel(name, operation, backend) for name in names}
    transports = {value["transport"] for value in parameters.values()}
    _require(len(transports) == 1, f"Contradictory {operation} transports: {sorted(transports)}")
    actual = next(iter(transports))
    _require(expected_transport is None or actual == expected_transport,
             f"{operation}: expected {expected_transport}, captured {actual}")
    return dict(actual_transport=actual, kernel_names=names,
                kernel_parameters=parameters, device_kernel_count=len(records),
                device_kernels=records)


def verify_case_metadata(kernel_parameters, operation, metadata):
    """Cross-check only case fields that a main device symbol actually encodes.

    B, deferred execution, and FP8 scale strides are runtime inputs and cannot be
    proved by this parser. They remain explicitly separate from template checks.
    A fresh dispatch cpu_sync=None means True; a cached call defaults to False.
    """
    operation = _operation(operation)
    metadata = metadata or {}
    checked = set()
    fields = (
        ("M", "num_max_tokens_per_rank"),
        ("num_ranks", "num_ranks"),
        ("num_topk", "num_topk"),
        ("num_vec_cores", "num_vec_cores"),
        ("num_ub_bytes", "num_ub_bytes"),
        ("barrier_in_prologue", "barrier_in_prologue"),
    )
    if operation == "dispatch":
        fields += (("cached", "cached_mode"), ("alignment", "expert_alignment"),
                   ("num_experts", "num_experts"))
    else:
        fields += (("hidden", "hidden"), ("weight_side_channel", "has_topk_weights"))

    for name, parameters in kernel_parameters.items():
        for case_field, parameter in fields:
            if case_field in metadata and metadata[case_field] is not None:
                _require(parameters[parameter] == metadata[case_field],
                         f"{operation} template {parameter}={parameters[parameter]} differs from "
                         f"case {case_field}={metadata[case_field]}: {name}")
                checked.add(case_field)
        if "min_num_ub_bytes" in metadata:
            _require(parameters["num_ub_bytes"] >= metadata["min_num_ub_bytes"],
                     f"{operation} UB {parameters['num_ub_bytes']} is below "
                     f"{metadata['min_num_ub_bytes']}")
            checked.add("min_num_ub_bytes")
        if operation == "dispatch":
            if "cpu_sync" in metadata:
                requested = metadata["cpu_sync"]
                _require(requested in (None, False, True), f"Invalid cpu_sync metadata: {requested}")
                effective = not parameters["cached_mode"] if requested is None else requested
                _require(parameters["do_cpu_sync"] == effective,
                         f"dispatch template do_cpu_sync={parameters['do_cpu_sync']} "
                         f"differs from case effective cpu_sync={effective}")
                checked.add("cpu_sync")
            if "dtype" in metadata:
                dtype = metadata["dtype"]
                _require(dtype in ("bf16", "fp8"), f"Unsupported native dispatch metadata dtype: {dtype}")
                has_scales = parameters["num_sf_packs"] > 0
                _require(has_scales == (dtype == "fp8"),
                         f"dispatch template num_sf_packs={parameters['num_sf_packs']} "
                         f"does not match dtype={dtype}")
                checked.add("dtype")
                if "hidden" in metadata:
                    expected_bytes = metadata["hidden"] * (1 if dtype == "fp8" else 2)
                    _require(parameters["num_hidden_bytes"] == expected_bytes,
                             f"dispatch template hidden bytes {parameters['num_hidden_bytes']} "
                             f"do not match H={metadata['hidden']}, dtype={dtype}")
                    checked.add("hidden")
        if metadata.get("expected_num_sf_packs") is not None and operation == "dispatch":
            _require(parameters["num_sf_packs"] == metadata["expected_num_sf_packs"],
                     f"dispatch template scale packs {parameters['num_sf_packs']} "
                     f"do not match {metadata['expected_num_sf_packs']}")
            checked.add("expected_num_sf_packs")
    semantic_fields = {"B", "M", "cached", "dtype", "hidden", "num_ranks", "num_topk",
                       "num_experts", "num_vec_cores", "num_ub_bytes", "min_num_ub_bytes",
                       "barrier_in_prologue", "cpu_sync", "alignment", "weight_side_channel",
                       "deferred", "scale_layout", "expected_num_sf_packs"}
    not_observable = sorted((semantic_fields & set(metadata)) - checked)
    return dict(checked_fields=sorted(checked), not_observable_from_symbol=not_observable)


def parse_profile_directory(directory, operation, backend="candidate", expected_transport=None,
                            metadata=None):
    """Prefer device CSVs; otherwise parse exported Chrome kernel events."""
    directory = Path(directory)
    csv_paths = sorted(directory.rglob("kernel_details*.csv"))
    trace_paths = sorted(set(directory.rglob("trace_view.json")) |
                         set(directory.rglob("chrome_trace.json")))
    paths = csv_paths or trace_paths
    _require(bool(paths), f"No device profiler CSV or Chrome trace under {directory}")
    reader = read_kernel_csv if csv_paths else read_chrome_trace
    records = [row for path in paths for row in reader(path, operation)]
    result = verify_kernel_records(records, operation, backend, expected_transport)
    checks = verify_case_metadata(result["kernel_parameters"], operation, metadata)
    result.update(template_metadata_verification=checks,
                  evidence_format="kernel_details_csv" if csv_paths else "chrome_kernel_events",
                  csv_files=[str(path) for path in csv_paths],
                  trace_files=[str(path) for path in trace_paths],
                  file_sha256={str(path): _sha256(path)
                               for path in sorted(set(csv_paths + trace_paths))})
    return result


class TraceCollector:
    """Collect one operation per profile; each collector must be rank-local.

    Wrap the API call *and its event wait* inside ``capture``. The optional
    profiler synchronizes on exit to flush device records; use it for targeted
    path verification, not for overlap or performance measurements.
    """

    def __init__(self, output_dir=None, *, enabled=False, backend="candidate"):
        self.enabled = bool(enabled)
        self.backend = _backend(backend)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        _require(not self.enabled or self.output_dir is not None,
                 "Profiling requires a rank-local output directory")
        self.records = []

    @contextmanager
    def capture(self, label, operation, expected_transport=None, metadata=None):
        operation = _operation(operation)
        _require(expected_transport in (None, "mte", "urma"),
                 f"Invalid expected transport: {expected_transport}")
        record = dict(label=str(label), operation=operation, backend=self.backend,
                      status="unverified", profiling_enabled=self.enabled,
                      expected_transport=expected_transport, actual_transport=None,
                      metadata=dict(metadata or {}))
        self.records.append(record)
        if not self.enabled:
            record["reason"] = "Device profiling disabled; actual transport was not observed"
            yield record
            return

        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")[:100] or operation
        directory = self.output_dir / f"{len(self.records):05d}_{safe_label}"
        directory.mkdir(parents=True, exist_ok=False)
        record["profile_directory"] = str(directory)
        try:
            # Runtime imports only; host parser tests do not need Torch/CANN.
            import torch
            import torch_npu

            profiler = torch_npu.profiler
            config = profiler._ExperimentalConfig(
                export_type=profiler.ExportType.Text,
                profiler_level=profiler.ProfilerLevel.Level1,
                data_simplification=False)
            with profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                    on_trace_ready=profiler.tensorboard_trace_handler(str(directory), async_mode=False),
                    record_shapes=False, profile_memory=False, with_stack=False,
                    with_flops=False, with_modules=False, experimental_config=config) as captured:
                with torch.profiler.record_function("ep_compat::" + str(label)):
                    yield record
                    torch.npu.synchronize()
            if not list(directory.rglob("trace_view.json")) and hasattr(captured, "export_chrome_trace"):
                captured.export_chrome_trace(str(directory / "chrome_trace.json"))
            record.update(parse_profile_directory(
                directory, operation, self.backend, expected_transport, record["metadata"]))
            record["status"] = "verified"
        except Exception as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            (directory / "evidence.json").write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def write_manifest(self, path=None):
        """Write all observed/unverified operations; this does not certify correctness."""
        if path is None:
            _require(self.output_dir is not None, "Specify a manifest path")
            path = self.output_dir / "trace_manifest.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = dict(schema_version=1, backend=self.backend, profiling_enabled=self.enabled,
                        records=self.records)
        path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path
