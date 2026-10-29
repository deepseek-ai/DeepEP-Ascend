# EP128 native transport: pre-merge checklist

This plan separates completed offline checks from pending device acceptance.
See the [offline review](ep128-offline-review.md) for results and limitations.
Host tests performed while preparing the change do not establish NPU correctness,
an actual Ascend build, or acceptance of the final commit. Device commands are
templates for a separately scheduled run; this document does not resume the
stopped experiments.

## 1. Scope, history, and upstream review

- [x] Statically check that the native automatic MTE policy is restricted to **global EP128**. Keep
  `1 <= B <= 32`, `B <= M <= 256`, H7168/K8/E128, BF16, arch3510, 64 AIV cores,
  and at least 190 KiB available UB. Other EP sizes and ineligible inputs retain
  the native URMA path. Do not remove the B/M gates on the strength of an EP128
  result. See [the policy](../../csrc/kernels/ep/low_latency_policy.hpp).
- [ ] Preserve the native public API, handle layout, routing metadata, cached
  dispatch, combine, events, and supported input semantics. FP8 dispatch remains
  URMA; its subsequent BF16 combine can be MTE when its original B/M and all
  other guards qualify.
- [x] Relative to the pinned baseline below, retain exactly two commits:
  retain `8d41d2f9738dbbb6f4e3aae9ac17d6c08be4d48e` unchanged and combine the
  subsequent work, including the former `53e8370` changes and this repair, into
  the new second commit. Preserve the old history through the recorded bundle
  and old remote ref. Do not claim that `53e8370` remains a separate final commit.
- [x] Statically review the complete two-commit diff, not only the final policy repair.
  Review DecodeBuffer/LowLatencyBuffer exports and `_C` registration separately:
  the native EP path does not require the standalone low-latency implementation,
  but removing only its kernels while keeping unconditional bindings breaks the
  build. Record whether those additions belong in this PR or require a separate
  API decision; do not silently remove already exposed functionality.
- [x] Statically review `build_support.py`, `setup.py`, `MANIFEST.in`, both build
  scripts, tests and README together. Verify the pinned backend-header overlay
  and mocked-SDK source-archive inventory; repair the issues recorded in the
  offline review.
- [ ] Verify actual supported build environments and distribution contents.
  Resolve third-party provenance and license notices for all retained copied
  code, including test references. Host packaging tests do not close these items.
- [x] Query and fetch `https://github.com/deepseek-ai/DeepEP-Ascend.git`.
  On 2026-10-21 its default branch and `main` both resolved to
  `3b25377d04b24fc6154698ded78a2bcb2c59afff`. It is an ancestor of the candidate;
  a merge simulation returned the candidate tree with no conflicts.
- [ ] Recheck the target immediately before upstream integration. Any later
  upstream change needs a new conflict review and acceptance evidence.
- [ ] Obtain maintainer agreement on the retained experimental APIs and resolve
  the third-party notice questions in the offline review before publication.

Run these history assertions only on the final clean candidate checkout:

```bash
BASE=3b25377d04b24fc6154698ded78a2bcb2c59afff
RETAINED=8d41d2f9738dbbb6f4e3aae9ac17d6c08be4d48e
test "$(git rev-parse HEAD~1)" = "$RETAINED"
test "$(git rev-list --count "$BASE..HEAD")" = 2
git diff --check "$BASE..HEAD"
git log --format=fuller "$BASE..HEAD"
git diff --stat "$BASE..HEAD"
```

## 2. Isolate the three identities

| Backend argument | Actual package to load | Transport switch |
| --- | --- | --- |
| `official` | Clean official baseline `3b25377d04b24fc6154698ded78a2bcb2c59afff` | Unset `EP_DISABLE_LOW_LATENCY_TRANSPORT` before launch |
| `candidate-disabled` | Clean final candidate, freshly built | Harness sets the switch to `1` |
| `candidate-auto` | The same final candidate build | Harness removes the switch |

- [ ] Use separate clean source checkouts/build artifacts for official and
  candidate, and separate output directories for all three backends and reruns.
  The backend option neither changes branches nor installs a package.
  Candidate-disabled is not a substitute for the official baseline.
- [ ] Freeze one external harness for all three: `test_ep_auto_compat.py`,
  `compat_config.py`, `compat_trace.py`, and `low_latency_reference.py` from
  `tests/ep`. Also freeze `summarize_compat.py` and the stream harness used below.
  Record file hashes. Do not copy candidate production code into the official
  checkout to make the harness run.
- [ ] Record the actual imported package path, source commit, tracked changes,
  extension SHA256, harness hashes, build log, Python/Torch/torch_npu versions,
  CANN/compiler/driver/firmware versions, relevant environment, ordered hosts,
  rank-to-device mapping, and device capabilities. Require a clean intended
  source tree; the harness records dirty tracked files but does not reject them.
- [ ] Verify `--expected-source-commit` against the imported source checkout.
  A build label alone is not identity proof; the commit check alone does not
  prove a stale extension was rebuilt. Keep the build log and binary hash too.
  Wheel imports outside a Git checkout need separately verified wheel provenance;
  do not pretend a label supplies a Git identity.

## 3. Build the source import and test the distributions

Run in the selected, isolated Linux build environment with its approved CANN,
Torch, torch_npu, compiler, and initialized dependencies. Repeat for each source
identity under review. `REPO` must be a clean Git checkout; `BUILD_OUT` must be a
new directory outside it. These commands intentionally fail on errors.

```bash
set -euo pipefail
: "${REPO:?Set the clean source checkout}"
: "${EXPECTED_COMMIT:?Set its full commit SHA}"
: "${BUILD_OUT:?Set a new absolute artifact directory}"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
mkdir "$BUILD_OUT"
export BUILD_OUT REPO
git status --porcelain > "$BUILD_OUT/source-status.txt"
git submodule status --recursive > "$BUILD_OUT/submodules.txt"
python setup.py build_ext --inplace 2>&1 | tee "$BUILD_OUT/build-inplace.log"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY' | tee "$BUILD_OUT/import-identity.txt"
import hashlib, pathlib, deep_ep, deep_ep._C
root = pathlib.Path(__import__('os').environ['REPO']).resolve()
for name, value in [('package', deep_ep.__file__), ('extension', deep_ep._C.__file__)]:
    path = pathlib.Path(value).resolve()
    assert path.is_relative_to(root), (name, path, root)
    print(name, path, hashlib.sha256(path.read_bytes()).hexdigest())
PY
python setup.py sdist --dist-dir "$BUILD_OUT/dist" 2>&1 | tee "$BUILD_OUT/build-sdist.log"
python setup.py bdist_wheel --dist-dir "$BUILD_OUT/dist" 2>&1 | tee "$BUILD_OUT/build-wheel.log"
```

`build.sh` runs `bdist_wheel`; it does not install the wheel or create an in-place
extension. Both build entry scripts now stop on failure. The explicit commands
above are the source-import acceptance path. Do not reuse an unrelated `_C.so`.

- [ ] Verify the candidate wheel contains its extension, every shipped `.h` and
  `.hpp` under `deep_ep/include`, and both third-party license files. Verify the
  sdist includes `build_support.py` and all dependencies needed for its build.
  Execute the following candidate distribution checks; a failure is an open
  packaging issue, not permission to skip an absent file:

```bash
python - <<'PY'
import os, pathlib, tarfile, zipfile
root = pathlib.Path(os.environ['REPO'])
out = pathlib.Path(os.environ['BUILD_OUT'])
wheels = list((out / 'dist').glob('*.whl'))
sdists = list((out / 'dist').glob('*.tar.gz'))
assert len(wheels) == len(sdists) == 1, (wheels, sdists)
with zipfile.ZipFile(wheels[0]) as archive:
    names = set(archive.namelist())
    assert any(n.startswith('deep_ep/_C') and n.endswith('.so') for n in names)
    headers = {p.relative_to(root).as_posix() for p in (root / 'deep_ep/include').rglob('*')
               if p.suffix in ('.h', '.hpp')}
    assert headers and headers <= names, sorted(headers - names)
    for suffix in ('cann.LICENSE', 'ascend-deepep.LICENSE'):
        assert any(n.endswith('/' + suffix) for n in names), suffix
with tarfile.open(sdists[0], 'r:gz') as archive:
    assert any(n.endswith('/build_support.py') for n in archive.getnames())
print('Candidate distribution inventory passed; rebuild/import checks remain separate.')
PY
SDIST=$(python -c 'import os,pathlib; p=list((pathlib.Path(os.environ["BUILD_OUT"])/"dist").glob("*.tar.gz")); assert len(p)==1; print(p[0])')
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$BUILD_OUT/from-sdist" "$SDIST" 2>&1 | tee "$BUILD_OUT/rebuild-sdist.log"
WHEEL=$(python -c 'import os,pathlib; p=list((pathlib.Path(os.environ["BUILD_OUT"])/"dist").glob("*.whl")); assert len(p)==1; print(p[0])')
python -m venv --system-site-packages "$BUILD_OUT/wheel-venv"
"$BUILD_OUT/wheel-venv/bin/python" -m pip install --no-deps --force-reinstall "$WHEEL"
(
  cd "$BUILD_OUT"
  env -u PYTHONPATH "$BUILD_OUT/wheel-venv/bin/python" -c \
    'import pathlib,sys,deep_ep,deep_ep._C; p=pathlib.Path(sys.prefix).resolve(); assert pathlib.Path(deep_ep.__file__).resolve().is_relative_to(p); assert pathlib.Path(deep_ep._C.__file__).resolve().is_relative_to(p); print(deep_ep.__file__, deep_ep._C.__file__)'
)
```

The dependency-bearing environment is reused by `--system-site-packages`; the
assertions require `deep_ep` itself to come from the new wheel environment.
Review the official baseline's distribution against its own manifest rather than
requiring candidate-only helper files there. Archive distribution SHA256 values.

## 4. Host gates

- [x] Run the host suites and standalone C++ policy/layout checks on the prepared
  source and save the output. The offline review records the result.
- [ ] Repeat applicable host checks in the target Linux build environment and
  save exit codes and skips against the final commit. No NPU result follows
  from these passes.

```bash
python -B -m unittest discover -s tests -p 'test_build*.py'
(cd tests/ep && python -B -m unittest \
  test_compat_config test_compat_trace test_ep_auto_compat_host test_summarize_compat)
(cd tests/comm && python -B -m unittest test_stream_host)
(cd tests/ep && python -B -m unittest test_low_latency_host test_decode_host test_public_api_compat_host)
HOST_BUILD=$(mktemp -d)
c++ -std=c++17 -I. tests/ep/low_latency_policy_test.cpp -o "$HOST_BUILD/ep-policy"
"$HOST_BUILD/ep-policy"
c++ -std=c++17 -Ideep_ep/include tests/ep/low_latency_layout_test.cpp -o "$HOST_BUILD/ll-layout"
"$HOST_BUILD/ll-layout"
```

The planner/parser/state checks use the standard library. Some synthetic tensor
checks require CPU Torch and explicitly skip without it; record those skips.
Preview each device plan with the same arguments plus `--dry-run` before launch.

## 5. Distributed launch contract

`WORLD_SIZE` is the **node count**, `RANK` is the **node index**, and
`--num-processes` is the number of local NPUs. Do not wrap these scripts in
`torchrun` or set the local process count to the global EP size.

| Global EP | Nodes (`WORLD_SIZE`) | Local NPUs |
| --- | ---: | ---: |
| 128 | 16 | 8 |
| 64 | 8 | 8 |
| 32 | 4 | 8 |
| 16 | 2 | 8 |
| 8 | 1 | 8 |
| 4 | 1 | 4 |
| 2 | 1 | 2 |

- [ ] Before each authorized device job, verify that **all eight NPUs on every
  participating host** are idle, including hosts used for EP2/4. Abort if any NPU
  is occupied or its idle state cannot be confirmed. Never terminate another
  user's processes.
- [ ] Use the same ordered hosts, devices, input plan, runtime, and harness for
  each three-way comparison. Set one shared `MASTER_ADDR` and `MASTER_PORT`, and
  the appropriate node `RANK` on every node. A global EP comparison is not an
  assertion that different EP topologies are equivalent.
- [ ] Use a controller with a bounded whole-job deadline, per-node logs and
  owned-process-group cleanup. The compatibility harness has a 600-second
  process-group timeout but no `--timeout` CLI option. A local timeout or local
  `mp.spawn` failure does not clean up siblings on other nodes. Any failed node
  must stop the other processes owned by that job before a new job starts.

Source the selected environment and set the variables below on **every** node.
Use a new `OUT_ROOT` for each backend, EP size, and rerun. The GNU `timeout`
budget shown is an example; choose and record a finite budget for the actual
plan. Exit 124/137, missing ranks, or missing final markers are failures.

```bash
set -euo pipefail
: "${REPO:?}" "${HARNESS:?}" "${BACKEND:?}" "${EXPECTED_COMMIT:?}"
: "${WORLD_SIZE:?}" "${RANK:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"
: "${LOCAL_NPUS:?}" "${OUT_ROOT:?}"
export WORLD_SIZE RANK MASTER_ADDR MASTER_PORT
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset EP_DISABLE_LOW_LATENCY_TRANSPORT
cd "$HARNESS"
COMMON=(
  --backend "$BACKEND" --num-processes "$LOCAL_NPUS"
  --expected-source-commit "$EXPECTED_COMMIT"
  --build-label "$BACKEND-$EXPECTED_COMMIT"
  --capacity 256 --capacity-mode fixed
  --cpu-sync false --barrier-in-prologue false
  --alignment 1 --deferred false --deterministic false
  --route random --dtype bf16 --scale-layout row --edge-routes off
)
run_compat() {
  local label="$1"; shift
  timeout --signal=TERM --kill-after=30s 60m \
    python -B "$HARNESS/test_ep_auto_compat.py" \
    "${COMMON[@]}" "$@" --output-dir "$OUT_ROOT/$label"
}
```

Each command below is a separate distributed job: launch it on all selected
nodes and collect the result before proceeding. The common arguments make
defaults explicit; later arguments override the selected scalar/list options.

## 6. Functional matrix: all three backends

| Pending gate | EP coverage | Acceptance focus |
| --- | --- | --- |
| [ ] Boundary/reuse | 128, 64, 32, 16, 8, 4, 2 | EP128 B32/33 switch; other EPs retain URMA; repeated small/large/small B and fixed/batch M |
| [ ] Feature combinations | 128 and the other supported EPs above | Alignment, deterministic layout, deferred completion, random/hotspot routes, optional weights and bias |
| [ ] FP8 | 128 and the other supported EPs above | Row/column scale packs, padding, cached payload/scale updates, BF16 combine |
| [ ] Uneven/empty ranks | 128, 64, 32, 16, 8, 4 | Empty ranks, masked/duplicate slots, unequal B with one group-wide M; EP128 mixed eligible/ineligible ranks |
| [ ] Capacity guard | 128 and a representative fallback EP | Small B with M257 must retain URMA |
| [ ] CPU sync true and API default | Separately on 128 and a fallback EP, then the supported deployment matrix | Explicit True and None behavior, not only False |
| [ ] Cross-stream/shared layer | 128, 64, 32, 16, 8; fallback shape at 4/2 | Producer/consumer ordering, events and allocation lifetime |

```bash
# 40 planned cases: retain the full sequence to exercise old-handle reuse.
run_compat boundary --bs 2 32 33 256 2 --capacity-mode both \
  --barrier-in-prologue both --deferred both

# 72 planned cases; weights, metadata, cached calls and bias are checked inside each case.
run_compat features --bs 2 33 2 --alignment 1 16 128 \
  --deferred both --deterministic both --route random hotspot

# 16 planned cases, with BF16 combine after FP8 dispatch.
run_compat fp8 --bs 2 32 33 2 --capacity-mode both \
  --dtype fp8 --scale-layout row col

# Four planned cases; valid only with global EP >= 4 and capacity >= 128.
run_compat edge --bs 128 --edge-routes only \
  --barrier-in-prologue both --deferred both

run_compat capacity-257 --bs 2 --capacity 257
run_compat cpu-sync-true --bs 2 --cpu-sync true
run_compat cpu-sync-default --bs 2 --cpu-sync default
```

The harness fixes H7168/K8/E128. `--capacity` is the allocation capacity of the
reused buffer. `--capacity-mode batch` passes M=max(B across ranks) to each call;
it does **not** recreate the allocation at M=B. Fixed mode passes M=capacity.
All planned B values must fit `--capacity`. FP8 test payloads require B<=256 and
BF16 payloads B<=4096; these are reference-data limits, not library limits.

- [ ] Confirm all embedded checks passed: per-rank/per-expert counts, padding,
  metadata and deterministic ordering; fresh and cached calls with/without
  weights; cached payload/scale/weight updates without metadata mutation;
  weighted/unweighted expert outputs, 0/1/2 bias tensors, optional returned
  weights without double weighting; deferred hook completion exactly once.
- [ ] Retain a first handle and consume it after intervening B/M changes in the
  same buffer session. Do not split every case into independent jobs and claim
  equivalent old-handle coverage. `--case-id` filters require the same originating
  plan and are useful for diagnosis, not a replacement for the reuse sequence.
- [ ] Treat CPU-sync acceptance as **open**. Earlier official and candidate
  `do_cpu_sync=True` jobs both timed out. Both failing is not a pass, and does not
  identify a candidate-only regression. Resolve or explicitly disposition the
  common failure before claiming full compatibility. Do not silently include an
  unreviewed CPU-sync repair in this policy change. Run None (`default`) separately;
  cached dispatch in this harness explicitly uses False.

## 7. Verify the actual device path

- [ ] Capture real device profiler evidence for fresh dispatch, cached dispatch,
  and combine. On EP128 use boundary, FP8, capacity and uneven-rank cases below;
  on the other EPs use the boundary sample to confirm URMA. Run all three backends.

```bash
run_compat paths-boundary --bs 2 32 33 2 --profile-paths
run_compat paths-boundary-barrier --bs 2 32 33 2 \
  --barrier-in-prologue true --profile-paths
run_compat paths-fp8 --bs 2 --dtype fp8 --scale-layout row col --profile-paths
run_compat paths-capacity-257 --bs 2 --capacity 257 --profile-paths
run_compat paths-edge --bs 128 --edge-routes only --profile-paths
```

Use `paths-edge` only at EP>=4. Profile samples are deliberately small because
each operation is profiled. The parser must find actual dispatch/combine device
kernels, validate template fields (including M, UB, cached, CPU sync, barrier,
dtype/scale packs and weights), and reject missing or unexpected transport.
CPU labels or an epilogue alone are not path evidence. Correct values with an
unexpected fallback do not pass the intended MTE path gate. Preserve raw traces,
`evidence.json`, and hashes, not only a console summary.

## 8. Streams and shared communication

Freeze `tests/comm/test_stream.py` from the same test revision. Set
`STREAM_HARNESS` to its containing directory; use the same source identity and
distributed environment as above. Record import/binary provenance externally:
this script has no `--expected-source-commit` or `--output-dir` option.

- [ ] Confirm the target runtime supplies a callable `torch.npu._sleep` before
  scheduling this harness. The previously inspected torch_npu 2.12 environment
  did not supply it. The harness explicitly registers torch_npu and rejects a
  missing delay primitive before distributed initialization; it does not supply
  an alternative device delay. A supported device-side delay implementation and
  its asynchronous behavior must be validated before using this test in that
  environment. A host sleep or a synchronized replacement would not exercise
  the intended ordering race. The commands below require this gate to be closed.

```bash
: "${STREAM_HARNESS:?}"
timeout --signal=TERM --kill-after=30s 30m \
  python -B "$STREAM_HARNESS/test_stream.py" \
  --num-processes "$LOCAL_NPUS" --backend "$BACKEND" \
  --ep-shape both --ep-bs 16 --ep-capacity 256 --barrier-in-prologue false
# Run as a separate job with --barrier-in-prologue true as well.
```

- [ ] At EP2/4 use `--ep-shape fallback` instead. At EP8/16/32/64 the legacy `mte`
  shape name means H7168/K8/E128 test inputs; the new policy still selects URMA.
- [ ] Keep the default EventOverlap and Bucket smoke checks; `--skip-bucket` is
  only a diagnostic isolation option. Check producer/consumer waits, delayed
  production, early input release, allocation pressure, immediate/deferred
  completion, and restoration of the current stream.
- [ ] Do not add the profiler to this stream test: its synchronization changes
  the behavior being tested. Establish transport separately with matching inputs.
  The correctness harness also synchronizes operations and cannot by itself prove
  overlap or asynchronous lifetime safety.
- [ ] Review and execute any required existing PP/Engram/shared-layer regressions
  affected by the full two-commit diff. A Bucket smoke pass is not their result.

## 9. Coverage requiring additional cases or manual review

- [ ] Extend statistics coverage beyond the existing two fresh dispatches, which
  already assert counts and then 2x counts. Check longer accumulation sequences,
  caller reset/reuse across changing B/M/routes, empty-rank behavior, and the
  documented unsupported/omitted-stats behavior of cached dispatch. Do not pass
  fresh-only statistics arguments to cached dispatch and call that compatibility.
- [ ] Resolve the inherited expanded-output capacity boundary for repeated expert
  IDs before claiming arbitrary duplicate-route support. With CPU sync disabled,
  EP128/E128/K8 and B=M=32, routing all eight slots to expert 0 needs 32768 rows
  on rank 0, while the baseline capacity formula allocates 4096 rows. This is a
  static counterexample, not a new device reproduction. The bounded edge fixture
  does not cover this worst case. See the offline review; do not run this case
  unguarded on a device or treat the existing fixture as proof of general safety.
- [ ] Exercise other supported H/K/E shapes and relevant hardware/core/UB
  configurations through existing native suites or explicit added cases. The
  compatibility CLI has no H/K/E flags. Check legal fallback behavior without
  inferring coverage from the fixed H7168/K8/E128 matrix. Include physical
  28-AIC/56-AIV versus 32-AIC/64-AIV configurations, legal smaller `num_sms`
  selections, rejection of invalid overrides, and cached calls retaining the
  core configuration saved in their handle. Do not claim MTE eligibility on
  physical 56-AIV hardware.
- [ ] Add public API negative tests for unsupported dtype/device/shape/stride,
  invalid routing/capacity, and invalid option combinations. Keep collective
  participation controlled so an expected local validation failure cannot leave
  peers hanging. Planner argument rejection alone is not native API validation.
- [x] Review constructor/method signatures, optional defaults, return arity,
  exported names and handle fields against the official API. The 13 signature
  fixtures and public-wrapper host tests passed.
- [ ] Verify ownership/lifetime on devices. Keep standalone/Decode API claims
  separate from native EPBuffer acceptance.
- [ ] Perform one controlled performance regression validation after functional
  acceptance: same final source identities, inputs, ordered topology, warmup/JIT
  exclusion and native timing method, with all-rank data and the documented max-
  rank aggregation. Preserve raw data and independent repeats for any decision.
  Profiler correctness runs and old-commit timings do not validate final-commit
  performance. This item does not authorize restarting the stopped matrix.

## 10. Evidence and final sign-off

Collect all nodes' globally named `rank-*.jsonl` files into one job directory,
without overwriting files or mixing reruns. After collection:

```bash
: "${COLLECTED_JOB:?Set the collected directory for one complete job}"
python -B "$HARNESS/summarize_compat.py" --input-dir "$COLLECTED_JOB"
# For a job launched with --profile-paths, require device evidence:
python -B "$HARNESS/summarize_compat.py" --input-dir "$COLLECTED_JOB" --require-paths
```

- [ ] Require every planned rank/case, valid identity, normal end markers, and
  successful checks. Without profiling, path status stays unverified. Missing
  ranks, truncated logs, started-but-unfinished cases, timeouts or failures must
  remain visible; never remove failed rows to make a summary pass.
- [ ] Preserve raw profiler files with the collected evidence. The summarizer
  validates embedded device evidence and hashes; it does not independently reopen
  every former node-local trace path. Independently check the archived raw files.
- [ ] Record per job: backend, commit and binary/harness hashes, topology, complete
  arguments/environment, case IDs, start/end, exit code, counts/skips, evidence
  location and checksum, and any unresolved issue. Reruns use new directories and
  retain the original failure history.
- [ ] Resolve or explicitly record every open build/API/functional/CPU-sync gap
  before sign-off. Keep a separate result record for the final commit. Do not mark
  this plan, its dry runs, or preparation-time host passes as completed NPU
  acceptance.
