# EP128 offline review

Reviewed on 2026-10-21 against official `main` at
`3b25377d04b24fc6154698ded78a2bcb2c59afff`. The official remote was queried and
fetched; the baseline is an ancestor of the candidate and the merge simulation
had no conflicts. Recheck upstream immediately before integration.

The [scope](ep128-scope.md) preserves the measured candidate strategy only for
native EP128, including its B/M, hardware and data-format guards. This review
did not change native dispatch/combine kernels or include the deferred CPU-sync
repair. The complete diff also retains the experimental APIs from `8d41d2f`;
it must not be described as a native-policy-only PR.

## Completed checks

| Check | Result and boundary |
| --- | --- |
| Python host suites | 130 tests passed, zero skipped: compatibility harness 71, stream harness 9, build/packaging 16, public/LL/Decode wrappers 34. Wrappers use fake extensions/tensors; harness checks do not execute collectives. |
| Official API signatures | 13 signature fixtures independently matched the official source. Existing arguments/defaults and native return/handle contracts were reviewed. |
| C++ policy and layout | Both standalone host programs compiled and passed with GCC 14.2. These do not compile Ascend kernels. |
| Syntax | 51 Python files parsed; `bash -n` passed for both build entry scripts. |
| Source distribution | Real setuptools archive generation with mocked SDK imports verified C++ sources/headers, device headers, a synthetic initialized DeepJIT header and declared notices. No extension was compiled. |
| DeepJIT overlay | The fixed backend-header hash matched; the isolated overlay was repeatable and left the source snapshot unchanged. |
| Native source review | Reviewed dispatch/combine completion, UB staging ownership, mixed eligible/ineligible ranks, cached handles, FP8, topology/core guards and stream captures. No definite new native MTE blocker was established statically. Hardware completion/visibility remains unverified. |

The host run used Windows, Python 3.14, GCC 14.2 and Git Bash. Bash could not
initialize inside the local sandbox; the affected local fixture tests and shell
syntax checks passed when rerun outside that sandbox. No server connection,
Ascend compilation, NPU execution or performance run was performed.

## Repairs made during this review

- `build.sh` and `develop.sh` now stop when a build fails. The development script
  no longer links a partial extension after a failing build, and selects the
  extension without a `find | head` pipeline. The wheel script's comment now
  describes its actual output.
- `MANIFEST.in` includes host-extension sources/headers and initialized DeepJIT
  headers needed to rebuild from a source archive.
- Generated environment configuration serializes values as Python literals,
  preserving quotes, backslashes and newlines.
- Added nine build/packaging regression tests and fifteen public-wrapper tests.
- Added stream-harness backend registration and device-delay preflight, with five
  host tests covering missing/callable delay APIs and failure before spawn/HCCL.

These repairs are included in the consolidated second commit. They do not
establish that an actual Ascend wheel or an sdist rebuild succeeds.

## Open review and acceptance items

1. **Device acceptance:** rebuild the final commit in the supported Ascend
   environment, verify source/binary identity and distributions, then run the
   three-backend functional, transport and stream matrix in the
   [pre-merge checklist](ep128-premerge-checklist.md). Repeat performance only
   after functional acceptance. Historical `53e8370` timings do not validate
   the final commit.
2. **Stream harness prerequisite:** the previously inspected torch_npu 2.12
   environment did not expose `torch.npu._sleep`. The harness now registers
   torch_npu explicitly and rejects an absent delay primitive before HCCL/spawn.
   It still needs a supported device-side delay in that environment; no host
   sleep or unvalidated timing substitute has been introduced.
3. **CPU synchronization:** fresh native dispatch still defaults from None to
   True. Earlier official and candidate True runs both timed out. False-only
   performance passes and host mocks do not close this issue. Validate True
   and the API default separately; the unmeasured repair remains outside this PR.
4. **Inherited repeated-expert capacity boundary:** static inspection found an
   official-baseline allocation problem when CPU synchronization is disabled.
   For EP128/E128/K8/H7168, B=M=32, alignment 1 and every token routing all eight
   slots to expert 0, rank 0 needs 32768 expanded rows, but the host formula
   `world_size * M * min(K, E / world_size)` allocates 4096. The corresponding
   BF16 tensor sizes are 469762048 and 58720256 bytes. The allocation in
   `csrc/buffers/ep.hpp` and expansion in
   `deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp` are unchanged
   from official `3b25377`. This is a static counterexample, not a device
   reproduction or a candidate-only regression. Do not run it unguarded, or
   infer arbitrary duplicate-route safety from the bounded edge fixture.
   Agree an explicit fix or supported-input disposition before making that claim.
5. **Retained API scope:** obtain maintainer agreement on the explicit
   low-latency/Decode APIs, topology helpers and build changes preserved by
   `8d41d2f`. The requested two-commit history keeps this commit unchanged;
   a narrower upstream PR would normally split these responsibilities.
6. **Third-party provenance:** the retained `comm_args.h` and
   `dispatch_dedup_types.h` under
   `deep_ep/include/deep_ep/impls/low_latency/dispatch/low_latency/common/`
   declare CANN Version 1.0, while `third-party/cann.LICENSE` is Version 2.0.
   The corresponding files at the documented Ascend/DeepEP source revision
   `8ffafebc0bf812e8479fda5139333bdd71c38a34` do not contain those CANN blocks.
   Confirm the original sources and applicable notices with maintainers; simply
   editing the version text would not establish provenance. The fixed DeepJIT
   revision `65f513952b8771408a19202f65e36398522db6db` also has no
   LICENSE/NOTICE/COPYING file in its tree, or license text in README/include.
   Clarify its distribution terms and preserve any required notices with the
   newly packaged headers. This is an unresolved provenance review, not a legal
   conclusion. No notices were removed or invented during this review.

Offline checks are complete for the recorded scope. Upstream approval,
provenance closure and device acceptance remain open; this document is not a
release sign-off. The checklist supplies executable host and device commands.
