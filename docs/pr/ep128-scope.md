# EP128 native transport scope

The default native `EPBuffer` now enables the measured candidate transport policy
only at EP128. The dispatch and combine implementations, wire layout, handles,
metadata and collective sequence remain those of candidate `53e8370`.

The shared policy allows MTE only for `1 <= B <= 32`, `B <= M <= 256`, arch3510,
64 AIVs, at least 190 KiB usable UB, H7168/K8/E128 and eligible BF16 data. All
other native cases use the existing URMA branch. FP8 dispatch uses URMA; the
following BF16 combine independently evaluates the original B and handle M.
`EP_DISABLE_LOW_LATENCY_TRANSPORT=1` still disables automatic MTE selection.

The EP128 comparison against official `3b25377` used candidate `53e8370` at
B=2/8/16/32/48/64/96/128/256/512/1024/2048/4096, with M=B, BF16, H7168/K8/E128,
alignment 1, no bias and CPU synchronization disabled. The first four points
used MTE; the remaining nine used URMA. This change preserves that strategy;
it does not claim that large-batch MTE or every public API combination was tested.
The final revision still needs its own device build, regression and performance
evidence. Follow the [pre-merge checklist](ep128-premerge-checklist.md).

## Commit and review boundaries

Relative to official baseline `3b25377d04b24fc6154698ded78a2bcb2c59afff`, the local
PR history contains two commits:

1. `8d41d2f9738dbbb6f4e3aae9ac17d6c08be4d48e`, preserved unchanged.
2. One consolidated commit containing the baseline-compatible test harness
   previously in `53e8370`, the EP128 restriction, matching host tests and docs,
   and the build-entry, source-distribution and configuration-serialization fixes
   identified during offline review.

Preserving the first commit also preserves its broader scope: the explicit
experimental low-latency API, Decode wrapper, topology/core validation and
build/packaging changes. The aggregate upstream diff therefore includes these
components as well as the native transport optimization. Reviewers must inspect
the full baseline-to-head diff, not just the second commit. The explicit
`low_latency_mode=True` API is not restricted to EP128 by this change.

The native MTE implementation is useful and should be retained. For a narrowly
scoped upstream performance PR, the independently useful build/topology changes
and experimental APIs would normally be split out. The requested two-commit
history deliberately keeps `8d41d2f` intact instead. Obtain maintainer agreement
on this aggregate scope before upstream submission; preserving a commit does
not establish that all its components have passed release validation.

The unmeasured CPU receive-count synchronization draft and personal design
files are excluded. The CPU-sync issue documented in the README remains an open
validation item, including the default API behavior. The passing CPU-sync-false
performance cases are not evidence that this issue is resolved.
