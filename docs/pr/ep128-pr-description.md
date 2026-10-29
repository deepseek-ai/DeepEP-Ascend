# Restrict native low-latency transport to the validated EP128 policy

The candidate transport was selected at multiple EP sizes although the measured
results do not support enabling it across those configurations. Restrict native
automatic MTE selection to EP128 while retaining the tested B/M, hardware and
format guards. Eligible EP128 inputs use MTE; other native inputs retain URMA.
Dispatch and combine share the policy and preserve their existing handle and
fallback semantics. FP8 dispatch remains URMA.

The aggregate diff also includes the experimental low-latency and Decode APIs,
topology/core validation and build changes from retained commit `8d41d2f`.
The second commit consolidates the compatibility harness, EP128 restriction,
host tests, documentation and build/packaging repairs. The unmeasured CPU-sync
draft and dynamic-routing research are excluded. See the
[scope and commit boundaries](ep128-scope.md).

Validation completed locally: 130 Python host tests with no skips, two standalone
C++ policy/layout programs, 51 Python syntax parses, shell syntax checks and a
source-archive inventory with mocked SDK dependencies. Official `main` resolved
to `3b25377d04b24fc6154698ded78a2bcb2c59afff` on 2026-10-21 and the merge
simulation had no conflicts.

Before merge, complete the [acceptance checklist](ep128-premerge-checklist.md).
Actual Ascend builds, device correctness/transport/stream tests and final-commit
performance remain pending. The [offline review](ep128-offline-review.md)
also records the unresolved CPU-sync default behavior, inherited repeated-expert
capacity boundary, retained-API scope decision and third-party provenance
questions. Local host passes do not close these items.
