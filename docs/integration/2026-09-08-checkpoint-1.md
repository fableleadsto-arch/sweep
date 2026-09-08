# SWEEP integration checkpoint 1

## Scope and provenance

Base commit: `8611d14022596d3a5e4dabaf06be8061cf602f9a`.
Development branch: `sweep/device-host-integration-20260908`. Main is not changed.

The full repository archive/clone could not be downloaded in the execution sandbox because GitHub DNS resolution failed. GitHub source inspection and branch writes work through the authenticated connection. This checkpoint uses TWO complete production source files, not a full checkout. Before editing, their bytes matched the GitHub blob hashes:

- `sweep_neural_mesh/sweep_api.py`: `3b8354f92801544b6603288c9f87ffc0af52b423`
- `sweep_neural_mesh/cortex_integration.py`: `a0fd7ee45edbb9d5fda354495c07e9b641ecc10d`

The changes reuse the earlier guarded reliability patch, rather than introducing another engine. They are a small first integration checkpoint; the wider architecture audit and reconciliation remain incomplete.

## Changes

1. SweepAPI honors explicit initialization failure, clears failed state, permits retry, and logs exception types rather than exception messages. Package-relative imports are preferred; standalone imports remain supported. This method no longer changes sys.path.
2. When no inference path returns a result, the existing cortex returns UNKNOWN / unavailable with zero compatibility confidence instead of echoing the query with confidence 0.3.
3. Added 16 tests that load the complete source modules. Dependency doubles are explicit; package __init__, heavy providers and full application startup are NOT exercised.

The zero compatibility score is NOT a calibrated probability of falsity. An initializer returning True is not proof of loaded/working models. Cortex initialization, direct inference exception handling, model isolation and calibration still need separate work.

## Initial architectural findings

- `sweep_neural_mesh/intent/__init__.py`: IntentMesh already connects classified intent to mesh capabilities, but executes only the first available capability. Registering several capabilities does not implement an ordered device plan.
- `companion/planning.py`: existing deterministic request/intent planning; keep it as an integration point, not a replacement brain.
- `companion/orchestrator.py`: existing ToolRegistry and plan/execute/observe loop. A returned dictionary is currently wrapped as ok=True regardless of a nested failure result; tool-input schemas are advertised but not enforced by this registry. Request tool filtering limits advertised specifications, not necessarily execution. These need adversarial tests before device actions are exposed. Iteration count does not bound every step in a model-generated batch.
- `companion/main.py` and `companion/config.py`: optional bearer authentication and wildcard CORS are unsuitable as the sole Device Host boundary. Dedicated device access must fail closed.
- `companion/execution.py`: generated-Python runner explicitly disclaims hostile-input containment. Do not use it to execute arbitrary model-generated OS commands. Its output truncation follows subprocess capture, not a hard capture-memory limit.
- The root pyproject testpaths points to tests, while inspected suites also live under companion/tests and sweep_neural_mesh/tests. Explicit suite discovery is needed; a default pytest result alone must not imply full coverage.
- Frontend files and entry points remain to be reconciled. app/main.py identifies itself as the C++ terminal UI backend, while an older audit describes React. Do not assume the older architecture document is current.

## Reconciliation ledger

- Prior neural reliability patch: API initialization and cortex fallback integrated here. Core node, execution engine and analysis-cache changes remain pending comparison/integration.
- Evidence storage/ingestion: earlier bundle retained; repository app/evidence currently has store.py and scoring.py. Real data contracts and research call sites must be reconciled before import.
- Strict quality gate: retained and regression-tested separately; not yet wired into this repository's evaluations.
- Workstation 0.1.1: retained and regression-tested separately; not merged as a second app package. Its standalone contract shim must not replace production app/core/types.py.
- Full cognitive platform and Device Host: not implemented by this checkpoint.

## Minimal Device Host direction (proposed, not implemented)

Extend existing planning/tool integration with structured action proposals. A separate local enforcement boundary owns action risk, schema validation, capabilities, permissions and approval state. The planner cannot approve itself or provide authoritative risk/confirmation fields. Each step must be independently authorized. Approvals must bind to the exact request, expire, and resist replay.

Start disabled, with simulation and dry-run separated from actual execution. Capability discovery must reflect the OS adapter's real availability. Do not mark a launch completed merely because a process-start request was accepted. Keep generated Python, credentials and arbitrary executable paths outside this API. Existing general-purpose routes must not become a bypass into the host. No host listeners or OS actions are enabled here.

## Test execution

Command:

```sh
python -m unittest discover -s sweep_neural_mesh/tests -p test_api_failure_contract.py -v
```

- Before patch: 16 test methods, 5 passed, 11 failed; unittest reports 14 failure events because one method has several failing subtests.
- After patch: all 16 passed (0.028 seconds in this run).
- Previous bundles rerun separately: reliability 48 passed, evidence/strict gate 68 passed, Workstation 36 passed. These are preservation checks, NOT the full production test suite or a model benchmark.
- Existing production neural contract tests were inspected, not run. They require model metadata, inference modules and (for inference tests) model dependencies/weights unavailable in this snapshot. No tests were weakened or replaced to conceal this limit.

## PHASE 0 MILESTONE REPORT

Phase: 0 — integration baseline.
Milestone: First two prior reliability fixes integrated into pinned production files.
Status: PARTIAL
Implementation: Two existing modules changed; no parallel engine.
Tests: 16 new source-module regressions; three earlier bundle suites rerun separately.
Integration: Actual API/cortex no-provider path tested together; full package/application integration untested.
Documentation: This audit, reconciliation ledger and milestone report.
Tests passed: 16 checkpoint tests; 48 + 68 + 36 separate preservation tests.
Tests failed: 0 after patch. Baseline had 11 failing methods / 14 failure events.
Demonstration: Missing providers return unavailable instead of an echoed Open Chrome request. This does not launch Chrome or prove any device action.
Measured results: Deterministic software regressions only.
Baseline: Initialization failure accepted and missing-provider query echoed.
Current: Explicit initialization failure rejected; unavailable inference reported.
Target: Honest failure propagation before permissioned actions are added.
Known limitations: Partial source snapshot, no full boot, no Windows execution, no models loaded, no accuracy/training measurement.
Known failures: Full checkout download blocked by sandbox DNS.
Dependencies satisfied: Repository read/write access, source integrity checks, guarded edits, targeted test execution.
Dependencies remaining: Complete audit/source environment, remaining reconciliation, full relevant suites, Device Host implementation and OS acceptance tests.
Ready for dependent phases: NO
