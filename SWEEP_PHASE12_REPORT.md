# SWEEP Phase 12 Report — Executive Controller

## PHASE NUMBER
12

## OBJECTIVE
Implement the central agent layer (spec module 13): goal interpretation, task state, plan decomposition, tool execution, verification-before-completion (module 17), error diagnosis (module 18), and replanning.

## WHAT CHANGED
Created `sweep_cognitive/executive/` with:

- **`ExecutiveController`** — drives the closed loop plan → execute → observe → verify → diagnose → replan (spec section 64)
- **`Plan` / `Step`** — steps carry an explicit `expected_state` verification contract; plans are versioned; superseded plans are kept
- **`ToolRegistry`** — the tool-schema boundary (hard logic is correct here per spec section 0); only registered handlers can execute
- **`TaskRecord`** — persistent task state with a full audit log
- **`ErrorClass`** — the 12-category failure taxonomy from module 18, with deterministic exception→class mapping
- **Replanning** — degraded plan versions (new version, old kept), step omission for missing tools, a replan budget, and safe user handoff (`NEEDS_USER`) when the budget is exhausted or verification cannot be met

## KEY DESIGN DECISIONS (all test-enforced)
1. **Verification gate:** a task only COMPLETED if no step failed/blocked AND at least one step actually succeeded *with real verification*. A plan where everything was skipped (all tools missing) produces nothing and must not "complete".
2. **No redefining success:** when a tool runs but the intended state never occurs, the controller does NOT degrade the verification contract to force completion — the honest outcome is user handoff with the diagnosis. (A test caught the original degraded-path hole: unverified "success" was passing the gate.)
3. **Unavailable modalities/tools fail honestly:** missing tool = PLANNING error, logged, never faked.
4. **Retries** happen only up to `max_attempts`, and only for transient tool/environment errors; verification failures are not retried into success.

## FILES CREATED
- `sweep_cognitive/executive/__init__.py`
- `sweep_cognitive/tests/test_executive.py` — 19 tests

## FILES MODIFIED
None outside the new package.

## ARCHITECTURE CHANGES
- The cognitive loop now has a driver. The controller's tool contract is deliberately engine-agnostic: any `handler(params) -> result` works, so the perception/memory/prediction/hypothesis modules can be wired as tools in later phases.
- Goal decomposition is template scaffolding (`retrieve → perform → verify`), explicitly marked; the learned planner slots into `_decompose` without changing contracts.

## MODEL CHANGES
None.

## DATA CHANGES
None.

## TRAINING CHANGES
None.

## BENCHMARK CHANGES
None yet (Phase 20). Serves the PLANNING, REPLANNING, ERROR RECOVERY, and VERIFICATION benchmarks.

## PERFORMANCE BEFORE
345 tests passing (end of Phase 11).

## PERFORMANCE AFTER
364 tests passing, 0.43s total runtime. 19 new tests.

## KNOWN FAILURES
None. Three real design bugs were caught by tests and fixed:
1. **All-skipped plans completed:** when no tools were registered, every step failed with PLANNING errors, was skipped in replanning, and `_all_critical_verified` counted all-skipped as success. Fixed: completion now also requires at least one genuinely verified success.
2. **Degraded verification passed the gate:** after replanning stripped `expected_state` from a failing step, its unverified success satisfied the gate via other template steps. Fixed: verification failures now lead to user handoff instead of contract degradation.
3. **Replan budget message overwritten:** the env-error strategy branch ran after the budget check and replaced the handoff summary. Fixed by moving the budget check first with early return.

## NEW RISKS
- Template decomposition only handles the investigation pattern; diverse task types need the learned planner or richer templates.
- `_classify_exception` is name/message based; ambiguous exceptions default to TOOL error.

## REGRESSIONS
None. Full suite green.

## NEXT BOTTLENECK
The controller plans and executes, but always uses the same computation depth. Phase 13 (adaptive routing) selects computation mode per task using the Phase 8 predictions.

## NEXT PHASE
Phase 13 — Adaptive computation routing.
