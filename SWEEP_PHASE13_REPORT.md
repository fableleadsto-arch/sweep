# SWEEP Phase 13 Report — Adaptive Computation Routing

## PHASE NUMBER
13

## OBJECTIVE
Implement spec module 14 (adaptive reasoning depth: FAST/ROUTINE/DELIBERATE/DEEP/EXPLORATORY with mode selection driven by ambiguity, consequence, uncertainty, complexity, time, compute, novelty, conflict) and spec section 12 (hardware-aware execution with graceful degradation).

## WHAT CHANGED
Created `sweep_cognitive/routing/` with:

- **`ComputationMode` + `ModeBudget`** — five modes, each with concrete budgets (retrieval depth, hypothesis count, simulation allowed, async allowed, verification strictness, time scale)
- **`RoutingSignals`** — explicit signal contract (complexity, ambiguity, uncertainty, consequence, novelty, evidence conflict, time pressure, compute headroom)
- **`AdaptiveRouter._decide`** — explainable decision table:
  1. time-pressure override (only for low-consequence work)
  2. consequence floor (≥0.6 → DELIBERATE; ≥0.8 → DEEP — destructive work is never rushed) and complexity floor (symmetric)
  3. weighted-score bands → FAST/ROUTINE/DELIBERATE
  4. DEEP conjunction rule (high complexity AND high uncertainty/conflict/consequence)
  5. EXPLORATORY split (DEEP + high ambiguity/novelty)
  6. compute ceiling (headroom < 0.3 caps at DELIBERATE)
  Every decision carries its reasons — routing is auditable and replayable.
- **`detect_hardware`** — stdlib-only detection (no new dependencies); unknowns stay unknown; operator overrides win; tier assignment FULL/STANDARD/CONSTRAINED/CPU/OFFLINE
- **`ExecutionProfile` per tier** — budgets scale by tier; mechanisms shrink, never vanish (spec: don't remove intelligence for weak hardware — route it)
- **`apply_to_step`** — non-destructive integration with Phase 12 executive Steps

## FILES CREATED
- `sweep_cognitive/routing/__init__.py`
- `sweep_cognitive/tests/test_routing.py` — 33 tests

## FILES MODIFIED
None outside the new package.

## ARCHITECTURE CHANGES
- "More intelligence per unit of compute" (spec section 14) is now a concrete, testable mechanism rather than an aspiration.
- Hardware awareness is a first-class input to routing, not an afterthought.

## MODEL CHANGES
None.

## DATA CHANGES
None.

## TRAINING CHANGES
None. Signal extraction is lexical scaffolding (marked); the decision table is the real, durable part.

## BENCHMARK CHANGES
None yet (Phase 20). Serves the LATENCY, TOKENS/COMPUTE PER SUCCESSFUL TASK, and CPU EFFICIENCY benchmarks.

## PERFORMANCE BEFORE
364 tests passing (end of Phase 12).

## PERFORMANCE AFTER
397 tests passing, 0.74s total runtime. 33 new tests.

## KNOWN FAILURES
None. Calibration bugs found by tests and fixed during development:
1. Weighted-score bands alone could never reach DEEP for lexical queries (weights + thresholds miscalibrated). Fixed by redesigning as an explicit decision table with floors and a DEEP conjunction rule rather than a single opaque score.
2. Marker-density-only complexity undersold terse marker-dense queries. Fixed with density + absolute-marker components.
3. Test bug: asserted tier factors ascending when FULL is intentionally the highest tier. Test corrected.

## NEW RISKS
- All thresholds (0.25/0.45/0.65/0.80, floors 0.6/0.7/0.8/0.85, compute cap 0.3) are hand-set and need empirical tuning against real workloads.
- Lexical signals are gameable/wrong for non-English input; learned estimators must replace them (prediction engine's Phase 8 outcome data feeds this).

## REGRESSIONS
None. Full suite green.

## NEXT BOTTLENECK
All modules exist but nothing has measured them end-to-end. Phase 20 builds the benchmark suite (understanding, evidence quality, source independence, calibration, routing efficiency, planning/replanning, verification) and the aggregate report.

## NEXT PHASE
Phase 20 — Full validation and benchmarking.
