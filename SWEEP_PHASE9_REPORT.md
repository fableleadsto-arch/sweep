# SWEEP Phase 9 Report — Hypothesis + Counterfactual Engine

## PHASE NUMBER
9

## OBJECTIVE
Implement Module 9 (multi-hypothesis reasoning: generate multiple explanations with supporting/conflicting/missing evidence, confidence, discriminating tests) and Module 22 (counterfactual engine) per the architecture spec.

## WHAT CHANGED
Created `sweep_cognitive/hypotheses/` with:

**Hypothesis engine:**
- `Hypothesis` — statement, status, confidence, prior, evidence list, missing evidence, `predictions_if_true`
- `EvidenceItem` — weight (negative = contradicting) × source reliability
- `HypothesisStatus` — OPEN / SUPPORTED / WEAKENED / ELIMINATED / SUPERSEDED
- `HypothesisEngine` — generation (direct + template-based from observations), evidence accumulation with logistic confidence updates, elimination on strong contradiction, `viable()` ranking with leader marking, abductive `best_explanation()` that **refuses to pick when top-2 are within 0.15** (hallucination control), and `recommend_discriminating_test()` (finds an observation one hypothesis predicts and the other doesn't)

**Counterfactual engine (over the world model):**
- `evaluate_premise` — checks whether a hypothetical relationship contradicts existing world knowledge (mutual-exclusion pairs, both directions)
- `observations_if_false` — what would disappear if a hypothesis were false
- `what_would_change` — propagation analysis: new neighbor + both endpoints' 1-hop neighborhoods

## FILES CREATED
- `sweep_cognitive/hypotheses/__init__.py`
- `sweep_cognitive/tests/test_hypotheses.py` — 31 tests

## FILES MODIFIED
None outside the new package.

## ARCHITECTURE CHANGES
- Hypotheses are now first-class cognitive objects with a lifecycle; they are never conclusions (spec section 18).
- The counterfactual engine consumes the Phase 7 world model directly — first cross-module consumer of the world model's relationship graph.

## MODEL CHANGES
None. Hypothesis *content* generation is template scaffolding (explicitly marked, with a test enforcing the marker). The bookkeeping around it — evidence weighting, logistic confidence updates, elimination thresholds, dominance ranking, discriminating-test selection — is real and tested.

## DATA CHANGES
None.

## TRAINING CHANGES
None.

## BENCHMARK CHANGES
None yet (Phase 20). The abductive-selection-refuses-ambiguity behavior is designed to serve the AMBIGUITY and HYPOTHESIS GENERATION benchmarks.

## PERFORMANCE BEFORE
258 tests passing (end of Phase 8).

## PERFORMANCE AFTER
289 tests passing, 0.35s total runtime. 31 new tests.

## KNOWN FAILURES
None. Three real bugs were found by tests and fixed:
1. Elimination threshold inconsistency: contradiction magnitude was compared unscaled against a threshold calibrated for scaled values. Fixed by scaling consistently with the confidence update (×2.0).
2. `CounterfactualEngine.evaluate_premise` called `query_relationships(subject_id=...)`, a parameter that doesn't exist. Fixed by filtering manually over all relationships.
3. `what_would_change` only reported the subject's *existing* neighbors and missed the object entirely — for a new edge (a, located_in, b) with no prior neighbors it returned an empty affected set. Fixed to always include the new neighbor plus both endpoints' neighborhoods.

## NEW RISKS
- The 0.15 ambiguity margin and the 1.5 elimination threshold are fixed constants; they need empirical tuning once real task data flows through.
- Template hypothesis generation produces structurally valid but shallow explanations; a learned generator must replace it before hypothesis quality can be measured meaningfully.

## REGRESSIONS
None. Full suite green.

## NEXT BOTTLENECK
Nothing automatically *generates* hypotheses from perceptions yet — wiring perception → hypothesis generation → evidence gathering is the Executive Controller's job (Phase 12).

## NEXT PHASE
Phase 10 — Cognitive reasoning over learned representations (harmonizing the existing logic engines with the new cognitive substrate).
