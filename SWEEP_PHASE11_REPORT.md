# SWEEP Phase 11 Report — Evidence Intelligence 2.0

## PHASE NUMBER
11

## OBJECTIVE
Implement source-independence analysis ("do not count ten copied sources as ten independent confirmations"), evidence quality scoring, and richer consensus per spec modules 10/11 and item 21.

## WHAT CHANGED
Created `sweep_cognitive/evidence2/` with:

**SourceRegistry:**
- Source registry with reliability priors and declared upstream dependence (ownership/syndication)
- Evidence items with source, stance (-1..+1), modality, timestamps, content hash
- Dependence graph: declared edges + observed edges (shared content hash, near-duplicate text via shingle containment)
- Lineage computation via union-find: `effective_independent_sources()` = connected lineages, transitively. Ten syndicated copies of one wire story → 1.
- **Ambiguity policy (important design decision):** near-identical text alone does NOT prove dependence — independent outlets quoting the same press release must not be merged. Near-duplicate content counts as copying only with a corroborating publish-time gap (> 60s; real syndication lags are minutes-to-hours). Same-timestamp identical content = independent quotation, stays independent.

**EvidenceEngine2:**
- Quality decomposition per item: reliability (35%) + independence factor (25%, 1/lineage-size) + corroboration (20%, saturating) + recency (20%, 7-day half-life)
- Independence-weighted consensus: contribution = stance × quality ÷ lineage size; modality agreement; contradiction severity; verdict mapping aligned with `cognition.uncertainty.EpistemicState` semantics (VERIFIED requires ≥2 effective independent sources): INSUFFICIENT_EVIDENCE / SINGLE_LINEAGE_ONLY / VERIFIED_TENDENCY / REFUTED_TENDENCY / CONTESTED

## FILES CREATED
- `sweep_cognitive/evidence2/__init__.py`
- `sweep_cognitive/tests/test_evidence2.py` — 33 tests

## FILES MODIFIED
None outside the new package.

## ARCHITECTURE CHANGES
- The reasoning layer's lexical evidence counting (Phase 10 scaffolding) now has a principled replacement to delegate to.
- First explicit dependence-graph data structure in the codebase.

## MODEL CHANGES
None. Duplicate detection is shingle-containment scaffolding (marked, replaceable behind `detect_shared_origin`).

## DATA CHANGES
None.

## TRAINING CHANGES
None.

## BENCHMARK CHANGES
None yet (Phase 20). Directly serves the SOURCE INDEPENDENCE and EVIDENCE QUALITY benchmarks.

## PERFORMANCE BEFORE
312 tests passing (end of Phase 10).

## PERFORMANCE AFTER
345 tests passing, 0.62s total runtime. 33 new tests.

## KNOWN FAILURES
None. Two design bugs caught by tests before they became silent wrongness:
1. Jaccard over-penalizes single-word edits in short texts (0.70 vs 0.75 threshold) → switched copy detection to containment (|A∩B|/min), the correct notion for copies-with-boilerplate.
2. **Over-merging:** five independent sources with identical text (same-timestamp ingestion) were merged into one lineage by the automatic dependence detection. This is the exact false positive the spec warns against in reverse — identical content is ambiguous evidence, not proof of copying. Fixed with the time-gap corroboration policy above.

## NEW RISKS
- The 60s copy-gap threshold is a heuristic; real-world feeds can repost within seconds. Needs tuning against real data.
- Shingle containment misses paraphrased copying (a learned semantic model is the fix, deferred).
- Consensus verdicts are tendencies, not final epistemic states; the UncertaintyEngine in cognition/ remains the state authority.

## REGRESSIONS
None. Full suite green.

## NEXT BOTTLENECK
All the intelligence modules exist but nothing orchestrates them. The Executive Controller (Phase 12) is now the critical path: it must drive perceive → remember → predict → hypothesize → reason → act → verify → learn.

## NEXT PHASE
Phase 12 — Executive controller with planning, replanning, and task-state management.
