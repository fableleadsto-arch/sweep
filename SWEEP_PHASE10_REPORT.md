# SWEEP Phase 10 Report — Cognitive Reasoning (Harmonized)

## PHASE NUMBER
10

## OBJECTIVE
Complete the reasoning harmonization: wire the real rule-based and logic-based engines from the legacy `cognition/` package into the unified reasoning layer, alongside the new Phase 8/9 prediction and hypothesis engines, with an honesty contract for unavailable modalities.

## WHAT CHANGED
`ReasoningEngine` (previously a partial harmonization layer with stub handlers) now:

1. **Real RuleEngine integration** — `cognition/rules.py` RuleEngine runs actual facts through actual rules with full traces. Facts are assembled from `request.context["facts"]` + evidence counts. Conclusion/verdict come from the fired rules; zero rules matched → `no_rule_fired` (not fabricated).
2. **Real LogicEngine integration** — `cognition/logic.py` LogicEngine performs transitive/symmetric closure with proof steps. World-model relationships load directly as logic facts. `context["claim"]` → `derive()` returns a proof (`proven`, with steps) or `not_provable`.
3. **Hypothesis modality wired** — Phase 9 `HypothesisEngine.best_explanation()` drives abductive conclusions; ambiguity → `ambiguous` (refuses to pick).
4. **Predictive modality wired** — Phase 8 `PredictionEngine.predict_task_complexity()` recommends the computation mode.
5. **Honesty contract** — any modality whose backing engine wasn't provided returns `conclusion="modality_unavailable", confidence=0.0` with an explanatory note. It never fabricates a result.
6. **Claim routing** — an explicit `context["claim"]` always routes to deductive logic regardless of lexical indicators.

## FILES CREATED
- `sweep_cognitive/tests/test_reasoning.py` — 23 tests

## FILES MODIFIED
- `sweep_cognitive/reasoning/__init__.py` — replaced rule/logic stubs with real integrations; added hypothesis + predictive modality handlers; engine constructor accepts optional `rule_engine`, `logic_engine`, `world_model`, `hypothesis_engine`, `prediction_engine`; claim routing in `_select_modalities`
- `sweep_cognitive/semantic/__init__.py` — entity extraction now also captures single capitalized mid-sentence words (confidence 0.4) in addition to multi-word names (0.5); sentence-start words are skipped as ambiguous

## ARCHITECTURE CHANGES
- The reasoning layer is now a genuine orchestrator across three generations of SWEEP code: legacy rules/logic (`cognition/`), the new cognitive substrate (`sweep_cognitive/` phases 1-9), and the representation layer.
- Explicit logic sits ABOVE the world model (its facts feed the logic engine), exactly as the architecture spec requires.

## MODEL CHANGES
None.

## DATA CHANGES
None.

## TRAINING CHANGES
None.

## BENCHMARK CHANGES
None yet (Phase 20).

## PERFORMANCE BEFORE
289 tests passing (end of Phase 9).

## PERFORMANCE AFTER
312 tests passing, 0.64s total runtime. 23 new reasoning tests.

## KNOWN FAILURES
None. Two real issues found and fixed by tests:
1. Semantic entity extraction missed single capitalized names ("Alice", "Bob") because the person pattern required two consecutive capitalized words. Fixed with a sentence-aware single-name pass (lower confidence, sentence-start excluded).
2. Logic modality never activated for requests with an explicit proof claim unless the query contained lexical indicators. Fixed: explicit `context["claim"]` always routes to deductive logic.

Also noted (not changed): `LogicEngine.derive()` returns `[]` for directly-held facts (proof is trivial). This is the legacy engine's documented contract and is treated as proven.

## NEW RISKS
- Single-name entity extraction will misclassify some capitalized non-name words (e.g., "Monday"). Mitigation: confidence 0.4 and marked as scaffolding until an NER model replaces it.
- Modality selection is still partly lexical; the adaptive routing controller (Phase 13) will replace it with prediction-driven selection.

## REGRESSIONS
None. Full suite green, including all prior phase tests.

## NEXT BOTTLENECK
Evidence aggregation inside the reasoning layer is still lexical scaffolding ("confirms"/"contradicts" word counting). Phase 11 (Evidence Intelligence 2.0 with source independence) replaces this with the real evidence machinery.

## NEXT PHASE
Phase 11 — Evidence intelligence 2.0: source independence analysis, evidence quality scoring, richer consensus.
