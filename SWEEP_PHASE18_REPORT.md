# SWEEP Phase 18 Report — Learned Feature Extraction (Text Estimator)

## Phase Number
18 (representation-level features from outcome data)

## Objective
Replace the hand-built behavioral feature ceiling with learned *content* weighting: an estimator over textual query content whose signal comes from accumulated outcome data rather than hand-enumerated marker lists — promoted only through the same baseline-beating gates as everything else.

## What Changed
- **`TextEstimator` + `text_features_from_query`**: hashed word unigram+bigram features (dim 512, blake2b-stable across processes, L2-normalized). HONEST SCOPE: the hashing *projection* is deterministic scaffolding; what is **learned** is the weighting of textual content — which fragments predict success/failure comes from outcome data, not from hand-picked marker lists. Trained dense embeddings remain future work.
- **`TextLearningPipeline`**: trains on accumulated experience with min-df vocabulary pruning (train-split-only — no holdout leakage), holdout evaluation, and a **three-way promotion gate**: the text model must strictly beat ALL of (1) the behavioral-feature estimator trained on the same split, (2) the lexical heuristic, (3) the majority-class floor. Rejection reasons name every baseline score.
- **`BlendedEstimator`**: combines behavioral + text estimators when both are promoted; falls back to the surviving component when one errors; raises when neither exists (callers' lexical fallbacks remain authoritative).
- **Router + PredictionEngine integration**: both now detect `predict_proba_blended` and use behavioral+text when available, behavioral-only otherwise. The learned note in prediction metadata states which kind ran (`learned(behavioral)` vs `learned(blended)`).
- **Phase 21 benchmark** `text_learning_gate`: accepts text on signal-bearing data AND refuses it on class-independent noise — both directions are required to pass.

## Files Created
- `sweep_cognitive/tests/test_phase18_demo.py` (live accumulation demo, also serves as the end-to-end test)
- `SWEEP_PHASE18_REPORT.md` (this file)

## Files Modified
- `sweep_cognitive/learning/__init__.py` — text feature space, `TextEstimator`, `TextTrainingReport`, `TextLearningPipeline` (MIN_DF=2, epochs=300, lr=0.5, l2=0.01), `BlendedEstimator`, `majority_baseline`, module docstring updated with honest scope
- `sweep_cognitive/routing/__init__.py` — `_apply_learned_signals` uses blended path when available
- `sweep_cognitive/prediction/__init__.py` — `predict_task_complexity` uses blended path when available; metadata records estimator kind
- `sweep_cognitive/benchmarks.py` — `bench_text_learning_gate` (suite now 12)
- `sweep_cognitive/tests/test_learning.py` — Phase 18 tests + blended integration tests

## Files Removed
None.

## Architecture Changes
None — learning remains a layer behind stable contracts. The `features_from_query` contract (behavioral space) is unchanged; the text space is additive. Consumers opt in via constructor injection and fall back automatically.

## Model Changes
New model class `TextEstimator` (hashed n-gram logistic). Promotion artifact includes its `TextTrainingReport` for audit. Key measured design choices:
- **min-df=2 vocabulary pruning is required**: without it, sparse record-specific bigrams memorize the training set (experimentally verified: test accuracy collapsed from ~0.92 to ~0.53 at n=50).
- Vocabulary is built from the train split only — the holdout never influences it.

## Data Changes
No schema change. `ExperienceRecord.goal` is the new training input for the text space; `query_features` still feeds the behavioral space.

## Training Changes
`TextLearningPipeline.train(epochs=300, lr=0.5, l2=0.01, min_df=None)`. Gates: MIN_RECORDS=40, MIN_TEST=10, holdout 30%, single-class rejection, three-baseline beating requirement.

## Benchmark Changes
Suite 11 → 12. `text_learning_gate` verifies BOTH directions: acceptance with genuine content signal (test accuracy 0.92 vs behavioral 0.42 / lexical 0.50 / majority 0.50 on the gate fixture) and honest refusal on noise data.

## Performance Before
- Tests: 586 passing | Benchmarks: 11/11 | Features: behavioral counts only (declared ceiling)

## Performance After
- Tests: **594 passing** | Benchmarks: **12/12** | Text estimator beats ALL baselines on signal-bearing holdout data; refused on noise; blended path exercised end-to-end (executive → store → train → promote → route)

## Known Failures
None outstanding.

## New Risks
1. **Hash collisions**: 512 dims means distinct grams can collide; at current vocabulary sizes (~70 active dims on fixtures) collision noise is negligible, but it grows with vocabulary. Mitigation: raise TEXT_DIM; monitor report weights count.
2. **Distribution shift**: the gate is i.i.d.-split; real drift (new task types over time) needs periodic retraining — the controlled pipeline supports this but nothing schedules it yet.
3. **Blend weight (0.5 default) is hand-set** — a calibrated weight is future work once both models have real-world track records.

## Regressions
- `TextTrainingReport` field had a truncated default (`= None` missing) — syntax error caught by import smoke test immediately.
- Benchmark closure had a stray `del rng` on an outer variable — caught by the benchmark runner.
- No behavioral regressions: full suite and all prior benchmarks pass unchanged.

## Honest Status
- ✅ Learned *content* weighting (no marker lists) — demonstrated by the noise-refusal benchmark: on class-independent text, no model is accepted.
- ✅ Promotion gated against behavioral + lexical + majority baselines on identical holdout.
- ⚠️ The hashing projection is still scaffolding (documented in module docstring); dense embeddings are Phase 19+ once data volume justifies them.
- ❌ AGI — still unclaimed.

## Next Bottleneck
(spec 60: better features → data bottleneck.) The gate now regularly REFUSES because accumulated real data is thin — which is the correct behavior. The system needs production task flow to accumulate honest records; scheduling periodic retraining (`promote` on a cadence) is the next infrastructure gap.

## Next Phase
Phase 19: scheduled retraining cadence + drift detection (compare fresh reports against the promoted report; alert when test accuracy decays), plus TEXT_DIM/blend-weight calibration from accumulated data.
