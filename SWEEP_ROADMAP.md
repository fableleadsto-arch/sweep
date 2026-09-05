# SWEEP Roadmap — Ranked Improvements

*Date: 2026-09-05. Ranking criteria per directive §48: impact × difficulty × resource requirement × risk × expected benchmark improvement. Rule enforced: no rewrite of functioning components; every change must show target-benchmark improvement + no unacceptable regression (§29).*

## P0 — Critical defect fixes (do first, this session) — ✅ COMPLETED 2026-09-05

| # | Item | Status | Result |
|---|---|---|---|
| 1 | **Retrain evidence classifier on premise/hypothesis pairs** | ✅ DONE | `train_evidence_pairs.py`: 1,092 pairs (disjoint vocab), 100% val on all 3 classes, 1,112 s CPU; artifact deployed with `input_format: pair`; old artifact backed up |
| 2 | **Abstention path** | ✅ DONE | All-evidence voting; neutral→`insufficient`; contradicting-evidence-on-claim→`mixed`; suite 668/668 after (no regression) |
| 3 | **Regression run** | ✅ DONE | Full suite fresh-process ×2 (pre/post retrain): 100.0% both; p50 0.45→0.60 ms; peak RAM 1322 MB |
| 4 | **Train/inference contract test** | ✅ DONE | `tests/test_neural_contract.py` 6/6 passing; all artifacts declare `input_format`; legacy single-sentence trainer fails contract by design |
| 5 | *(added during session)* **Assessment gate** on neural fast path | ✅ DONE | Non-assessment questions reach the full pipeline; fixed early-return interception of arbitrary queries |

## P1 — Trust infrastructure (next 1-2 sessions) — updated 2026-09-05

| # | Item | Impact | Difficulty | Risk |
|---|---|---|---|---|
| 7 | **Hidden test set + dev/val split + contamination labels** — ✅ **DONE 2026-09-05**: `sweep_benchmark/hidden.py` (270 cases, 12 families, SHA-frozen manifest, protocol-enforced `run-hidden --confirm`, reference run recorded in REPORT addendum §A5). Reference result: content-generalization 100%, novel skills 0–45%, abstention-first behavior (76% abstain, 8.5% committed-wrong). | ✅ DONE | — | — |
| 5 | **Temperature scaling calibration** — ✅ **DONE 2026-09-05**: `training/neural_calibration.py`; T=1.1065 shipped in `calibration.json`, auto-applied at inference. Finding: new artifact already well-calibrated (ECE 0.048 vs 0.175 legacy); fresh-draw accuracy 85.5% recorded as the honest artifact number. | ✅ DONE | — | — |
| 6 | **Model artifact versioning**: single canonical training entrypoint, data hashes in metadata, deprecate the 5 duplicate training scripts. | HIGH (reproducibility, §42) | MEDIUM | LOW |
| 6a | *(added after hidden reference run)* **Implement the missing hidden-set skills**, validated against fresh-seed held-out slices during development; final verification ONLY on a regenerated **hidden_v2** (never hidden_v1 again): modulo/min-max/set-count arithmetic, duration arithmetic over evidence, ISO-date event ordering, quantifier grounding (most/none). Hidden_v1 evidence/temporal groups show the right instinct (100% abstention) — now add the capabilities. | HIGH | MEDIUM-HIGH | LOW |
| 6b | **Contradiction-detector quality retrain**: paraphrase augmentation (currently mislabels some paraphrase pairs as `contradiction` @ ~0.67). | MEDIUM-HIGH | MEDIUM | LOW |
| 6c | **Intent classifier retrain**: current artifact near-chance (21.5% val, 65 examples, 13 classes). | LOW-MEDIUM | LOW | LOW |
| 6d | **Fix 12 pre-existing test failures** (7× adaptive-depth API mismatch, 2× full-brain interception assumptions, 3× timing-dependent) and decide the interception-vs-full-pipeline contract explicitly. | MEDIUM | MEDIUM | LOW |
| 8 | **Resource manager**: max_runtime/max_memory guards around research runs + partial-report output on limit (§36). | MEDIUM | MEDIUM | LOW |
| 9 | **Audit store**: persist query/plan/decision/evidence references per run (§34) — extend existing reasoning traces with a JSONL sink. | MEDIUM | LOW-MEDIUM | LOW |

## P2 — Research/verification engines (the platform's differentiators)

| # | Item | Impact | Difficulty |
|---|---|---|---|
| 10 | **Source-dependency graph**: detect syndication/duplication via SimHash (already in normalize layer) + URL/citation chains; collapse derivative confirmations. (§6) | HIGH | MEDIUM |
| 11 | **Provenance store**: URL/title/publisher/date/retrieval-date/content-hash/excerpt per claim; snapshot-on-retrieve where lawful. (§20-21) | HIGH | MEDIUM |
| 12 | **Evidence scoring formula** (SourceQuality × Independence × Recency × EntityMatch × Directness × Corroboration), configurable weights, explainable output. (§17) | HIGH | MEDIUM |
| 13 | **Web research benchmark suite** (retrieval + citation accuracy + contradiction-across-sources) — prerequisite for any competitor comparison. (§3B/3C) | HIGH | HIGH (network determinism) |
| 14 | **Contradiction engine across sources**: temporal/entity-scope reasoning ("2023 vs 2025 ≠ contradiction"). (§8-9) | HIGH | HIGH |
| 15 | **Entity resolution** with match probabilities + supporting/conflicting features. (§16) | MEDIUM-HIGH | HIGH |

## P3 — Later (after P0–P2 are measured)

- Multimodal evaluation suites (§22) — deps already declared; zero measurement exists.
- Model router + execution profiles AUTO/CPU_LOW/CPU_STANDARD/GPU_STANDARD/GPU_HIGH (§23-24).
- Saturation-driven research loop with information-gain stopping (§12, §37) — requires P2's engines first.
- Hypothesis engine with disconfirming-search policy (§14).
- Human blind evaluation pipeline (§45).
- Tor transport layer with source classification and policy enforcement at tool layer (§11, §33) — implement only with explicit owner approval and legal review.

## Explicitly NOT planned (per directive)

- No "search the entire internet" feature (§12 — implement MAXIMUM_RELEVANT_COVERAGE instead).
- No authentication bypass, paywall defeat, or private-data collection (§33).
- No benchmark contamination or "beats everyone" marketing (§40, §47).

## Definition of done for each P0 item

`BUILD → TEST (pytest) → BENCHMARK (full suite, fresh process) → REGRESSION COMPARE → DOCUMENT (results into benchmark/REPORT.md addendum) → COMMIT`
