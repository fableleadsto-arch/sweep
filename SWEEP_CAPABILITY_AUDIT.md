# SWEEP Capability Audit

*Date: 2026-09-05. Method: direct code inspection + local benchmark run (`benchmark/REPORT.md`, Git SHA 5375b35f, suite SHA-256 e16e29d7…). No marketing claims; every number below is measured or marked NOT MEASURED.*

## 1. What Sweep currently is

Two coexisting systems:

1. **`sweep_neural_mesh/`** — a CPU-first hybrid neural/rule reasoning engine (`ReasoningCortex`, `reason(query, evidence, sources, context)`), with 3 fine-tuned BERT classifiers, 5 neural cores, logic engines, and a self-evolution layer.
2. **Web platform (`src/`, `companion/`, `sweep_core/`)** — multi-engine search, scraping, extraction, normalization, entity/evidence data contracts, React UI, FastAPI ML companion.

The benchmark system (`sweep_benchmark/`) evaluates system (1). This audit therefore focuses there; the web platform's research capabilities are **NOT MEASURED** by any current suite.

## 2. Verified capability inventory (neural mesh)

| Capability | Status | Evidence |
|---|---|---|
| Deterministic logic (arith/sets/booleans/syllogisms) | **STRONG** | 94.7% (125/132), perfect yes/no P/R on `basic_logic` |
| Multi-step deduction/transitivity | **COMPETENT** | 80.0% (84/105) |
| Adversarial (negation, distractors) | **COMPETENT** | 81.8% (90/110) |
| Contradiction detection (pairs) | **STRONG** | 100% P/R/F1 on contradiction cases; pair-trained model works |
| Evidence classification (claim vs evidence) | **BROKEN** | 66.7% group accuracy; deployed model label-insensitive (see §3) |
| Ambiguity handling / abstention | **FAILING** | 0.0% (0/108); 0 abstentions on ambiguous inputs; 4% abstention overall |
| Calibration | **POOR** | ECE 0.1746; wrong answers at conf ≥0.75 = 77 (37.6% of errors) |
| Generalization to novel structures | **MODERATE** | 61.9% (65/105) |
| Latency (offline reasoning) | **STRONG** | p50 50.4 ms, 26.1 cases/s (warm, CPU-only) |
| Memory | **HEAVY** | peak RSS 1805.6 MB; ~24 s cold model load; ~30 s×3 neural cold cost per process |
| Web research / retrieval | **NOT MEASURED** | offline benchmark disabled live retrieval by design |
| Multimodal (image/audio/video) | **NOT MEASURED** | deps declared in `requirements.txt`; no eval suite exists |
| Source verification / provenance / citation | **NOT MEASURED** | data contracts exist (`src/lib/web-intelligence`); no verification engine or eval |
| Entity resolution | **NOT MEASURED** | extraction exists; no ER quality eval |
| Long-context synthesis | **NOT MEASURED** | no suite |

## 3. Verified critical defect: evidence classifier train/inference mismatch

- Training: `sweep_neural_mesh/training/retrain_all.py` (`train_evidence_classifier`, line 1607) tokenizes **single sentences**: `texts = [t for t, _ in data]` → `tokenizer(train_texts, …)`. `EVIDENCE_DATA` is ~180 single sentences about clinical drug trials. Saved metadata confirms `training_examples: 180`.
- Inference: `sweep_neural_mesh/neurons/neural_engine.py:140` `classify_evidence(premise, hypothesis)` encodes a **pair**: `tokenizer(premise, hypothesis, …)`, called from `cortex.py:713` as `(evidence_text, query)`.
- Consequence (measured, REPORT §11): the deployed model collapses to one label (`refutes`) at 0.95–0.99 confidence for arbitrary inputs; single statements with no claim are asserted `refuted` at ~0.98 instead of abstaining. This one artifact drives most of the 100 `EVIDENCE_ERROR` failures (43.1% of all failures).
- Contrast: the contradiction detector was trained on true sentence pairs and scores 100% P/R — pair-format training works.

## 4. Gaps vs the platform directive (summary)

- No verification layer, source-dependency graph, contradiction-across-sources engine, or provenance store.
- No research/web/OSINT/multimodal evaluation suites; no dev/validation/hidden dataset split; no contamination labels.
- No automated continuous-improvement loop wired to the failure database (error taxonomy exists in the benchmark; nothing consumes it).
- Resource limits, Tor support, source classification (PUBLIC/AUTHENTICATED/…), audit logging: declared in roadmap docs, not present as enforceable tool-layer policy.

Full details: `SWEEP_BENCHMARK_GAP_ANALYSIS.md`, `SWEEP_ROADMAP.md`.

---

## 5. SESSION UPDATE (2026-09-05, post-audit measurement + P0 fixes) — read before §2 table

Re-measuring the **current working tree** before changing anything revealed the §2 table was stale: an uncommitted deterministic claim-verification layer (`neurons/claim_evidence.py`, runs before the neural path) already bypasses the degenerate classifier on this suite. Measured fresh-process runs:

- Current tree BEFORE this session's changes: **668/668 = 100.0%**, p50 0.45 ms, 230 cases/s, peak 1322.7 MB.
- After this session's P0 fixes (below): **668/668 = 100.0%** — no regression, p50 0.60 ms, 191.6 cases/s.

Updated capability status:

| Capability | Was (§2) | Now (measured) |
|---|---|---|
| Evidence classification | BROKEN | Suite-saturated; deployed artifact **fixed this session** (pair-retrained, 100% val, contract-tested) |
| Ambiguity/abstention | FAILING | 108/108 on the suite — but only because a template analyzer covers exactly these families; true abstention generalization is UNTESTED (needs a hidden set) |
| Overall (offline suite) | 65.3% | 100.0% — **suite saturated; not evidence of general superiority** (directive §4/§40) |

P0 fixes applied this session: (1) pair-format evidence retrain via `training/train_evidence_pairs.py` (artifact + backup + `input_format` metadata); (2) honest abstention semantics in the neural fast path (all-evidence voting, neutral→`insufficient`, contradicting-evidence-on-claim→`mixed`); (3) assessment gate so non-evidence questions reach the full pipeline; (4) `tests/test_neural_contract.py` (6 tests, all passing) preventing the A1 defect class; (5) `input_format` declarations on all deployed artifacts.

Known quality debt (queued P1): contradiction detector mislabels some paraphrase pairs at ~0.67 conf; intent classifier artifact near-chance (21.5% val, unused for gating); calibration not re-measured on new artifacts; 12 pre-existing pytest failures documented in benchmark/REPORT.md addendum §A3 (verified unrelated to this session's changes).

### 5.1 Hidden-set reference measurement (end of session)

A frozen hidden evaluation set (`hidden_v1`, 270 cases, new skills + new vocabulary) was built and run once (REPORT addendum §A5). The honest capability picture it adds:

| Capability | Dev suite (saturated) | Hidden v1 (frozen) |
|---|---|---|
| Claim verification, unseen vocabulary | 100% | **100%**, 0 committed-wrong |
| Quantifier grounding (most/none) | — | 45% |
| Modulo / min-max / set-count | — | 0% (77% abstain) |
| Duration arithmetic over evidence | — | 0% (100% abstain, 0 wrong) |
| ISO-date event ordering | — | 0% (100% abstain, 0 wrong) |
| Calibration (fresh draws) | ECE 0.048 (was 0.175 legacy) | accuracy 85.5% on fresh-draw pairs |

Failure mode matches directive §47: on unknown skills the system abstains (76% overall) instead of fabricating — only 8.5% of hidden cases were confidently wrong. The gaps are capability gaps, not honesty gaps.
