# Sweep Benchmark Results

*Status: local, CPU-only, offline benchmark run on the machine below. Not an MLPerf submission; no MLPerf certification is claimed.*

## 1. Executive Summary

- **Overall accuracy: 65.3%** (436/668 cases)
  - adversarial: 81.8%
  - evidence: 66.7%
  - generalization: 61.9%
  - basic_logic: 94.7%
  - ambiguity: 0.0%
  - multi_step: 80.0%
- **Median latency (p50): 50.4 ms** | p95: 89.7 ms | p99: 131.7 ms
- **Throughput: 26.1 cases/sec**
- **Peak RAM: 1805.6 MB**

## 2. Hardware

- OS: Windows-11-10.0.26200-SP0
- CPU: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel / Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
- Physical cores: 10 | Logical cores: 12
- RAM: 16.88 GB total, 0.67 GB available at capture
- GPU: ['none detected']

## 3. Software

- Python: 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]
- torch: 2.9.1+cu126
- tensorflow: 2.21.0
- transformers: 4.57.0
- numpy: 2.3.3
- sklearn: 1.7.2
- psutil: 7.2.2
- pandas: 2.3.3
- pytest: 9.1.1
- Git SHA: 5375b35fe5c069ebc7fa73690058abe7335fa7f2 (main, dirty)

## 4. Dataset

- File: benchmark/datasets/suite_v1.json
- SHA-256: e16e29d75d580d8af9197b5f630869ba6a400190723458db0d6639f4811dcc1b
- Cases: 668 | per group: {"adversarial": {"total": 110}, "evidence": {"total": 108}, "generalization": {"total": 105}, "basic_logic": {"total": 132}, "ambiguity": {"total": 108}, "multi_step": {"total": 105}}
- Generated post-hoc with synthetic content (Phase 4); never routed into Sweep's training pipeline.

## 5. Accuracy Results (by category)

| Group | Accuracy | Correct/Total |
|---|---|---|
| basic_logic | 94.7% | 125/132 |
| multi_step | 80.0% | 84/105 |
| ambiguity | 0.0% | 0/108 |
| evidence | 66.7% | 72/108 |
| adversarial | 81.8% | 90/110 |
| generalization | 61.9% | 65/105 |

## 5b. Detailed Accuracy (precision/recall/F1, abstention)

**adversarial** (accuracy 81.8%):
- consistent: precision=—% recall=—% F1=—%
- contradiction: precision=100.0% recall=100.0% F1=100.0%
- unknown: precision=—% recall=—% F1=—%
**evidence** (accuracy 66.7%):
- supported: precision=50.0% recall=100.0% F1=66.7%
- refuted: precision=—% recall=0.0% F1=—%
- unknown: precision=—% recall=—% F1=—%
**generalization** (accuracy 61.9%):
- yes: precision=100.0% recall=85.7% F1=92.3%
- no: precision=—% recall=0.0% F1=—%
- unknown: precision=0.0% recall=—% F1=—%
**basic_logic** (accuracy 94.7%):
- yes: precision=100.0% recall=100.0% F1=100.0%
- no: precision=100.0% recall=100.0% F1=100.0%
- unknown: precision=—% recall=—% F1=—%
**ambiguity** (accuracy 0.0%):
- supported: precision=0.0% recall=—% F1=—%
- refuted: precision=0.0% recall=—% F1=—%
- unknown: precision=—% recall=0.0% F1=—%
- Abstention (correct refusal on 108 ambiguous cases): 0 -> 0.0%
**multi_step** (accuracy 80.0%):
- yes: precision=84.2% recall=100.0% F1=91.4%
- no: precision=65.2% recall=100.0% F1=79.0%
- unknown: precision=—% recall=0.0% F1=—%
- Abstention (correct refusal on 21 ambiguous cases): 0 -> 0.0%

**Confidence calibration:**
- mean confidence when correct: 0.895 | when wrong: 0.764
- ECE: 0.1746
- wrong answers at conf ≥0.75 (hallucination): 77 (37.6% of all errors)
- Contradiction rate: 19.2% | abstention rate: 4.0%

## 6. Latency Results

- p50: **50.4 ms**
- p90: 80.2 ms
- p95: 89.7 ms
- p99: 131.7 ms
- mean: 38.3 ms | min: 0.3 ms | max: 205.9 ms | stdev: 37.8 ms
- Startup (cortex+import): 1077.9 ms; warm-up block: 24056.0 ms

## 7. Throughput

- 26.1 cases/sec (measured block 25.6 s, 668 cases)

## 8. Memory

- Peak RSS: 1805.6 MB
- Mean RSS during run: 1312.9 MB

## 9. CPU Scaling

*Note: each row runs the same 108-case stratified subset (seed 7) in a fresh process. Accuracy is identical across configs (same sample, same engine) — only latency changes with thread count. This is a different (smaller) sample than the 668-case headline in Section 5, so 61.1% is not directly comparable to 65.3%.*

| Config | threads | accuracy | median ms | p95 ms | p99 ms | cases/s |
|---|---|---|---|---|---|---|
| 1 | 1 | 61.1% | 163.6 | 327.8 | 365.4 | 7.8 |
| 2 | 2 | 61.1% | 103.7 | 215.1 | 293.7 | 12.5 |
| 4 | 4 | 61.1% | 68.5 | 104.9 | 125.4 | 21.5 |
| physical(10) | 10 | 61.1% | 60.1 | 146.5 | 288.2 | 19.4 |
| logical(12) | 12 | 61.1% | 80.3 | 261.4 | 516.6 | 13.1 |

## 10. Ablation

*Note: all configs run the same 108-case stratified subset (seed 11) with a fresh cortex per config (no state leaks). This subset is different from the headline suite, so absolute accuracies are not directly comparable to Section 5 — the deltas between configs are the meaningful signal.*

| Config | accuracy | vs baseline | mean latency ms | peak RSS MB |
|---|---|---|---|---|
| A_baseline_rules | 47.2% | +0.000 | 12.1 | 1229.6 |
| B_gi | 47.2% | +0.000 | 16.0 | 1230.7 |
| C_neural_mesh | 50.9% | +0.037 | 56.8 | 1250.8 |
| D_full_logic | 64.8% | +0.176 | 8.4 | 1250.8 |
| E_full_sweep | 68.5% | +0.213 | 55.0 | 1251.1 |

## 11. Error Analysis

| Error type | Count | % of failures |
|---|---|---|
| EVIDENCE_ERROR | 100 | 43.1% |
| PARSING_ERROR | 27 | 11.6% |
| CONTRADICTION_ERROR | 27 | 11.6% |
| HALLUCINATION | 27 | 11.6% |
| LOGIC_ERROR | 21 | 9.1% |
| NEGATION_ERROR | 20 | 8.6% |
| ARITHMETIC_ERROR | 10 | 4.3% |

### Root cause: the deployed evidence-classifier model is degenerate

The single largest failure cluster is driven by one deployed artifact, not by the architecture. Verified by direct experiments on this machine:

1. **Training format (single sentences).** `sweep_neural_mesh/training/retrain_all.py` builds `EVIDENCE_DATA` as ~180 *single sentences* about clinical drug trials ("The study confirmed the drug's efficacy beyond doubt." → supports) and fine-tunes the evidence classifier with `tokenizer(texts, ...)` — one text per example.
2. **Inference format (sentence pairs).** `sweep_neural_mesh/neurons/neural_engine.py:classify_evidence()` runs `tokenizer(premise, hypothesis, ...)` — `[CLS] evidence [SEP] claim [SEP]`. Every inference input is therefore out-of-distribution relative to the fine-tuning data, and the domain (drug-trial statements) does not overlap the benchmark's claim/evidence content at all.
3. **The model does not classify at all.** Feeding the same content in both formats produced label-insensitive output: a training-domain "supports" sentence ("The study confirmed the drug's efficacy beyond doubt.") was classified `refutes` at 0.95 confidence, alongside every other input. Through the engine path in the benchmark it collapses to one label with 0.6–0.8 confidence regardless of whether the evidence supports or refutes the claim.
4. **Contrast:** the *contradiction detector* (trained on true sentence pairs) scores 100% precision/recall on contradiction cases in this suite — the pair-trained model works; the single-sentence-trained evidence model does not. The fix is retraining the evidence classifier on the same premise/hypothesis pair format used at inference.

Top failures (by confidence):

- `sb_0137` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9896: The gloop is connected. — degenerate evidence classifier: single statements with no claim are asserted `refuted` at ~0.98 confidence instead of abstaining (`unknown`).
- `sb_0296` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9885: The quark is active.
- `sb_0396` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9883: The wibble is open.
- `sb_0539` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9869: The gloop is charged.
- `sb_0603` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9864: The quark is open.
- `sb_0174` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9863: The gloop is awake.
- `sb_0490` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9863: The zorp is running.
- `sb_0331` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9852: The quark is connected.
- `sb_0562` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.985: The troon is open.
- `sb_0215` [CONTRADICTION_ERROR] expected `unknown` got `refuted` conf=0.9837: The troon is connected.

## 12. Generalization

- Accuracy on unseen structures: **61.9%** (65/105)
- Note: cases use novel content (entities/values) and novel structures (deep syllogisms, 3-term arithmetic, nested booleans, multi-hop transitivity).

## 13. Adversarial Robustness

- Accuracy: **81.8%** (90/110)

## 14. Public Comparison

Sweep was evaluated on a **private synthetic benchmark**, so no published result shares its dataset. Per the fairness check, every comparison below is **NOT DIRECTLY COMPARABLE**. Reported for context only.

| Model | Benchmark | Accuracy | Source | Comparable |
|---|---|---|---|---|
| GPT-4o | MMLU (5-shot) | 88.7% | https://openai.com/index/hello-gpt-4o/ | NOT DIRECTLY COMPARABLE |
| Claude 3.5 Sonnet | MMLU | 88.7% | https://www.anthropic.com/news/claude-3-5-sonnet | NOT DIRECTLY COMPARABLE |
| Gemini 1.5 Pro | MMLU | 85.3% | https://storage.googleapis.com/deepmind-media/gemini/gemini_1_5_pro_report.pdf | NOT DIRECTLY COMPARABLE |
| GPT-4 | MMLU | 86.4% | https://arxiv.org/abs/2303.08774 | NOT DIRECTLY COMPARABLE |
| DeepSeek-V3 | MMLU (CoT) | 88.4% | https://arxiv.org/abs/2412.19437 | NOT DIRECTLY COMPARABLE |
| LLaMA-3-70B | MMLU | 82.0% | https://ai.meta.com/blog/meta-llama-3/ | NOT DIRECTLY COMPARABLE |
| BERT-large (fine-tuned) | MNLI (GLUE) | 86.7% | https://arxiv.org/abs/1810.04805 | NOT DIRECTLY COMPARABLE |
| distilbert-base | MNLI | 82.4% | https://arxiv.org/abs/1910.01108 | NOT DIRECTLY COMPARABLE |

## 15. Limitations

- **Offline mode**: live Wikipedia/Wikidata/LLM retrieval was disabled for determinism. Real-world questions that Sweep can only answer via live lookup are not exercised; in this mode they abstain.
- **Synthetic content**: entity names are invented so no KB memorisation is possible; that also means these are not MMLU-style knowledge questions.
- **CPU-only, 1 process, enable_ml=False**: sentiment/NER embedder engines in the full pipeline were not enabled (they add latency, not accuracy here).
- **Standard ML workloads (BERT/ResNet on public data): NOT RUN — RESOURCE LIMITATION** (no GPU, ~2.4 GB free RAM, CPU-only policy).
- **Throughput measured on a warm process**; first-ever cold start includes ~24 s of model loading.

## 16. Conclusion

This benchmark measures what Sweep's `ReasoningCortex` actually does on 668 deterministic reasoning cases **without internet access**, on this CPU-only machine.

**What the evidence supports:**

- **Basic deterministic logic is genuinely strong**: 94.7% on arithmetic/comparison/ordering/boolean/set/deduction — with perfect yes/no precision and recall, and 100% F1 on contradiction pairs. These cases are answered correctly, in milliseconds, offline, at $0 cost.
- **Multi-step and adversarial reasoning are competent**: 80.0% and 81.8%, respectively. The neural mesh adds a measurable +3.7 pts over rules-only on identical cases, and the logic engines add +17.6 pts — the component contributions are real (ablation on a fixed subset).
- **The system generalizes to unseen structures but not to abstention**: 61.9% on novel-content/novel-structure cases is respectable for a non-LLM engine, but it never refuses — it asserts an answer on 100% of ambiguous inputs (0/108 abstentions) and only 4% of cases overall are flagged abstain, which is the single largest accuracy liability after the evidence classifier.

**What the evidence refutes:**

- **The deployed evidence classifier does not work.** It collapses to a single label regardless of evidence/claim semantics (verified by direct model experiments), because it was fine-tuned on single sentences but inferred on sentence pairs. This one artifact accounts for most of the evidence-group failures (66.7%) and many hallucination-class errors.
- **Claims of LLM-level or MLPerf-class performance are not supported by these measurements.** 65.3% overall on deterministic reasoning, 0% on ambiguity, no abstention mechanism, and every public comparison is NOT DIRECTLY COMPARABLE because no published model shares this synthetic dataset. Nothing here should be marketed against MMLU scores.

**Bottom line:** Sweep is a fast, deterministic, offline reasoner that is genuinely good at well-specified logic and contradiction detection, and measurably improved by its neural mesh and logic engines — but its evidence-classification layer is currently broken and it cannot yet handle ambiguity. Strengths are real; the headline "beats GPT-4o" claims are not supported by this benchmark.

---

# ADDENDUM — 2026-09-05 session (post-REPORT verification & fixes)

*All runs below: same machine, CPU-only, offline, `python -m sweep_benchmark.cli run --profile cpu`, full 668-case suite, fresh process per run.*

## A1. Baseline re-measurement (critical correction to §1–§16)

The REPORT above was generated from an **older tree**. The current working tree (still Git 5375b35f, dirty) contains an uncommitted deterministic claim-verification layer (`sweep_neural_mesh/neurons/claim_evidence.py` + `_try_claim_evidence_fast_path`, runs BEFORE the neural fast path) that was built to mitigate the degenerate evidence classifier diagnosed in §11.

Measured on the current tree BEFORE this session's changes: **668/668 = 100.0% accuracy**, p50 0.45 ms, 230 cases/s, peak RAM 1322.7 MB.

Consequences:
- Sections 1–16 describe historical behavior, not the current tree. The "evidence classifier does not work" conclusion remains true of the **deployed artifact**, but the claim-evidence fast path already bypassed its failure modes on this suite.
- **Fairness warning (directive §4/§40): 100% means the suite is SATURATED.** It was generated post-hoc and tuned against — it can no longer detect regressions in the families it covers, and it must not be cited as evidence of general superiority. A hidden set (new content/structures, never used in development) is required before any further capability claims. This is the top item in SWEEP_ROADMAP.md P1.

## A2. Fixes applied this session (P0 of SWEEP_ROADMAP.md)

1. **Pair-format evidence classifier retrain** — `sweep_neural_mesh/training/train_evidence_pairs.py` (new canonical trainer; 1,092 unique pairs; subjects/predicates disjoint from the benchmark vocabulary; CPU-only, top-2-encoder-layer fine-tune, 6 epochs, batch 8, seed 13; 1,112 s wall). Validation **100%** with all three label classes correct (L0 41/41, L1 32/32, L2 42/42). Artifact deployed with `input_format: "pair"` in metadata; old degenerate artifact preserved at `evidence_classifier_v1_backup/`. Model SHA-256 (first 16): f61e576a29d788e7. Production-path self-check 3/4 (one borderline probe at conf 0.44 — a non-collapse; the degenerate artifact scored 0/4 with ~0.98 wrong-label confidences).
2. **Honest abstention semantics in the neural fast path** (`cortex.py`) — evidence votes aggregate over ALL evidence items (was: first item only); unanimous-direction votes decide; mixed directions → `mixed`; neutral-only → `insufficient` (abstain) instead of `mixed` (claim-mode); contradicting evidence on a bare claim → `mixed` (undetermined), never a false `refuted`.
3. **Assessment gate** — the neural contradiction and evidence branches only short-circuit for evidence-assessment queries (bare claims or consistency/contrast questions). Previously, ANY question with ≥1 evidence item was answered by classifying evidence against it, e.g. "Is Python fast or slow?" returned verdicts. Non-assessment queries now reach the full neuronal pipeline (matching the documented architecture).
4. **Train/inference format contract tests** — `sweep_neural_mesh/tests/test_neural_contract.py` (6 tests): every deployed artifact must declare `input_format` matching the inference encoding, and must produce label-sensitive output. All deployed artifacts annotated (contradiction: pair ✓; intent: single ✓ — also flagged as a weak artifact, best_val_acc 21.5% on 13 classes). Legacy `retrain_all.py` evidence trainer honestly marked `input_format: "single"` so it FAILS the contract if rerun.
5. **metadata `input_format` declarations** added to all three deployed artifacts (provenance-verified by code inspection).

## A3. Post-fix verification (all re-run this session)

| Check | Result |
|---|---|
| Full suite (fresh process, post-retrain + post-patch) | **668/668 = 100.0%**, p50 0.60 ms, 191.6 cases/s, peak 1322.0 MB — no regression vs A1 baseline |
| Contract tests (`test_neural_contract.py`) | 6/6 pass |
| `sweep_neural_mesh/tests` full pytest | 500 passed / 12 failed — **all 12 verified pre-existing** (7× `TestAdaptivePipelineDepth` AttributeError from an unrelated module API mismatch; 2× full-brain interception mismatch where earlier fast paths — one predating this session — legitimately return before hindbrain; 3× timing-dependent when BERT load finishes mid-test) |

## A4. Honest limitations after this session

- 100% on a saturated, development-visible suite — NOT evidence of general superiority (see A1 fairness warning).
- The deployed contradiction detector still mislabels some paraphrase pairs as `contradiction` at ~0.67 confidence (just above the 0.6 gate) — quality retrain queued (P1).
- The intent classifier artifact is near-chance (21.5% val) and unused for gating — retrain queued (P1).
- Calibration (ECE 0.1746 in §5b) has not been re-measured on the new artifacts; temperature scaling queued (P1).
- All web-research, multimodal, OSINT, and cross-source-verification capabilities remain NOT TESTED (no suites exist; see SWEEP_BENCHMARK_GAP_ANALYSIS.md §5).

*Audit documents produced alongside this addendum: SWEEP_CAPABILITY_AUDIT.md, SWEEP_ARCHITECTURE_AUDIT.md, SWEEP_COMPETITIVE_MATRIX.md, SWEEP_BENCHMARK_GAP_ANALYSIS.md, SWEEP_ROADMAP.md.*

## A5. Hidden evaluation set (fairness protocol, directive §4/§31) — reference run

Built `sweep_benchmark/hidden.py` + `benchmark/datasets/hidden_v1.json` (270 cases, 5 groups, 12 families) with properties the dev suite no longer has:

- **Novel skills** absent from all 21 suite_v1 families: modulo arithmetic, min/max, set counting, duration arithmetic over evidence, ISO-date event ordering, most/none quantifier grounding.
- **Novel vocabulary** disjoint from suite_v1 AND from sweep_neural_mesh training data (including the new pair trainer).
- A content-generalization block: suite-style claim verification rendered in the new vocabulary.
- Per-case contamination labels: 0 KNOWN_CONTAMINATION, 270 POSSIBLY_CONTAMINATED (honest default), 0 CLEAN.
- Protocol enforcement: dataset SHA-256 frozen in `hidden_v1_manifest.json`; `run-hidden` refuses modified datasets and requires explicit `--confirm`; every consumption is counted.

**Reference run (run #1, 2026-09-05, offline CPU, fresh process — results in `benchmark/results/hidden_runs/run_01.json`):**

| Group | Accuracy | Committed-wrong | Abstained |
|---|---|---|---|
| hidden_content (suite-style, new vocab) | 100% (60/60) | 0 | 36 (all expected-unknown) |
| hidden_quantifier | 45.0% (18/40) | 2 | 20 |
| hidden_logic | 0% (0/90) | 21 | 69 |
| hidden_evidence (duration grounding) | 0% (0/40) | 0 | 40 |
| hidden_temporal (ISO ordering) | 0% (0/40) | 0 | 40 |
| **Overall** | **28.9% (78/270)** | **23 (8.5%)** | **205 (76%)** |

Reading (per directive §47 — never trade truth for confidence):
- Claim verification **fully generalizes to unseen vocabulary** (100%, zero committed-wrong).
- The missing skills (modulo/min-max/set-count, duration arithmetic, date ordering, quantifier grounding) are genuine capability gaps — and the system **abstained on 76% of all cases instead of fabricating answers**, with only 8.5% confidently-wrong. On hidden_evidence and hidden_temporal it never asserted a wrong fact.
- This 28.9% is the honest generalization baseline; the 100% dev-suite number must always be reported alongside it.
- **Do not iterate against hidden_v1.** Post-improvement evaluation requires a fresh hidden_v2 (new seed/vocabulary), because a fixed hidden set degrades into a dev set after tuning.

## A6. Confidence calibration (directive §18/§25)

Added `sweep_neural_mesh/training/neural_calibration.py` (temperature scaling, Guo et al. 2017 / arXiv:1706.04599; fit on fresh-seed pairs — never training instances — 50/50 fit/eval split; written to `evidence_classifier/calibration.json` and applied automatically in `neural_engine.classify_evidence`).

Measured (572 held-out fresh-draw pairs):
- Old single-sentence artifact era ECE (REPORT §5b): **0.1746**
- New pair-trained artifact, raw: **ECE 0.0477** — already well-calibrated
- After temperature scaling (T=1.1065): **ECE 0.0515** (no material ECE gain; T kept as the NLL-optimal parameter)
- Held-out accuracy on fresh draws: **85.5%** — the honest generalization number for the artifact (its 100% training-time validation was in-distribution; recorded to avoid overclaiming)

Dev suite re-checked after wiring calibration: 668/668 (p50 0.49 ms) — no regression. Contract tests 6/6.
