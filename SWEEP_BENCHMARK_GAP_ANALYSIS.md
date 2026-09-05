# SWEEP Benchmark Gap Analysis

*Date: 2026-09-05. Inputs: `benchmark/REPORT.md` (668-case suite, measured 2026-09), `benchmark/results/error_analysis.json`, code inspection. Every weakness below is measured, not assumed.*

## 1. Headline numbers and what they mean

| Group | Accuracy | Verdict |
|---|---|---|
| basic_logic | 94.7% | genuine strength |
| adversarial | 81.8% | competent |
| multi_step | 80.0% | competent |
| evidence | 66.7% | artifact-driven (see §2) |
| generalization | 61.9% | moderate |
| ambiguity | **0.0%** | total failure mode |
| **Overall** | **65.3%** | — |

Calibration: ECE 0.1746; 77 wrong answers at conf ≥ 0.75 = 37.6% of all errors — the system is most confidently wrong exactly where the directive's §47 ("never trade truth for confidence") bites hardest.

## 2. Gap G1 — Degenerate evidence classifier (largest measured defect)

- **Evidence**: 100 `EVIDENCE_ERROR` failures = 43.1% of all failures. Root cause verified by direct experiment (REPORT §11): trained on ~180 single sentences (clinical-trial domain), inferred on premise/hypothesis pairs; model is label-insensitive, collapsing to `refutes` at 0.95–0.99 regardless of input.
- **Impact**: caps `evidence` group at 66.7% and pollutes `ambiguity` + `hallucination` error classes (single statements with no claim asserted `refuted` at ~0.98).
- **Fix**: retrain on pair-format data matching inference (`tokenizer(premise, hypothesis)`) — the contradiction detector proves this format works (100% P/R).
- **Risk**: LOW (training-only change; no inference code changes required). Constraint: ~1.5 GB free RAM — must train on CPU with small batch/short sequences.
- **Verification protocol**: re-run the full 668-case suite in a fresh process; require evidence-group accuracy ≥ 80% and no regression > 2 pts in any other group; keep old artifact as `evidence_classifier_v1_backup`.

## 3. Gap G2 — Zero abstention (second largest)

- **Evidence**: 0/108 correct abstentions on ambiguous inputs; 0.0% on `ambiguity`; abstention rate 4.0% overall.
- **Contributing causes (verified in code)**:
  1. Neural fast path label mapping has no `insufficient` branch (`cortex.py:717-724`): neutral→`mixed`, which `scoring.py:88` maps to `unknown` for claim-mode only via the regex, but `decision="mixed"` alone maps to `unknown` only if the reasoning regex matches — the mapping is fragile.
  2. Confidence gate 0.6 commits whatever the (broken) classifier says.
- **Fix**: after retraining the evidence model on pairs, the `neutral` label becomes meaningful → map `neutral` → decision `insufficient` (→ canonical `unknown`). Add a low-confidence abstention rule (conf < threshold → `insufficient`) rather than a hard commit at 0.6.
- **Verification**: ambiguity-group accuracy > 0 and abstention rate on ambiguous cases > 50%; multi_step/basic_logic must not regress beyond 2 pts (over-abstention risk is real — the logic engines' `insufficient` handling at `cortex.py:1037-1039, 1070-1073` shows deliberate calibration of when abstention is accepted).

## 4. Gap G3 — Calibration

- **Evidence**: mean conf 0.895 when correct vs 0.764 when wrong; ECE 0.1746.
- **Fix path**: temperature scaling on a held-out validation split of the benchmark; store calibration parameters with model artifacts. (After G1/G2, because calibration of a broken model is meaningless.)

## 5. Gap G4 — Suite coverage vs directive

| Directive benchmark area | Current coverage |
|---|---|
| A. General reasoning | Partial (logic, multi-step) — no causal/counterfactual/spatial suites |
| B. Research | **None** (offline suite cannot measure retrieval) |
| C. Web investigation | **None** |
| D. OSINT reasoning | **None** |
| E. Multimodal | **None** |
| F. Evidence reasoning | Partial (single-claim pairs; no multi-source corroboration, no independence weighting) |
| G. Adversarial | Partial (text-level; no SEO-spam/circular-sourcing suites) |

Also missing per directive §4: dev/validation/hidden dataset split, contamination labels (§31), hidden set (the single suite is used both to develop against and to report).

## 6. Error taxonomy (current distribution, REPORT §11)

| Error type | Count | % of failures | Maps to gap |
|---|---|---|---|
| EVIDENCE_ERROR | 100 | 43.1% | G1 |
| PARSING_ERROR | 27 | 11.6% | G2/G3 (overcommit) |
| CONTRADICTION_ERROR | 27 | 11.6% | G1 (false `refuted` on unknowns) |
| HALLUCINATION | 27 | 11.6% | G1+G2+G3 |
| LOGIC_ERROR | 21 | 9.1% | future suite work |
| NEGATION_ERROR | 20 | 8.6% | adversarial hardening |
| ARITHMETIC_ERROR | 10 | 4.3% | logic engine edge cases |

Post-G1/G2, the expected dominant classes are LOGIC_ERROR and NEGATION_ERROR — that is the next improvement frontier, not more neural-model work.

## 7. Fairness protections to add before tuning further

1. Split `suite_v1` into dev/validation; author a **hidden set** (new content, new structures, never used during development) before any further tuning.
2. Add contamination labels to dataset manifest (CLEAN / POSSIBLY_CONTAMINATED / UNKNOWN).
3. Freeze the protocol (already good: offline mode, synthetic predicates, deterministic scoring) and record prompt/config versions with results (§42 partially met — needs model-artifact versioning, see architecture audit A3).

---

## 8. SESSION UPDATE (2026-09-05) — gap resolution and a new critical gap

- **G1 (degenerate evidence classifier): RESOLVED at the root.** Pair-format retrain deployed (`training/train_evidence_pairs.py`, 100% val on all three classes, contract-tested). The previously-unmeasured fact: the working tree's uncommitted `claim_evidence.py` analyzer had ALREADY lifted suite accuracy to 100% before this fix — the artifact fix addresses the defect itself, not just the symptom.
- **G2 (zero abstention): RESOLVED on this suite** (108/108) via the template analyzer + honest neural-path abstention semantics. Generalization beyond template families: UNTESTED.
- **G4 (suite coverage): unchanged** — research/web/OSINT/multimodal suites still do not exist.
- **NEW CRITICAL GAP — suite saturation:** the suite now scores 100% and is development-visible (generated post-hoc, tuned against). It can no longer (a) measure improvement, (b) detect regressions in covered families, or (c) support any comparative claim. The dev/validation/hidden split (§7.1) moves from "recommended" to **blocking** for all further tuning: until a hidden set exists, model-quality changes must be validated on held-out slices of NEW data (as done this session: 90/10 split of freshly generated pair data), and the 100% must be reported as "suite saturated", never as "system perfect".
- **RESOLVED SAME SESSION:** hidden_v1 built and frozen (`sweep_benchmark/hidden.py`, SHA-verified manifest, protocol-enforced consumption). Reference run (REPORT addendum §A5): content-generalization 100%, novel-skill groups 0–45%, abstention-first failure mode (76% abstain, 8.5% confidently-wrong). **New gap exposed:** missing skills — modulo/min-max/set-count, duration arithmetic over evidence, ISO-date event ordering, quantifier grounding — queued as P1 #6a with the rule that verification happens on a fresh hidden_v2, never by re-tuning against hidden_v1.
