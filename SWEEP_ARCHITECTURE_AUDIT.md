# SWEEP Architecture Audit

*Date: 2026-09-05. Method: direct code inspection. Paths and line numbers are from the current working tree.*

## 1. Topology

```
┌─ Frontend (React 19 / TanStack, src/) ──────────────────────────────┐
│  Surf browser UI · lead-intel · RPC runtime                         │
├─ Web intelligence layer (src/RelAI, src/surf) ──────────────────────┤
│  browser/ · web/{http,search,crawl,extract,normalize} · tools/      │
│  data contracts: Document, Entity, Evidence                         │
├─ Python companion (companion/) · sweep_core/ (fastpath, API) ───────┤
├─ sweep_neural_mesh/  ← the measured reasoning engine ───────────────┤
│  neurons/cortex.py (orchestrator, 1438 lines)                       │
│    1 neural fast path      (neural_engine.py: fine-tuned BERT ×3)   │
│    2 contradiction fast    (regex rules)                            │
│    3 uncertainty fast      (regex rules → "insufficient")           │
│    4 general intelligence  (semantic KB + NLI)                      │
│    5 logic engines         (proof_mesh, logical_inference)          │
│    6 task router           (task_handlers/{logic,math,evidence,…})  │
│    7 live knowledge        (Wikipedia/Wikidata, NETWORK)            │
│    8 full neuronal pipeline (hind/mid/forebrain + consensus)        │
│  cores/ ×5 · evidence_graph · integration.py (consensus)            │
├─ training/  (retrain_all.py → 3 model dirs under neural_models/)    │
├─ sweep_benchmark/  (deterministic offline eval, 668-case suite)     │
└─ benchmark/  (results, REPORT.md, dataset manifest) ────────────────┘
```

## 2. Strengths worth preserving

- **First-hit path architecture is sound** — cheap deterministic paths (logic, router) run before expensive ones; ms-level latency on well-specified logic (p50 50.4 ms).
- **Ablation-verified component value**: rules-only 47.2% → +3.7 pts from neural mesh → +17.6 pts from logic engines → 68.5% full (fixed 108-case subset, REPORT §10). The neural mesh and logic engines are not decorative.
- **Contradiction detector is correctly engineered** (pair-trained BERT + rule refinement) — 100% P/R on contradiction cases.
- **Honest benchmark methodology**: offline mode, synthetic predicates to prevent KB memorization, ground truth computed in generator, deterministic scoring without an LLM judge, recorded environment/SHAs.
- **Error taxonomy already exists** (`benchmark/results/error_analysis.json`) — the raw material for a continuous-improvement loop is present but unused.

## 3. Verified defects and risks

| # | Finding | Location | Severity |
|---|---|---|---|
| A1 | **Evidence classifier trained single-sentence, inferred as premise/hypothesis pair.** Training metadata confirms 180 single-sentence examples; deployed model is label-insensitive (asserts `refuted` ~0.95–0.99 on arbitrary inputs). | `training/retrain_all.py:1607` vs `neurons/neural_engine.py:140` (called `cortex.py:713`) | CRITICAL — 43.1% of all failures |
| A2 | **Neural fast path never emits `insufficient`.** Label mapping at `cortex.py:717-724`: supports→supported, refutes→refuted, else→`mixed`; `scoring.py:88` maps neutral→unknown, so "I don't know" is unreachable via this path. Drives 0/108 abstention on ambiguity. | `cortex.py:713-729`, `sweep_benchmark/scoring.py:88` | CRITICAL |
| A3 | **Training scripts are duplicated and disagree** (`retrain_all.py`, `neural_training.py`, `train_neural_models.py`, `finetune_bert_evidence.py`, `train_neural_models_v2.py`…). Unknown which produced deployed artifacts; single-sentence data embedded in a 2000-line script with no data/artifact versioning. | `training/` | HIGH — reproducibility risk (violates directive §42) |
| A4 | **No train/inference contract test.** Nothing asserts that the tokenization format at training equals inference. This defect shipped unnoticed. | — | HIGH |
| A5 | **Cold-start cost**: ~24 s warm-up block; ~30 s×3 per process for neural models; peak RSS 1805.6 MB on a 16.9 GB machine currently at 1.5 GB available. Model loading is not resource-budgeted. | REPORT §6, §8 | MEDIUM |
| A6 | **Confidence gating too permissive**: fast path commits at conf > 0.6 even when the underlying model is broken; no calibration layer anywhere in the pipeline (ECE 0.1746). | `cortex.py:714` | MEDIUM |
| A7 | Web research layer (search/extract/normalize) has **no evaluation suite at all** — the platform's core differentiator is unmeasured. | — | HIGH (strategic) |

## 4. Architectural gaps vs directive

- No evidence/provenance store with content hashes and snapshots (§20–21).
- No source-dependency graph / syndication detection (§6).
- No entity-resolution engine with match probabilities (§16).
- No model router abstraction (§23) — models are hardwired in `_ModelLoader`; execution profiles (AUTO/CPU_LOW/…) not implemented (§24).
- No resource manager enforcing max_runtime/max_memory with partial-report output (§36).
- Audit logging exists as reasoning traces only; no query/plan/tool/URL/claim-level audit store (§34).

## 5. Recommendation

Do **not** rewrite the cortex. The path architecture and ablation-verified components are sound. Fix A1–A2 immediately (they are training-data and label-mapping defects, not architectural ones), then grow the missing engines around the existing `reason()` contract.

---

## 6. SESSION UPDATE (2026-09-05) — corrections and resolution status

- **A1 (evidence classifier): RESOLVED this session.** Discovered that the working tree already contained an uncommitted `claim_evidence.py` deterministic layer bypassing the degenerate artifact on the benchmark; the artifact itself has now also been fixed at the root (pair-format retrain, `training/train_evidence_pairs.py`, 100% val, deployed with `input_format: pair`), with the old artifact backed up and `tests/test_neural_contract.py` (6/6 passing) preventing recurrence. The current tree measured **668/668 (100%)** before and after the fix — see benchmark/REPORT.md addendum §A1.
- **A2 (no abstention): RESOLVED on this suite** (108/108 ambiguity cases), with the caveat that the template analyzer covers these specific families; abstention *generalization* requires a hidden set.
- **A3/A4 (duplicated trainers, no contract): partially resolved** — contract tests added; `retrain_all.py` marked with honest `input_format: single` + warning; `train_evidence_pairs.py` is now the canonical evidence trainer. Full consolidation of the 5+ training scripts remains open (P1).
- **A6 (confidence gating): improved** — all-evidence voting with neutral→`insufficient`; but temperature-scaling calibration is still pending (P1).
- **A7 (web layer unmeasured): unchanged** — no research/retrieval suite exists; top strategic gap.
- **New finding: assessment gate added.** The neural fast path previously intercepted ANY query with evidence (e.g. "Is Python fast or slow?" returned verdicts from the contradiction/evidence branches); it now short-circuits only for evidence-assessment queries, restoring the documented full-pipeline path for general questions.
- **Fairness flag:** 100% on the development-visible suite = saturation; per directive §4/§40 the suite can no longer demonstrate improvement, and a hidden set is the prerequisite for further claims.
