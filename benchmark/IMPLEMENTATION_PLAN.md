# Sweep Benchmark — Implementation Plan (Phase 1 Findings)

Date: 2026-09-03
Git SHA (at plan time): 5375b35fe
Status: Draft — updated as phases complete.

## 1. What Sweep actually is (from code inspection)

Sweep (`sweep_neural_mesh/`) is a **hybrid neuronal/rule-based reasoning system**.
The orchestrator is `ReasoningCortex` (`neurons/cortex.py`), whose public entry
is `reason(query, evidence, sources, context) -> ReasoningResult`.

A single `reason()` call tries these paths **in order** and returns the first hit:

| # | Path | Module | Notes |
|---|------|--------|-------|
| 1 | Neural fast path | `neural_engine.py` (fine-tuned BERT classifiers) | contradiction detect (2+ evidence) / evidence classify (1 evidence). Models load in a background thread; **~30s×3 cold cost per process**, then <10ms/sample |
| 2 | Contradiction fast path | `cortex.py` regex rules | only when query contains contradiction keywords + ≥2 evidence |
| 3 | Uncertainty fast path | `cortex.py` regex rules | opinion/future queries with no evidence |
| 4 | General intelligence | `general_intelligence.py` | knowledge lookup / semantic match (`fact_lookup`, `known_fact`, …) |
| 5 | Logic engines | `proof_mesh.py`, `logical_inference.py` | syllogisms, transitivity, modus ponens/tollens (parses premises from query text) |
| 6 | Task router | `task_handlers/` (logic, math, evidence, temporal, causal) | deterministic handlers; emits `Task router (cat/sub): ANSWER` in `reasoning` |
| 7 | Live knowledge | `retrieve_live_knowledge()` | **NETWORK** (Wikipedia + multi-tier GeneralKnowledge engine incl. local LLM fallback) |
| 8 | Full neuronal pipeline | hindbrain→midbrain→forebrain→consensus | falls back to a verdict (supported/refuted/insufficient/mixed) |

Important behavioral findings (measured):
- Pure synthetic-predicate logic (gorbs/greeps/…) is answered **fast and deterministically**
  by paths 5–6 (ms-level after warm-up).
- Queries containing real-world words often fall through to path 7 (**live network
  retrieval**), which is slow (1–8 s), flaky, and non-deterministic.
- Verdict-style outputs use the vocabulary: `supported / refuted / insufficient / mixed`.
- One `ReasoningCortex` instance keeps state (synapses, working memory, traces),
  so a single long-lived instance is the realistic deployment shape.
- `enable_ml=False` skips sentiment/NER/embedder in the full pipeline; neural
  fast path models load regardless.

## 2. Benchmark design decisions (CPU-only, reproducible)

1. **Offline mode**: path 7 (live network retrieval) is disabled by replacing
   `cortex.retrieve_live_knowledge`. This makes runs deterministic and
   reproducible on machines without network/GPU — matching the CPU-only mandate.
2. **Synthetic predicates** (zorbs, fleeps, …) for logic families so no knowledge
   base can memorise the answers; only genuine reasoning over the supplied text
   can produce them.
3. **One process per configuration**: model cold-load (~90 s) is amortised by
   warming up once, then running the whole suite in that process.
4. **Ground truth is computed in the generator** (pure Python), never by Sweep.
5. **Deterministic scoring**: no LLM judge. Answers are compared canonically
   (number / yes-no / set / token) by parsing Sweep's emitted `reasoning`,
   `explanation_data`, and `decision`.

## 3. Test groups (Phase 3)

| Group | id | Cases | Coverage |
|-------|----|-------|----------|
| Basic Logic | `basic_logic` | ≥100 | arithmetic, boolean, sequences, sets, simple syllogisms, classification |
| Multi-Step Reasoning | `multi_step` | ≥100 | chained deduction, transitivity, modus ponens/tollens, conditional, relational chains |
| Ambiguity | `ambiguity` | ≥100 | ambiguous wording, incomplete info, conflicting evidence → correct abstention |
| Evidence Reasoning | `evidence` | ≥100 | support/refute/neutral, corroboration, contradiction, irrelevance |
| Adversarial | `adversarial` | ≥100 | negation, double negation, distractors, false premises, near-identical alternatives |
| Generalization | `generalization` | ≥100 | structures/entity-pools unseen in Sweep's KB and in the other groups |

## 4. Workloads executed per profile

- `cpu`: full suite (≈700+ cases) × 1 measured pass (after warm-up) in one process;
  latency/RAM sampling on every case; 2 repeat passes for CI on a representative subset.
- `scaling`: representative latency subset, 10 warm-up + 30 measured iterations per
  thread config {1, 2, 4, 10, 12} in fresh subprocesses with `OMP/TF` thread envs.
- `ablation`: balanced subset (≈120 cases) × component toggles
  (no-neural / no-logic / no-GI / no-router / rules-only baseline / full).
- Standard ML workloads (BERT/ResNet on public data): **NOT RUN — RESOURCE
  LIMITATION** (CPU-only, ~2.4 GB free RAM, no GPU; documented in REPORT).

## 5. Deliverables

`benchmark/dataset_manifest.json`, `benchmark/datasets/*.json`,
`benchmark/results/{raw,summary.json,results.csv,environment.json,
error_analysis.json,ablation.json,scaling.json,comparison.json,
public_references.json}`, `benchmark/REPORT.md`.
