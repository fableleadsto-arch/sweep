# Sweep Cognition — Architecture

This document describes the layered architecture of the `cognition` package
that Sweep uses to reason, remember, verify, explain, and learn.

## Placement

```
app/                    neural-backed application layers (existing)
sweep_neural_mesh/      working neural mesh (retained, NOT rewritten)
cognition/              deterministic cognitive core (this package)
```

The rule of layering: **neural output never becomes a trusted fact until it
has passed the contract (schema) gate and confidence gate.** Deterministic
reasoning is the ground truth that neural results are checked against.

## Package layout

| File              | Core responsibility                                   |
|-------------------|-------------------------------------------------------|
| `schema.py`       | Unified data contract (Observation/Entity/Evidence/Claim/Hypothesis/ReasoningEvent) with lossless serialization |
| `store.py`        | Append-only JSONL persistence, thread-safe, id-indexed |
| `evidence_graph.py`| Entity -> Relationship -> Claim -> Evidence -> Source/Provenance graph |
| `memory.py`       | Working memory (bounded) + Long-term memory (persistent), independent of context window |
| `logic.py`        | Deterministic logic engine (transitive/symmetric closure, proofs) |
| `rules.py`        | Inspectable, versioned rule engine with trace |
| `bridge.py`       | NeuralOutput -> validated Contract objects (schema + confidence gate) |
| `router.py`       | Task-aware route selection with logged decisions and fallbacks |
| `hypotheses.py`   | Competing hypothesis engine (ranking, confidence, uncertainty, no forced conclusion) |
| `contradiction.py`| Explicit conflict representation; both claims always preserved |
| `beliefs.py`      | Evidence-driven belief revision with full history retention |
| `uncertainty.py`  | Explicit epistemic states (VERIFIED/PROBABLE/POSSIBLE/CONTESTED/UNRESOLVED/INSUFFICIENT_EVIDENCE) |
| `verification.py` | Independent multi-path verification (VERIFIED/CONTESTED/UNRESOLVED) |
| `prediction.py`   | Prediction/observation error loop; failed predictions never become observations |
| `feedback.py`     | Error classification -> targeted learning action, fully logged |
| `resources.py`    | CPU/RAM budget enforcement with graceful degradation |
| `loop.py`         | Unified end-to-end cognitive pipeline |
| `explain.py`      | `explain_conclusion()` reconstructable reasoning traces |
| `bench.py`        | Reproducible 12-measurement capability benchmark |
| `ids.py`          | Stable identity + content-derived ids |

## Data flow

See `COGNITIVE_LOOP.md` for the exact stage order. In summary:

```
INPUT -> PERCEPTION -> OBSERVATION -> WORKING MEMORY -> RETRIEVAL
     -> HYPOTHESIS -> NEURAL REASONING -> LOGIC REASONING
     -> EVIDENCE CHECK -> CONTRADICTION CHECK -> VERIFICATION
     -> UNCERTAINTY -> CONCLUSION -> FEEDBACK -> LONG-TERM MEMORY
```

## Persistence

All contract objects persist through `CognitiveStore` as JSONL. Graph edges are
compound records (`_ctype`), memory items are `_memory_records()`. A fresh
process reloads everything from the same directory.

## Retained existing subsystems

- `sweep_neural_mesh` — untouched, its test suite still passes.
- Existing benchmark suites remain the baseline; `cognition/bench.py` is the new
  deterministic capability benchmark (see `BENCHMARKS.md`).