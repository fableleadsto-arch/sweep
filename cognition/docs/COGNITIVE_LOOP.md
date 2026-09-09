# Sweep Cognition — The Unified Cognitive Loop

The loop orders every cognitive stage into a single end-to-end pipeline by
`CognitiveLoop.run(text, claims=..., hypotheses=...)` in `cognition/loop.py`.

## Stage order

```
INPUT -> PERCEPTION -> OBSERVATION -> WORKING MEMORY -> RETRIEVAL
     -> HYPOTHESIS -> NEURAL REASONING -> LOGIC REASONING
     -> EVIDENCE CHECK -> CONTRADICTION CHECK -> VERIFICATION
     -> UNCERTAINTY -> CONCLUSION -> FEEDBACK -> LONG-TERM MEMORY
```

Each stage records a `StageRecord(name, status, detail, output)` and a resource
sample, so a run is fully replayable (Phase 17 `explain_conclusion()`).

## Stage responsibilities

1. **PERCEPTION** — capture the raw task text.
2. **OBSERVATION** — promote raw input into a schema `Observation`, persisted.
3. **WORKING MEMORY** — write the percept into the bounded working set.
4. **RETRIEVAL** — recall prior long-term memories relevant to the task.
5. **HYPOTHESIS** — register candidate explanations, each with explicit
   initial confidence.
6. **NEURAL REASONING** — route to a model/tool via `Router`, run the handler,
   log the decision; on failure, degrade (never fabricate).
7. **LOGIC REASONING** — feed facts to `LogicEngine` for deterministic
   transitive/symmetric derivation.
8. **EVIDENCE CHECK** — persist each claim's evidence; corroborating evidence
   strengthens the hypotheses it supports.
9. **CONTRADICTION CHECK** — compare all claim pairs; store explicit
   `Contradiction` objects; **both claims always remain available**.
10. **VERIFICATION** — run every registered independent check
    (neural/logic/evidence). Two+ agreeing checks => VERIFIED; any
    disagreement => CONTESTED; otherwise UNRESOLVED.
11. **UNCERTAINTY** — classify the run into an explicit epistemic state
    (see `UNCERTAINTY.md`). Contradiction count feeds the signal so genuinely
    contested input never yields a confident conclusion.
12. **CONCLUSION** — the leading hypothesis is chosen only if it clears the
    confidence/uncertainty gates and the epistemic state is not CONTESTED.
    Otherwise the loop emits *"insufficient evidence to draw a conclusion"* —
    never a fabricated one.
13. **FEEDBACK** — classify (contradictions / insufficient evidence) and record
    the targeted learning action.
14. **LONG-TERM MEMORY** — persist the conclusion with provenance.

## Invariants

- The 14 stage names are returned in order by `result.stage_names()`.
- Failed predictions (Phase 13) are never written back as observations.
- A run with no usable evidence always ends in an explicit uncertainty state.
- The pipeline is deterministic: identical inputs produce identical stage
  structure and conclusions.