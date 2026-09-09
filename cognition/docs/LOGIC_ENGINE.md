# Sweep Cognition — Logic Engine

`cognition/logic.py` is a pure, **deterministic** reasoning core: no random
state, no LLM. Repeated execution produces identical results.

## Model

- `Fact(relation, subject, object)` — ground truth `relation(subject, object)`.
- `transitive_relations` — default: `before, after, older-than, north-of,
  ancestor-of, subclass-of, is-part-of`. `R(a,b) & R(b,c) => R(a,c)`.
- `symmetric_relations` — default: `adjacent-to, sibling-of, related-to`.
  `R(a,b) => R(b,a)`.

## Operations

| Method | Behavior |
|--------|----------|
| `add_fact(rel, s, o)` / `add(fact)` | insert ground truth |
| `holds(rel, s, o)` | direct or derived truth (uses full closure) |
| `derive(rel, s, o)` | return `[ProofStep]` if derivable else `None` |
| `closure()` | full fixpoint closure (deterministic iteration) |
| `reachable(subject, rel=None)` | all objects reachable from subject |

`ProofStep` records conclusion, premises, rule name, and detail so every
derived conclusion is justifiable.

## Known answers (regression)

`before(A,B)`, `before(B,C)`, `before(C,D)` derives `before(A,C)`,
`before(A,D)`, `before(B,D)`; never `before(D,A)` or `before(C,B)`.
(north-of facts with self are not reflexive.)

## Guarantees

- Determinism: same facts -> same closure, same proofs (see
  `test_schema_*` / logic engine tests).
- Proof reconstruction: every derived conclusion has a proof path recorded.