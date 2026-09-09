# Sweep Cognition — Evidence Model

The evidence chain spans

```
Entity -> Relationship -> Claim -> Evidence -> Source / Provenance
```

implemented in `cognition/evidence_graph.py` on top of `CognitiveStore`.

## Contract objects

Defined once in `cognition/schema.py` (Phase 1):

- `Observation` — a raw percept.
- `Entity` — a thing being reasoned about.
- `Evidence` — content, source, retrieval_time, content_hash, entity links,
  observation link, metadata.
- `Claim` — statement + subject/predicate/object roles, time context,
  confidence, status, linked evidence ids.
- `Hypothesis` — candidate explanation with competing confidence.
- `ReasoningEvent` — replayable reasoning step.

All contract objects serialize losslessly via `object_to_dict` /
`object_from_dict`; identity and relationships survive the round-trip.

## Graph operations

- `add_entity / add_claim / add_evidence` persist and index.
- `link(claim_id, evidence_id, relation, confidence, reason)` records
  `supports` / `refutes` / `neutral` edges (compound `_ctype` records).
- `supporting_evidence(claim)` / `contradicting_evidence(claim)` reconstruct
  the two sides of a conflict from persisted edges.
- `evidence_with_provenance(claim)` returns Claim -> Evidence -> Source with
  relation confidence and content hash — the audit trail.
- `reconstruct_from_entity(entity_id)` restores an entity's claims plus
  supporting and contradicting evidence.
- `relationship_chain(entity_id, max_depth)` walks Entity -> Relationship edges.

## Invariants

- A claim's evidence can be independently recovered after a process restart.
- Every evidence item's source and retrieval time are retained.
- Contradiction detection never deletes either conflicting claim
  (see `contradiction.py` and its tests).