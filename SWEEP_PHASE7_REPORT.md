# SWEEP Phase 7 Report — World Model

## PHASE NUMBER
7

## OBJECTIVE
Implement a versioned, auditable world model with entities, relationships, events, confidence, provenance, entity resolution, contradiction detection, and related-entity traversal.

## WHAT CHANGED
Created `sweep_cognitive/world/` package implementing the world model per the architecture spec (Module 4):
- Versioned state (every mutation creates a new snapshot)
- Confidence on every fact
- Provenance on every fact
- Contradiction detection (PART_OF vs DIFFERENT_FROM on same pair)
- Entity resolution by name/alias
- BFS traversal for related entities

## FILES CREATED
- `sweep_cognitive/world/__init__.py` — WorldEntity, WorldRelationship, WorldEvent, WorldState, WorldStateVersion, WorldModel
- `sweep_cognitive/tests/test_world_model.py` — 30 tests

## FILES MODIFIED
None outside the new package.

## FILES REMOVED
None.

## ARCHITECTURE CHANGES
- World model added as the central state representation other modules will reference.
- Version history is append-only; `get_version(n)` retrieves any snapshot.

## MODEL CHANGES
None (no learned models in this phase; structure only).

## DATA CHANGES
None.

## TRAINING CHANGES
None.

## BENCHMARK CHANGES
None yet (benchmark suite is Phase 20).

## PERFORMANCE BEFORE
160 tests passing (end of Phase 5/6), 195 after Phase 6.

## PERFORMANCE AFTER
225 tests passing, 0.24s total runtime. 30 new world-model tests.

## KNOWN FAILURES
None. One real bug was found and fixed during development (see REGRESSIONS note).

## NEW RISKS
- `WorldStateVersion.entities` stores dict copies (shallow). Entity objects are shared between versions; mutation of an entity mutates all snapshots pointing to it. Deep-copy on snapshot would be safer but slower. Deferred: entity mutation is currently done via `update_property` which appends provenance; acceptable for now, revisit if audit trails require immutable snapshots.

## REGRESSIONS
**Bug found and fixed:** Relationship IDs were generated from `int(time.time() * 1000)` only. Two relationships created in the same millisecond collided on ID, so the second silently overwrote the first in the state dict. This was exposed by the contradiction test (PART_OF and DIFFERENT_FROM added back-to-back). Fixed by introducing a module-level `itertools.count` counter combined with the timestamp (`_uid(prefix)`), guaranteeing uniqueness. Entities and events use the same generator.

## NEXT BOTTLENECK
Perception → world model integration: nothing currently writes observations into the world model. The Executive Controller (Phase 12) will need a world-model update rule (observation → candidate update → conflict analysis → decision) per spec section 17.

## NEXT PHASE
Phase 8 — Prediction engine with uncertainty.
