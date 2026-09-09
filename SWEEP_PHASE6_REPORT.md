# SWEEP Phase 6 — Memory Hierarchy Report

**Date:** 2026-09-09  
**Phase:** 6 of 20 (Memory Hierarchy)  
**Status:** COMPLETE

---

## 1. Objective

Create a memory hierarchy that separates:
- **Episodic Memory** — Past experiences, events, episodes (what happened)
- **Semantic Memory** — Generalized knowledge, concepts (what we know)
- **Procedural/Skill Memory** — How to do things (reusable workflows)
- **Failure Memory** — Known failure patterns (what goes wrong)

With:
- Memory consolidation (episodes → semantic knowledge)
- Skill extraction (successes → reusable skills)
- Failure learning (failures → patterns)
- Selective persistence (importance, relevance, novelty, reliability, recurrence)
- Memory provenance
- Expiration/downgrading for stale information

**Acceptance Criteria:** Past experience, generalized knowledge, and procedural skill are separated.

---

## 2. What Changed

### 2.1 New Files Created

| File | Purpose |
|------|---------|
| `sweep_cognitive/memory/hierarchy.py` | Complete memory hierarchy |
| `sweep_cognitive/tests/test_memory_hierarchy.py` | 35 tests for memory hierarchy |

### 2.2 Modified Files

| File | Change |
|------|--------|
| `sweep_cognitive/memory/__init__.py` | Already includes hierarchy exports |
| `sweep_cognitive/tests/test_memory.py` | No changes (associative memory tests still pass) |

---

## 3. Architecture Changes

### 3.1 Memory Types (NEW ENUMS)

```python
class MemoryType(str, Enum):
    EPISODIC = "episodic"      # Personal experiences
    SEMANTIC = "semantic"      # Generalized knowledge
    PROCEDURAL = "procedural"  # How to do things
    SKILL = "skill"            # Reusable workflows
    FAILURE = "failure"        # Known failure patterns
    SOURCE = "source"          # Source reliability

class MemoryImportance(str, Enum):
    CRITICAL = "critical"      # Must persist
    HIGH = "high"              # Should persist
    MEDIUM = "medium"          # May persist
    LOW = "low"                # Candidate for expiration
    TRIVIAL = "trivial"        # Likely to expire

class MemoryStatus(str, Enum):
    ACTIVE = "active"          # Currently relevant
    DORMANT = "dormant"        # Retained but not current
    STALE = "stale"            # May be out of date
    ARCHIVED = "archived"      # Kept for history
    EXPIRED = "expired"        # Past expiration
```

### 3.2 Episodic Memory (NEW)

Stores past experiences with:
- Episode types (task, investigation, error, success, interaction)
- Time-stamped entries
- Actions taken, observations, outcomes
- Success/failure tracking
- Importance levels (for persistence decisions)
- Status tracking (active, dormant, archived, expired)
- Consolidation candidates
- Indexing by type, task, outcome
- Recent retrieval with filtering

### 3.3 Semantic Memory (NEW)

Stores generalized knowledge:
- Concepts with types and properties
- Related concept retrieval (by type, property overlap)
- Consolidation from episodes
- Access tracking
- Separate from KnowledgeBase (which is about facts/entities)

### 3.4 Skill Registry (NEW)

Manages learned skills:
- Skill metadata (name, description, purpose, steps, tools)
- Success rate tracking
- Usage counting
- Versioning
- Reliable skill filtering
- Skill extraction from successful episodes

### 3.5 Failure Memory (NEW)

Tracks known failure patterns:
- Pattern name, description, type
- Symptoms, causes, avoidance strategies
- Detection cues
- Occurrence counting
- Pattern detection from symptoms
- Severity tracking

### 3.6 Memory Hierarchy (NEW ORCHESTRATOR)

Orchestrates all memory types:
- `record_episode()` — Add episodic memory
- `consolidate()` — Convert old episodes to semantic knowledge + extract skills
- `learn_from_failure()` — Extract failure patterns from failures
- `retrieve_relevant_memory()` — Cross-type retrieval by query
- `stats()` — Complete hierarchy statistics

---

## 4. How It Works

### 4.1 Episode Recording

```python
episode = EpisodicMemoryEntry(
    episode_type="task",
    description="Analyzed data",
    task_id="task_123",
    goal="Understand X",
    actions_taken=["searched", "analyzed"],
    observations=["Found pattern X"],
    outcome="Successfully identified X",
    success=True,
    importance=MemoryImportance.HIGH,
)
hierarchy.record_episode(episode)
```

### 4.2 Consolidation

Periodically (or on demand), old important successful episodes are consolidated:

```python
# Convert old episodes to semantic concepts
concepts = hierarchy.consolidate(max_age_days=7)

# Extract skills from successes
# Update episodes to archived status
```

### 4.3 Failure Learning

```python
# From a failure episode, learn a pattern
pattern = hierarchy.learn_from_failure(failure_episode)

# Later, detect similar failures
matches = hierarchy.failures.detect(["timeout", "slow response"])
```

### 4.4 Cross-Type Retrieval

```python
# Search across all memory types
results = hierarchy.retrieve_relevant_memory(
    "cat",
    memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.SKILL],
    top_k=5,
)
```

---

## 5. Model Changes

None. Pattern-based concept extraction is temporary scaffolding.

---

## 6. Data Changes

None.

---

## 7. Test Results

**Total tests: 195 (all passing)**

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `test_representation.py` | 25 | Representation layer |
| `test_perception.py` | 24 | Text perception |
| `test_knowledge.py` | 24 | Knowledge base |
| `test_document_perception.py` | 27 | Document/structured perception |
| `test_semantic.py` | 30 | Semantic understanding |
| `test_memory.py` | 30 | Associative + working memory |
| `test_memory_hierarchy.py` | 35 | Memory hierarchy |

**Execution time:** ~0.29 seconds

### Memory Hierarchy Test Coverage

**EpisodicMemoryEntry:**
- Basic entry creation
- Entry with full details
- Entry serialization

**EpisodicMemory:**
- Add entry
- Add duplicate (update)
- Get by type
- Get by task
- Get successes
- Get failures
- Get recent (with/without filter)
- Update status
- Stats

**SemanticMemory:**
- Add concept
- Get concept
- Get nonexistent concept
- Related concepts
- Consolidate from episode
- Stats

**SkillRegistry:**
- Register skill
- Register duplicate ID (update)
- Get by name
- List skills
- Get reliable skills
- Update success rate
- Stats

**FailureMemory:**
- Record pattern
- Detect pattern
- Detect no match
- Stats

**MemoryHierarchy:**
- Record episode
- Consolidate
- Learn from failure
- Retrieve relevant memory
- Full stats

---

## 8. Known Failures

None. All 195 tests passing.

---

## 9. New Risks

1. **Concept extraction is simplistic** — Uses first word of description. Real concept extraction needs NLP. Mitigation: scaffolding.

2. **Consolidation is basic** — Only chains episodes → concepts → skills. Real consolidation needs more sophistication. Mitigation: scaffolding.

3. **No forgetting mechanism beyond capacity** — Real memory has sophisticated forgetting. Mitigation: future enhancement.

4. **No memory retrieval noise handling** — As memory grows, retrieval may return irrelevant results. Mitigation: Phase 4 associative retrieval helps, but more needed.

---

## 10. Regressions

None. All existing tests still pass.

---

## 11. Next Bottleneck

**Prediction:** Better memory hierarchy will expose a world model bottleneck — having episodic, semantic, procedural, and failure memory is great, but without a coherent world model that tracks entities, relationships, and state over time, the memories remain disconnected.

**Mitigation:** Phase 7 will build the world model.

---

## 12. Next Phase

**Phase 7 — World Model**

Implement a world model that represents:
- Entities with properties
- Relationships between entities
- Events with time and participants
- States that change over time
- Confidence and provenance
- Contradictions

This provides the "state of the world" that all memory types can reference.

---

## 13. Phase 6 Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Episodic memory | ✓ DONE | EpisodicMemory with 11 tests |
| Semantic memory | ✓ DONE | SemanticMemory with 6 tests |
| Procedural/skill memory | ✓ DONE | SkillRegistry with 7 tests |
| Failure memory | ✓ DONE | FailureMemory with 4 tests |
| Memory consolidation | ✓ DONE | MemoryHierarchy.consolidate() |
| Skill extraction | ✓ DONE | Skills extracted from successes |
| Selective persistence | ✓ DONE | Importance levels, status tracking |
| Memory provenance | ✓ DONE | Source episode IDs tracked |
| Expiration/downgrading | ✓ DONE | Status enum, importance-based eviction |
| All tests passing | ✓ DONE | 195/195 tests passing |

---

## 14. Directive Compliance

- ✓ Did not store everything permanently — capacity limits, importance-based eviction
- ✓ Memory has provenance — source episode IDs, timestamps
- ✓ Memory supports expiration — status tracking, age-based consolidation
- ✓ Memory is selective — importance levels determine persistence

---

## 15. Phase Summary

| Phase | Status | Tests | Key deliverable |
|-------|--------|-------|-----------------|
| 0 | ✓ Done | — | Architecture audit |
| 1 | ✓ Done | 73 | Representation interfaces |
| 2 | ✓ Done | 100 | Multimodal perception |
| 3 | ✓ Done | 130 | Semantic understanding |
| 4 | ✓ Done | 160 | Associative memory + reasoning harmonization |
| 5 | ✓ Done | 160 | Harmonized working memory |
| 6 | ✓ Done | 195 | Memory hierarchy (episodic/semantic/procedural/failure) |
| 7-20 | — | — | In progress |
| **Total** | **6/20** | **195** | — |

---

*End of Phase 6 Report*
