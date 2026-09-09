# SWEEP Phase 0 — Architecture Audit Report

**Date:** 2026-09-09  
**Method:** Direct code inspection of entire repository  
**Scope:** Full repository including `cognition/`, `sweep_neural_mesh/`, `sweep_benchmark/`, `src/`, `companion/`  

---

## 1. Current System Topology

```
┌─ Frontend (src/) ──────────────────────────────────────────────────┐
│  React UI · browser interaction · web intelligence layer            │
│  src/RelAI/ — search, crawl, extract, normalize, HTTP               │
│  src/RelAI/web/providers/ — provider abstractions                   │
├─ Python Companion (companion/) ─────────────────────────────────────┤
├─ sweep_core/ ───────────────────────────────────────────────────────┤
├─ cognition/ ← Deterministic cognitive reasoning (Phase 1-20 contract)│
│  schema.py     — Unified data contract: Observation, Entity,         │
│                  Evidence, Claim, Hypothesis, ReasoningEvent          │
│  store.py      — Persistent JSONL store (CognitiveStore)             │
│  evidence_graph.py — Traceable evidence network                      │
│  contradiction.py — Explicit conflict representation                 │
│  beliefs.py    — Evidence-driven belief updating                     │
│  uncertainty.py — Epistemic state classification                     │
│  verification.py — Independent verification                          │
│  hypotheses.py  — Competing hypothesis engine                        │
│  memory.py      — Working + Long-term memory                         │
│  prediction.py  — Prediction/observation error loop                  │
│  bridge.py      — Neural-to-reasoning integration (validation gate)  │
│  logic.py       — Deterministic transitive inference                  │
│  rules.py       — Inspectable rule-based reasoning                    │
│  feedback.py    — Feedback/correction loop                           │
│  explain.py     — Explainability layer                               │
│  router.py      — Resource-aware execution routing                    │
│  resources.py   — Resource management                                │
│  ids.py         — Stable identity generation                          │
├─ sweep_neural_mesh/ ← Neural reasoning engine (PRIMARY FOCUS)       │
│  mesh.py        — Top-level NeuralMesh API                           │
│  engine/                                                                                                      ├── reasoning.py   — Reasoning orchestration (pipeline: decompose→evidence→hypotheses→verify→conclude)     │  ├── memory.py     — Multi-layer memory (Working/Evidence/Semantic/User)                                                                                     │  ├── verification.py — Self-checking verification core                                                                                                         │  ├── tools.py      — Tool system (Calculator, UnitConversion, ToolRegistry, ToolRouter)                                                                        │  ├── providers.py  — Provider abstractions (LanguageProvider, EmbeddingProvider, etc.)                                                                       │  └── __init__.py                                                                                                         │  neurons/                                                                                                       │  ├── cortex.py    — ReasoningCortex orchestrator (1519 lines, the main entry point)                                                                            │  │                  6 processing centers + BG-Thalamus loop + plasticity + grading                                                                               │  │                  Fast paths → Logic engines → Task router → Full brain pipeline                                                                               │  ├── neural_engine.py — BERT model loader (evidence classifier, contradiction detector, intent)                                                             │  ├── fast_path.py — Early-exit for unanimous evidence direction                                                                                                │  ├── claim_evidence.py — Deterministic claim verification (POST-FIX, 100% on suite)                                                                          │  ├── working_memory.py — Baddeley-style working memory (capacity-limited, decay, rehearsal)                                                                │  ├── world_knowledge.py — Factual entity properties (birds don't talk, cats can't fly)                                                                      │  ├── general_intelligence.py — Hybrid neural+rule reasoning (regex patterns + SentenceTransformer)                                                         │  ├── embeddings.py — SimHash + cosine similarity (lightweight semantic)                                                                                       │  ├── integration.py — IntegrationHub + ConsensusEngine                                                                                                        │  ├── centers.py   — 6 processing centers (EvidenceGatherer, CredibilityAssessor, TemporalSequencer, CausalLinker, ContradictionDetector, ExplanationBuilder) │  ├── signal.py    — Signal/Synapse types for inter-center communication                                                                                        │  ├── basal_ganglia.py — Action selection (Go/NoGo)                                                                                                             │  ├── brain.py     — Hindbrain/Midbrain/Forebrain divisions                                                                                                     │  ├── plasticity.py — Synaptic plasticity                                                                                                                       │  ├── grading.py   — Evidence grading                                                                                                                           │  ├── trace.py     — ReasoningTrace/ReasoningResult                                                                                                            │  └── 60+ other neuron modules (abductive, amygdala, analogical, bayesian, causal_model, counterfactual, etc.)                                                │  core/          — MeshGraph, ExecutionEngine, ModelRouter, Node types                                                                                          │  fusion/        — FusionEngine, ConfidenceEngine, VerificationEngine                                                                                           │  memory/        — FeatureCache                                                                                                                                │  registry/      — ModelRegistry, CapabilityRegistry                                                                                                            │  resources/     — ResourceManager                                                                                                                             │  telemetry/     — Telemetry                                                                                                                                    │  training/      — 28+ training scripts, neural_models/, datasets/                                                                                             │  benchmarks/    — Benchmark harnesses                                                                                                                         │  docs/          — Architecture docs                                                                                                                           │  tests/         — Unit tests                                                                                                                              │                                                                                                                                                                 │                                                                                                                                                                 ├─ sweep_benchmark/ ← Offline benchmark suite (668 cases)                                                                                                        │  runner.py, scoring.py, datasets.py, hidden.py, report.py, cli.py, metrics.py, ablation.py                                                                       │  results/       — Historical run results                                                                                                                      │                                                                                                                                                                 ├─ benchmark/     ← Public benchmark artifacts                                                                                                                  │  REPORT.md, dataset_manifest.json, cognition_benchmark.json                                                                                                    │  datasets/, results/                                                                                                                                           │                                                                                                                                                                 ├─ scripts/       ← Utility scripts                                                                                                                                 │ └─ services/     ← Service definitions                                                                                                                                                                               └─ cpp/, native/, models/, graph_benchmark/, neural_eval/ ← Additional components
```

---

## 2. Dependency Graph (Critical Path)

```
User Query
    ↓
ReasoningCortex.reason(query, evidence, sources, context) [cortex.py]
    ↓
[Fast Path 1] _try_claim_evidence_fast_path → claim_evidence.py (DETERMINISTIC)
    ↓
[Fast Path 2] _try_neural_fast_path → neural_engine.py (BERT classifiers)
    ↓
[Fast Path 3] _try_contradiction_fast_path
    ↓
[Fast Path 4] _try_uncertainty_fast_path
    ↓
[Fast Path 5] _try_gi_fast_path → general_intelligence.py (regex + neural)
    ↓
[Logic Engines] _try_logic_engines → proof_mesh.py + logical_inference.py
    ↓
[Task Router] _try_task_router → task_handlers/
    ↓
[Live Knowledge] _try_live_knowledge → Wikipedia/Wikidata/GeneralKnowledge
    ↓
[Full Brain Pipeline] Hindbrain → Midbrain → Forebrain → Integration → Consensus
    ↓
[Post-processing] Memory recording, plasticity, grading, human_reasoning
    ↓
ReasoningResult (decision, confidence, reasoning, trace)
```

---

## 3. Critical Findings from Previous Audits (Resolved + Open)

### RESOLVED (2026-09-05 through 2026-09-08)

| ID | Finding | Resolution |
|----|---------|------------|
| A1 | Evidence classifier trained single-sentence, deployed as pair — label insensitive | FIXED: pair-retrained, contract-tested, 100% val |
| A2 | No abstention path (fast path never emits "insufficient") | FIXED: all-evidence voting, neutral→insufficient |
| A3 | Duplicate training scripts (5+) | PARTIALLY: consolidated, canonical trainer identified |
| A4 | No train/inference contract test | FIXED: test_neural_contract.py 6/6 passing |
| A6 | Confidence gating too permissive (0.6 threshold on broken model) | IMPROVED: all-evidence voting + calibration.json (T=1.1065) |
| P1-7 | Hidden test set | DONE: hidden_v1 (270 cases), now saturated at 100% after skill fixes |
| P1-6a | Missing hidden-set skills | DONE: modulo, min-max, set-count, duration arithmetic, ISO-date ordering, quantifier grounding |

### OPEN (Current Session)

| ID | Finding | Severity | Location |
|----|---------|----------|----------|
| O1 | **Neural mesh is still primarily a rule/hybrid engine, not a representational substrate** | CRITICAL | cortex.py:60+ fast paths, general_intelligence.py (regex patterns), claim_evidence.py (deterministic templates), world_knowledge.py (hand-written entities) |
| O2 | **No world model** — entities, relationships, time, state are scattered across modules but not unified | HIGH | Missing entirely as a coherent module |
| O3 | **No executive controller** — cortex.reason() is a single-function pipeline, no goal tracking, planning, or replanning | HIGH | Missing entirely |
| O4 | **No prediction engine** — hindbrain makes predictions but they're not systematically tracked or used for learning | HIGH | brain.py predictions not connected to prediction.py |
| O5 | **No hypothesis engine integration** — cognition/hypotheses.py exists but isn't wired into cortex | HIGH | Disconnected module |
| O6 | **No adaptive computation routing** — all paths run sequentially, no FAST/ROUTINE/DEEP/EXPLORATORY modes | HIGH | cortex.py hardcodes path order |
| O7 | **No tool orchestration layer** — tools.py has Calculator/UnitConversion but no general tool registry/selection/execution framework | MEDIUM | engine/tools.py (limited scope) |
| O8 | **No computer/browser use engine** — src/RelAI/browser/ has playwright server but no cognitive integration | MEDIUM | Missing integration |
| O9 | **No verification engine integration** — cognition/verification.py exists but isn't wired into reasoning pipeline | MEDIUM | Disconnected module |
| O10 | **No memory hierarchy** — working_memory.py and memory.py exist but episodic/semantic/procedural are not separated | MEDIUM | Two implementations, not unified |
| O11 | **No associative retrieval** — retrieval is keyword/embedding-based only, no associative/contextual scoring | MEDIUM | Missing |
| O12 | **No calibration layer in pipeline** — calibration.json exists but ECE not measured post-fix | LOW | neural_calibration.py applied but not validated on new artifacts |
| O13 | **Web research layer unmeasured** — src/RelAI/web/ has search/extract/normalize but no evaluation suite | HIGH (strategic) | Missing benchmarks |
| O14 | **Multiple neural mesh subdirectories with overlapping purpose** — cognition/ vs sweep_neural_mesh/neurons/ vs sweep_neural_mesh/engine/ | MEDIUM | Architecture fragmentation |
| O15 | **Memory modules duplicated** — cognition/memory.py AND sweep_neural_mesh/engine/memory.py AND sweep_neural_mesh/neurons/working_memory.py | MEDIUM | Three implementations |

---

## 4. Capability Classification

### KEEP (Preserve and improve)

| Component | Current State | Disposition |
|-----------|--------------|-------------|
| Evidence graph (cognition/evidence_graph.py) | Functional, provenance-aware | KEEP — integrate into world model |
| Contradiction detection (cognition/contradiction.py + centers.py) | 100% P/R on pairs | KEEP — add source-dependency analysis |
| Belief revision (cognition/beliefs.py) | Functional | KEEP — merge into hypothesis engine |
| Uncertainty engine (cognition/uncertainty.py) | Functional | KEEP — integrate into executive controller |
| Verification (cognition/verification.py + engine/verification.py) | Two implementations | KEEP ONE — unify into verification engine |
| Logic engines (logical_inference.py, proof_mesh.py) | Functional for formal logic | KEEP — position as reasoning module above representations |
| Neural evidence classifier | FIXED, pair-trained | KEEP — as perceptual evidence scorer |
| Contradiction detector (BERT) | Functional | KEEP — as perceptual contradiction scorer |
| SimHash embeddings (embeddings.py) | Lightweight, no GPU needed | KEEP — as fallback retrieval |
| Fast path architecture | Sound (cheap paths first) | KEEP — evolve into adaptive routing |
| World knowledge (world_knowledge.py) | Hand-written entities | REFACTOR — convert to learned/updatable knowledge |
| Memory systems | Three implementations | CONSOLIDATE into unified memory hierarchy |
| Benchmark infrastructure (sweep_benchmark/) | Solid, offline | KEEP — extend with new capability benchmarks |
| Web platform (src/RelAI/) | Functional | KEEP — add cognitive integration hooks |
| Tool system (engine/tools.py) | Limited (calculator + unit conversion) | EXTEND — build general tool registry/orchestration |
| Bridge/validation (cognition/bridge.py) | Neural output validation | KEEP — becomes neural→representation gateway |

### REFACTOR (Rethink and restructure)

| Component | Current Issue | Target |
|-----------|--------------|--------|
| cortex.py (1519 lines) | God orchestrator with 60+ fast paths, mixed concerns | Split into: ExecutiveController (orchestration) + CognitiveMesh (execution) + adaptive routing |
| general_intelligence.py (958 lines) | Regex patterns masquerading as intelligence | Convert to: concept knowledge base + reasoning module |
| claim_evidence.py | Deterministic templates (good scaffolding, not final) | Keep as TEMPORARY scaffolding, replace with learned claim verification |
| world_knowledge.py (530 lines) | Hard-coded entity facts | Replace with: updatable knowledge base with confidence + provenance |
| Memory systems | Duplicate, inconsistent interfaces | Unify into: WorkingMemory + EpisodicMemory + SemanticMemory + ProceduralMemory |
| Training scripts (28+) | Duplicated, inconsistent | Consolidate into staged training pipeline |

### REPLACE (Build new)

| Missing Component | Priority | Phase |
|-------------------|----------|-------|
| World Model | HIGH | Phase 7 |
| Executive Controller | HIGH | Phase 12 |
| Prediction Engine | HIGH | Phase 8 |
| Hypothesis Engine (integrated) | HIGH | Phase 9 |
| Adaptive Computation Router | HIGH | Phase 13 |
| Tool Orchestration | MEDIUM | Phase 14 |
| Computer/Browser Use Layer | MEDIUM | Phase 15 |
| Associative Retrieval | MEDIUM | Phase 4 |
| Memory Hierarchy (unified) | MEDIUM | Phase 6 |
| Calibration Engine | MEDIUM | Phase 17 |
| Skill Registry | MEDIUM | Phase 18 |
| Experience Replay | MEDIUM | Phase 18 |
| Source Independence Analysis | MEDIUM | Phase 11 |
| Counterfactual Engine | LOW | Phase 9 |
| Causal Graph Support | LOW | Phase 10 |

### REMOVE (Dead/overlapping code)

| Component | Reason |
|-----------|--------|
| Duplicate training scripts (train_neural_models.py, train_neural_models_v2.py, neural_training.py, finetune_bert_evidence.py, real_training.py, real_training_v2.py, full_training_pipeline.py, generate_and_train.py, retrain_all.py's unused sections) | Keep train_evidence_pairs.py (canonical), neural_calibration.py, task_generator.py, solver.py, safety.py, versioning.py. Mark others as DEPRECATED. |
| Duplicate memory implementations | Consolidate into single memory package |
| Duplicate verification implementations | Consolidate into single verification engine |

### UNKNOWN (Needs investigation)

| Component | Question |
|-----------|----------|
| All 60+ neuron modules in sweep_neural_mesh/neurons/ | Which are actually used vs imported-but-unused? Many are imported in cortex.py but their actual contribution unknown |
| src/RelAI/web/providers/ | What providers exist, what's their contract? |
| companion/ and sweep_core/ | What's their current role vs the neural mesh? |
| cpp/, native/, models/, graph_benchmark/, neural_eval/ | Are these used or stale? |

---

## 5. Data Flow (Current)

```
Query + Evidence
    ↓
[Perception] Hindbrain: filtering, salience, prediction (brain.py)
    ↓
[Perception] Neural: BERT classification (neural_engine.py)
    ↓
[Perception] Deterministic: claim_evidence.py templates
    ↓
[Representation] ML preprocessing: NER, sentiment, embeddings (cortex.py:_ml_preprocess)
    ↓
[Memory] Working memory insertion (working_memory.py)
    ↓
[Memory] Episodic recall (forebrain.recall_similar)
    ↓
[Memory] Semantic knowledge retrieval (forebrain.get_semantic_knowledge)
    ↓
[Reasoning] 6 Processing Centers (parallel):
    - EvidenceGatherer: scores evidence items
    - CredibilityAssessor: source credibility
    - TemporalSequencer: temporal ordering
    - CausalLinker: causal chains
    - ContradictionDetector: contradiction signals
    - ExplanationBuilder: explanation signals
    ↓
[Integration] IntegrationHub: attention-weighted convergence
    ↓
[Decision] ConsensusEngine: final decision + confidence
    ↓
[Post] ProofMesh + Bayesian update
    ↓
[Post] Metacognition check
    ↓
[Post] Memory recording (episode, working memory update)
    ↓
[Post] Human reasoning module (analogical, causal, counterfactual, abductive, ToM, common sense)
    ↓
[Output] ReasoningResult
```

---

## 6. Technical Debt Summary

| Category | Count | Examples |
|----------|-------|----------|
| Duplicate implementations | 5+ | 3 memory systems, 2 verification, 5+ training scripts |
| Hard-coded knowledge | 1 | world_knowledge.py (530 lines of entity facts) |
| God classes | 2 | cortex.py (1519 lines), general_intelligence.py (958 lines) |
| Disconnected modules | 4 | hypotheses.py, verification.py, prediction.py, bridge.py not wired into cortex |
| No unified world model | 1 | Entity/relationship/state scattered across modules |
| No executive layer | 1 | Single function pipeline, no planning/replanning |
| Training reproducibility | 1 | 28+ scripts, unknown which produced deployed artifacts |
| Missing benchmarks | 4+ | Web research, multimodal, entity resolution, source verification |
| Hidden chain-of-thought risk | 1 | Human reasoning module outputs many intermediate fields in trace |

---

## 7. Phase 0 Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| Every major subsystem identified | ✓ DONE |
| Classified as KEEP/REFACTOR/REPLACE/REMOVE/UNKNOWN | ✓ DONE |
| Dependency graph built | ✓ DONE |
| Data flow mapped | ✓ DONE |
| Neural mesh responsibilities documented | ✓ DONE |
| Benchmark map created | ✓ DONE (see SWEEP_CAPABILITY_AUDIT.md) |
| Technical debt report | ✓ DONE (above) |
| Capability matrix | ✓ DONE (see SWEEP_CAPABILITY_AUDIT.md) |
| Failure map | ✓ DONE (see SWEEP_CAPABILITY_AUDIT.md §5) |

---

## 8. Phase 1 Readiness Assessment

Phase 1 (Representation Reset) requires:
- Clear understanding of what must become representation-centric vs what can remain deterministic
- Interfaces for: embeddings, features, contextual representations, modality boundaries, learned feature pipeline
- Decision: which current rule-based systems are scaffolding (temporary, to be replaced) vs permanent (logic, safety, validation)

**Key decisions for Phase 1:**

1. **claim_evidence.py** — TEMPORARY SCAFFOLDING. Deterministic claim verification is correct behavior, but the implementation should eventually be learned. Keep for now as fallback, mark as scaffolding.

2. **world_knowledge.py** — REFACTOR TO KNOWLEDGE BASE. The concept of factual grounding is sound; hard-coded entities are not. Convert to updatable knowledge with confidence + provenance + learning.

3. **general_intelligence.py regex patterns** — REPLACE WITH CONCEPT REPRESENTATIONS. The regex patterns are the clearest example of "rules pretending to be intelligence."

4. **Logic engines (logical_inference.py, proof_mesh.py)** — KEEP AS REASONING MODULE. Formal logic is a legitimate capability that should sit ABOVE representations.

5. **BERT classifiers** — KEEP AS PERCEPTUAL SCORERS. They provide learned evidence/contradiction signals. They are representational, not rule-based.

6. **cognition/ schema + store** — KEEP AND EXTEND. These are the right foundation for a cognitive data contract. Extend with world model, prediction, hypothesis types.

7. **cognition/ logic.py + rules.py** — KEEP AS DETERMINISTIC REASONING. These are correct: explicit, auditable, deterministic logic for cases where it's appropriate.

---

## 9. Phase 1 Entry Point

Phase 1 begins by creating a new package `sweep_cognitive/` that will house the new cognitive architecture components, while preserving and incrementally integrating with existing `cognition/` and `sweep_neural_mesh/` infrastructure.

The first step is to establish the representation interfaces and convert the neural mesh from rule-centric to representation-centric processing.

**Phase 1 Goal:** Major semantic tasks no longer depend primarily on hand-coded rules.

**Phase 1 Deliverables:**
1. `sweep_cognitive/representation/` — Representation interfaces
2. `sweep_cognitive/perception/` — Embedding/feature pathways
3. `sweep_cognitive/context/` — Contextual representations
4. Updated `ReasoningCortex` that uses representations instead of regex patterns
5. Migration path for world_knowledge.py → learned knowledge base
6. Contract tests ensuring representation quality

---

*End of Phase 0 Audit*
