# Sweep Neural Mesh — Architecture

## Overview

Sweep is a **neural reasoning system** built on a biologically-inspired architecture. It processes queries through multiple specialized modules that mirror brain regions, then uses multi-core parallel processing with consensus voting to produce accurate, well-calibrated answers.

The system has been transformed from a rule-based engine to a **true neural engine** using pre-trained transformer models for knowledge retrieval, reasoning, and question answering.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWEEP NEURAL MESH                            │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ General  │   │  World   │   │  Live    │   │  Multi-  │   │
│  │ Intel.   │   │ Knowl.   │   │ Knowl.   │   │  Core    │   │
│  │(neural)  │   │(embedded)│   │ (APIs)   │   │(5 cores) │   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   │
│       │              │              │              │           │
│       └──────────────┴──────────────┴──────────────┘           │
│                            │                                   │
│                    ┌───────┴───────┐                           │
│                    │    Cortex     │                           │
│                    │ (orchestrator)│                           │
│                    └───────┬───────┘                           │
│                            │                                   │
│       ┌────────────────────┼────────────────────┐              │
│       │                    │                    │              │
│  ┌────┴────┐         ┌─────┴─────┐        ┌────┴────┐        │
│  │Hindbrain│         │ Midbrain  │        │Forebrain │        │
│  │(filter) │         │ (route)   │        │(process) │        │
│  └─────────┘         └───────────┘        └──────────┘        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Neural Models                               │   │
│  │  SentenceTransformer │ NLI │ QA │ Embeddings            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Neural General Intelligence (`general_intelligence.py`)

**Purpose**: Neural fast-path answering using pre-trained transformer models.

- **SentenceTransformer** (`all-MiniLM-L6-v2`) for semantic knowledge retrieval
- **Cross-encoder NLI** (`cross-encoder/nli-deberta-v3-base`) for deductive/abductive reasoning
- **QA model** (`distilbert-base-cased-distilled-squad`) for factual extraction
- **Embedded knowledge base** with 500+ entries from authoritative sources
- **Latency**: ~50-200ms per query (model inference)
- **Confidence**: 0.7 – 0.95 depending on retrieval similarity and NLI scores

```
Query → Embed → Semantic Search → Top-K Knowledge → NLI Verification → Answer
```

**Key Features**:
- Semantic similarity search instead of regex pattern matching
- Natural Language Inference for reasoning (supports/refutes/neutral)
- Neural contradiction detection
- Graceful fallback to structured knowledge when models unavailable

### 2. World Knowledge (`knowledge_training.py` + `knowledge_supplement.py`)

**Purpose**: Structured knowledge base with 722+ entries across 29 domains.

- Entries have: topic, answer, source, confidence, category, relationship.
- Loaded into the Cortex at init time.
- Embedded into vector space for neural retrieval.
- Used for evidence grounding and fact verification.

### 3. Live Knowledge (`live_knowledge.py`)

**Purpose**: Real-time knowledge retrieval from external APIs.

- **Wikipedia API** — encyclopedic knowledge.
- **Wikidata API** — structured entity data.
- LRU cache (2000 entries) to avoid redundant calls.
- 3s timeout with connection pooling.

### 3a. Web Scraper (`web_scraper/`)

**Purpose**: Multi-source web scraping and content extraction.

```
neurons/web_scraper/
├── __init__.py         — Package exports
├── scraper.py          — Core WebScraper: multi-strategy fetching
├── content.py          — ContentExtractor: HTML → clean text
└── researcher.py       — WebResearcher: multi-query research
```

**Sources**:
| Source | Method | Speed | Coverage |
|--------|--------|-------|----------|
| Wikipedia | API (structured) | Fast | Encyclopedia |
| Wikidata | API (structured) | Fast | Entity data |
| arXiv | API (XML) | Medium | Academic papers |
| OpenAlex | API (JSON) | Medium | Academic works |
| Generic HTML | HTTP + regex parsing | Slow | Any webpage |

### 4. Multi-Core Neural Processing (`cores/`)

**Purpose**: Parallel specialized processing with consensus voting.

```
┌─────────────────────────────────────────────────────────┐
│  MultiCoreCoordinator                                   │
│                                                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐            │
│  │ Factual   │ │ Reasoning │ │ Evidence  │            │
│  │ Core      │ │ Core      │ │ Core      │            │
│  │ 80+ facts │ │ 30+ logic │ │ relevance │            │
│  └───────────┘ └───────────┘ └───────────┘            │
│  ┌───────────┐ ┌───────────┐                           │
│  │ Temporal  │ │ Causal    │                           │
│  │ Core      │ │ Core      │                           │
│  │ dates/time│ │ cause→eff │                           │
│  └───────────┘ └───────────┘                           │
│       ↓              ↓              ↓                   │
│  ┌─────────────────────────────────────────────┐       │
│  │  Consensus Engine (voting + agreement)      │       │
│  └─────────────────────────────────────────────┘       │
│                     ↓                                   │
│              Final Decision                             │
└─────────────────────────────────────────────────────────┘
```

| Core | File | Responsibility |
|------|------|----------------|
| **FactualCore** | `cores/factual_core.py` | Knowledge lookup, number extraction |
| **ReasoningCore** | `cores/reasoning_core.py` | Logic, common sense, yes/no, math |
| **EvidenceCore** | `cores/evidence_core.py` | Evidence relevance, definition extraction |
| **TemporalCore** | `cores/temporal_core.py` | Dates, historical events |
| **CausalCore** | `cores/causal_core.py` | Cause-effect chains |

All cores implement `NeuralCoreProtocol` from `core_protocol.py`.

### 5. Cortex (`cortex.py`)

**Purpose**: Master orchestrator that runs the full reasoning pipeline.

Implements a **three-division brain architecture**:

| Division | Components | Function |
|----------|-----------|----------|
| **Hindbrain** | Brainstem, Cerebellum | Fast filtering, reflexes, energy gating |
| **Midbrain** | Thalamus, VTA | Signal routing, attention, reward prediction |
| **Forebrain** | Cortex, Basal Ganglia, Hippocampus | Processing centers, memory, decision-making |

**Reasoning flow**:

```
Raw Input
    ↓
Neural GI Fast Path → semantic retrieval + NLI → answer? → return
    ↓
Live Knowledge → answer? → return
    ↓
Hindbrain → filter → reflex check → energy gate
    ↓
Midbrain → value predict → salience → inhibit → route
    ↓
Forebrain → workspace → working memory → processing centers
    ↓
Cortex-BG-Thalamus loop → metacognition → output
```

### 6. Neural Training Pipeline (`training/`)

**Purpose**: Fine-tune and manage neural models for Sweep.

| Model | File | Purpose |
|-------|------|---------|
| **Evidence Classifier** | `neural_models/evidence_classifier/` | Classifies evidence as supports/refutes/neutral |
| **Contradiction Detector** | `neural_models/contradiction_detector/` | Detects contradictions between statements |
| **Intent Classifier** | `neural_models/intent_classifier/` | Classifies query intent (13 categories) |

**Training**: `python -m sweep_neural_mesh.training.neural_training`

### 7. Self-Evolution (`evolution/`)

**Purpose**: Enables the system to learn and adapt from interactions.

| Module | File | Function |
|--------|------|----------|
| **LearningModule** | `evolution/learning.py` | Track successes/failures, learn from feedback |
| **EvolutionEngine** | `evolution/engine.py` | Mutate failing patterns, cross-pollinate |
| **KnowledgeAcquisition** | `evolution/knowledge.py` | Acquire new knowledge, validate, deduplicate |
| **PerformanceTracker** | `evolution/tracker.py` | Monitor metrics, calibrate confidence, suggest optimisations |
| **Coordinator** | `evolution/coordinator.py` | Orchestrate all the above |

---

## Data Flow

### Fast Path (common questions)

```
Query → Neural GI (semantic search + NLI) → answer with confidence ≥ 0.75 → return
```

### Medium Path (questions needing evidence)

```
Query → Cortex → Hindbrain → Forebrain → Processing Centers → answer
```

### Slow Path (complex reasoning)

```
Query → Cortex → Full brain pipeline → Multi-Core parallel → Consensus → answer
```

### Learning Path (after answering)

```
Answer → Self-Evolution → learn → evolve → acquire → optimise
```

---

## Neural Models

### Pre-trained Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `all-MiniLM-L6-v2` | Semantic embeddings | 80MB |
| `cross-encoder/nli-deberta-v3-base` | Natural Language Inference | 400MB |
| `distilbert-base-cased-distilled-squad` | Question Answering | 65MB |
| `facebook/bart-large-mnli` | Fallback NLI | 1.6GB |

### Knowledge Base

- 500+ entries from `knowledge_training.py` (authoritative sources)
- 80+ common-sense facts
- Embedded using SentenceTransformer
- Fast cosine similarity search

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `cortex.py` | ~1200 | Master orchestrator |
| `general_intelligence.py` | ~500 | Neural knowledge retrieval + NLI reasoning |
| `trace.py` | ~200 | ReasoningTrace + ReasoningResult data classes |
| `fast_path.py` | ~140 | Early-exit for simple queries |
| `evidence_pipeline.py` | ~135 | Cross-referencing and corroboration |
| `complexity.py` | ~90 | Adaptive pipeline depth classification |
| `human_reasoning.py` | ~160 | Common sense, abductive, ToM, narrative, analogical, causal, counterfactual |
| `core_protocol.py` | ~120 | NeuralCoreProtocol + CoreResult + ConsensusResult |
| `cores/*.py` | ~100 each | Individual neural cores (factual, reasoning, evidence, temporal, causal) |
| `evolution/*.py` | ~80 each | Self-evolution modules (learning, engine, knowledge, tracker) |
| `knowledge_training.py` | ~1500 | 500-entry knowledge base |
| `training/neural_training.py` | ~400 | BERT fine-tuning pipeline |
| `training/neural_integration.py` | ~280 | Neural layer integration |

---

## Neural Transformation (2026-09-02)

The General Intelligence module was transformed from a rule-based system to a neural engine:

**Before (Rule-based)**:
- 922+ pre-compiled regex patterns
- Keyword index → regex match → answer
- Hardcoded reasoning rules
- ~0.1ms latency

**After (Neural)**:
- SentenceTransformer embeddings for semantic retrieval
- Cross-encoder NLI for reasoning (supports/refutes/neutral)
- QA model for factual extraction
- 500+ embedded knowledge entries
- ~50-200ms latency

**Benefits**:
- **Generalizes** to novel combinations of facts
- **Semantic understanding** instead of pattern matching
- **Calibrated confidence** from model probabilities
- **Extensible** — new knowledge automatically embedded
- **Graceful degradation** — falls back to structured knowledge when models unavailable

---

## Design Principles

1. **Protocol-driven**: All cores implement `NeuralCoreProtocol`.
2. **Neural-first**: Use pre-trained models for reasoning, fall back to rules only when needed.
3. **Lazy-loaded**: Expensive modules (ML engines, live APIs) loaded on first use.
4. **Fail-safe**: External failures (APIs, ML) never crash the pipeline.
5. **Observable**: Every result includes latency, confidence, and reasoning trace.
6. **Evolvable**: The system learns from interactions and adapts over time.
7. **Modular**: Each concern lives in its own file (~100-200 lines).
