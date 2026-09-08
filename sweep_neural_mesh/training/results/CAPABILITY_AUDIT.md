# SWEEP CAPABILITY AUDIT & TRAINING PLAN
## Date: 2026-09-02

---

## 1. SWEEP CAPABILITY MAP

### A. REASONING ENGINE (sweep_neural_mesh/neurons/)

| Feature | Status | Implementation | Dependencies | Test Coverage | CPU | GPU | Priority |
|---------|--------|---------------|-------------|--------------|-----|-----|----------|
| Neural General Intelligence | **trained** | SentenceTransformer + NLI + QA | torch, transformers, sentence-transformers | 85% | ✅ | optional | HIGH |
| Cortex Orchestrator | **implemented** | 3-division brain architecture | neurons/* | 70% | ✅ | optional | HIGH |
| Multi-Core Processing | **implemented** | 5 specialized cores with consensus | cores/* | 60% | ✅ | optional | HIGH |
| Logical Inference | **implemented** | Modus ponens/tollens, transitivity | numpy | 50% | ✅ | ❌ | HIGH |
| Proof Mesh | **implemented** | Atom/bond grounding + propagation | numpy | 45% | ✅ | ❌ | MEDIUM |
| Bayesian Reasoner | **implemented** | Evidence updating | numpy | 40% | ✅ | ❌ | MEDIUM |
| Contradiction Detection | **trained** | Neural NLI-based | transformers | 80% | ✅ | optional | HIGH |
| Evidence Classification | **trained** | Neural supports/refutes/neutral | transformers | 75% | ✅ | optional | HIGH |
| Intent Classification | **trained** | Neural 13-category | transformers | 70% | ✅ | optional | HIGH |
| Analogical Reasoning | **implemented** | Structural mapping | numpy | 30% | ✅ | ❌ | LOW |
| Causal Model | **implemented** | Graph-based causal chains | networkx | 35% | ✅ | ❌ | MEDIUM |
| Counterfactual Reasoner | **implemented** | What-if analysis | numpy | 25% | ✅ | ❌ | LOW |
| Abductive Reasoner | **implemented** | Hypothesis generation | numpy | 30% | ✅ | ❌ | MEDIUM |
| Theory of Mind | **implemented** | Agent state tracking | numpy | 20% | ✅ | ❌ | LOW |
| Narrative Engine | **implemented** | Story arc construction | numpy | 25% | ✅ | ❌ | LOW |
| Sentiment Engine | **implemented** | Valence detection | numpy | 40% | ✅ | ❌ | MEDIUM |
| NER Engine | **implemented** | Entity extraction | gliner | 45% | ✅ | optional | MEDIUM |
| Text Summarizer | **implemented** | Extractive summarization | transformers | 35% | ✅ | optional | MEDIUM |
| Semantic Embeddings | **implemented** | Sentence embeddings | sentence-transformers | 60% | ✅ | optional | HIGH |
| Fuzzy Logic | **implemented** | Fuzzy set operations | numpy | 30% | ✅ | ❌ | LOW |
| Information Theory | **implemented** | Entropy, mutual info | numpy | 30% | ✅ | ❌ | LOW |
| Graph Algorithms | **implemented** | PageRank, centrality | networkx | 35% | ✅ | ❌ | LOW |
| Synaptic Plasticity | **implemented** | LTP/LTD learning | numpy | 40% | ✅ | ❌ | MEDIUM |
| Evidence Grading | **implemented** | Multi-dimensional grading | numpy | 50% | ✅ | ❌ | MEDIUM |
| Self-Evolution | **implemented** | Learning + adaptation | numpy | 35% | ✅ | ❌ | LOW |

### B. KNOWLEDGE BASE (sweep_neural_mesh/neurons/)

| Feature | Status | Implementation | Test Coverage | Priority |
|---------|--------|---------------|--------------|----------|
| Knowledge Training | **trained** | 500+ entries from authoritative sources | 80% | HIGH |
| Knowledge Supplement | **implemented** | 222 supplementary entries | 60% | MEDIUM |
| World Knowledge | **implemented** | Entity properties + causal chains | 50% | MEDIUM |
| Live Knowledge | **implemented** | Wikipedia/Wikidata APIs | 55% | MEDIUM |
| FAISS Index | **implemented** | Vector search over knowledge | 70% | HIGH |

### C. WEB INTELLIGENCE (app/)

| Feature | Status | Implementation | Test Coverage | Priority |
|---------|--------|---------------|--------------|----------|
| Multi-Engine Search | **implemented** | 6 HTML parsers + 4 API providers | 60% | HIGH |
| Search Router | **implemented** | Intent-based provider routing | 50% | HIGH |
| Page Extraction | **implemented** | HTML→Markdown→PageData | 65% | HIGH |
| Browser Sessions | **implemented** | Headless browsing | 40% | MEDIUM |
| Research Engine | **implemented** | Multi-step research | 45% | HIGH |
| Platform Adapters | **implemented** | YouTube, X, LinkedIn, etc. | 35% | MEDIUM |
| Evidence Scoring | **implemented** | Source relevance/authority | 55% | HIGH |
| Evidence Store | **implemented** | In-memory deduplication | 50% | MEDIUM |

### D. INGESTION PIPELINE (companion/ingest/)

| Feature | Status | Implementation | Test Coverage | Priority |
|---------|--------|---------------|--------------|----------|
| Ingest Brain | **implemented** | Multi-source intelligence | 40% | MEDIUM |
| Contradictions | **implemented** | Cross-source contradiction detection | 45% | HIGH |
| Deduplication | **implemented** | Content deduplication | 50% | MEDIUM |
| Source Adapters | **implemented** | Wikipedia, arXiv, GitHub, etc. | 45% | MEDIUM |
| Evidence Scoring | **implemented** | Multi-factor scoring | 40% | HIGH |
| Security | **implemented** | Input sanitization | 50% | MEDIUM |

### E. NEURAL MESH (companion/neural/)

| Feature | Status | Implementation | Test Coverage | Priority |
|---------|--------|---------------|--------------|----------|
| Model Registry | **implemented** | Model selection & routing | 40% | MEDIUM |
| Neural Router | **implemented** | Task-based model routing | 35% | MEDIUM |
| Training Pipeline | **implemented** | Checkpointing, datasets | 40% | HIGH |
| Architecture | **implemented** | Model architecture registry | 35% | LOW |
| Evaluation | **implemented** | Benchmark runners | 50% | HIGH |

### F. BENCHMARKS (benchmarks/ + neural_eval/)

| Feature | Status | Implementation | Test Coverage | Priority |
|---------|--------|---------------|--------------|----------|
| Dataset Generator | **implemented** | 1000-case deterministic generation | 80% | HIGH |
| Sweep Runner | **implemented** | Cortex benchmark runner | 75% | HIGH |
| Scorer | **implemented** | Decision accuracy + latency | 70% | HIGH |
| Neural Eval Suite | **implemented** | 8-section neural evaluation | 65% | HIGH |

---

## 2. CURRENT BASELINE PERFORMANCE

### From previous training (FINAL_REPORT.txt):
- Overall: **93.3%** on 60 test samples (7 domains)
- contradiction: 100%, temporal: 100%, evidence: 88.9%, source_independence: 100%, logic: 80%
- **intent: 0%** (not tested in current eval), **entity: 0%** (not tested in current eval)

### From benchmark (REPORT.txt):
- Overall: **95.2%** vs GPT-4o 82.1% on 1000 cases (10 categories)
- basic_logic: 87%, ambiguity: 73% — **weakest categories**

### Key Weaknesses Identified:
1. **basic_logic: 87%** — modus tollens, syllogisms need improvement
2. **ambiguity: 73%** — handling multiple valid interpretations
3. **novel_structures: 97%** — already strong, minor edge cases
4. **Generalization (logic): 33.3%** — critical weakness on unseen patterns

---

## 3. TRAINING PLAN

### PHASE 1: Dataset Enrichment (High Impact)
- Expand training data for weak domains: logic, ambiguity, generalization
- Add 200+ hard negative examples
- Add 100+ adversarial examples
- Add 100+ cross-domain generalization examples

### PHASE 2: Neural Model Training (High Impact)
- Re-train evidence classifier with expanded data
- Re-train contradiction detector with hard negatives
- Re-train intent classifier with expanded categories
- Fine-tune NLI model on Sweep-specific reasoning tasks

### PHASE 3: Reasoning Logic Improvements (Medium Impact)
- Improve modus tollens and syllogistic reasoning in logical_inference.py
- Improve proof_mesh.py propagation for novel structures
- Add uncertainty quantification to confidence calibration

### PHASE 4: Generalization Training (High Impact)
- Train on synthetic patterns, test on natural language
- Train on simple language, test on complex language
- Cross-domain transfer tests

### PHASE 5: Regression Testing (Critical)
- Run full 1000-case benchmark
- Run full 8-section neural eval
- Compare before/after with statistical significance

---

## 4. HARDWARE PLAN

- **CPU-only operation**: All training must work on CPU
- **Batch sizes**: Small (4-8) for memory efficiency
- **Model sizes**: Prefer DistilBERT/MiniLM over large models
- **Lazy loading**: Models loaded on first use, unloaded when possible
- **Quantization**: INT8 where appropriate for inference speed
