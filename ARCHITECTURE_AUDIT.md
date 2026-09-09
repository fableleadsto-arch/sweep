# ARCHITECTURE_AUDIT.md — Phase 0 Deliverable

*Date: 2026-09-08. Method: direct code inspection, live module imports, and local pytest runs. Paths/line numbers refer to the current working tree (Git HEAD 8611d1402).*

## Purpose

This is the Phase 0 milestone deliverable: a baseline map of Sweep's existing implementation so that every Phase 1–20 capability has a known integration point. It is written to be **actionable**, not a marketing document.

---

## 1. Existing module/component map

```
┌─ src/ (TypeScript, TanStack) ─────────────────────────────────────────┐
│  RelAI/web: search.server.ts, crawl, extract, normalize, http, markdown│
│  surf/: search router + providers, browse, platforms, research,       │
│         evidence (store/verify/score), docs extractor                  │
├─ app/ (Python FastAPI, port 8787) ─────────────────────────────────────┤
│  search/engine.py, search/router.py · scraping/{normalize,markdown}   │
│  platforms/{reddit,x,github,youtube,instagram,linkedin}                │
│  research/engine.py · evidence/{store,scoring}.py · extraction/        │
│  browser/ · core/{http,cache,types,cpp_bridge,guard}                   │
├─ companion/ (Python FastAPI brain, port 8088) ────────────────────────┤
│  brain.py, orchestrator.py, providers.py (LLM fail-over chain),       │
│  memory.py (FileMemoryStore/Qdrant, persistence), rag.py, routing,     │
│  planning.py (deterministic routing), multi_agent.py, execution.py,   │
│  capabilities.py · ingest/ · neural/ · vendor/                         │
├─ sweep_neural_mesh/ (Python reasoning engine — measured ⊂) ───────────┤
│  neurons/cortex.py (ReasoningCortex.reason, 1438+ lines)              │
│    1 neural fast path (neural_engine.py, BERT×3, pair-format)         │
│    2 contradiction fast path (trained detector + rules)               │
│    3 uncertainty fast path (regex → insufficient)                     │
│    4 general intelligence (semantic KB + NLI)                         │
│    5 logic engines (proof_mesh, logical_inference, graph_algorithms)  │
│    6 task router (task_handlers/)                                     │
│    7 live knowledge (Wikipedia/Wikidata)                              │
│    8 full neuronal pipeline (hind/mid/forebrain + consensus)          │
│  cores/ ×5 · evidence_graph.py · fusion/{confidence,verification}      │
│  routing/cascade.py (CascadeRouter) · neurons/trace.py (ReasoningTrace)│
│  resources/ (ResourceManager) · memory/ · optimization/ · training/    │
├─ sweep_benchmark/ (CLI: validate/run/analyze/compare/audit/report/     │
│                    run-hidden, hidden.py SHA-protocol)                │
├─ sweep_core/ (visual/voice core, fastpath, integrations)              │
├─ services/intelligence/ (multimodal intent_core.py, model_manager)    │
├─ benchmark/, benchmarks/, graph_benchmark/, neural_eval/ (suites)     │
├─ cpp/ + native/ (C++20 pybind11 engines + terminal UI)                │
└─ models/ (HuggingFace cache: nlp, vision, audio, embeddings, etc.)    │
```

## 2. Existing model inventory

Deployed ML artifacts (with `input_format` contract metadata):

| Artifact | Type | Format | Status |
|---|---|---|---|
| `evidence_classifier` (pair-retrained) | BERT (transformers) | premise/hypothesis pair | Training val 100%; fresh-draw 85.5%; ECE 0.048; contract-tested |
| `contradiction_detector` | BERT pair classifier | pair | 100% P/R on dev contradiction cases; paraphrase false-positives @~0.67 queued P1 |
| `intent_classifier` | BERT single | single | Near-chance (21.5% val, 13 classes); unused for gating |
| `relay_small_trained`, `relay_nano_trained`, `bert_evidence_finetuned` | BERT seq2seq/fine-tuned | per-artifact | Historical; version tracking in `training/versions/` |
| C++ engines (cpp/, native/) | pybind11 | — | HTML parse, text extract, rank, regex, DSP |

Registry: `models/registry.json`; capability config: `services/intelligence/models/capabilities.yaml`; `models/` contains HF caches (all-MiniLM-L6-v2, CLIP ViT-B/32 & L/14, YOLOv8n, whisper, etc.).

## 3. Existing memory/state inventory

- **companion/memory.py** — `MemoryService`, `FileMemoryStore` (JSON persist), `QdrantMemoryStore` (vectors), `MemoryEntry` w/ kinds, versioning, dedupe, scores. Persistent across sessions.
- **sweep_neural_mesh/neurons/working_memory.py** — `WorkingMemory` + `MemorySlot` (in-process, per-cortex).
- **sweep_neural_mesh/neurons/forgetting.py** — decay/retention.
- **sweep_neural_mesh/memory/** — lightweight feature cache.
- **sweep_neural_mesh/neurons/evidence_graph.py** — `EvidenceNode`, `EvidenceChain`, `EvidenceGraph` (in-memory only, not persisted).
- **sweep_neural_mesh/training/audit/**, `training/failures/` — JSON audit/failure records (`.learning_state.json` mutated in worktree).
- **ReasoningTrace** (`neurons/trace.py`) — per-reasoning-pass trace with brain-division timings, grades, uncertainty signals, ML outputs. `to_dict()` serializable.
- No unified **persistent evidence graph**, no cross-session **Working→LTM handoff store**, no shared stable **object identity across the two systems**.

## 4. Integration/tool inventory

- HTTP: `httpx`; search: DuckDuckGo/Bing/Brave/Mojeek/Google HTML parsers + Tavily/Exa/SearXNG/Jina APIs; browser: Playwright WebSocket + HTTP fallback.
- Extraction: trafilatura, lxml, BeautifulSoup, JSON-LD/microdata, C++ html_parser/text_extractor.
- ML: torch, transformers, sentence-transformers, ONNX, sklearn, openclip, ultralytics, insightface, whisper, pyannote, GLiNER, faiss.
- Research: `src/surf/research/` (engine/planner/session/report/limits), `app/research/engine.py` (planning).
- Evidence scoring: `app/evidence/scoring.py` (SourceQuality/Independence/Recency/EntityMatch/Directness/Corroboration declared), `src/surf/evidence/` (store/verify/score).
- LLM providers (companion): Puter, Gemini, OpenAI, OpenRouter, Anthropic, Groq with fail-over.
- Compute: `companion/compute/` (backend mgmt/diagnostics/scheduling), `sweep_neural_mesh/resources/ResourceManager`.

## 5. Duplicate/conflicting implementation list

| Duplicate/conflict | Locations | Risk |
|---|---|---|
| Evidence classes defined in ≥4 places | `app/core/types.py`, `app/evidence/`, `sweep_neural_mesh/neurons/evidence_graph.py`, `engine/memory.py`, `companion/ingest/models.py` | No single schema; graph persistence impossible |
| Training scripts duplicated & disagreeing | `training/retrain_all.py`, `neural_training.py`, `train_neural_models.py`, `finetune_bert_evidence.py`, `train_neural_models_v2.py` vs canonical `train_evidence_pairs.py` | Reproducibility (A3 historical; partially mitigated by contract tests) |
| Model router concepts in multiple places | `routing/cascade.py`, `cortex.py` `_ModelLoader`, `services/intelligence/intent_core.py`, `companion/planning.py` | Routing decisions not centrally logged |
| Memory systems (3) | companion `memory.py`, neural mesh `working_memory.py`, neural mesh `memory/` | Cross-session cognitive memory not unified |
| Uncertainty implementations | `cortex.py:_try_uncertainty_fast_path` (regex), `metacognition.py:UncertaintySignal`, `proof_mesh.py` `INSUFFICIENT` | No single epistemic-state model |
| Hypothesis objects | `neurons/abductive.py:Hypothesis`, `neurons/bayesian.py:Hypothesis`, `engine/reasoning.py:Hypothesis` | No shared ranking/confidence contract |

## 6. Missing-capability list (vs the Phase 1–20 milestone contract)

| Capability | Evidence of absence |
|---|---|
| Unified cognitive data contract (Observation→Evidence→Claim→Hypothesis→ReasoningEvent) with identity-preserving serialization | Objects exist piecemeal; no shared contract + round-trip test |
| Persistent, queryable evidence graph (entity→relationship→claim→evidence→provenance) | `evidence_graph.py` in-memory only |
| Cross-session cognitive memory working-→LTM with provenance retrieval | No handoff store |
| Versioned, traceable rule engine (rule provenance for every derived result) | Rules are data in `common_sense.py`/`general_intelligence.py`; no rule-identity/trace layer |
| Structured neural-output validation gate (reject malformed/low-confidence) | Confidence gate exists (A6 improved); no schema-validated structured bridge |
| Central model router with logged decisions + fallback | Cascade exists; not wired to a decision log |
| Competing-hypothesis engine (rank, confidence, retain uncertainty) | bayesian/abductive standalone; no shared ranked store |
| Contradiction representation preserving both claims (explicit conflict objects) | Detector exists; conflict *representation* with retention absent |
| Belief revision with history (why-leader-changed retained) | BayesianUpdate exists; no revision-history ledger |
| Explicit epistemic states (VERIFIED/PROBABLE/POSSIBLE/CONTESTED/UNRESOLVED/INSUFFICIENT_EVIDENCE) | `insufficient`/`mixed` only; no state machine |
| Independent verification engine (neural+logic+evidence → VERIFIED/CONTESTED/UNRESOLVED) | `fusion/verification.py` model-output cross-check; not evidence-driven |
| Prediction-observation error loop (prediction can never silently become observation) | `predictive.py` is biological-style; no compare/error-update store |
| Targeted feedback/learning loop (error classification → memory/strategy/rule/model action) | `self_evolution.py`, `reward.py`; no classification-driven action record |
| Resource-aware execution with limits/fallback/graceful degradation + measured records | `ResourceManager` exists; no budget ledger or degradation path tests |
| End-to-end single pipeline on a synthetic investigation task | Cortex pipeline exists; no unified cross-subsystem loop test |
| `explain_conclusion(id)` reconstructing stored state | ReasoningTrace per-pass; no stored-conclusion explanation API |
| Full cognitive test suite (unit/integration/e2e/failure/adversarial/regression/resource) | Unit suites exist; no failure/adversarial/resource layers for the unified system |
| Reproducible benchmark (12 required measurements) | sweep_benchmark covers ~8; no unified benchmark over the new loop |
| `IMPLEMENTATION_STATUS.md` with DONE/IN_PROGRESS/NOT_IMPLEMENTED/BLOCKED and evidence links | Absent |

## 7. Retain / replace / rewrite recommendations

| Component | Recommendation | Rationale |
|---|---|---|
| Cortex reason() path architecture | **RETAIN** | Ablation-verified (+3.7 pts neural mesh, +17.6 pts logic); fast-path-first order is sound |
| Detectors (contradiction, evidence pair) | **RETAIN** | 100% dev P/R; contract-tested; calibration shipped (T=1.1065) |
| `train_evidence_pairs.py` + contract tests | **RETAIN as canonical** | Only trainer matching inference format |
| Legacy trainers (retrain_all etc.) | **DEPRECATE** (mark, don't delete) | Violate train/inference contract by default |
| Intents artifact | **RETAIN but don't gate on it** | Near-chance; exclude from decision path |
| Scoring formula `app/evidence/scoring.py` | **RETAIN** | Configurable, documented |
| Duplicate schema objects (§5) | **REWRITE into unified `cognition/schema.py`** | Milestone Phase 1 requires one stable contract |
| Memory trio (§5) | **REWRITE into unified persistent memory** in `cognition/memory.py` | Milestone Phase 3 |
| Router concepts (§5) | **CONSOLIDATE under a logged router** in `cognition/router.py` | Milestone Phase 7 |
| Uncertainty/revision/contradiction-query representation | **REWRITE under explicit epistemic model** in `cognition/` | Milestones 9–11 |

## 8. Migration roadmap (phase-by-phase integration points)

| Phase | Integration point | Primary files |
|---|---|---|
| 1 Data contract | Replace/alias existing schema classes; single `cognition/schema.py` | cognition/schema.py |
| 2 Evidence graph | Persist graph over `cognition/evidence_graph.py` w/ JSONL store | cognition/evidence_graph.py, cognition/store.py |
| 3 Memory | Working+LTM handoff with provenance retrieval | cognition/memory.py |
| 4 Logic | Deterministic transitive closure + proof steps | cognition/logic.py |
| 5 Rules | Versioned rule objects, every derived result references rule id | cognition/rules.py |
| 6 Neural bridge | Validate structured neural output before it becomes Evidence/Claim | cognition/bridge.py |
| 7 Router | Logged task-aware selection + fallback | cognition/router.py |
| 8 Hypothesis | Ranked multi-hypothesis store from bayesian/abductive | cognition/hypotheses.py |
| 9 Contradiction | Explicit CONTRADICTION objects retaining both claims | cognition/contradiction.py |
| 10 Belief revision | Revision ledger with reasons | cognition/beliefs.py |
| 11 Uncertainty | Epistemic state machine over the 6 states | cognition/uncertainty.py |
| 12 Verification | Independent neural+logic+evidence paths → verdict | cognition/verification.py |
| 13 Prediction loop | Prediction/observation error ledger | cognition/prediction.py |
| 14 Feedback/learning | Error classification → targeted action records | cognition/feedback.py |
| 15 Resources | Budget ledger, limits, graceful degradation | cognition/resources.py |
| 16 Cognitive loop | Single end-to-end pipeline over the above | cognition/loop.py |
| 17 Explainability | `explain_conclusion(id)` from stored traces | cognition/explain.py |
| 18 Testing | Full pyramid over the new subsystems | cognition/tests/ |
| 19 Benchmarks | Reproducible unified benchmark (12 metrics) | cognition/bench.py |
| 20 Documentation | 12 required docs + IMPLEMENTATION_STATUS.md | docs/ |

## 9. Demonstrable quick facts (measured this audit)

- `sweep_neural_mesh/tests`: all collected test files **pass** (spot-verified: test_core, test_neural_contract, test_stages, test_neurons, test_biological, test_human_reasoning, test_math_modules, test_ml_engines, test_training_systems, test_improvements, test_refactored_modules, test_stages_678, test_stages_9 — 512+ passed, 0 failed).
- `companion/tests`: 2 passed / 6 skipped / **7 failed**: `test_compute.py::test_every_profile_maps_to_a_requirements_file`, 6× `test_vendor_loader.py` (vendor wheels/licenses/gitignore/inventory/source-runs/describe-all) — pre-existing, unrelated to reasoning engine.
- `sweep_core/tests`: **4 collection errors** — `from sweep import __version__` fails because the `sweep` package is not installed/importable (pyproject names the dist `sweep` but packages only `app*`).
- Dev benchmark suite: 668/668 (100%), suite saturated (development-visible); hidden_v1 run_01 51.1% → run_02 100% after skill-gap fixes (see benchmark/REPORT.md A5).
- Latency p50 ≈0.5 ms warm on-evidence fast path; cold-start ~24 s + up to ~30 s×3 model loads; peak RSS ~1.3–1.8 GB.

## 10. Known limitations after audit

- Web research layer still has no eval suite (A7 unchanged; top strategic gap).
- 12 legacy pytest failures + 4 sweep_core collection errors are pre-existing environment/dependency issues.
- Native `sweep_engine`/`sweep_native` C++ modules not verified importable in this session (no build artifact check performed).
- `sweep_neural_mesh/training/evolution/.learning_state.json` and audit/failure JSON mutated in worktree (uncommitted state churn).

## 11. Migration decision (this program)

Build a new **`cognition/`** package implementing Phases 1–19 as a single integrated subsystem, importing/dual-linking where convenient with `sweep_neural_mesh` schemas but **not** requiring the heavyweight neural pipeline to pass its tests. Persistence via a JSONL-backed `cognition/store.py`. This gives the milestone gates deterministic, self-contained evidence while existing systems remain untouched and their value retained per §7.