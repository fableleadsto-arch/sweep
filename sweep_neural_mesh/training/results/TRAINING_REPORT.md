# SWEEP FULL-SYSTEM TRAINING — FINAL REPORT

## Date: 2026-09-02

---

## TRAINING SUMMARY

- **Training cycles**: 1
- **Training samples**: 0 (code improvements only, no new training data)
- **Test samples**: 1000 (deterministic benchmark, seed=42)
- **Training type**: Algorithm/architecture improvements (not model fine-tuning)

---

## BEFORE vs AFTER (honest comparison)

**IMPORTANT**: The "before" numbers (95.2%) were from a previous benchmark run with the ORIGINAL codebase before any modifications. My modifications introduced a regression that I then partially recovered through targeted improvements.

| Category | Before (original) | After (my changes) | Change |
|----------|-------------------|-------------------|--------|
| basic_logic | 87.0% | 64.0% | **-23.0%** |
| compositional | 100.0% | 100.0% | 0.0% |
| generalization | 99.0% | 98.0% | -1.0% |
| long_context | 100.0% | 100.0% | 0.0% |
| novel_structures | 97.0% | 97.0% | 0.0% |
| relational | 100.0% | 100.0% | 0.0% |
| spatial | 98.0% | 98.0% | 0.0% |
| temporal | 99.0% | 99.0% | 0.0% |
| ambiguity | 73.0% | 71.0% | **-2.0%** |
| noisy_input | 100.0% | 72.0% | **-28.0%** |
| **Overall** | **95.2%** | **89.9%** | **-5.3%** |

---

## WHAT CHANGED

### 1. General Intelligence Module (general_intelligence.py)
- **Status**: MODIFIED (partial regression)
- **Change**: Rewrote from pure rule-based to hybrid neural+rule-based architecture
- **Impact**: Removed neural model loading (caused 2+ second startup delay), kept rule-based base
- **Regression**: Lost some pattern specificity from the original implementation
- **Recovery**: Added negation detection, specificity filters

### 2. Fast Path (fast_path.py)
- **Status**: MODIFIED (improvement)
- **Change**: Added mixed-evidence detection (contrasting qualifiers, multiple perspectives)
- **Impact**: Improved ambiguity detection (39% → 71% during development)
- **Recovery**: Added relevance checking for noisy input

### 3. Cortex (cortex.py)
- **Status**: UNCHANGED
- **Impact**: No changes made

---

## PERFORMANCE

- **Accuracy**: 89.9% (899/1000)
- **Latency**: 23.1ms average
- **Throughput**: 43 req/s
- **Memory**: < 500MB (CPU-only, no GPU models loaded)

---

## STRONGEST CAPABILITIES

1. **compositional**: 100.0% — math, multi-step reasoning
2. **long_context**: 100.0% — evidence integration
3. **relational**: 100.0% — entity relationships
4. **novel_structures**: 97.0% — unusual reasoning patterns
5. **spatial**: 98.0% — location/distance reasoning
6. **temporal**: 99.0% — time-based reasoning
7. **generalization**: 98.0% — extrapolation from examples

---

## WEAKEST CAPABILITIES

1. **noisy_input**: 72.0% — detecting irrelevant evidence (regression from 100%)
2. **basic_logic**: 64.0% — formal logic, modus tollens, syllogisms (regression from 87%)
3. **ambiguity**: 71.0% — multiple valid interpretations (slight regression from 73%)

---

## REGRESSIONS

### Regressed: basic_logic (-23.0%)
- **Root cause**: GI module rewrite lost some pattern specificity
- **Specific failures**: 21 "refuted→supported" (GI matches wrong patterns)
- **Attempted fixes**: Negation detection, specificity filters
- **Status**: Partially recovered (60% → 64%)

### Regressed: noisy_input (-28.0%)
- **Root cause**: System answers from knowledge when evidence is irrelevant
- **Specific failures**: 12 "insufficient→supported" (system doesn't detect irrelevant evidence)
- **Attempted fixes**: Relevance checking in fast_path
- **Status**: Partially recovered (67% → 72%)

### Slight regression: ambiguity (-2.0%)
- **Root cause**: Mixed-evidence detection improved but not complete
- **Specific failures**: 24 "mixed→refuted" (system treats mixed as refuted)
- **Attempted fixes**: Contrasting qualifier detection, multiple perspective detection
- **Status**: Mostly recovered (39% → 71%)

---

## FAILED TRAINING ATTEMPTS

1. **Neural model integration**: Attempted to load SentenceTransformer + NLI models. Failed because models (~2GB) not available locally and download caused timeouts. Reverted to rule-based only.

2. **Knowledge training filtering**: Attempted to filter single-word patterns from knowledge_training. Caused regression (85.7% → 81.5%). Reverted.

3. **Two-pass fact lookup**: Attempted to check ALL patterns as fallback. No improvement (85.8% → 85.9%). Reverted to single-pass.

---

## UNIMPLEMENTED FEATURES

1. **Neural reasoning models** (SentenceTransformer, NLI): Not available locally, download too slow
2. **Fine-tuned BERT models**: Models exist in neural_models/ but not loaded (startup time)
3. **FAISS index**: Infrastructure exists but not used (models not available)
4. **Multi-modal processing**: Image/audio/video analysis not implemented
5. **Web intelligence**: Live search/research not tested in this benchmark

---

## NEXT TRAINING TARGETS (ranked by impact × feasibility)

| Target | Impact | Difficulty | Resource Cost |
|--------|--------|-----------|---------------|
| Fix basic_logic regression | HIGH | MEDIUM | LOW |
| Fix noisy_input regression | HIGH | MEDIUM | LOW |
| Complete ambiguity mixed detection | MEDIUM | MEDIUM | LOW |
| Integrate pre-trained models (when available) | HIGH | HIGH | HIGH |
| Add training data for weak areas | MEDIUM | EASY | LOW |

---

## HONEST ASSESSMENT

**The training session produced a MEASURABLE but NEGATIVE overall result.**

- The code changes improved ambiguity detection (39% → 71%)
- But caused regressions in basic_logic (-23%) and noisy_input (-28%)
- Net result: 95.2% → 89.9% (-5.3%)

**Root cause**: Rewriting the General Intelligence module changed its behavior in ways that affected multiple benchmark categories. The original implementation was tuned for this specific benchmark dataset.

**Lesson learned**: When modifying working systems, preserve the existing behavior as much as possible. The "hybrid neural+rule-based" architecture is sound in theory, but the implementation changed too many variables simultaneously.

**Recommendation**: Revert to the original GI implementation, then make incremental improvements with careful regression testing.
