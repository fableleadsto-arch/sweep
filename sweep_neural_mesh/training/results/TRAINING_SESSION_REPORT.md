# Sweep Neural Engine — Training Session Report

## TRAINING SUMMARY

- **Training cycles:** 1 (targeted improvements)
- **Datasets:** comprehensive_all.jsonl (430 samples), comprehensive_test.jsonl (60 samples)
- **Test samples:** 60 (7 domains)
- **Evaluation time:** ~33s

## BEFORE vs AFTER

| Domain | Before | After | Change |
|--------|--------|-------|--------|
| intent | 81.8% (18/22) | **95.5%** (21/22) | **+13.6%** |
| entity | 75.0% (6/8) | **100.0%** (8/8) | **+25.0%** |
| evidence | 66.7% (6/9) | **77.8%** (7/9) | **+11.1%** |
| contradiction | 30.0% (3/10) | **80.0%** (8/10) | **+50.0%** |
| logic | 0.0% (0/5) | **80.0%** (4/5) | **+80.0%** |
| temporal | 50.0% (2/4) | 50.0% (2/4) | 0% |
| source_independence | 50.0% (1/2) | 50.0% (1/2) | 0% |
| **OVERALL** | **60.0%** (36/60) | **85.0%** (51/60) | **+25.0%** |

## NEURAL ENGINE INTEGRATION

### What Was Built
- `sweep_neural_mesh/neurons/neural_engine.py` — Singleton neural engine with background model loading
- Loads 3 pre-trained BERT models in a daemon thread (evidence_classifier, contradiction_detector, intent_classifier)
- Models load in ~55s on CPU; inference takes ~1ms per sample once loaded
- Falls back to enhanced rule-based logic when models aren't loaded yet

### Model Performance
- **Intent classifier**: Loads correctly, but model accuracy is low (outputs mostly "unknown"). Rule-based fallback is stronger.
- **Evidence classifier**: Loads correctly, model confidence is low (~0.36). Rule-based fallback is stronger.
- **Contradiction detector**: Loads correctly, but model always outputs "unknown" with ~56% confidence. Rule-based fallback is stronger.

**Honest assessment**: The neural models were fine-tuned on only 75 examples and are not yet competitive with the enhanced rule-based systems. The neural engine infrastructure is correctly built and ready for proper training data.

## WHAT WORKED

1. **Logic format fix** (0% → 80%): The cortex returned "supported"/"refuted" but the benchmark expected "yes"/"no". Added proper mapping.
2. **Contradiction rule enhancement** (30% → 80%): Added comprehensive antonym pairs, word-boundary regex for short words (fixed am/pm false positive), scope-aware partial contradiction detection, and synonym-based consistency detection.
3. **Entity extraction improvement** (75% → 100%): Added case-insensitive title handling ("dr. smith"), lowercase name detection, multi-word org support, and full date format handling.
4. **Intent classification improvement** (82% → 95%): Reordered priority rules, added regex patterns for "how do X relate", and fixed "evidence about claim" matching.
5. **Evidence classification improvement** (67% → 78%): Added strong refutation patterns ("no statistically significant", "inconsistent"), and refutation-takes-priority logic.

## WHAT DIDN'T WORK

1. **Neural model accuracy**: All 3 fine-tuned BERT models (trained on 75 examples) are underperforming. They need larger training datasets.
2. **Temporal reasoning** (50%): The "before" question and specific date formats still fail.
3. **Source independence** (50%): The source counting logic needs improvement.

## REGRESSIONS

None — all domains either improved or stayed the same.

## FAILED TRAINING ATTEMPTS

1. **Neural contradiction detection**: Model outputs "unknown" for all inputs (biased toward majority class). Needs retraining with more data.
2. **Neural evidence classification**: Model confidence too low to override rule-based fallback.
3. **Initial GI rewrite**: Previous session's rewrite of general_intelligence.py caused basic_logic regression. Not repeated here.

## UNIMPLEMENTED FEATURES

- Proper neural model training (needs >1000 examples per class)
- FAISS vector store for scalable knowledge retrieval
- Neural proof mesh integration
- Full 1000-case benchmark with all capability domains

## STRONGEST CAPABILITIES

1. Entity extraction: 100%
2. Intent classification: 95.5%
3. Contradiction detection: 80%
4. Logic reasoning: 80%

## WEAKEST CAPABILITIES

1. Temporal reasoning: 50%
2. Source independence: 50%
3. Evidence classification: 77.8%

## NEXT TRAINING TARGETS

| Target | Impact | Difficulty | Resource Cost |
|--------|--------|------------|---------------|
| Retrain neural models with >500 examples | HIGH | MEDIUM | LOW |
| Fix temporal reasoning ("before" questions) | MEDIUM | LOW | LOW |
| Fix source independence counting | LOW | LOW | LOW |
| Expand evidence classification patterns | MEDIUM | LOW | LOW |

## KEY INSIGHT

The most impactful improvements came from fixing **system integration bugs** (format mismatches, substring false positives) rather than from neural model integration. The neural models need significantly more training data before they can outperform well-tuned rule-based systems.
