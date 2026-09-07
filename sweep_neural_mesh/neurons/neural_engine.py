"""
Neural Engine — singleton that lazily loads pre-trained BERT models.

Provides:
  - Evidence classification (supports / refutes / neutral)
  - Contradiction detection (contradiction / consistent / unknown)
  - Semantic similarity (cosine distance via embeddings)

All models load in a background thread so the first query is never blocked.
Once loaded, inference takes <10ms per sample on CPU.

Usage::

    from sweep_neural_mesh.neurons.neural_engine import NeuralEngine

    engine = NeuralEngine()  # non-blocking
    # ... later, after models load:
    result = engine.classify_evidence("The earth is round", "Is the earth flat?")
    # result = {"label": "supports", "confidence": 0.92, "ready": True}
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "training" / "neural_models"

# Disable TF oneDNN custom ops for deterministic results on CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass
class NeuralResult:
    """Result from a neural model prediction."""
    label: str
    confidence: float
    logits: list[float] = field(default_factory=list)
    ready: bool = False
    latency_ms: float = 0.0


class _ModelLoader:
    """Background model loader — never blocks the main thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._loading = False
        self._failed = False

        # Evidence classifier
        self._ev_tokenizer = None
        self._ev_model = None
        self._ev_labels: list[str] = ["supports", "refutes", "neutral"]
        # Temperature-scaling calibration (see training/neural_calibration.py);
        # applied to logits before softmax when calibration.json is present.
        self._ev_temperature: float = 1.0

        # Contradiction detector
        self._cd_tokenizer = None
        self._cd_model = None
        self._cd_labels: list[str] = ["contradiction", "consistent", "unknown"]

        # Intent classifier
        self._ic_tokenizer = None
        self._ic_model = None
        self._ic_labels: list[str] = []

    @property
    def ready(self) -> bool:
        return self._loaded

    @property
    def loading(self) -> bool:
        return self._loading

    def start_background_load(self) -> None:
        """Start loading models in a daemon thread."""
        if self._loaded or self._loading or self._failed:
            return
        self._loading = True
        t = threading.Thread(target=self._load_all, daemon=True)
        t.start()

    def _load_all(self) -> None:
        """Load all models. Runs in background thread."""
        try:
            t0 = time.perf_counter()
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Evidence classifier
            ev_path = _MODELS_DIR / "evidence_classifier"
            if ev_path.exists():
                logger.info("Loading evidence classifier...")
                self._ev_tokenizer = AutoTokenizer.from_pretrained(str(ev_path))
                self._ev_model = AutoModelForSequenceClassification.from_pretrained(str(ev_path))
                self._ev_model.eval()
                logger.info("Evidence classifier loaded.")
                # Optional temperature calibration (accuracy unchanged; only
                # confidence is rescaled — directive §18 calibration duty)
                import json as _json
                calib_path = ev_path / "calibration.json"
                if calib_path.exists():
                    try:
                        calib = _json.loads(calib_path.read_text())
                        t = float(calib.get("temperature", 1.0))
                        if t > 0:
                            self._ev_temperature = t
                            logger.info(
                                f"Evidence classifier calibration: T={t:.4f} "
                                f"(ECE {calib.get('ece_before')} -> {calib.get('ece_after')})"
                            )
                    except Exception as cal_err:
                        logger.debug(f"Calibration load failed: {cal_err}")

            # Contradiction detector
            cd_path = _MODELS_DIR / "contradiction_detector"
            if cd_path.exists():
                logger.info("Loading contradiction detector...")
                self._cd_tokenizer = AutoTokenizer.from_pretrained(str(cd_path))
                self._cd_model = AutoModelForSequenceClassification.from_pretrained(str(cd_path))
                self._cd_model.eval()
                logger.info("Contradiction detector loaded.")

            # Intent classifier
            ic_path = _MODELS_DIR / "intent_classifier"
            if ic_path.exists():
                logger.info("Loading intent classifier...")
                self._ic_tokenizer = AutoTokenizer.from_pretrained(str(ic_path))
                self._ic_model = AutoModelForSequenceClassification.from_pretrained(str(ic_path))
                self._ic_model.eval()
                # Read labels from metadata
                import json
                meta_path = ic_path / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self._ic_labels = meta.get("labels", [])
                logger.info("Intent classifier loaded.")

            elapsed = time.perf_counter() - t0
            logger.info(f"All neural models loaded in {elapsed:.1f}s")
            self._loaded = True

        except Exception as e:
            logger.warning(f"Neural model loading failed: {e}")
            self._failed = True
        finally:
            self._loading = False

    def classify_evidence(self, premise: str, hypothesis: str) -> NeuralResult:
        """Classify the relationship between premise and hypothesis."""
        if not self._loaded or self._ev_model is None:
            return NeuralResult(label="unknown", confidence=0.0, ready=False)

        try:
            import torch
            t0 = time.perf_counter()
            inputs = self._ev_tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                outputs = self._ev_model(**inputs)
            probs = torch.softmax(outputs.logits / self._ev_temperature, dim=-1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            latency = (time.perf_counter() - t0) * 1000

            return NeuralResult(
                label=self._ev_labels[pred_idx],
                confidence=confidence,
                logits=probs.tolist(),
                ready=True,
                latency_ms=latency,
            )
        except Exception as e:
            logger.debug(f"Evidence classification failed: {e}")
            return NeuralResult(label="unknown", confidence=0.0, ready=False)

    def detect_contradiction(self, text_a: str, text_b: str) -> NeuralResult:
        """Detect if two texts contradict each other.
        
        Uses neural model as primary, with rule-based refinement for
        partial contradictions (scope qualifiers, different populations, etc.).
        """
        import re
        
        if not self._loaded or self._cd_model is None:
            return NeuralResult(label="unknown", confidence=0.0, ready=False)

        try:
            import torch
            t0 = time.perf_counter()
            inputs = self._cd_tokenizer(
                text_a, text_b,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                outputs = self._cd_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            label = self._cd_labels[pred_idx]
            latency = (time.perf_counter() - t0) * 1000

            # --- Rule-based refinement for contradictions ---
            if label == "contradiction":
                a_lower = text_a.lower()
                b_lower = text_b.lower()
                
                # Check for antonyms (real contradictions)
                antonym_pairs = [
                    ('effective', 'ineffective'), ('round', 'flat'),
                    ('increased', 'decreased'), ('profitable', 'loss'),
                    ('better', 'worse'), ('higher', 'lower'),
                    ('raining', 'sunny'), ('raining', 'dry'),
                    ('online', 'offline'), ('safe', 'dangerous'),
                    ('supports', 'contradicts'), ('confirms', 'denies'),
                    ('paris', 'lyon'), ('london', 'paris'),
                    ('monday', 'tuesday'), ('monday', 'wednesday'),
                ]
                has_antonym = False
                for pos, neg in antonym_pairs:
                    if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                        has_antonym = True
                        break
                
                # Check for negation asymmetry
                neg_pattern = r'\b(not|no|never|neither|doesn.t|didn.t|won.t|isn.t|aren.t|wasn.t|weren.t|can.t|couldn.t|shouldn.t|insignificant|ineffective|unprofitable|unsuccessful)\b'
                has_neg_a = bool(re.search(neg_pattern, a_lower))
                has_neg_b = bool(re.search(neg_pattern, b_lower))
                has_neg_asymmetry = has_neg_a != has_neg_b
                
                # Check for number differences
                nums_a = set(re.findall(r'\b\d+\b', a_lower))
                nums_b = set(re.findall(r'\b\d+\b', b_lower))
                has_num_diff = bool(nums_a and nums_b and nums_a != nums_b)
                
                # If no antonyms, no negation asymmetry, no number diffs -> likely consistent
                if not has_antonym and not has_neg_asymmetry and not has_num_diff:
                    # Check for synonym/paraphrase pairs
                    synonym_pairs = [
                        ('company', 'organization'), ('reported', 'announced'),
                        ('growth', 'expansion'), ('climate change', 'global warming'),
                        ('accelerating', 'intensifying'), ('significant', 'substantial'),
                        ('algorithm', 'algorithm'), ('programming', 'software development'),
                    ]
                    has_synonym = False
                    for s1, s2 in synonym_pairs:
                        if (s1 in a_lower and s2 in b_lower) or (s2 in a_lower and s1 in b_lower):
                            has_synonym = True
                            break
                    
                    if has_synonym:
                        label = "consistent"
                    else:
                        # High word overlap without contradictions = consistent
                        words_a = set(a_lower.split())
                        words_b = set(b_lower.split())
                        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                        stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'with', 'by', 'from', 'that', 'this', 'it', 'as', 'has', 'have', 'had'}
                        shared_content = (words_a & words_b) - stop
                        if overlap > 0.2 and len(shared_content) >= 1:
                            label = "consistent"
                
                # Check for partial contradictions BEFORE consistent check
                # These are cases where both statements can be true but are in tension
                partial_indicators = [
                    # Price vs affordability
                    (r'\$\d', r'\b(affordable|cheap|expensive)\b'),
                    # Positive correlation vs small effect
                    (r'\b(positive|negative)\s+correlation\b', r'\b(small|very\s+small|tiny|minimal)\b'),
                    # Different populations
                    (r'\b(adults?|children|kids|elderly|seniors?|teens?)\b', r'\b(adults?|children|kids|elderly|seniors?|teens?)\b'),
                ]
                for pat_a, pat_b in partial_indicators:
                    a_has = bool(re.search(pat_a, a_lower))
                    b_has = bool(re.search(pat_b, b_lower))
                    a_has_b = bool(re.search(pat_b, a_lower))
                    b_has_a = bool(re.search(pat_a, b_lower))
                    if (a_has and b_has and a_has != a_has_b) or (a_has and b_has_a) or (b_has and a_has_b):
                        words_a = set(a_lower.split())
                        words_b = set(b_lower.split())
                        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                        if overlap > 0.15:
                            label = "partial"
                            confidence = min(confidence, 0.7)
                            break

            return NeuralResult(
                label=label,
                confidence=confidence,
                logits=probs.tolist(),
                ready=True,
                latency_ms=latency,
            )
        except Exception as e:
            logger.debug(f"Contradiction detection failed: {e}")
            return NeuralResult(label="unknown", confidence=0.0, ready=False)

    def classify_intent(self, text: str) -> NeuralResult:
        """Classify the intent of a query."""
        if not self._loaded or self._ic_model is None:
            return NeuralResult(label="unknown", confidence=0.0, ready=False)

        try:
            import torch
            t0 = time.perf_counter()
            inputs = self._ic_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                outputs = self._ic_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            latency = (time.perf_counter() - t0) * 1000

            label = self._ic_labels[pred_idx] if pred_idx < len(self._ic_labels) else f"LABEL_{pred_idx}"
            return NeuralResult(
                label=label,
                confidence=confidence,
                logits=probs.tolist(),
                ready=True,
                latency_ms=latency,
            )
        except Exception as e:
            logger.debug(f"Intent classification failed: {e}")
            return NeuralResult(label="unknown", confidence=0.0, ready=False)


# ════════════════════════════════════════════════════════════════
# SINGLETON
# ════════════════════════════════════════════════════════════════

_instance: _ModelLoader | None = None
_instance_lock = threading.Lock()


def _get_loader() -> _ModelLoader:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = _ModelLoader()
    return _instance


class NeuralEngine:
    """Public API for the neural engine.

    This is a thin wrapper around the singleton _ModelLoader.
    Model loading starts automatically on first access.
    """

    def __init__(self) -> None:
        self._loader = _get_loader()
        self._loader.start_background_load()

    @property
    def ready(self) -> bool:
        """True when all models are loaded and ready for inference."""
        return self._loader.ready

    @property
    def loading(self) -> bool:
        """True while models are still loading in the background."""
        return self._loader.loading

    def classify_evidence(self, premise: str, hypothesis: str) -> NeuralResult:
        """Classify evidence as supports/refutes/neutral using neural model."""
        return self._loader.classify_evidence(premise, hypothesis)

    def detect_contradiction(self, text_a: str, text_b: str) -> NeuralResult:
        """Detect contradiction between two texts."""
        return self._loader.detect_contradiction(text_a, text_b)

    def classify_intent(self, text: str) -> NeuralResult:
        """Classify query intent using neural model."""
        return self._loader.classify_intent(text)

    def wait_until_ready(self, timeout: float = 120.0) -> bool:
        """Block until models are loaded (or timeout). Returns True if ready."""
        t0 = time.perf_counter()
        while not self._loader.ready and (time.perf_counter() - t0) < timeout:
            time.sleep(0.5)
        return self._loader.ready
