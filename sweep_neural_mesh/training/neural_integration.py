"""
Neural Integration — Connects trained BERT models to the Sweep cortex.

Replaces rule-based components with fine-tuned neural models.
Falls back to rule-based if neural models are unavailable.

Usage:
    from sweep_neural_mesh.training.neural_integration import NeuralLayer
    layer = NeuralLayer()
    result = layer.classify_evidence("Studies confirm the drug works")
"""
from __future__ import annotations

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Any

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logger = logging.getLogger("neural_integration")


class NeuralLayer:
    """
    Neural layer that replaces rule-based components with fine-tuned BERT models.
    
    Provides:
    - Evidence classification (supports/refutes/neutral)
    - Contradiction detection (contradiction/consistent/neutral)
    - Intent classification (13 categories)
    
    Falls back to rule-based if neural models are unavailable.
    """
    
    def __init__(self, models_dir: str | None = None):
        """
        Initialize the neural layer.
        
        Args:
            models_dir: Path to trained models directory.
                       Defaults to sweep_neural_mesh/training/neural_models
        """
        if models_dir is None:
            models_dir = str(Path(__file__).parent / "neural_models")
        
        self._models_dir = models_dir
        self._evidence_model = None
        self._contradiction_model = None
        self._intent_model = None
        self._initialized = False
        
        # Track which models are neural vs rule-based
        self._model_types = {
            "evidence": "unknown",
            "contradiction": "unknown",
            "intent": "unknown",
        }
    
    def initialize(self):
        """Initialize neural models (lazy loading)."""
        if self._initialized:
            return
        
        # Try to load evidence classifier
        evidence_path = os.path.join(self._models_dir, "evidence_classifier")
        if os.path.exists(evidence_path):
            try:
                from sweep_neural_mesh.training.neural_training import NeuralEvidenceClassifier
                self._evidence_model = NeuralEvidenceClassifier(evidence_path)
                self._evidence_model.load()
                self._model_types["evidence"] = "neural"
                logger.info("Loaded neural evidence classifier")
            except Exception as e:
                logger.warning(f"Failed to load neural evidence classifier: {e}")
        
        # Try to load contradiction detector
        contradiction_path = os.path.join(self._models_dir, "contradiction_detector")
        if os.path.exists(contradiction_path):
            try:
                from sweep_neural_mesh.training.neural_training import NeuralContradictionDetector
                self._contradiction_model = NeuralContradictionDetector(contradiction_path)
                self._contradiction_model.load()
                self._model_types["contradiction"] = "neural"
                logger.info("Loaded neural contradiction detector")
            except Exception as e:
                logger.warning(f"Failed to load neural contradiction detector: {e}")
        
        # Try to load intent classifier
        intent_path = os.path.join(self._models_dir, "intent_classifier")
        if os.path.exists(intent_path):
            try:
                from sweep_neural_mesh.training.neural_training import NeuralIntentClassifier
                self._intent_model = NeuralIntentClassifier(intent_path)
                self._intent_model.load()
                self._model_types["intent"] = "neural"
                logger.info("Loaded neural intent classifier")
            except Exception as e:
                logger.warning(f"Failed to load neural intent classifier: {e}")
        
        self._initialized = True
        logger.info(f"Neural layer initialized: {self._model_types}")
    
    def classify_evidence(self, text: str) -> dict[str, Any]:
        """
        Classify evidence as supports/refutes/neutral.
        
        Args:
            text: Evidence text to classify
            
        Returns:
            Dict with 'answer', 'confidence', 'method'
        """
        self.initialize()
        
        if self._evidence_model is not None:
            try:
                result = self._evidence_model.classify(text)
                result["method"] = "neural"
                return result
            except Exception as e:
                logger.warning(f"Neural evidence classification failed: {e}")
        
        # Fallback to rule-based
        return self._rule_based_evidence(text)
    
    def detect_contradiction(self, text_a: str, text_b: str) -> dict[str, Any]:
        """
        Detect contradiction between two texts.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Dict with 'answer', 'confidence', 'method'
        """
        self.initialize()
        
        if self._contradiction_model is not None:
            try:
                result = self._contradiction_model.detect(text_a, text_b)
                result["method"] = "neural"
                return result
            except Exception as e:
                logger.warning(f"Neural contradiction detection failed: {e}")
        
        # Fallback to rule-based
        return self._rule_based_contradiction(text_a, text_b)
    
    def classify_intent(self, text: str) -> dict[str, Any]:
        """
        Classify intent of a query.
        
        Args:
            text: Query text
            
        Returns:
            Dict with 'answer', 'confidence', 'method'
        """
        self.initialize()
        
        if self._intent_model is not None:
            try:
                result = self._intent_model.classify(text)
                result["method"] = "neural"
                return result
            except Exception as e:
                logger.warning(f"Neural intent classification failed: {e}")
        
        # Fallback to rule-based
        return self._rule_based_intent(text)
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all models."""
        self.initialize()
        return {
            "initialized": self._initialized,
            "model_types": self._model_types,
            "models_dir": self._models_dir,
        }
    
    # ════════════════════════════════════════════════════════════════
    # RULE-BASED FALLBACKS
    # ════════════════════════════════════════════════════════════════
    
    def _rule_based_evidence(self, text: str) -> dict[str, Any]:
        """Rule-based evidence classification fallback."""
        import re
        text_lower = text.lower()
        
        support_patterns = [
            r'\bconfirm', r'\bdemonstrat', r'\bsupport', r'\bprove',
            r'\bbeneficial', r'\bimprov', r'\beffective', r'\bpositive',
        ]
        refute_patterns = [
            r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bfail', r'\brefute',
            r'\bcontradict', r'\binconsisten',
        ]
        neutral_patterns = [
            r'\bmixed\b', r'\buncertain\b', r'\bpreliminar',
            r'\bneed.*more research', r'\binsufficient\b',
        ]
        
        support_count = sum(1 for p in support_patterns if re.search(p, text_lower))
        refute_count = sum(1 for p in refute_patterns if re.search(p, text_lower))
        neutral_count = sum(1 for p in neutral_patterns if re.search(p, text_lower))
        
        if support_count > refute_count and support_count > neutral_count:
            return {"answer": "supports", "confidence": 0.6, "method": "rule_based"}
        elif refute_count > support_count and refute_count > neutral_count:
            return {"answer": "refutes", "confidence": 0.6, "method": "rule_based"}
        else:
            return {"answer": "neutral", "confidence": 0.4, "method": "rule_based"}
    
    def _rule_based_contradiction(self, text_a: str, text_b: str) -> dict[str, Any]:
        """Rule-based contradiction detection fallback."""
        import re
        a_lower = text_a.lower()
        b_lower = text_b.lower()
        
        # Check for opposites
        opposites = [
            ("effective", "ineffective"), ("safe", "dangerous"),
            ("increase", "decrease"), ("true", "false"),
            ("yes", "no"), ("round", "flat"),
        ]
        for pos, neg in opposites:
            if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                return {"answer": "contradiction", "confidence": 0.7, "method": "rule_based"}
        
        # Check word overlap
        words_a = set(a_lower.split())
        words_b = set(b_lower.split())
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        
        if overlap > 0.5:
            return {"answer": "consistent", "confidence": 0.6, "method": "rule_based"}
        
        return {"answer": "neutral", "confidence": 0.4, "method": "rule_based"}
    
    def _rule_based_intent(self, text: str) -> dict[str, Any]:
        """Rule-based intent classification fallback."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["investigate", "look into", "research"]):
            return {"answer": "investigation", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["search", "find", "look up"]):
            return {"answer": "search", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["contradict", "conflict", "consistent"]):
            return {"answer": "contradiction_analysis", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["evidence", "proof", "support"]):
            return {"answer": "evidence_analysis", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["compare", "difference", "versus"]):
            return {"answer": "comparison", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["timeline", "when", "first"]):
            return {"answer": "timeline", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["summarize", "overview", "summary"]):
            return {"answer": "summarization", "confidence": 0.6, "method": "rule_based"}
        elif any(kw in text_lower for kw in ["extract", "find all", "what organizations"]):
            return {"answer": "extraction", "confidence": 0.6, "method": "rule_based"}
        else:
            return {"answer": "unknown_ambiguous", "confidence": 0.3, "method": "rule_based"}


# Singleton instance
_neural_layer: NeuralLayer | None = None


def get_neural_layer() -> NeuralLayer:
    """Get the singleton neural layer."""
    global _neural_layer
    if _neural_layer is None:
        _neural_layer = NeuralLayer()
    return _neural_layer
