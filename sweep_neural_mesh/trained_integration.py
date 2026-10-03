"""
Sweep Trained Model Integration - bridges trained classifiers into the cortex.
"""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("sweep.trained_models")

_EXPERIMENT_DIR = Path(__file__).parent / "experiments"


@dataclass
class TrainedClassification:
    task: str
    predicted_label: str
    confidence: float
    all_probs: dict[str, float]
    latency_ms: float = 0.0
    model_source: str = "joint_multitask"


class TrainedModelRouter:
    def __init__(self):
        self._model = None
        self._embedder = None
        self._label_maps = {
            "logic": {0: "VALID", 1: "INVALID"},
            "math": {0: "COMPUTE", 1: "INVALID"},
            "evidence": {0: "SUPPORTS", 1: "REFUTES", 2: "NEUTRAL"},
            "recognition": {0: "EXTRACT", 1: "SKIP"},
        }
        self._initialized = False
    
    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            import torch
            _parent = str(Path(__file__).resolve().parent)
            if _parent not in sys.path:
                sys.path.insert(0, _parent)
            from neurons.semantic_embeddings import SemanticEmbedder
            
            checkpoint_path = _EXPERIMENT_DIR / "checkpoints_scaled" / "joint_model.pt"
            if not checkpoint_path.exists():
                logger.warning(f"No trained model at {checkpoint_path}")
                return False
            
            _exp_parent = str(_EXPERIMENT_DIR)
            if _exp_parent not in sys.path:
                sys.path.insert(0, _exp_parent)
            from scale_and_integrate import MultiTaskClassifier
            ckpt = torch.load(str(checkpoint_path), weights_only=True, map_location="cpu")
            
            self._model = MultiTaskClassifier(
                input_dim=ckpt["input_dim"], shared_dim=128, task_heads=ckpt["task_heads"],
            )
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()
            self._embedder = SemanticEmbedder()
            self._initialized = True
            logger.info(f"Trained model loaded: {sum(p.numel() for p in self._model.parameters()):,} params")
            return True
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return False
    
    def classify(self, query: str, context: str = "", task: str = None):
        if not self._initialized and not self.initialize():
            return None
        import torch, numpy as np
        t0 = time.perf_counter()
        emb1 = self._embedder.embed(query)
        if context:
            emb2 = self._embedder.embed(context)
            vec = np.array(emb1.vector) + np.array(emb2.vector)
        else:
            vec = np.array(emb1.vector)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        
        if task and task in self._model.heads:
            with torch.no_grad():
                logits = self._model(x, task)
                probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            lm = self._label_maps.get(task, {})
            predicted = lm.get(pred_idx, f"class_{pred_idx}")
            all_probs = {lm.get(i, f"c{i}"): round(probs[i].item(), 4) for i in range(len(probs))}
            return TrainedClassification(task=task, predicted_label=predicted, confidence=confidence, all_probs=all_probs, latency_ms=(time.perf_counter()-t0)*1000)
        
        best = None
        for tn in self._model.heads:
            with torch.no_grad():
                logits = self._model(x, tn)
                probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            if best is None or confidence > best.confidence:
                lm = self._label_maps.get(tn, {})
                predicted = lm.get(pred_idx, f"class_{pred_idx}")
                all_probs = {lm.get(i, f"c{i}"): round(probs[i].item(), 4) for i in range(len(probs))}
                best = TrainedClassification(task=tn, predicted_label=predicted, confidence=confidence, all_probs=all_probs, latency_ms=(time.perf_counter()-t0)*1000)
        return best


_trained_router = None

def get_trained_router():
    global _trained_router
    if _trained_router is None:
        _trained_router = TrainedModelRouter()
    return _trained_router
