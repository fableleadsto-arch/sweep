"""
Sweep Cortex Integration v2 — trained models + seq2seq + logic engines + RAG

Priority order:
1. Logic engines (for formal reasoning)
2. RAG pipeline (retrieve + generate)
3. Seq2seq generation (standalone)
4. Rule-based fallback
"""
from __future__ import annotations

import os
import sys
import time
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.cortex_integration")

_SWEEP_DIR = Path(__file__).parent


@dataclass
class InferenceResult:
    """Unified result from the inference pipeline."""
    answer: str
    confidence: float
    method: str
    task: str = ""
    reasoning: str = ""
    all_probs: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    components_used: list = field(default_factory=list)


class SweepInferencePipeline:
    """Unified inference pipeline with RAG support."""

    def __init__(self):
        self._trained_model = None
        self._seq2seq_model = None
        self._seq2seq_tokenizer = None
        self._embedder = None
        self._logic_engine = None
        self._proof_mesh = None
        self._rag = None
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            import torch
            _parent = str(_SWEEP_DIR)
            if _parent not in sys.path:
                sys.path.insert(0, _parent)

            # Load embedder
            try:
                from neurons.semantic_embeddings import SemanticEmbedder
                self._embedder = SemanticEmbedder()
            except Exception:
                pass

            # Load trained classifier
            try:
                from trained_integration import get_trained_router
                self._trained_model = get_trained_router()
                self._trained_model.initialize()
            except Exception:
                pass

            # Load seq2seq
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                seq2seq_path = Path(__file__).parent / "experiments" / "checkpoint_seq2seq_expanded" / "best_model"
                if seq2seq_path.exists():
                    self._seq2seq_tokenizer = AutoTokenizer.from_pretrained(str(seq2seq_path))
                    self._seq2seq_model = AutoModelForCausalLM.from_pretrained(str(seq2seq_path))
                    self._seq2seq_model.eval()
                    if self._seq2seq_tokenizer.pad_token is None:
                        self._seq2seq_tokenizer.pad_token = self._seq2seq_tokenizer.eos_token
            except Exception:
                pass

            # Load logic engines
            try:
                from neurons.logical_inference import LogicalInferenceEngine
                from neurons.proof_mesh import NeuralProofMesh
                self._logic_engine = LogicalInferenceEngine()
                self._proof_mesh = NeuralProofMesh()
            except Exception:
                pass

            # Load RAG
            try:
                from rag_pipeline import get_rag
                self._rag = get_rag()
                self._rag.initialize()
                logger.info("RAG pipeline loaded")
            except Exception as e:
                logger.warning(f"RAG not available: {e}")

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False

    def _try_logic_engines(self, query, evidence):
        """Try formal logic engines."""
        t0 = time.perf_counter()
        if self._proof_mesh and evidence:
            try:
                pr = self._proof_mesh.solve(query, evidence)
                if pr.conclusion in ("supported", "refuted", "mixed"):
                    return InferenceResult(
                        answer=pr.conclusion, confidence=pr.confidence,
                        method="logic_engine", task="proof_mesh",
                        reasoning=pr.reasoning[0] if pr.reasoning else "",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        components_used=["proof_mesh"],
                    )
            except Exception:
                pass
        if self._logic_engine:
            try:
                lr = self._logic_engine.analyze(query, evidence)
                if lr.conclusion in ("supported", "refuted", "mixed"):
                    return InferenceResult(
                        answer=lr.conclusion, confidence=lr.confidence,
                        method="logic_engine", task="logical_inference",
                        reasoning=lr.reasoning,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        components_used=["logical_inference"],
                    )
            except Exception:
                pass
        return None

    def _try_rag(self, query):
        """Try RAG pipeline (retrieve + generate)."""
        if not self._rag:
            return None
        try:
            result = self._rag.query(query)
            if result.answer and len(result.answer) > 3:
                return InferenceResult(
                    answer=result.answer, confidence=result.confidence,
                    method="rag", task="rag_generation",
                    reasoning=f"Sources: {', '.join(result.sources)}",
                    latency_ms=result.latency_ms,
                    components_used=["rag", "embeddings", "seq2seq"],
                )
        except Exception:
            pass
        return None

    def _try_seq2seq(self, query):
        """Try seq2seq generation."""
        if not self._seq2seq_model or not self._seq2seq_tokenizer:
            return None
        try:
            import torch
            t0 = time.perf_counter()
            input_ids = self._seq2seq_tokenizer.encode(f"Human: {query} Assistant:", return_tensors="pt")
            with torch.no_grad():
                output = self._seq2seq_model.generate(
                    input_ids, max_length=100, do_sample=True,
                    top_k=50, top_p=0.95, temperature=0.7,
                )
            response = self._seq2seq_tokenizer.decode(output[0], skip_special_tokens=True)
            answer = response.split("Assistant:")[-1].strip()
            if answer and len(answer) > 3 and not answer.startswith("VALID"):
                return InferenceResult(
                    answer=answer, confidence=0.7, method="seq2seq",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    components_used=["seq2seq"],
                )
        except Exception:
            pass
        return None

    def infer(self, query, evidence=None, context=""):
        """Run the full inference pipeline."""
        if not self._initialized:
            self.initialize()
        evidence = evidence or []

        # 1. Logic engines
        result = self._try_logic_engines(query, evidence)
        if result:
            return result

        # 2. RAG (retrieve + generate)
        result = self._try_rag(query)
        if result:
            return result

        # 3. Seq2seq standalone
        result = self._try_seq2seq(query)
        if result:
            return result

        # 4. Fallback
        t0 = time.perf_counter()
        return InferenceResult(
            answer="UNKNOWN: no inference provider produced a result.",
            confidence=0.0, method="unavailable",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = SweepInferencePipeline()
    return _pipeline
