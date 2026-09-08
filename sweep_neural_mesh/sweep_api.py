"""
Sweep Unified Inference API — single entry point for all reasoning.

Usage:
    from sweep_api import SweepAPI
    api = SweepAPI()
    result = api.query("What is the capital of France?")
    print(result)
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.api")


@dataclass
class QueryResult:
    """Result from a unified query."""
    query: str
    answer: str
    confidence: float
    method: str
    latency_ms: float
    task: str = ""
    reasoning: str = ""
    evidence_used: list = field(default_factory=list)
    components: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SweepAPI:
    """Unified API for Sweep's neural engine.

    Provides a single interface that routes through:
    1. Trained classifiers (fast, accurate)
    2. Seq2seq generation (for open-ended answers)
    3. Logic engines (for formal reasoning)
    4. Web search (for live information)
    5. Knowledge base (for factual lookups)
    """

    def __init__(self):
        self._pipeline = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            # Package import first; preserve explicit script-mode compatibility.
            if __package__:
                from .cortex_integration import get_pipeline
            else:
                from cortex_integration import get_pipeline
            pipeline = get_pipeline()
            if pipeline.initialize() is not True:
                self._pipeline = None
                self._initialized = False
                logger.error("Pipeline initialization returned failure")
                return
            self._pipeline = pipeline
            self._initialized = True
        except Exception as exc:
            self._pipeline = None
            self._initialized = False
            logger.error("Pipeline initialization failed (%s)", type(exc).__name__)

    def query(
        self,
        question: str,
        evidence: list = None,
        context: str = "",
    ) -> QueryResult:
        """Ask Sweep a question.

        Args:
            question: The question to answer
            evidence: Optional evidence to evaluate
            context: Optional additional context

        Returns:
            QueryResult with answer, confidence, and metadata
        """
        self._ensure_init()
        t0 = time.perf_counter()

        if self._pipeline:
            result = self._pipeline.infer(question, evidence=evidence, context=context)
            return QueryResult(
                query=question, answer=result.answer, confidence=result.confidence,
                method=result.method, latency_ms=(time.perf_counter() - t0) * 1000,
                task=result.task, reasoning=result.reasoning,
                evidence_used=evidence or [], components=result.components_used,
            )

        return QueryResult(
            query=question, answer="Pipeline not available", confidence=0.0,
            method="error", latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def batch_query(self, questions: list) -> list:
        """Process multiple questions."""
        return [self.query(q) for q in questions]

    def status(self) -> dict:
        """Get API status."""
        self._ensure_init()
        return {
            "initialized": self._initialized,
            "pipeline": self._pipeline is not None,
            "trained_model": self._pipeline._trained_model is not None if self._pipeline else False,
            "seq2seq": self._pipeline._seq2seq_model is not None if self._pipeline else False,
            "logic_engines": self._pipeline._logic_engine is not None if self._pipeline else False,
        }


# Convenience function
_api = None

def ask(question: str, evidence: list = None) -> QueryResult:
    """Quick query to Sweep."""
    global _api
    if _api is None:
        _api = SweepAPI()
    return _api.query(question, evidence=evidence)
