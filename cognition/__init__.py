"""Cognition — a deterministic cognitive-reasoning subsystem for Sweep.

Implements the Phase 1–20 milestone contract: a unified data model, persistent
memory and evidence graph, logic and rule engines, hypothesis/contradiction/
belief-revision machinery, uncertainty and verification, prediction feedback,
resource-aware execution, explainability, tests, and a reproducible benchmark.
"""

from .schema import (  # noqa: F401
    Claim,
    Entity,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Observation,
    ReasoningEvent,
)
from .store import CognitiveStore  # noqa: F401

__version__ = "1.0.0"