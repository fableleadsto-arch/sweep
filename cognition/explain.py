"""Phase 17 — Explainability / Reconstructable Reasoning Trace.

A conclusion can be explained: reconstruct every step of the route that
produced it — input, observations, hypotheses, route, verification,
uncertainty — from the recorded state. `explain_conclusion(conclusion_id)`
returns a structured, deterministic, human-readable explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .loop import CognitiveLoop


@dataclass
class ExplanationStep:
    """One step of a reconstructed reasoning trace."""

    stage: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "detail": self.detail, "data": self.data}


@dataclass
class Explanation:
    """Full human-readable explanation of a conclusion."""

    conclusion_id: str
    conclusion: str
    confidence: float
    epistemic_state: str
    steps: List[ExplanationStep] = field(default_factory=list)
    source_run: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "epistemic_state": self.epistemic_state,
            "steps": [s.to_dict() for s in self.steps],
            "source_run": self.source_run,
        }

    def render(self) -> str:
        """Human-readable multiline trace."""
        lines = [f"Conclusion {self.conclusion_id}: {self.conclusion}",
                 f"  confidence:    {self.confidence}",
                 f"  epistemic state: {self.epistemic_state}",
                 "  trace:"]
        for s in self.steps:
            lines.append(f"    - {s.stage}: {s.detail}")
        return "\n".join(lines)


class Explainer:
    """Reconstructs and explains conclusions from a completed CognitiveLoop."""

    def __init__(self, loop: CognitiveLoop):
        self.loop = loop

    def explain_conclusion(self, conclusion_id: str) -> Explanation:
        """Build a deterministic explanation for a prior loop conclusion."""
        result = None
        for r in self.loop.results:
            if r.conclusion_id == conclusion_id or r.run_id == conclusion_id:
                result = r
                break
        if result is None:
            raise KeyError(f"no conclusion with id '{conclusion_id}'")

        steps: List[ExplanationStep] = []
        for rec in result.stages:
            steps.append(ExplanationStep(
                stage=rec.name,
                detail=rec.detail,
                data=rec.output,
            ))
        return Explanation(
            conclusion_id=result.conclusion_id,
            conclusion=result.conclusion,
            confidence=result.conclusion_confidence,
            epistemic_state=result.epistemic_state,
            steps=steps,
            source_run=result.run_id,
        )