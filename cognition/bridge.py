"""Phase 6 — Structured Neural-to-Reasoning Integration.

Neural models provide structured output (Claims / Evidence / Entities).
The bridge validates every neural output before it becomes a trusted fact.
Malformed or low-confidence output is REJECTED or downgraded to a
pending/untrusted level and cannot silently become trusted facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import Claim, Entity, Evidence


@dataclass
class NeuralOutput:
    """Raw output from a neural model."""

    raw: str
    structured: Dict[str, Any]
    confidence: float
    kind: str = "claim"
    model_id: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "structured": self.structured,
            "confidence": self.confidence,
            "kind": self.kind,
            "model_id": self.model_id,
        }


@dataclass
class ValidationResult:
    """Outcome of validating a neural output."""

    accepted: bool
    reason: str
    schema_ok: bool = False
    confidence_ok: bool = False
    reason_rejected: str = ""
    transformed: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)


class NeuralBridge:
    """Validates structured neural output and hands it to the deterministic world.

    Rules:
    - Statement must be non-empty string
    - Kind must be one of claim/evidence/entity
    - Structured dict must contain the fields required for that kind
    - Confidence must be >= min_confidence
    - Low-confidence then by the marker to trust, else rejected
    """

    REQUIRED_FIELDS = {
        "claim": {"statement"},
        "evidence": {"content", "source"},
        "entity": {"name"},
    }

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self.log: List[Dict[str, Any]] = []

    def validate(self, output: NeuralOutput) -> ValidationResult:
        """Validate a neural output; accepted outputs are transformed into
        contract objects (Claim / Evidence / Entity). Never silently trusted."""
        reasons: List[str] = []
        res = ValidationResult(accepted=False, reason="", schema_ok=False, confidence_ok=False)

        # 1) schema check
        if output.kind not in self.REQUIRED_FIELDS:
            res.reason = f"unknown kind '{output.kind}'"
            res.reason_rejected = res.reason
            self._log(output, False, res)
            return res
        if not isinstance(output.structured, dict):
            res.reason = "structured output is not a dict"
            res.reason_rejected = res.reason
            self._log(output, False, res)
            return res
        required = self.REQUIRED_FIELDS[output.kind]
        missing = required - set(output.structured.keys())
        if missing:
            res.reason = f"missing required field(s): {sorted(missing)}"
            res.reason_rejected = res.reason
            self._log(output, False, res)
            return res
        if not isinstance(output.raw, str) or not output.raw.strip():
            res.reason = "empty raw text"
            res.reason_rejected = res.reason
            self._log(output, False, res)
            return res
        res.schema_ok = True

        # 2) confidence check
        if output.confidence < self.min_confidence:
            res.reason = f"confidence {output.confidence:.3f} below threshold {self.min_confidence}"
            res.reason_rejected = res.reason
            res.confidence_ok = False
            self._log(output, False, res)
            return res
        res.confidence_ok = True

        # 3) transform into a contract object only if schema+confidence pass
        res.accepted = True
        res.transformed = self._transform(output)
        res.reason = "accepted"
        self._log(output, True, res)
        return res

    def _transform(self, output: NeuralOutput) -> Any:
        s = output.structured
        if output.kind == "claim":
            return Claim(
                statement=s["statement"],
                subject=s.get("subject", ""),
                predicate=s.get("predicate", ""),
                object=s.get("object", ""),
                confidence=output.confidence,
                status="unverified",
            )
        if output.kind == "evidence":
            return Evidence(
                content=s["content"],
                source=s["source"],
                retrieval_time=s.get("retrieval_time", ""),
                metadata={"neural_model": output.model_id, "raw": output.raw[:200]},
            )
        if output.kind == "entity":
            ent = Entity(name=s["name"], kind=s.get("kind", "unknown"))
            ent.attributes = {k: v for k, v in s.items() if k not in ("name", "kind")}
            return ent
        raise ValueError(f"cannot transform kind {output.kind}")

    def _log(self, output: NeuralOutput, accepted: bool, res: ValidationResult) -> None:
        self.log.append({
            "model_id": output.model_id,
            "kind": output.kind,
            "confidence": output.confidence,
            "accepted": accepted,
            "reason": res.reason,
        })

    def accepted_count(self) -> int:
        return sum(1 for e in self.log if e["accepted"])

    def rejected_count(self) -> int:
        return sum(1 for e in self.log if not e["accepted"])