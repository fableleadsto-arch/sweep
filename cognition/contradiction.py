"""Phase 9 — Explicit Conflict Representation.

Sweep identifies and preserves conflicting information. Both claims remain
available for investigation and revision; the conflict is an explicit object
with a reason and confidence, never resolved by deleting either side.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ids import new_id


@dataclass
class Contradiction:
    """An explicit representation of two (or more) incompatible claims."""

    claim_a_id: str
    claim_b_id: str
    claim_a_text: str
    claim_b_text: str
    reason: str
    confidence: float = 0.8
    context: str = "same-time-same-location"     # e.g. temporal/entity-scope posture
    status: str = "OPEN"                          # OPEN / INVESTIGATED / RESOLVED
    id: str = field(default_factory=lambda: new_id("ctr"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim_a_id": self.claim_a_id,
            "claim_b_id": self.claim_b_id,
            "claim_a_text": self.claim_a_text,
            "claim_b_text": self.claim_b_text,
            "reason": self.reason,
            "confidence": self.confidence,
            "context": self.context,
            "status": self.status,
        }


@dataclass
class ContradictionSpec:
    """Declarative, testable rule for a contradiction family."""

    name: str
    version: str
    detect: Any  # callable(subject, predicate, obj, other_predicate, other_obj) -> reason|None


class BaseContradictionDetector(abc.ABC):
    """Detector contract: both claims stay available regardless of verdict."""

    @abc.abstractmethod
    def detect(self, subject: str, predicate: str, obj: str,
               other_predicate: str, other_obj: str,
               time_eq: bool = True, location_eq: bool = True) -> Optional[str]:
        """Return a human-readable reason if the two claims contradict, else None."""


class RuleContradictionDetector(BaseContradictionDetector):
    """Deterministic, rule-based contradiction detection.

    Families:
    - Location: same subject, same time, different incompatible locations
    - Foreign key: subject cannot be in two distinct valued predicates (e.g. status A vs B)
    - Boolean: `is gate open` vs `is gate closed`
    """

    LOCATION_PREDICATES = {"located-at", "in", "at", "inside", "position", "occupies"}
    STATUS_PREDICATES = {"status", "state", "is-open", "is-closed", "mode", "active"}
    BOOLEAN_TRUE = {"open", "closed", "on", "off", "running", "stopped", "true", "false"}

    def detect(self, subject: str, predicate: str, obj: str,
               other_predicate: str, other_obj: str,
               time_eq: bool = True, location_eq: bool = True) -> Optional[str]:
        # --- location family ------------------------------------------------
        if predicate in self.LOCATION_PREDICATES and other_predicate in self.LOCATION_PREDICATES:
            if time_eq and location_eq and obj != other_obj:
                return (f"{subject} cannot be at both {obj} and {other_obj} at the same time "
                        f"(same location context, same time)")
        # --- boolean status family ------------------------------------------
        if predicate in self.STATUS_PREDICATES and other_predicate in self.STATUS_PREDICATES:
            if obj != other_obj and obj in self.BOOLEAN_TRUE and other_obj in self.BOOLEAN_TRUE:
                return f"{subject} has conflicting {predicate}: {obj} vs {other_obj}"
        return None


class ContradictionEngine:
    """Records and preserves all detected contradictions."""

    def __init__(self, detector: BaseContradictionDetector | None = None):
        self.detector = detector or RuleContradictionDetector()
        self._contradictions: Dict[str, Contradiction] = {}
        self._by_claim: Dict[str, List[str]] = {}

    def compare(self, claim_a_id: str, text_a: str, subject_a: str, pred_a: str, obj_a: str,
                claim_b_id: str, text_b: str, subject_b: str, pred_b: str, obj_b: str,
                time_eq: bool = True, location_eq: bool = True,
                context: str = "") -> Optional[Contradiction]:
        """Detect a contradiction between two claims; if found, store and return
        the explicit CONTRADICTION object. Both claims remain unchanged."""
        subject = subject_a or subject_b or ""
        reason = self.detector.detect(subject, pred_a, obj_a, pred_b, obj_b, time_eq, location_eq)
        if reason is None:
            return None
        cid = new_id("ctr")
        c = Contradiction(
            claim_a_id=claim_a_id, claim_b_id=claim_b_id,
            claim_a_text=text_a, claim_b_text=text_b,
            reason=reason, confidence=0.9,
            context=context or "same-time-same-location",
        )
        self._contradictions[c.id] = c
        self._by_claim.setdefault(claim_a_id, []).append(c.id)
        self._by_claim.setdefault(claim_b_id, []).append(c.id)
        return c

    def all(self) -> List[Contradiction]:
        return list(self._contradictions.values())

    def for_claim(self, claim_id: str) -> List[Contradiction]:
        return [self._contradictions[i] for i in self._by_claim.get(claim_id, [])
                if i in self._contradictions]

    def count(self) -> int:
        return len(self._contradictions)

    def resolve(self, cid: str, note: str = "") -> None:
        c = self._contradictions.get(cid)
        if c:
            c.status = "RESOLVED"

    def to_dict(self) -> Dict[str, Any]:
        return {"contradictions": [c.to_dict() for c in self.all()]}