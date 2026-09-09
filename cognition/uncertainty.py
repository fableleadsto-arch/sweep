"""Phase 11 — Explicit Epistemic State.

Sweep distinguishes between what it knows, suspects, cannot establish, and
what is contested. Required states:

    VERIFIED, PROBABLE, POSSIBLE, CONTESTED, UNRESOLVED, INSUFFICIENT_EVIDENCE

An ambiguous or incomplete task produces an explicit uncertainty state instead
of fabricated certainty.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EpistemicState(str, enum.Enum):
    VERIFIED = "VERIFIED"
    PROBABLE = "PROBABLE"
    POSSIBLE = "POSSIBLE"
    CONTESTED = "CONTESTED"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class ClaimEpistemicStatus:
    """Epistemic status of a single claim."""

    claim_id: str
    statement: str
    state: EpistemicState
    confidence: float
    reasons: List[str] = field(default_factory=list)
    contested_against: List[str] = field(default_factory=list)  # claim ids
    supporting_evidence: int = 0
    contradicting_evidence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "contested_against": self.contested_against,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
        }


@dataclass
class EvidenceSignal:
    """A signal for computing epistemic state from evidence."""

    claim_id: str
    statement: str
    supporting: int = 0
    contradicting: int = 0
    source_diversity: int = 0     # distinct independent sources
    neutral: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "supporting": self.supporting,
            "contradicting": self.contradicting,
            "source_diversity": self.source_diversity,
            "neutral": self.neutral,
        }


class UncertaintyEngine:
    """Deterministic mapping from evidence signals to epistemic states."""

    def __init__(self):
        self._statuses: Dict[str, ClaimEpistemicStatus] = {}

    def assess(self, signal: EvidenceSignal) -> ClaimEpistemicStatus:
        """Classify a claim into an explicit epistemic state."""
        sup, con, neutral, div = signal.supporting, signal.contradicting, signal.neutral, signal.source_diversity
        reasons: List[str] = []
        contested_against: List[str] = []

        if con > 0 and sup > 0:
            state = EpistemicState.CONTESTED
            reasons.append(f"{sup} supporting vs {con} contradicting")
        elif con > 0:
            state = EpistemicState.UNRESOLVED
            reasons.append(f"all {con} evidence contradicts; nothing supports")
        elif sup == 0 and con == 0 and neutral == 0:
            state = EpistemicState.INSUFFICIENT_EVIDENCE
            reasons.append("no evidence either way")
        elif sup == 0:
            state = EpistemicState.POSSIBLE
            reasons.append(f"{neutral or 0} neutral items; consistent but unverified")
        elif div >= 2 and sup >= 2:
            state = EpistemicState.VERIFIED
            reasons.append(f"{sup} supporting evidence across {div} independent sources")
        else:
            state = EpistemicState.PROBABLE
            reasons.append(f"{sup} supporting evidence; weak source independence")

        status = ClaimEpistemicStatus(
            claim_id=signal.claim_id,
            statement=signal.statement,
            state=state,
            confidence=_confidence_for(state),
            reasons=reasons,
            contested_against=contested_against,
            supporting_evidence=sup,
            contradicting_evidence=con,
        )
        self._statuses[signal.claim_id] = status
        return status

    def status(self, claim_id: str) -> Optional[ClaimEpistemicStatus]:
        return self._statuses.get(claim_id)

    def all(self) -> List[ClaimEpistemicStatus]:
        return list(self._statuses.values())

    def assert_state(self, claim_id: str) -> EpistemicState:
        s = self._statuses.get(claim_id)
        if s is None:
            return EpistemicState.INSUFFICIENT_EVIDENCE
        return s.state


def _confidence_for(state: EpistemicState) -> float:
    return {
        EpistemicState.VERIFIED: 0.95,
        EpistemicState.PROBABLE: 0.75,
        EpistemicState.POSSIBLE: 0.55,
        EpistemicState.CONTESTED: 0.5,
        EpistemicState.UNRESOLVED: 0.4,
        EpistemicState.INSUFFICIENT_EVIDENCE: 0.2,
    }.get(state, 0.4)