"""Phase 10 — Evidence-Driven Belief Updating.

Sweep changes an existing conclusion when new evidence arrives. The original
conclusion and the reason for every revision remain in history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ids import new_id


@dataclass
class BeliefState:
    """A current belief (a leading conclusion) with confidence."""

    conclusion: str
    confidence: float
    hypothesis_id: str = ""
    created_at: float = 0.0
    evidence_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        import time as _t
        return {
            "conclusion": self.conclusion,
            "confidence": round(self.confidence, 4),
            "hypothesis_id": self.hypothesis_id,
            "created_at": self.created_at or _t.time(),
            "evidence_seen": self.evidence_seen,
        }


@dataclass
class RevisionRecord:
    """A history record: what the belief was, why it changed."""

    revision_id: str
    timestamp: float
    previous: Optional[Dict[str, Any]]
    new: Dict[str, Any]
    reason: str
    triggering_evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "timestamp": self.timestamp,
            "previous": self.previous,
            "new": self.new,
            "reason": self.reason,
            "triggering_evidence": self.triggering_evidence,
        }


@dataclass
class BeliefEvidence:
    """New evidence bearing on a conclusion."""

    content: str
    supports: bool = True
    strength: float = 1.0     # how strongly this evidence counts
    source: str = ""
    hypothesis_target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "supports": self.supports,
            "strength": self.strength,
            "source": self.source,
            "hypothesis_target": self.hypothesis_target,
        }


class BeliefRevisionEngine:
    """Tracks a current leading belief and revises it as evidence arrives.

    The original conclusion and every revision are recorded in history.
    """

    def __init__(self):
        self.current: Optional[BeliefState] = None
        self.history: List[RevisionRecord] = []
        self._candidate_scores: Dict[str, float] = {}
        self._candidate_names: Dict[str, str] = {}

    # ------------------------------------------------------------- seeding
    def set_initial(self, conclusion: str, confidence: float,
                    hypothesis_id: str = "", evidence_seen: int = 0) -> BeliefState:
        state = BeliefState(conclusion=conclusion, confidence=confidence,
                            hypothesis_id=hypothesis_id or new_id("hyp"),
                            evidence_seen=evidence_seen)
        self.current = state
        self._candidate_scores[state.hypothesis_id] = confidence
        self._candidate_names[state.hypothesis_id] = conclusion
        return state

    def register_candidate(self, conclusion: str, confidence: float,
                           hypothesis_id: str = "") -> str:
        """Register a competing candidate belief with an initial confidence."""
        hid = hypothesis_id or new_id("hyp")
        self._candidate_scores[hid] = confidence
        self._candidate_names[hid] = conclusion
        return hid

    # ------------------------------------------------------------- revision
    def consider(self, evidence: BeliefEvidence) -> Optional[RevisionRecord]:
        """Apply new evidence; if it flips the leading conclusion, record it."""
        if self.current is None:
            raise ValueError("no initial belief set")

        target = evidence.hypothesis_target or self.current.hypothesis_id

        # update the candidate score for the targeted hypothesis
        delta = (0.25 if evidence.supports else -0.25) * evidence.strength
        old = self._candidate_scores.get(target, 0.0)
        new_score = max(0.0, min(1.0, old + delta))
        self._candidate_scores[target] = new_score

        # scoring update to current belief
        current_was_hyp = target == self.current.hypothesis_id
        if current_was_hyp:
            self.current.confidence = new_score
            self.current.evidence_seen += 1

        # If the current belief is no longer leading, switch.
        new_leader = max(self._candidate_scores, key=lambda k: self._candidate_scores[k])
        if new_leader != self.current.hypothesis_id:
            reason = (f"new evidence {'supports' if evidence.supports else 'contradicts'} "
                      f"'{self._candidate_names[new_leader] or target}'")
            return self._switch(new_leader, reason, evidence)
        return None

    def _switch(self, new_hyp_id: str, reason: str, evidence: BeliefEvidence) -> RevisionRecord:
        prev = self.current.to_dict() if self.current else None
        name = self._candidate_names.get(new_hyp_id, "unknown conclusion")
        new_state = BeliefState(
            conclusion=name,
            confidence=self._candidate_scores[new_hyp_id],
            hypothesis_id=new_hyp_id,
            evidence_seen=self.current.evidence_seen if self.current else 0,
        )
        record = RevisionRecord(
            revision_id=new_id("rev"),
            timestamp=__import__("time").time(),
            previous=prev,
            new=new_state.to_dict(),
            reason=reason,
            triggering_evidence=evidence.content,
        )
        self.history.append(record)
        self.current = new_state
        return record

    def leading(self) -> Optional[BeliefState]:
        return self.current

    def leading_hypothesis(self) -> str:
        if not self.current:
            return ""
        return self.current.conclusion

    def revision_history(self) -> List[RevisionRecord]:
        return list(self.history)

    def count_revisions(self) -> int:
        return len(self.history)