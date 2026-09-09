"""Phase 8 — Competing Hypothesis Reasoning.

Maintains multiple explanations simultaneously. Evidence updates each
hypothesis's confidence; ranking and uncertainty are explicit. No conclusion
is forced when evidence is insufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ids import new_id


@dataclass
class HypothesisScore:
    """Score of a single hypothesis."""

    hypothesis_id: str
    statement: str
    confidence: float
    support_weight: float = 0.0
    oppose_weight: float = 0.0
    uncertainty: float = 0.0
    status: str = "ACTIVE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "confidence": round(self.confidence, 4),
            "support_weight": round(self.support_weight, 4),
            "oppose_weight": round(self.oppose_weight, 4),
            "uncertainty": round(self.uncertainty, 4),
            "status": self.status,
        }


@dataclass
class EvidenceUpdate:
    """One evidence event affecting hypotheses."""

    evidence_id: str
    content: str
    supports: List[str] = field(default_factory=list)      # hypothesis ids
    opposes: List[str] = field(default_factory=list)       # hypothesis ids
    weight: float = 1.0
    strength: float = 0.8    # how strongly the evidence counts
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "supports": self.supports,
            "opposes": self.opposes,
            "weight": self.weight,
            "strength": self.strength,
            "source": self.source,
        }


class HypothesisEngine:
    """Competing-hypothesis engine with explicit ranking and retention of
    uncertainty when evidence is insufficient."""

    def __init__(self):
        self._scores: Dict[str, HypothesisScore] = {}
        self._updates: List[EvidenceUpdate] = []

    # ------------------------------------------------------------- lifecycle
    def add_hypothesis(self, statement: str, initial: float = 0.5, status: str = "ACTIVE") -> HypothesisScore:
        hid = new_id("hyp")
        score = HypothesisScore(hypothesis_id=hid, statement=statement,
                                confidence=initial, uncertainty=1.0 - initial, status=status)
        self._scores[hid] = score
        return score

    def all(self) -> List[HypothesisScore]:
        return list(self._scores.values())

    def get(self, hid: str) -> Optional[HypothesisScore]:
        return self._scores.get(hid)

    # ------------------------------------------------------------- evidence
    def incorporate(self, update: EvidenceUpdate) -> None:
        """Apply an evidence update to every affected hypothesis."""
        self._updates.append(update)
        for hid in self._scores:
            supports = hid in update.supports
            opposes = hid in update.opposes
            w = update.weight * update.strength
            if supports and not opposes:
                self._boost(hid, w)
            elif opposes and not supports:
                self._reduce(hid, w)
            elif not supports and not opposes:
                self._uncertainty_up(hid)

    def _boost(self, hid: str, w: float) -> None:
        s = self._scores[hid]
        s.support_weight += w
        s.confidence = min(1.0, s.confidence + 0.2 * w)
        s.uncertainty = max(0.0, 0.382 * (1.0 - s.confidence))
        self._recheck_status(s)

    def _reduce(self, hid: str, w: float) -> None:
        s = self._scores[hid]
        s.oppose_weight += w
        s.confidence = max(0.0, s.confidence - 0.2 * w)
        s.uncertainty = min(1.0, s.uncertainty + 0.1 * w)
        self._recheck_status(s)

    def _uncertainty_up(self, hid: str) -> None:
        s = self._scores[hid]
        s.uncertainty = min(1.0, s.uncertainty + 0.05)

    def _recheck_status(self, s: HypothesisScore) -> None:
        if s.confidence >= 0.8:
            s.status = "SUPPORTED"
        elif s.confidence <= 0.3:
            s.status = "WEAK"
        else:
            s.status = "ACTIVE"

    # ------------------------------------------------------------- ranking
    def rank(self, key: str = "confidence") -> List[HypothesisScore]:
        return sorted(self._scores.values(), key=lambda h: getattr(h, key), reverse=True)

    def leading(self) -> Optional[HypothesisScore]:
        ranked = self.rank()
        if not ranked:
            return None
        lead = ranked[0]
        # No conclusion forced when evidence is insufficient: require a
        # meaningful confidence margin AND low uncertainty.
        if lead.confidence < 0.6 or lead.uncertainty > 0.5:
            return None
        return lead

    def no_conclusion(self) -> bool:
        """True when no hypothesis has enough support to be a conclusion."""
        return self.leading() is None

    # ------------------------------------------------------------- json
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypotheses": [h.to_dict() for h in self.all()],
            "updates": [u.to_dict() for u in self._updates],
            "leading": self.leading().to_dict() if self.leading() else None,
            "no_conclusion": self.no_conclusion(),
        }