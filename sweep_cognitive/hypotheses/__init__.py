"""
Hypothesis & Counterfactual Engine — Phase 9.

Module 9 (spec): generate multiple explanations when ambiguity exists.
Each hypothesis carries supporting evidence, conflicting evidence,
confidence, missing evidence, and discriminating tests.

Module 22 (spec): counterfactual reasoning — "what would have to be
true for H to hold?" and "what would we observe if H were false?"

IMPLEMENTATION STATUS (honest):
- REAL: Hypothesis lifecycle (generation, evidence accumulation,
  confidence updates, dominance ranking), discriminating-test
  recommendation (which observation would separate competing
  hypotheses), abductive selection, and counterfactual simulation
  over the world model's relationship graph.
- TEMPORARY SCAFFOLDING: hypothesis *content* generation is a
  structured template built from observations, not a learned
  generator. The bookkeeping around it (evidence weights, Bayesian
  confidence updates, dominance, test recommendation) is real.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


# ======================================================================
# HYPOTHESES
# ======================================================================

class HypothesisStatus(str, Enum):
    """Lifecycle of a hypothesis (spec section 18 distinguishes these
    from FACT/OBSERVATION/INFERENCE — a hypothesis is an explanation
    under consideration, never a conclusion)."""
    OPEN = "open"
    SUPPORTED = "supported"       # best current explanation
    WEAKENED = "weakened"         # contradicted but not eliminated
    ELIMINATED = "eliminated"     # ruled out by contradicting evidence
    SUPERSEDED = "superseded"     # merged/absorbed into another hypothesis


@dataclass
class EvidenceItem:
    """A piece of evidence attached to a hypothesis.

    weight: how strongly this item supports (positive) or contradicts
    (negative) the hypothesis. Reliability scales the weight.
    """
    description: str
    weight: float                    # negative = contradicting
    source: str = ""
    reliability: float = 0.8         # source reliability, 0-1
    created_at: float = field(default_factory=time.time)
    item_id: str = field(default_factory=lambda: _uid("evi"))

    @property
    def effective_weight(self) -> float:
        return self.weight * self.reliability

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "description": self.description,
            "weight": self.weight,
            "source": self.source,
            "reliability": self.reliability,
            "effective_weight": self.effective_weight,
        }


@dataclass
class DiscriminatingTest:
    """A test whose outcome would separate competing hypotheses."""
    description: str
    # hypothesis_id -> expected outcome if that hypothesis were true
    expectations: dict[str, str] = field(default_factory=dict)
    cost: float = 0.5                # relative cost of running the test
    test_id: str = field(default_factory=lambda: _uid("test"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "description": self.description,
            "expectations": dict(self.expectations),
            "cost": self.cost,
        }


@dataclass
class Hypothesis:
    """One explanation among several (spec module 9).

    confidence is maintained by Bayesian-style updates in
    HypothesisEngine._recompute, not set by hand.
    """
    statement: str
    hypothesis_id: str = field(default_factory=lambda: _uid("hyp"))
    status: HypothesisStatus = HypothesisStatus.OPEN
    confidence: float = 0.5
    prior: float = 0.5
    evidence: list[EvidenceItem] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    # What we'd expect to observe if this hypothesis were true
    predictions_if_true: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def supporting(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.effective_weight > 0]

    @property
    def contradicting(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.effective_weight < 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence,
            "prior": self.prior,
            "supporting_count": len(self.supporting),
            "contradicting_count": len(self.contradicting),
            "missing_evidence": list(self.missing_evidence),
            "predictions_if_true": list(self.predictions_if_true),
            "metadata": self.metadata,
        }


class HypothesisEngine:
    """
    Generates, tracks, and ranks competing explanations.

    Confidence updates: each evidence item contributes its effective
    weight; the raw score is squashed through a logistic function onto
    (0,1). This is a real, monotone update rule — more/stronger support
    raises confidence, contradiction lowers it — but it is a fixed
    functional form, not a learned one.
    """

    def __init__(self):
        self._hypotheses: dict[str, Hypothesis] = {}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        statement: str,
        prior: float = 0.5,
        predictions_if_true: Optional[list[str]] = None,
        missing_evidence: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Hypothesis:
        """Register a candidate explanation. SCAFFOLDING note: callers
        currently supply the statement; a learned generator would sit
        here."""
        hyp = Hypothesis(
            statement=statement,
            prior=prior,
            confidence=prior,
            predictions_if_true=list(predictions_if_true or []),
            missing_evidence=list(missing_evidence or []),
            metadata=dict(metadata or {}),
        )
        self._hypotheses[hyp.hypothesis_id] = hyp
        return hyp

    def generate_from_observations(
        self, observations: list[str], context: str = ""
    ) -> list[Hypothesis]:
        """TEMPLATE-BASED generation from observations (scaffolding).
        Produces:
        - one hypothesis per observation ("O explains the situation")
        - one alternative hypothesis (the negation/alternative frame)
        Real generation would propose *causes*; this gives the ranking
        machinery something structurally valid to work with.
        """
        hyps: list[Hypothesis] = []
        if not observations:
            return hyps
        for obs in observations:
            hyp = self.generate(
                statement=f"{obs.strip().rstrip('.')}" + (f" (in context: {context})" if context else ""),
                prior=0.4,
                predictions_if_true=[f"We would observe more evidence consistent with: {obs}"],
                missing_evidence=["Independent corroboration", "Causal mechanism"],
                metadata={"generated_from": "observation_template", "observation": obs},
            )
            hyps.append(hyp)
        # One explicit alternative so there are always >= 2 competitors
        alt = self.generate(
            statement=(
                "Alternative explanation: the observations above have a "
                "different common cause" + (f" (in context: {context})" if context else "")
            ),
            prior=0.3,
            predictions_if_true=["Evidence would point to a shared unobserved cause"],
            missing_evidence=["Identification of the common cause"],
            metadata={"generated_from": "alternative_template"},
        )
        hyps.append(alt)
        return hyps

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def add_evidence(
        self, hypothesis_id: str, description: str, weight: float,
        source: str = "", reliability: float = 0.8,
    ) -> EvidenceItem:
        """Add supporting (weight > 0) or contradicting (weight < 0)
        evidence and update confidence."""
        hyp = self.get(hypothesis_id)
        if hyp.status in (HypothesisStatus.ELIMINATED, HypothesisStatus.SUPERSEDED):
            raise ValueError(f"Cannot add evidence to {hyp.status.value} hypothesis")
        item = EvidenceItem(
            description=description, weight=weight,
            source=source, reliability=reliability,
        )
        hyp.evidence.append(item)
        self._recompute(hyp)
        return item

    def _recompute(self, hyp: Hypothesis) -> None:
        """Squash accumulated evidence through a logistic onto (0,1)."""
        import math
        raw = hyp.prior * 4.0 - 2.0  # map prior 0..1 to logit space -2..2
        for item in hyp.evidence:
            raw += item.effective_weight * 2.0
        hyp.confidence = round(1.0 / (1.0 + math.exp(-raw)), 4)

        # Status transitions (scaled consistently with the logit update:
        # strong_contra uses the same ×2.0 scaling)
        if hyp.contradicting:
            strong_contra = sum(-e.effective_weight * 2.0 for e in hyp.contradicting)
            if strong_contra >= 1.5:
                hyp.status = HypothesisStatus.ELIMINATED
                hyp.confidence = 0.0
            else:
                hyp.status = HypothesisStatus.WEAKENED
        elif hyp.status == HypothesisStatus.WEAKENED and not hyp.contradicting:
            hyp.status = HypothesisStatus.OPEN

    # ------------------------------------------------------------------
    # Queries and ranking
    # ------------------------------------------------------------------

    def get(self, hypothesis_id: str) -> Hypothesis:
        if hypothesis_id not in self._hypotheses:
            raise KeyError(f"Unknown hypothesis: {hypothesis_id}")
        return self._hypotheses[hypothesis_id]

    def all(self, status: Optional[HypothesisStatus] = None) -> list[Hypothesis]:
        hyps = list(self._hypotheses.values())
        if status is not None:
            hyps = [h for h in hyps if h.status == status]
        return hyps

    def viable(self) -> list[Hypothesis]:
        """Hypotheses still in contention (open/supported/weakened),
        ranked by confidence."""
        keep = {HypothesisStatus.OPEN, HypothesisStatus.SUPPORTED, HypothesisStatus.WEAKENED}
        ranked = [h for h in self.all() if h.status in keep]
        ranked.sort(key=lambda h: h.confidence, reverse=True)
        # Mark the leader
        if ranked:
            for h in ranked:
                if h.status == HypothesisStatus.SUPPORTED:
                    h.status = HypothesisStatus.OPEN
            ranked[0].status = HypothesisStatus.SUPPORTED
        return ranked

    def best_explanation(self) -> Optional[Hypothesis]:
        """Abductive selection: the highest-confidence viable hypothesis,
        provided it is meaningfully ahead of the runner-up."""
        ranked = self.viable()
        if not ranked:
            return None
        if len(ranked) == 1:
            return ranked[0]
        if ranked[0].confidence - ranked[1].confidence < 0.15:
            return None  # ambiguous — refuse to pick (hallucination control)
        return ranked[0]

    def recommend_discriminating_test(self) -> Optional[DiscriminatingTest]:
        """
        Find an observation that separates the top-2 viable hypotheses:
        something one predicts and the other does not. Prefers cheaper
        tests. Returns None if no discriminating observation exists.
        """
        ranked = self.viable()
        if len(ranked) < 2:
            return None
        top, second = ranked[0], ranked[1]
        only_top = [p for p in top.predictions_if_true if p not in second.predictions_if_true]
        if not only_top:
            return None
        return DiscriminatingTest(
            description=f"Check whether: {only_top[0]}",
            expectations={top.hypothesis_id: only_top[0]},
            cost=0.5,
        )

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for h in self._hypotheses.values():
            by_status[h.status.value] = by_status.get(h.status.value, 0) + 1
        return {"total": len(self._hypotheses), "by_status": by_status}


# ======================================================================
# COUNTERFACTUAL ENGINE (spec module 22)
# ======================================================================

@dataclass
class CounterfactualResult:
    """Outcome of a counterfactual simulation."""
    question: str
    is_consistent: bool              # world model contains no contradiction with the premise
    requires_relationships: list[tuple[str, str, str]]  # triples the premise needs
    blocking_contradictions: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "is_consistent": self.is_consistent,
            "requires_relationships": [list(t) for t in self.requires_relationships],
            "blocking_contradictions": self.blocking_contradictions,
            "notes": self.notes,
        }


class CounterfactualEngine:
    """
    Answers counterfactual questions against the world model:
    - "Would X hold if Y were true?" — consistency check of a hypothetical
      premise against current world knowledge.
    - "What would we observe if H were false?" — returns the hypotheses'
      predictions_if_true as the observations that would disappear.

    The consistency check and propagation analysis are graph operations
    over explicit relationships — NOT a learned world simulator.
    """

    def __init__(self, world_model):
        # world_model: sweep_cognitive.world.WorldModel (duck-typed to
        # avoid a hard import cycle)
        self.world = world_model

    def evaluate_premise(
        self, subject_id: str, predicate, object_id: str, question: str = ""
    ) -> CounterfactualResult:
        """
        Check whether asserting a hypothetical relationship would
        contradict the world model. A premise is inconsistent if the
        world already holds a mutually-exclusive relationship for the
        same (subject, object) pair (e.g., premise PART_OF vs existing
        DIFFERENT_FROM).
        """
        state = self.world.state
        requires = [(subject_id, predicate.value, object_id)]
        blocking: list[dict[str, Any]] = []

        exclusive_pairs = {
            ("part_of", "different_from"),
            ("different_from", "part_of"),
        }
        for rel in state.query_relationships():
            involved = (rel.subject_id == subject_id and rel.object_id == object_id) or \
                       (rel.subject_id == object_id and rel.object_id == subject_id)
            if not involved:
                continue
            existing = (rel.predicate.value, predicate.value)
            if existing in exclusive_pairs:
                blocking.append({
                    "existing_relationship": list(rel.to_triple()),
                    "conflicts_with": [subject_id, predicate.value, object_id],
                })

        notes = []
        if blocking:
            notes.append("Premise contradicts existing world knowledge")
        else:
            notes.append("Premise is consistent with current world knowledge")
        return CounterfactualResult(
            question=question or f"Would ({subject_id}, {predicate.value}, {object_id}) hold?",
            is_consistent=not blocking,
            requires_relationships=requires,
            blocking_contradictions=blocking,
            notes=notes,
        )

    def observations_if_false(self, hypothesis: Hypothesis) -> list[str]:
        """If the hypothesis were FALSE, its predictions_if_true would
        not be observed — i.e., their absence is evidence for falsity."""
        return list(hypothesis.predictions_if_true)

    def what_would_change(
        self, subject_id: str, predicate, object_id: str
    ) -> dict[str, Any]:
        """Report the downstream entities a new relationship would touch:
        the new neighbor itself plus both endpoints' existing 1-hop
        neighborhoods (i.e., where the change would propagate)."""
        affected: list[str] = []

        # The new neighbor itself is always affected
        if object_id not in affected:
            affected.append(object_id)

        # Propagation through both endpoints' existing neighborhoods
        for endpoint in (subject_id, object_id):
            for entity in self.world.get_related_entities(endpoint, max_depth=1):
                if entity.id not in affected:
                    affected.append(entity.id)

        return {
            "new_relationship": [subject_id, predicate.value, object_id],
            "affected_entities": affected,
            "affected_count": len(affected),
        }


__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "HypothesisEngine",
    "EvidenceItem",
    "DiscriminatingTest",
    "CounterfactualEngine",
    "CounterfactualResult",
]
