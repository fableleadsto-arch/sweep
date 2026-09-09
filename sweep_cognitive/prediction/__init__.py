"""
Prediction Engine — Phase 8.

SWEEP should not only answer "what is this?" — it should continuously estimate:
- what is likely to happen next
- what information is missing
- which action will probably succeed
- what can go wrong
- how much computation a task needs

Core rules (from the architecture spec):
- Predictions must carry uncertainty.
- Never convert predictions into facts automatically.
- Predictions can be verified when outcomes arrive; the resulting
  track record feeds calibration.

IMPLEMENTATION STATUS (honest):
- REAL: Prediction records, registry, verification, expiry, accuracy
  tracking, and confidence-bucket calibration. All tested.
- TEMPORARY SCAFFOLDING: the individual heuristic predictors
  (predict_task_complexity, predict_action_success, predict_action_risk,
  predict_missing_evidence). They are lexical/statistical priors that
  exercise the pipeline end-to-end. They must be replaced with learned
  models as outcome data accumulates; the Prediction/verify/calibration
  abstraction is designed so predictors can be swapped without changing
  callers.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    """Generate a guaranteed-unique ID (counter + timestamp)."""
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


class PredictionCategory(str, Enum):
    """What level of the system the prediction is about (spec section 10)."""
    TASK = "task"              # complexity, depth, duration, failure points
    ENVIRONMENT = "environment"  # next UI state, tool results, env changes
    KNOWLEDGE = "knowledge"    # missing evidence, likely sources, contradictions
    REASONING = "reasoning"    # promising hypothesis, discriminating evidence
    RESOURCE = "resource"      # compute, memory, latency needs
    ACTION = "action"          # action success, side effects, reversibility, risk
    LEARNING = "learning"      # novelty, skill-worthiness, belief updates


class PredictionStatus(str, Enum):
    """Lifecycle of a prediction. Predictions are NEVER facts; they are
    estimates that are pending, verified against an outcome, expired, or
    unverifiable in principle."""
    PENDING = "pending"
    VERIFIED_TRUE = "verified_true"
    VERIFIED_FALSE = "verified_false"
    EXPIRED = "expired"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Prediction:
    """
    A single prediction with uncertainty.

    confidence: believed probability (0-1) that the prediction is true.
    uncertainty: epistemic spread (0-1); high means we don't know how
        reliable our own estimate is. Kept separate from confidence
        deliberately — a confident guess can still be high-uncertainty.
    horizon_seconds: how long the prediction is meaningful for. None =
        no specific horizon (never auto-expires).
    basis: IDs of evidence/observations the prediction is grounded in.
    """
    id: str = field(default_factory=lambda: _uid("pred"))
    category: PredictionCategory = PredictionCategory.TASK
    content: str = ""
    confidence: float = 0.5
    uncertainty: float = 0.2
    created_at: float = field(default_factory=time.time)
    horizon_seconds: Optional[float] = None
    basis: list[str] = field(default_factory=list)
    status: PredictionStatus = PredictionStatus.PENDING
    verified_at: Optional[float] = None
    outcome_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(f"uncertainty must be in [0,1], got {self.uncertainty}")

    def is_stale(self, now: Optional[float] = None) -> bool:
        """A pending prediction past its horizon is stale."""
        if self.status != PredictionStatus.PENDING:
            return False
        if self.horizon_seconds is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.created_at) > self.horizon_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "content": self.content,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "created_at": self.created_at,
            "horizon_seconds": self.horizon_seconds,
            "basis_count": len(self.basis),
            "status": self.status.value,
            "verified_at": self.verified_at,
            "outcome_note": self.outcome_note,
            "metadata": self.metadata,
        }


class PredictionEngine:
    """
    Stores predictions, verifies them against outcomes, and reports
    accuracy and calibration.

    The engine NEVER marks a prediction true on its own — verification
    always comes from an explicit observed outcome via verify().
    """

    # Confidence buckets used for calibration reporting.
    CONFIDENCE_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    def __init__(self, learned_estimator: Optional["object"] = None):
        self._predictions: dict[str, Prediction] = {}
        # Optional Phase 15 learned estimator; lexical heuristics remain
        # the fallback whenever this is None or errors.
        self.learned_estimator = learned_estimator

    # ------------------------------------------------------------------
    # Core registry
    # ------------------------------------------------------------------

    def predict(
        self,
        category: PredictionCategory,
        content: str,
        confidence: float,
        uncertainty: float = 0.2,
        horizon_seconds: Optional[float] = None,
        basis: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Prediction:
        """Create and register a prediction."""
        pred = Prediction(
            category=category,
            content=content,
            confidence=confidence,
            uncertainty=uncertainty,
            horizon_seconds=horizon_seconds,
            basis=list(basis or []),
            metadata=dict(metadata or {}),
        )
        self._predictions[pred.id] = pred
        return pred

    def get(self, prediction_id: str) -> Prediction:
        if prediction_id not in self._predictions:
            raise KeyError(f"Unknown prediction: {prediction_id}")
        return self._predictions[prediction_id]

    def pending(self, category: Optional[PredictionCategory] = None) -> list[Prediction]:
        preds = [p for p in self._predictions.values() if p.status == PredictionStatus.PENDING]
        if category is not None:
            preds = [p for p in preds if p.category == category]
        return preds

    # ------------------------------------------------------------------
    # Verification (the ONLY way a prediction leaves PENDING besides expiry)
    # ------------------------------------------------------------------

    def verify(self, prediction_id: str, outcome: bool, note: str = "") -> Prediction:
        """Record the observed outcome for a prediction."""
        pred = self.get(prediction_id)
        if pred.status != PredictionStatus.PENDING:
            raise ValueError(
                f"Prediction {prediction_id} is {pred.status.value}, cannot verify"
            )
        pred.status = (
            PredictionStatus.VERIFIED_TRUE if outcome else PredictionStatus.VERIFIED_FALSE
        )
        pred.verified_at = time.time()
        pred.outcome_note = note
        return pred

    def mark_unverifiable(self, prediction_id: str, note: str = "") -> Prediction:
        """Mark a prediction as never having an observable outcome."""
        pred = self.get(prediction_id)
        if pred.status != PredictionStatus.PENDING:
            raise ValueError(
                f"Prediction {prediction_id} is {pred.status.value}, cannot mark unverifiable"
            )
        pred.status = PredictionStatus.UNVERIFIABLE
        pred.verified_at = time.time()
        pred.outcome_note = note
        return pred

    def expire_stale(self, now: Optional[float] = None) -> list[str]:
        """Expire pending predictions whose horizon has passed."""
        now = now if now is not None else time.time()
        expired: list[str] = []
        for pred in self._predictions.values():
            if pred.is_stale(now):
                pred.status = PredictionStatus.EXPIRED
                expired.append(pred.id)
        return expired

    # ------------------------------------------------------------------
    # Track record and calibration
    # ------------------------------------------------------------------

    def accuracy(self, category: Optional[PredictionCategory] = None) -> dict[str, Any]:
        """Accuracy over verified predictions, optionally per category."""
        verified = [
            p for p in self._predictions.values()
            if p.status in (PredictionStatus.VERIFIED_TRUE, PredictionStatus.VERIFIED_FALSE)
        ]
        if category is not None:
            verified = [p for p in verified if p.category == category]
        n = len(verified)
        true_n = sum(1 for p in verified if p.status == PredictionStatus.VERIFIED_TRUE)
        return {
            "verified": n,
            "true": true_n,
            "false": n - true_n,
            "accuracy": (true_n / n) if n else None,
        }

    def calibration(self) -> list[dict[str, Any]]:
        """
        Bucket verified predictions by stated confidence and compare with
        the actual true-rate in each bucket. calibration_error = |actual
        rate - bucket midpoint|. This is the raw material for the
        calibration engine (spec item 20); it cannot fix calibration by
        itself, it only measures it.
        """
        report = []
        verified = [
            p for p in self._predictions.values()
            if p.status in (PredictionStatus.VERIFIED_TRUE, PredictionStatus.VERIFIED_FALSE)
        ]
        for lo, hi in self.CONFIDENCE_BUCKETS:
            in_bucket = [p for p in verified if lo <= p.confidence < hi]
            n = len(in_bucket)
            mid = (lo + hi) / 2
            if n:
                rate = sum(1 for p in in_bucket if p.status == PredictionStatus.VERIFIED_TRUE) / n
                error = abs(rate - mid)
            else:
                rate = None
                error = None
            report.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "expected": mid,
                "count": n,
                "actual_rate": rate,
                "calibration_error": error,
            })
        return report

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for p in self._predictions.values():
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
            by_category[p.category.value] = by_category.get(p.category.value, 0) + 1
        return {
            "total": len(self._predictions),
            "by_status": by_status,
            "by_category": by_category,
        }

    # ==================================================================
    # TEMPORARY SCAFFOLDING — heuristic predictors
    # ------------------------------------------------------------------
    # The predictors below are lexical/statistical priors, NOT learned
    # models. They exist to exercise the predict → verify → calibrate
    # pipeline and to give the executive controller (Phase 12) something
    # to consume. Each returns a registered Prediction so outcomes can
    # later be fed back via verify() and the heuristics replaced.
    # ==================================================================

    COMPLEXITY_MARKERS = (
        "investigate", "analyze", "compare", "why", "how", "explain",
        "research", "verify", "prove", "design",
    )
    AMBIGUITY_MARKERS = (
        "maybe", "possibly", "might", "could be", "unclear", "not sure",
        "perhaps", "or something",
    )

    def predict_task_complexity(self, query: str) -> Prediction:
        """Estimate required reasoning depth. Lexical heuristics are the
        SCAFFOLDING fallback; when a learned estimator (Phase 15) is
        attached AND it was validated against held-out experience data,
        its calibrated probability contributes to the score.
        """
        lower = query.lower()
        words = len(query.split())

        score = 0.30
        if words > 30:
            score += 0.20
        elif words > 12:
            score += 0.10
        markers = sum(1 for m in self.COMPLEXITY_MARKERS if m in lower)
        score += 0.10 * min(markers, 3)
        ambiguity = sum(1 for m in self.AMBIGUITY_MARKERS if m in lower)
        score += 0.10 * min(ambiguity, 2)
        score = min(0.95, score)

        learned_note = None
        if self.learned_estimator is not None:
            try:
                from ..learning import features_from_query
                feats = features_from_query(query)
                # BlendedEstimator (Phase 18): behavioral + text combined;
                # plain LogisticEstimator: behavioral only.
                blend = getattr(
                    self.learned_estimator, "predict_proba_blended", None
                )
                if blend is not None:
                    p_success = blend(feats, query)
                else:
                    p_success = self.learned_estimator.predict_proba(feats)
                # Documented approximation: difficulty = 1 - p(success),
                # blended 50/50 with the lexical score.
                score = min(0.95, 0.5 * score + 0.5 * (1.0 - p_success))
                kind = "blended" if blend is not None else "behavioral"
                learned_note = (
                    f"learned({kind}) p(success)={p_success:.3f} "
                    "blended into complexity"
                )
            except Exception:  # noqa: BLE001 — learned path must never break prediction
                learned_note = "learned estimator errored; lexical score used"

        if ambiguity >= 2:
            mode = "EXPLORATORY"
        elif score < 0.35:
            mode = "FAST"
        elif score < 0.55:
            mode = "ROUTINE"
        elif score < 0.75:
            mode = "DELIBERATE"
        else:
            mode = "DEEP"

        return self.predict(
            category=PredictionCategory.TASK,
            content=f"Task requires {mode} reasoning depth",
            confidence=round(min(0.9, 0.5 + 0.1 * min(markers, 3)), 3),
            uncertainty=round(min(0.9, 0.30 + 0.05 * ambiguity), 3),
            metadata={
                "complexity_score": round(score, 3),
                "mode": mode,
                "query_words": words,
                "markers": markers,
                "ambiguity_markers": ambiguity,
                **({"learned": learned_note} if learned_note else {}),
            },
        )

    # Base success priors per action type. TEMPORARY hand-set priors —
    # replace with observed success rates once action history exists.
    ACTION_BASE_SUCCESS = {
        "read": 0.90,
        "search": 0.75,
        "navigate": 0.70,
        "write": 0.80,
        "execute": 0.65,
        "ui_click": 0.60,
        "api_call": 0.75,
    }

    def predict_action_success(
        self, action_type: str, history_success_rate: Optional[float] = None
    ) -> Prediction:
        """Probability an action of the given type succeeds. SCAFFOLDING."""
        known = action_type in self.ACTION_BASE_SUCCESS
        base = self.ACTION_BASE_SUCCESS.get(action_type, 0.5)
        if history_success_rate is not None:
            # Blend the prior with the observed rate (equal weight until
            # a proper Bayesian shrinkage replaces this).
            p = 0.5 * base + 0.5 * history_success_rate
        else:
            p = base
        uncertainty = 0.15 if history_success_rate is None else 0.10
        if not known:
            uncertainty += 0.20
        return self.predict(
            category=PredictionCategory.ACTION,
            content=f"Action of type {action_type!r} will succeed",
            confidence=round(min(1.0, max(0.0, p)), 3),
            uncertainty=round(min(0.9, uncertainty), 3),
            metadata={
                "action_type": action_type,
                "known_type": known,
                "history_rate": history_success_rate,
            },
        )

    DESTRUCTIVE_MARKERS = ("delete", "remove", "overwrite", "format", "drop", " rm ", "wipe")

    def predict_action_risk(self, action_description: str) -> Prediction:
        """Risk (incl. reversibility) of an action from its description. SCAFFOLDING."""
        lower = f" {action_description.lower()} "
        destructive = any(m in lower for m in self.DESTRUCTIVE_MARKERS)
        risk = 0.80 if destructive else 0.20
        return self.predict(
            category=PredictionCategory.ACTION,
            content=f"Action risk assessed: {action_description[:80]!r}",
            confidence=0.70,
            uncertainty=0.35,  # lexical risk detection is crude
            metadata={"destructive": destructive, "risk": risk},
        )

    def predict_missing_evidence(self, claim: str, known_source_count: int) -> Prediction:
        """Knowledge-level: is independent corroboration missing? SCAFFOLDING."""
        if known_source_count < 2:
            conf = 0.85
            content = (
                f"Claim likely lacks independent corroboration "
                f"({known_source_count} source(s)): {claim[:60]!r}"
            )
        else:
            conf = max(0.10, 0.60 - 0.10 * (known_source_count - 2))
            content = f"Claim may still lack corroboration: {claim[:60]!r}"
        return self.predict(
            category=PredictionCategory.KNOWLEDGE,
            content=content,
            confidence=round(conf, 3),
            uncertainty=0.25,
            metadata={"claim": claim[:200], "source_count": known_source_count},
        )


__all__ = [
    "Prediction",
    "PredictionCategory",
    "PredictionStatus",
    "PredictionEngine",
]
