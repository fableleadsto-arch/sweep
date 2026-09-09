"""Phase 13 — Prediction / Observation Error Loop.

Sweep makes an explicit prediction from its current belief; when a new
observation arrives it compares against the prediction. A failed prediction
is an error that drives revision — but the failed prediction is never
written into memory as if it were a real observation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MatchResult(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    NEUTRAL = "NEUTRAL"


@dataclass
class Prediction:
    """A concrete, testable statement extracted from current belief."""

    prediction_id: str
    content: str
    belief_id: str
    belief_text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "content": self.content,
            "belief_id": self.belief_id,
            "belief_text": self.belief_text,
            "confidence": self.confidence,
        }


@dataclass
class Observation:
    """A real-world observation to compare against the prediction."""

    observation_id: str
    content: str
    agrees: bool = False      # independent judgement whether it agrees
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "content": self.content,
            "agrees": self.agrees,
            "source": self.source,
        }


@dataclass
class PredictionError:
    """Record of a prediction vs observation mismatch."""

    prediction_id: str
    observation_id: str
    prediction: str
    observation: str
    verdict: MatchResult
    error_magnitude: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "observation_id": self.observation_id,
            "prediction": self.prediction,
            "observation": self.observation,
            "verdict": self.verdict.value,
            "error_magnitude": self.error_magnitude,
        }


class PredictionEngine:
    """Tracks predictions; a failed prediction is flagged, never conflated with
    a true observation."""

    def __init__(self):
        self._predictions: Dict[str, Prediction] = {}
        self.errors: List[PredictionError] = []
        self._events: List[Dict[str, Any]] = []

    def submit(self, prediction: Prediction) -> None:
        self._predictions[prediction.prediction_id] = prediction
        self._events.append({"event": "prediction_submitted", "id": prediction.prediction_id})

    def compare(self, observation: Observation) -> PredictionError:
        """Compare an observation against the latest prediction."""
        pred = self.latest_prediction()
        if pred is None or not pred.prediction_id:
            raise ValueError("no prediction available to compare")
        verdict = MatchResult.CONFIRMED if observation.agrees else MatchResult.CONTRADICTED
        error_magnitude = 0.0 if observation.agrees else pred.confidence
        error = PredictionError(
            prediction_id=pred.prediction_id,
            observation_id=observation.observation_id,
            prediction=pred.content,
            observation=observation.content,
            verdict=verdict,
            error_magnitude=error_magnitude,
        )
        self.errors.append(error)
        self._events.append({"event": "compared", "verdict": verdict.value,
                             "prediction_id": pred.prediction_id})
        return error

    def latest_prediction(self) -> Optional[Prediction]:
        if not self._predictions:
            return None
        return list(self._predictions.values())[-1]

    def prediction_count(self) -> int:
        return len(self._predictions)

    def error_count(self) -> int:
        return len([e for e in self.errors if e.verdict is MatchResult.CONTRADICTED])

    def failed_predictions(self) -> List[PredictionError]:
        return [e for e in self.errors if e.verdict is MatchResult.CONTRADICTED]

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)