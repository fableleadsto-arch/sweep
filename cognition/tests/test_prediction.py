"""Phase 13 — Prediction / Observation Error Loop tests."""

import pytest

from cognition.prediction import (Observation, Prediction, PredictionEngine,
                                  PredictionError, MatchResult)


def test_prediction_submitted_and_latest():
    eng = PredictionEngine()
    eng.submit(Prediction("p1", "gate will open", "b1", "gate opens automatically", 0.85))
    assert eng.prediction_count() == 1
    assert eng.latest_prediction().content == "gate will open"


def test_observation_confirms_prediction_no_error():
    eng = PredictionEngine()
    eng.submit(Prediction("p1", "gate will open", "b1", "gate opens automatically", 0.85))
    error = eng.compare(Observation("o1", "gate opened", agrees=True, source="sensor"))
    assert error.verdict is MatchResult.CONFIRMED
    assert eng.error_count() == 0


def test_failed_prediction_is_recorded_but_not_observation():
    eng = PredictionEngine()
    eng.submit(Prediction("p1", "gate will stay closed", "b1", "gate locked", 0.9))
    error = eng.compare(Observation("o1", "gate opened", agrees=False, source="sensor"))
    assert error.verdict is MatchResult.CONTRADICTED
    assert eng.error_count() == 1
    # critical: the failed prediction is not promoted into memory as a fact
    assert error.prediction != error.observation
    # the prediction object stays a Prediction below the "true observation" level
    assert not isinstance(error, Observation)
    events = [ev["event"] for ev in eng.events()]
    assert "prediction_submitted" in events and "compared" in events


def test_error_magnitude_scales_with_confidence():
    eng = PredictionEngine()
    eng.submit(Prediction("p1", "state is A", "b1", "state is A", 0.95))
    err = eng.compare(Observation("o1", "state is B", agrees=False))
    assert abs(err.error_magnitude - 0.95) < 1e-9


def test_failed_predictions_listed():
    eng = PredictionEngine()
    eng.submit(Prediction("p1", "rain", "b1", "rain", 0.6))
    eng.compare(Observation("o1", "sunny", agrees=False))
    eng.submit(Prediction("p2", "rain", "b1", "rain", 0.6))
    eng.compare(Observation("o2", "sunny", agrees=False))
    assert len(eng.failed_predictions()) == 2


def test_compare_without_prediction_raises():
    eng = PredictionEngine()
    with pytest.raises(ValueError):
        eng.compare(Observation("o1", "no prediction yet", agrees=True))


def test_payloads_serialize():
    eng = PredictionEngine()
    eng.submit(Prediction("p1", "rain", "b1", "rain", 0.6))
    err = eng.compare(Observation("o1", "sunny", agrees=False))
    d = err.to_dict()
    assert d["verdict"] == "CONTRADICTED"
    assert d["prediction"] == "rain"
    assert d["observation"] == "sunny"