"""
Tests for the prediction engine (Phase 8).

Verifies:
1. Predictions carry uncertainty and never auto-become facts
2. Lifecycle: pending → verified_true/false / expired / unverifiable
3. Accuracy and calibration reporting
4. Heuristic predictors produce registered, verifiable predictions
5. Scaffolding honesty: heuristics are marked as such
"""

import pytest
import time

from sweep_cognitive.prediction import (
    Prediction,
    PredictionCategory,
    PredictionStatus,
    PredictionEngine,
)


class TestPrediction:
    """Tests for Prediction records."""

    def test_basic_prediction(self):
        pred = Prediction(
            category=PredictionCategory.ACTION,
            content="click will succeed",
            confidence=0.7,
            uncertainty=0.2,
        )
        assert pred.status == PredictionStatus.PENDING
        assert 0.0 <= pred.confidence <= 1.0
        assert 0.0 <= pred.uncertainty <= 1.0

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValueError):
            Prediction(confidence=1.5)
        with pytest.raises(ValueError):
            Prediction(confidence=-0.1)
        with pytest.raises(ValueError):
            Prediction(uncertainty=1.2)

    def test_predictions_are_never_facts(self):
        """A fresh prediction is PENDING regardless of how confident it is."""
        pred = Prediction(confidence=0.99, uncertainty=0.0)
        assert pred.status == PredictionStatus.PENDING

    def test_ids_are_unique(self):
        a = Prediction()
        b = Prediction()
        assert a.id != b.id

    def test_staleness(self):
        pred = Prediction(horizon_seconds=1.0)
        assert not pred.is_stale(now=pred.created_at)
        assert pred.is_stale(now=pred.created_at + 2.0)

    def test_no_horizon_never_stale(self):
        pred = Prediction(horizon_seconds=None)
        assert not pred.is_stale(now=pred.created_at + 10_000)

    def test_stale_only_if_pending(self):
        pred = Prediction(horizon_seconds=1.0, status=PredictionStatus.VERIFIED_TRUE)
        assert not pred.is_stale(now=pred.created_at + 100.0)

    def test_to_dict(self):
        pred = Prediction(content="x", confidence=0.6)
        d = pred.to_dict()
        assert d["status"] == "pending"
        assert d["confidence"] == 0.6
        assert "uncertainty" in d


class TestLifecycle:
    """Tests for verification and expiry."""

    def setup_method(self):
        self.engine = PredictionEngine()

    def test_verify_true(self):
        pred = self.engine.predict(
            PredictionCategory.ACTION, "action succeeds", confidence=0.8
        )
        result = self.engine.verify(pred.id, True, note="it worked")
        assert result.status == PredictionStatus.VERIFIED_TRUE
        assert result.verified_at is not None
        assert result.outcome_note == "it worked"

    def test_verify_false(self):
        pred = self.engine.predict(PredictionCategory.ACTION, "fails", confidence=0.3)
        result = self.engine.verify(pred.id, False)
        assert result.status == PredictionStatus.VERIFIED_FALSE

    def test_cannot_verify_twice(self):
        pred = self.engine.predict(PredictionCategory.ACTION, "x", confidence=0.5)
        self.engine.verify(pred.id, True)
        with pytest.raises(ValueError):
            self.engine.verify(pred.id, False)

    def test_cannot_verify_unknown(self):
        with pytest.raises(KeyError):
            self.engine.verify("pred_nope", True)

    def test_mark_unverifiable(self):
        pred = self.engine.predict(PredictionCategory.REASONING, "hypothesis", confidence=0.5)
        result = self.engine.mark_unverifiable(pred.id, note="no observable outcome")
        assert result.status == PredictionStatus.UNVERIFIABLE

    def test_expire_stale(self):
        pred = self.engine.predict(
            PredictionCategory.ENVIRONMENT, "screen changes", confidence=0.6,
            horizon_seconds=1.0,
        )
        # Force creation time into the past
        pred.created_at -= 10.0
        expired = self.engine.expire_stale()
        assert pred.id in expired
        assert pred.status == PredictionStatus.EXPIRED

    def test_pending_filters(self):
        p1 = self.engine.predict(PredictionCategory.ACTION, "a", confidence=0.5)
        p2 = self.engine.predict(PredictionCategory.TASK, "t", confidence=0.5)
        self.engine.verify(p2.id, True)
        pending = self.engine.pending()
        assert p1 in pending and p2 not in pending
        assert self.engine.pending(PredictionCategory.ACTION) == [p1]


class TestAccuracyCalibration:
    """Tests for track record and calibration."""

    def setup_method(self):
        self.engine = PredictionEngine()

    def _seed(self, confidences, outcomes):
        for c, o in zip(confidences, outcomes):
            p = self.engine.predict(PredictionCategory.ACTION, "x", confidence=c)
            self.engine.verify(p.id, o)

    def test_accuracy_empty(self):
        acc = self.engine.accuracy()
        assert acc["verified"] == 0
        assert acc["accuracy"] is None

    def test_accuracy(self):
        self._seed([0.9, 0.9, 0.2, 0.2], [True, True, False, True])
        acc = self.engine.accuracy()
        assert acc["verified"] == 4
        assert acc["true"] == 3
        assert acc["false"] == 1
        assert acc["accuracy"] == 0.75

    def test_accuracy_by_category(self):
        pa = self.engine.predict(PredictionCategory.ACTION, "x", confidence=0.8)
        pt = self.engine.predict(PredictionCategory.TASK, "y", confidence=0.8)
        self.engine.verify(pa.id, True)
        self.engine.verify(pt.id, False)
        acc_a = self.engine.accuracy(PredictionCategory.ACTION)
        assert acc_a["accuracy"] == 1.0
        acc_t = self.engine.accuracy(PredictionCategory.TASK)
        assert acc_t["accuracy"] == 0.0

    def test_calibration_buckets(self):
        # 4 high-confidence predictions all true; 2 low-confidence, half true
        self._seed([0.9, 0.85, 0.95, 0.88], [True, True, True, True])
        self._seed([0.3, 0.3], [True, False])
        report = self.engine.calibration()
        assert len(report) == 5
        high = next(b for b in report if b["bucket"] == "0.8-1.0")
        low = next(b for b in report if b["bucket"] == "0.2-0.4")
        assert high["count"] == 4
        assert high["actual_rate"] == 1.0
        assert high["calibration_error"] == pytest.approx(0.1, abs=0.01)
        assert low["count"] == 2
        assert low["actual_rate"] == 0.5

    def test_calibration_empty_buckets(self):
        report = self.engine.calibration()
        assert all(b["count"] == 0 and b["actual_rate"] is None for b in report)

    def test_stats(self):
        self.engine.predict(PredictionCategory.TASK, "x", confidence=0.5)
        self.engine.predict(PredictionCategory.ACTION, "y", confidence=0.5)
        stats = self.engine.stats()
        assert stats["total"] == 2
        assert stats["by_status"]["pending"] == 2
        assert stats["by_category"]["task"] == 1
        assert stats["by_category"]["action"] == 1


class TestHeuristicPredictors:
    """
    Tests for the scaffolding predictors. These verify the PIPELINE:
    predictor → registered prediction → verifiable outcome. The heuristic
    content itself is intentionally not asserted too precisely since it
    will be replaced with learned models.
    """

    def setup_method(self):
        self.engine = PredictionEngine()

    def test_simple_query_maps_to_fast(self):
        pred = self.engine.predict_task_complexity("what is 2+2")
        assert pred.category == PredictionCategory.TASK
        assert pred.metadata["mode"] == "FAST"

    def test_complex_query_maps_to_deeper_mode(self):
        pred = self.engine.predict_task_complexity(
            "investigate why the distributed system fails and analyze the root cause"
        )
        assert pred.metadata["mode"] in ("DELIBERATE", "DEEP", "EXPLORATORY")
        assert pred.metadata["markers"] >= 2

    def test_ambiguity_raises_uncertainty(self):
        simple = self.engine.predict_task_complexity("sum these numbers")
        ambiguous = self.engine.predict_task_complexity(
            "maybe summarize this possibly unclear report, or something"
        )
        assert ambiguous.uncertainty > simple.uncertainty

    def test_complexity_prediction_is_verifiable(self):
        """End-to-end: predict, then record the outcome."""
        pred = self.engine.predict_task_complexity("research and compare the options")
        assert pred.status == PredictionStatus.PENDING
        self.engine.verify(pred.id, True, note="task indeed needed deep reasoning")
        assert pred.status == PredictionStatus.VERIFIED_TRUE

    def test_action_success_known_type(self):
        pred = self.engine.predict_action_success("read")
        assert pred.metadata["known_type"] is True
        assert pred.confidence >= 0.85

    def test_action_success_unknown_type_less_confident(self):
        known = self.engine.predict_action_success("read")
        unknown = self.engine.predict_action_success("teleport")
        assert unknown.metadata["known_type"] is False
        assert unknown.uncertainty > known.uncertainty

    def test_action_success_uses_history(self):
        no_hist = self.engine.predict_action_success("ui_click")
        with_hist = self.engine.predict_action_success("ui_click", history_success_rate=0.95)
        assert with_hist.confidence > no_hist.confidence

    def test_risk_detects_destructive(self):
        risky = self.engine.predict_action_risk("delete the production database")
        safe = self.engine.predict_action_risk("read the config file")
        assert risky.metadata["destructive"] is True
        assert risky.metadata["risk"] > safe.metadata["risk"]

    def test_missing_evidence_few_sources(self):
        pred = self.engine.predict_missing_evidence("X causes Y", known_source_count=1)
        assert pred.category == PredictionCategory.KNOWLEDGE
        assert pred.confidence >= 0.8

    def test_missing_evidence_many_sources(self):
        few = self.engine.predict_missing_evidence("claim", known_source_count=1)
        many = self.engine.predict_missing_evidence("claim", known_source_count=5)
        assert many.confidence < few.confidence

    def test_scaffolding_honestly_marked(self):
        """The scaffolding markers must stay in the module docstring so
        nobody mistakes heuristics for learned models."""
        import sweep_cognitive.prediction as mod
        doc = mod.__doc__ or ""
        assert "TEMPORARY SCAFFOLDING" in doc

    def test_all_predictions_verify_through_pipeline(self):
        """Every heuristic predictor output flows through the same
        predict/verify/calibrate machinery."""
        preds = [
            self.engine.predict_task_complexity("analyze the logs deeply"),
            self.engine.predict_action_success("execute"),
            self.engine.predict_action_risk("overwrite the file"),
            self.engine.predict_missing_evidence("claim Z", known_source_count=0),
        ]
        for p in preds:
            assert p.status == PredictionStatus.PENDING
            self.engine.verify(p.id, True)
        acc = self.engine.accuracy()
        assert acc["verified"] == 4
        assert acc["accuracy"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
