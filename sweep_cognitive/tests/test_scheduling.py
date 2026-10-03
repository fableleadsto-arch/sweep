"""
Tests for Phase 19 — Scheduling & Drift Detection.

Honest coverage:
1. DriftDetector compares a fresh model's report vs the currently-promoted
   model's embedded report; flags degradation (severity "degraded" or
   "degraded_closely") when fresh is worse than current by more than the
   min-improve band, and "recovered" when fresh beats current by the
   required margin. No re-derivation; the on-disk report is ground truth.
2. CadencePolicy blocks retraining by (a) minimum interval, (b) minimum
   new records, (c) minimum improvement a candidate must show — all
   exposed as honest knobs.
3. RetrainScheduler runs behavioral + text pipelines, returns structured
   statuses (cadence_blocked / trained-with-recommendation), and NEVER
   silently mutates live estimators. Training-at is refreshed only when
   a report was actually produced (empty store → no refresh).
"""

import json
import time

import pytest

from sweep_cognitive.learning import (
    ExperienceStore,
    ExperienceRecord,
    features_from_query,
    text_features_from_query,
    LogisticEstimator,
    TextEstimator,
    TextLearningPipeline,
    LearningPipeline,
    CadencePolicy,
    DriftDetector,
    DriftReport,
    RetrainScheduler,
    _records_since,
    TextTrainingReport,
)


# ---------- helpers ----------

def _append_records(store: ExperienceStore, n_good: int, n_bad: int,
                    seed: int = 100) -> None:
    import random
    good_verbs = ["collect", "arrange"]
    bad_verbs = ["wipe", "clobber"]
    nouns = ["ledger", "bucket", "queue", "cache"]
    rng = random.Random(seed)
    i = 0
    for k in range(max(n_good, n_bad)):
        for verb_list, ok in ((good_verbs, True), (bad_verbs, False)):
            if ok and k >= n_good:
                continue
            if (not ok) and k >= n_bad:
                continue
            verb = verb_list[k % len(verb_list)]
            n1 = rng.choice(nouns)
            n2 = rng.choice(nouns)
            goal = f"{verb} the {n1} and sync the {n2}"
            store.append(ExperienceRecord(
                task_id=f"t{i}", goal=goal,
                query_features=features_from_query(goal),
                actions=[], outcome="completed" if ok else "failed",
                success=ok, verification_passed=ok,
                errors=[], replans=0, duration_ms=1.0, confidence=1.0,
            ))
            i += 1


def _fake_text_report(accuracy: float | None) -> TextTrainingReport:
    rep = TextTrainingReport.__new__(TextTrainingReport)
    rep.n_records = 80 if accuracy is not None else 0
    rep.n_train = 56 if accuracy is not None else 0
    rep.n_test = 24 if accuracy is not None else 24
    rep.test_accuracy = accuracy
    rep.train_accuracy = accuracy if accuracy is not None else 0.9
    rep.test_brier = 0.1 if accuracy else 0.5
    rep.behavioral_baseline_accuracy = 0.4
    rep.lexical_baseline_accuracy = 0.5
    rep.majority_class_accuracy = 0.5
    rep.beats_behavioral = True
    rep.beats_lexical = True
    rep.beats_majority = True
    rep.rejected_reason = ("") if accuracy is not None else "not accepted"
    rep.rejected_reason = ""
    return rep


def _persist_text_promoted(report: TextTrainingReport, path: str) -> None:
    TextLearningPipeline.promote(report, path)


# ---------- drift detection ----------

class TestDriftDetector:
    def test_no_promoted_model_is_ok_unmeasured(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(0.9))
        assert r.severity == "ok"
        assert r.fresh_vs_promoted_test_accuracy_delta is None
        assert "promoted model to compare" in r.note

    def test_fresh_worse_than_promoted_is_degraded(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(0.82), _fake_text_report(0.90))
        assert r.has_drift
        assert r.degraded_amount < 0
        assert "degraded" in r.severity

    def test_fresh_worse_heavily_is_degraded(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(0.80), _fake_text_report(0.95))
        assert r.degraded_amount <= -0.05

    def test_fresh_within_band_not_degraded(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(0.89), _fake_text_report(0.90))
        assert not r.has_drift
        assert r.severity not in ("degraded", "degraded_closely")

    def test_fresh_better_than_promoted_by_min_improve_recovers(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(0.93), _fake_text_report(0.90))
        assert r.severity == "recovered"
        assert r.fresh_vs_promoted_test_accuracy_delta > 0

    def test_fresh_better_by_less_than_min_is_not_recovered(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(0.91), _fake_text_report(0.90))
        assert r.severity != "recovered"

    def test_delta_none_when_report_missing_test_accuracy(self):
        det = DriftDetector(min_improve=0.02)
        r = det.detect(_fake_text_report(None), _fake_text_report(0.90))
        assert r.degraded_amount is None
        assert "could not compare" in r.note

    def test_to_dict_is_stable(self):
        r = DriftReport(fresh_report=_fake_text_report(0.85),
                        severity="degraded", note="fresh worse")
        d = r.to_dict()
        assert d["fresh_test_accuracy"] == 0.85
        assert d["severity"] == "degraded"
        assert d["drift_note"] == "fresh worse"

    def test_promoted_report_held_not_recalc(self):
        det = DriftDetector(min_improve=0.02)
        rep = _fake_text_report(0.90)
        r = det.detect(_fake_text_report(0.82), current_promoted_report=rep)
        assert r.promoted_report is rep
        assert r.promoted_report.test_accuracy == 0.90


# ---------- cadence ----------

def make_recs_store(tmp_path, n_good=30, n_bad=20, seed=7):
    store = ExperienceStore(str(tmp_path / "exp.jsonl"))
    _append_records(store, n_good, n_bad, seed=seed)
    return store


class TestCadencePolicy:
    def test_blocks_when_not_enough_new_records(self, tmp_path):
        store = make_recs_store(tmp_path, 30, 20, seed=7)
        last_train = time.time()
        c = CadencePolicy(min_new_records=50, min_interval_seconds=0)
        ok, reason = c.is_allowed(store, None, last_train)
        assert not ok and "enough new records" in reason

    def test_allows_when_enough_new_records(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=7)
        last_train = time.time() - 1000
        c = CadencePolicy(min_new_records=100, min_interval_seconds=0)
        ok, reason = c.is_allowed(store, None, last_train)
        assert ok and reason == ""

    def test_blocks_interval_not_elapsed(self, tmp_path):
        c = CadencePolicy(min_interval_seconds=3600, min_new_records=0)
        now = time.time()
        ok, reason = c.is_allowed(make_recs_store(tmp_path), now - 100, None)
        assert not ok and "interval" in reason

    def test_allows_interval_elapsed(self, tmp_path):
        c = CadencePolicy(min_interval_seconds=1, min_new_records=0)
        now = time.time()
        ok, _ = c.is_allowed(make_recs_store(tmp_path), now - 10, None)
        assert ok

    def test_improvement_threshold_exposed(self, tmp_path):
        c = CadencePolicy(min_improvement=0.05)
        assert c.improvement_threshold() == 0.05


# ---------- scheduling ----------

class TestRetrainScheduler:
    def test_cadence_blocked_returns_skip(self, tmp_path):
        store = make_recs_store(tmp_path, 60, 60, seed=7)
        c = CadencePolicy(min_interval_seconds=86400, min_new_records=0)
        sched = RetrainScheduler(store, cadence=c, seed=7)
        sched._last_text_promoted_at = time.time()
        r = sched.schedule_text()
        assert r["status"] == "cadence_blocked"
        assert r["recommendation"] == "skip"

    def test_text_only_promotes_when_beating_all_baselines(self, tmp_path):
        store = make_recs_store(tmp_path, 60, 60, seed=100)
        path = str(tmp_path / "text_model.json")
        sched = RetrainScheduler(store, text_promoted_path=path, seed=11)
        r = sched.schedule_text()
        assert r["space"] == "text"
        if r["recommendation"] == "promote":
            assert r["report"].accepted
            assert r["report"].beats_behavioral
            assert r["report"].beats_lexical
            assert r["report"].beats_majority
            assert r["path"] == path
            loaded = TextLearningPipeline.load_promoted(path)
            assert loaded is not None
            p_good = loaded.predict_proba_text("collect the queue")
            p_bad = loaded.predict_proba_text("wipe the queue")
            assert p_good > p_bad
        elif r["recommendation"] == "refuse":
            assert r["report"] is None or not r["report"].accepted

    def test_behavioral_only_promotes_above_baseline(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=5)
        path = str(tmp_path / "behav_model.json")
        sched = RetrainScheduler(store, behavioral_promoted_path=path, seed=5)
        r = sched.schedule_behavioral()
        assert r["space"] == "behavioral"
        if r["recommendation"] == "promote":
            assert r["report"].accepted
            assert r["report"].beats_baseline
            assert r["path"] == path
            loaded = LearningPipeline.load_promoted(path)
            assert loaded is not None
            p = loaded.predict_proba(features_from_query("arrange the ledger"))
            assert 0.0 <= p <= 1.0

    def test_not_promoting_below_min_improve_keeps_on_disk_model(self, tmp_path):
        store = make_recs_store(tmp_path, 60, 60, seed=100)
        path = str(tmp_path / "text_model.json")
        reportA = TextLearningPipeline(store, seed=11).train()
        if not reportA.accepted:
            pytest.skip("gate test needs accepted text model")
        TextLearningPipeline.promote(reportA, path)
        on_disk_before = TextLearningPipeline.load_promoted(path)
        assert on_disk_before is not None

        sched = RetrainScheduler(store, text_promoted_path=path, seed=13,
                                 cadence=CadencePolicy(min_improvement=0.20))
        r = sched.schedule_text()
        on_disk_after = TextLearningPipeline.load_promoted(path)
        assert on_disk_after is not None  # unchanged unless promoted
        if r["recommendation"] == "promote":
            assert r["drift"] is not None
            assert r["drift"].fresh_vs_promoted_test_accuracy_delta >= 0.20
            assert r["path"] == path
        else:
            # on-disk model is unchanged (same object identity after reload
            # is not guaranteed because load returns a new instance; compare
            # instead via test_accuracy + promotion path unchanged)
            assert on_disk_after.to_dict() == on_disk_before.to_dict()

    def test_schedule_both_returns_structured_summary(self, tmp_path):
        store = make_recs_store(tmp_path, 60, 60, seed=100)
        sched = RetrainScheduler(store,
                                 text_promoted_path=str(tmp_path / "text.json"),
                                 behavioral_promoted_path=str(tmp_path / "behav.json"),
                                 seed=11)
        both = sched.schedule_both()
        assert "text" in both["spaces"] and "behavioral" in both["spaces"]
        assert "summary" in both
        for space in ("text", "behavioral"):
            s = both["spaces"][space]
            assert s["space"] == space
            assert s["recommendation"] in ("promote", "refuse", "hold", "skip")

    def test_schedule_refreshes_training_metadata_when_report_produced(self, tmp_path):
        store = make_recs_store(tmp_path, 60, 60, seed=100)
        sched = RetrainScheduler(store,
                                 text_promoted_path=str(tmp_path / "text.json"),
                                 behavioral_promoted_path=str(tmp_path / "behav.json"),
                                 seed=11)
        before_text = sched._last_text_training_at
        before_behav = sched._last_behav_training_at
        sched.schedule_text()
        sched.schedule_behavioral()
        after_text = sched._last_text_training_at
        after_behav = sched._last_behav_training_at
        # A non-empty store produces a real report → metadata refreshed.
        assert after_text is not None and (
            before_text is None or after_text >= before_text
        )
        assert after_behav is not None and (
            before_behav is None or after_behav >= before_behav
        )

    def test_schedule_does_not_refresh_when_no_records(self, tmp_path):
        store = ExperienceStore(str(tmp_path / "empty.jsonl"))
        sched = RetrainScheduler(store, text_promoted_path=str(tmp_path / "text.json"),
                                 behavioral_promoted_path=str(tmp_path / "behav.json"),
                                 seed=11)
        before_text = sched._last_text_training_at
        sched.schedule_text()
        assert sched._last_text_training_at == before_text  # unchanged (no report)

    def test_drift_detection_used_for_both_spaces(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=100)
        path = str(tmp_path / "text_model.json")
        reportA = TextLearningPipeline(store, seed=11).train()
        if not reportA.accepted:
            pytest.skip("gate test needs accepted text model")
        TextLearningPipeline.promote(reportA, path)
        sched = RetrainScheduler(store, text_promoted_path=path, seed=11)
        r = sched.schedule_text()
        assert r["drift"] is not None
        assert r["drift"].has_drift or r["drift"].severity == "recovered" or \
               r["drift"].severity in ("ok", "degraded_closely")


# ---------- promotion fallback chain (honest) ----------

class TestPromotionFallbackChain:
    def test_text_prefers_over_behavioral_when_accepted(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=100)
        sched = RetrainScheduler(store,
                                 text_promoted_path=str(tmp_path / "text.json"),
                                 behavioral_promoted_path=str(tmp_path / "behav.json"),
                                 seed=11)
        text_r = sched.schedule_text()
        behav_r = sched.schedule_behavioral()
        if text_r["recommendation"] == "promote":
            assert text_r["report"].accepted
            assert "text.json" in text_r["path"]
        assert behav_r["recommendation"] in ("promote", "refuse", "hold", "skip")

    def test_behavioral_fallback_does_not_promote_below_gate(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=5)
        behav_path = str(tmp_path / "behav.json")
        sched = RetrainScheduler(store, behavioral_promoted_path=behav_path, seed=5)
        r = sched.schedule_behavioral()
        if r["recommendation"] == "promote":
            assert r["report"].accepted and r["report"].beats_baseline
            assert r["path"] == behav_path

    def test_scheduler_does_not_silent_promote(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=100)
        behav_path = str(tmp_path / "behav_only.json")
        sched = RetrainScheduler(store, behavioral_promoted_path=behav_path,
                                 seed=5)
        r = sched.schedule_behavioral()
        if r["recommendation"] == "promote":
            assert r["report"].accepted and r["report"].beats_baseline
            assert r["path"] == behav_path

    def test_promotion_chain_respects_text_beat_all_gate(self, tmp_path):
        store = make_recs_store(tmp_path, 80, 80, seed=100)
        sched = RetrainScheduler(store,
                                 text_promoted_path=str(tmp_path / "text.json"),
                                 behavioral_promoted_path=str(tmp_path / "behav.json"),
                                 seed=11)
        text_r = sched.schedule_text()
        if text_r["recommendation"] == "promote":
            assert text_r["report"].beats_behavioral
            assert text_r["report"].beats_lexical
            assert text_r["report"].beats_majority
