"""Phase 19 — Benchmark reproducibility tests."""

import json

import pytest

from cognition.bench import BenchRunner, BenchReport, run_benchmark, write_report


REQUIRED = ["logical_accuracy", "evidence_attribution", "contradiction_detection",
            "memory_retrieval", "hypothesis_ranking", "belief_revision",
            "uncertainty_calibration", "hallucination_error_rate",
            "router_accuracy", "latency_ms", "cpu_usage", "ram_usage_mb"]


def test_report_contains_all_12_measurements():
    rep = run_benchmark(trials=3)
    assert set(rep.measurements.keys()) == set(REQUIRED)


def test_capability_measurements_are_100_percent():
    # deterministic tasks with known answers must score at ceiling
    rep = run_benchmark(trials=3)
    for name in ("logical_accuracy", "evidence_attribution",
                 "contradiction_detection", "memory_retrieval",
                 "hypothesis_ranking", "belief_revision",
                 "uncertainty_calibration", "router_accuracy"):
        assert rep.measurements[name] == 100.0, name


def test_hallucination_rate_is_zero():
    rep = run_benchmark(trials=3)
    assert rep.measurements["hallucination_error_rate"] == 0.0


def test_report_serializes():
    rep = run_benchmark(trials=3)
    d = rep.to_dict()
    assert set(d["measurements"].keys()) == set(REQUIRED)
    assert d["extra"]["trials"] == 3


def test_write_report(tmp_path):
    rep = BenchReport(measurements={"a": 1.0})
    path = write_report(rep, str(tmp_path / "report.json"))
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["measurements"]["a"] == 1.0


def test_latency_is_positive_float():
    rep = BenchRunner(trials=2)
    lat = rep._latency()
    assert lat > 0
    assert isinstance(lat, float)