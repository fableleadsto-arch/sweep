"""Phase 9 — Explicit Conflict Representation tests."""

import pytest

from cognition.contradiction import ContradictionEngine, RuleContradictionDetector


def test_location_contradiction_detected():
    eng = ContradictionEngine()
    c = eng.compare("c1", "X at Location A", "x", "located-at", "Location A",
                    "c2", "X at Location B", "x", "located-at", "Location B",
                    time_eq=True, location_eq=True)
    assert c is not None
    assert "cannot be at both" in c.reason
    assert eng.count() == 1


def test_both_claims_preserved():
    eng = ContradictionEngine()
    c = eng.compare("c1", "Entity X at Location A", "x", "located-at", "Location A",
                    "c2", "Entity X at Location B", "x", "located-at", "Location B",
                    time_eq=True, location_eq=True)
    assert c.claim_a_text == "Entity X at Location A"
    assert c.claim_b_text == "Entity X at Location B"
    # both claims remain available for investigation
    assert eng.for_claim("c1")
    assert eng.for_claim("c2")


def test_different_times_no_contradiction():
    eng = ContradictionEngine()
    c = eng.compare("c1", "X at A at 10:00", "x", "located-at", "A",
                    "c2", "X at B at 11:00", "x", "located-at", "B",
                    time_eq=False, location_eq=True)
    assert c is None  # different time -> not a contradiction


def test_status_boolean_contradiction():
    eng = ContradictionEngine()
    c = eng.compare("c1", "gate is open", "gate", "is-open", "open",
                    "c2", "gate is closed", "gate", "is-closed", "closed",
                    time_eq=True, location_eq=True)
    assert c is not None
    assert "conflicting" in c.reason


def test_consistent_claims_no_contradiction():
    eng = ContradictionEngine()
    c = eng.compare("c1", "X at A", "x", "located-at", "A",
                    "c2", "X at A too", "x", "located-at", "A",
                    time_eq=True, location_eq=True)
    assert c is None


def test_contradiction_explicit_object_shape():
    eng = ContradictionEngine()
    c = eng.compare("c1", "X at Location A", "x", "located-at", "Location A",
                    "c2", "X at Location B", "x", "located-at", "Location B")
    d = c.to_dict()
    assert d["reason"]
    assert d["status"] == "OPEN"
    assert c.id
    assert eng.to_dict()["contradictions"][0]["reason"] == d["reason"]


def test_rule_detector_direct():
    d = RuleContradictionDetector()
    reason = d.detect("x", "located-at", "A", "located-at", "B", time_eq=True, location_eq=True)
    assert reason is not None