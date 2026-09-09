"""Phase 11 — Explicit Epistemic State tests."""

import pytest

from cognition.uncertainty import (ClaimEpistemicStatus, EpistemicState,
                                   EvidenceSignal, UncertaintyEngine)


def test_verified_when_multiple_independent_sources():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c1", "floor is wet", supporting=3, contradicting=0,
                                  source_diversity=3))
    assert s.state == EpistemicState.VERIFIED
    assert s.confidence > 0.9


def test_probable_single_source():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c2", "floor probably wet", supporting=1, source_diversity=1))
    assert s.state == EpistemicState.PROBABLE


def test_possible_consistent_but_unverified():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c3", "door may be open", supporting=0, contradicting=0, neutral=2))
    assert s.state == EpistemicState.POSSIBLE


def test_contested_with_mixed_evidence():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c4", "gate is open", supporting=1, contradicting=2,
                                  source_diversity=3))
    assert s.state == EpistemicState.CONTESTED


def test_unresolved_all_against():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c5", "thing happened", supporting=0, contradicting=2))
    assert s.state == EpistemicState.UNRESOLVED


def test_insufficient_evidence_empty():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c6", "mystery claim", supporting=0, contradicting=0, neutral=0))
    assert s.state == EpistemicState.INSUFFICIENT_EVIDENCE


def test_neutral_only_is_possible_not_fabricated_certainty():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c7", "maybe", supporting=0, contradicting=0, neutral=4))
    assert s.state in (EpistemicState.POSSIBLE, EpistemicState.INSUFFICIENT_EVIDENCE)
    assert s.state is not EpistemicState.VERIFIED  # no invention of certainty


def test_all_six_states_are_instantiated():
    # milestone proof: the system produces an explicit uncertainty state
    # instead of fabricating certainty on ambiguous input
    eng = UncertaintyEngine()
    eng.assess(EvidenceSignal("v", "verified", 4, 0, 3))
    eng.assess(EvidenceSignal("pr", "probable", 1, 0, 1))
    eng.assess(EvidenceSignal("po", "possible", 0, 0, 0, 2))
    eng.assess(EvidenceSignal("co", "contested", 2, 2, 2))
    eng.assess(EvidenceSignal("un", "unresolved", 0, 2))
    eng.assess(EvidenceSignal("io", "insufficient", 0, 0, 0))
    states = {s.state for s in eng.all()}
    assert states == set(EpistemicState)


def test_status_roundtrip_dict():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("c8", "roundtrip", supporting=1, contradicting=1))
    d = s.to_dict()
    assert d["state"] == "CONTESTED"
    assert "contradicting" in " ".join(d["reasons"])
    assert d["confidence"] == ClaimEpistemicStatus(**d).confidence
    assert d["supporting_evidence"] == 1
    assert d["contradicting_evidence"] == 1