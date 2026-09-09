"""Phase 8 — Competing Hypothesis Reasoning tests."""

import pytest

from cognition.hypotheses import EvidenceUpdate, HypothesisEngine


def _three_hypotheses():
    eng = HypothesisEngine()
    h1 = eng.add_hypothesis("alice stole the data", initial=0.5)
    h2 = eng.add_hypothesis("system glitch exfiltrated it", initial=0.5)
    h3 = eng.add_hypothesis("contractor took it", initial=0.5)
    return eng, h1, h2, h3


def test_multiple_hypotheses_maintained():
    eng, h1, h2, h3 = _three_hypotheses()
    assert len(eng.all()) == 3


def test_evidence_supports_one_rank_changes():
    eng, h1, h2, h3 = _three_hypotheses()
    eng.incorporate(EvidenceUpdate(evidence_id="ev1", content="badge 4481 logged in alice's hands",
                                   supports=[h1.hypothesis_id], opposes=[], weight=3.0))
    eng.incorporate(EvidenceUpdate(evidence_id="ev2", content="HR directory: badge to alice",
                                   supports=[h1.hypothesis_id], opposes=[], weight=2.0))
    ranked = eng.rank()
    assert ranked[0].hypothesis_id == h1.hypothesis_id
    assert ranked[0].confidence > 0.5


def test_evidence_opposes_another():
    eng, h1, h2, h3 = _three_hypotheses()
    eng.incorporate(EvidenceUpdate(evidence_id="ev3", content="glitch was ruled out by sysadmins",
                                   supports=[], opposes=[h2.hypothesis_id], weight=3.0))
    assert eng.get(h2.hypothesis_id).confidence < 0.5


def test_no_conclusion_when_insufficient():
    eng, h1, h2, h3 = _three_hypotheses()
    # only weak, conflicting evidence
    eng.incorporate(EvidenceUpdate(evidence_id="e", content="maybe", supports=[h1.hypothesis_id],
                                   opposes=[h2.hypothesis_id], weight=0.2))
    assert eng.no_conclusion() is True
    assert eng.leading() is None


def test_no_forced_conclusion_retains_uncertainty():
    eng, h1, h2, h3 = _three_hypotheses()
    eng.incorporate(EvidenceUpdate(evidence_id="e1", content="x", supports=[h1.hypothesis_id],
                                   opposes=[h2.hypothesis_id], weight=0.4))
    state = eng.to_dict()
    assert state["no_conclusion"] is True
    assert any(h["uncertainty"] > 0.0 for h in state["hypotheses"])


def test_clear_winner_reached():
    eng, h1, h2, h3 = _three_hypotheses()
    # strong, consistent support for h1
    for i in range(5):
        eng.incorporate(EvidenceUpdate(evidence_id=f"e{i}", content=f"supports {i}",
                                       supports=[h1.hypothesis_id], opposes=[], weight=2.0))
    lead = eng.leading()
    assert lead is not None
    assert lead.hypothesis_id == h1.hypothesis_id
    assert lead.status == "SUPPORTED"


def test_rank_by_confidence_stable():
    eng, h1, h2, h3 = _three_hypotheses()
    eng.incorporate(EvidenceUpdate("e", "c", supports=[h3.hypothesis_id], weight=2))
    eng.incorporate(EvidenceUpdate("f", "c", supports=[h3.hypothesis_id], weight=2))
    r1 = [h.hypothesis_id for h in eng.rank()]
    r2 = [h.hypothesis_id for h in eng.rank()]
    assert r1 == r2  # deterministic


def test_to_dict_roundtrip_shape():
    eng, h1, h2, h3 = _three_hypotheses()
    d = eng.to_dict()
    assert len(d["hypotheses"]) == 3
    assert set(d["hypotheses"][0].keys()) == {"hypothesis_id", "statement", "confidence",
                                              "support_weight", "oppose_weight", "uncertainty", "status"}