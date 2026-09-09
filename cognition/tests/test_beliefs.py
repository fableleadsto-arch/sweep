"""Phase 10 — Evidence-Driven Belief Updating tests."""

import pytest

from cognition.beliefs import BeliefEvidence, BeliefRevisionEngine


def test_initial_belief_set():
    eng = BeliefRevisionEngine()
    eng.set_initial("alice broke the lock", confidence=0.8)
    assert eng.leading_hypothesis() == "alice broke the lock"


def test_conflicting_evidence_switches_leader():
    eng = BeliefRevisionEngine()
    eng.set_initial("alice broke the lock", confidence=0.9, hypothesis_id="hyp-alice")
    eng.register_candidate("glitch caused the lock failure", confidence=0.5, hypothesis_id="hyp-glitch")
    eng.register_candidate("contractor broke it", confidence=0.4, hypothesis_id="hyp-co")
    # Initial evidence: H1 = leading hypothesis
    assert eng.leading_hypothesis() == "alice broke the lock"
    # new contradictory evidence about H1
    eng.consider(BeliefEvidence(content="alibi verified", supports=False, strength=1.0,
                                source="witness", hypothesis_target="hyp-alice"))
    # H1 confidence decreases
    assert eng.current.confidence < 0.9
    # new support for H2
    eng.consider(BeliefEvidence(content="glitch signature found in lock logs", supports=True,
                                strength=1.0, source="log-analysis", hypothesis_target="hyp-glitch"))
    # H2 becomes leading hypothesis
    assert eng.leading_hypothesis() == "glitch caused the lock failure"


def test_original_conclusion_retained_in_history():
    eng = BeliefRevisionEngine()
    eng.set_initial("alice broke the lock", confidence=0.9, hypothesis_id="hyp-alice")
    eng.register_candidate("glitch caused the lock failure", 0.5, "hyp-glitch")
    eng.consider(BeliefEvidence(content="alibi verified", supports=False, strength=1.0,
                                hypothesis_target="hyp-alice"))
    eng.consider(BeliefEvidence(content="glitch signature in lock logs", supports=True, strength=1.0,
                                hypothesis_target="hyp-glitch"))
    assert eng.count_revisions() >= 1
    first = eng.history[0]
    assert first.previous["conclusion"] == "alice broke the lock"
    assert first.new["conclusion"] == "glitch caused the lock failure"
    assert first.reason
    assert first.triggering_evidence  # the evidence that produced the switch


def test_no_revision_when_belief_stays_valid():
    eng = BeliefRevisionEngine()
    eng.set_initial("gate is open", confidence=0.85, hypothesis_id="h1")
    revision = eng.consider(BeliefEvidence(content="sensor confirms open", supports=True,
                                           strength=0.3, hypothesis_target="h1"))
    assert revision is None
    assert eng.leading_hypothesis() == "gate is open"


def test_confidence_decreases_with_contradictory_evidence():
    eng = BeliefRevisionEngine()
    eng.set_initial("H1 is the leading hypothesis", confidence=0.9, hypothesis_id="h1")
    eng.consider(BeliefEvidence(content="new evidence contradicts", supports=False, strength=1.0,
                                hypothesis_target="h1"))
    # confidence decreases
    _, score = max(eng._candidate_scores.items(), key=lambda kv: kv[1])

    with_contradiction = eng.current.confidence
    eng2 = BeliefRevisionEngine()
    eng2.set_initial("H1 is the leading hypothesis", confidence=0.9, hypothesis_id="h1")
    assert eng2.current.confidence > with_contradiction


def test_clear_revision_reason_in_history():
    eng = BeliefRevisionEngine()
    eng.set_initial("A", confidence=0.9, hypothesis_id="a")
    eng.register_candidate("B", 0.5, "b")
    eng.consider(BeliefEvidence(content="A is false", supports=False, strength=1.0,
                                hypothesis_target="a"))
    eng.consider(BeliefEvidence(content="B is true after all", supports=True, strength=1.0,
                                hypothesis_target="b"))
    assert len(eng.history) >= 1
    assert eng.history[-1].reason
    assert "supports" in eng.history[-1].reason.lower()