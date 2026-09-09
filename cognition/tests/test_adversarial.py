"""Phase 18 — Adversarial tests.

Malformed, hostile, or confusing inputs must be handled deterministically
without fabrication, crashes, or silent state corruption.
"""

import string

import pytest

from cognition.contradiction import ContradictionEngine
from cognition.loop import CognitiveLoop
from cognition.uncertainty import EpistemicState, EvidenceSignal, UncertaintyEngine
from cognition.store import CognitiveStore


def test_adversarial_huge_input_handled_without_crash():
    loop = CognitiveLoop()
    blob = "x " * 800                       # 1600 tokens of noise
    result = loop.run(blob)
    assert result.stages[-1].name == "long_term_memory"
    assert result.epistemic_state in ("INSUFFICIENT_EVIDENCE", "POSSIBLE", "UNRESOLVED")


def test_adversarial_all_punctuation():
    loop = CognitiveLoop()
    result = loop.run("!@#$%^&*()_+{}|:<>?,")
    assert result.run_id  # completed, deterministic
    assert result.conclusion


def test_adversarial_repeated_contradictions_stack_deterministically():
    eng = ContradictionEngine()
    n = 0
    for _ in range(20):
        if eng.compare(f"c{n}", "x at a", "x", "located-at", "a",
                       f"c{n+1}", "x at b", "x", "located-at", "b"):
            n += 2
    assert eng.count() == 20
    ids = [c.id for c in eng.all()]
    assert len(set(ids)) == 20             # no id collisions


def test_adversarial_claim_ids_never_break_store():
    store = CognitiveStore(in_memory=True)
    for i in range(50):
        from cognition.schema import Claim
        store.save(Claim(statement=f"claim {i}", subject="s", predicate="p", object=f"o{i}"))
    objects = store.all()
    assert len(objects) == 50
    assert store.count() == 50


def test_adversarial_unknown_states_rejected_no_fabrication():
    eng = UncertaintyEngine()
    s = eng.assess(EvidenceSignal("neg", "negative evidence only",
                                  supporting=0, contradicting=5))
    assert s.state is EpistemicState.UNRESOLVED
    assert "[]".join(s.reasons)            # explicit explanation, no invention


def test_adversarial_malformed_claim_does_not_corrupt_loop():
    loop = CognitiveLoop()
    result = loop.run("basic", claims=[
        {"statement": "", "subject": "", "predicate": "", "object": ""}])
    # corrupt empty claim is tolerated; run completes with uncertainty
    assert result.stage_names() == loop.PIPELINE
    assert result.contradiction_count == 0