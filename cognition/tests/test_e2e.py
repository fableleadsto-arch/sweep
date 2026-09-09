"""Phase 18 — End-to-end scenario tests.

Full loop runs end-to-end: input -> reasoning -> persistent memory -> explain.
"""

import tempfile

import pytest

from cognition.explain import Explainer
from cognition.loop import CognitiveLoop
from cognition.store import CognitiveStore


def test_e2e_strong_evidence_verified_conclusion_explained():
    with tempfile.TemporaryDirectory() as tmp:
        store = CognitiveStore(path=tmp)
        loop = CognitiveLoop(store=store)
        result = loop.run(
            "the meeting is at noon",
            claims=[
                {"statement": "calendar says noon", "subject": "meeting",
                 "predicate": "at", "object": "noon", "source": "calendar"},
                {"statement": "invite confirms noon", "subject": "meeting",
                 "predicate": "at", "object": "noon", "source": "invite"},
            ],
            hypotheses=["the meeting is at noon"],
            hypothesis_initial={"the meeting is at noon": 0.6},
        )
        # stored, concluded, explainable, memory retained
        assert result.conclusion == "the meeting is at noon"
        explanation = Explainer(loop).explain_conclusion(result.conclusion_id)
        assert "the meeting is at noon" in explanation.render()
        assert store.count() > 0


def test_e2e_contradiction_loop_never_deletes_a_claim():
    with tempfile.TemporaryDirectory() as tmp:
        store = CognitiveStore(path=tmp)
        loop = CognitiveLoop(store=store)
        result = loop.run(
            "drone location",
            claims=[
                {"statement": "drone in hangar", "subject": "drone",
                 "predicate": "located-at", "object": "hangar", "source": "gps"},
                {"statement": "drone on runway", "subject": "drone",
                 "predicate": "located-at", "object": "runway", "source": "radar"},
            ],
            hypotheses=["drone in hangar", "drone on runway"],
        )
        assert result.contradiction_count == 1
        claims = store.by_type("Claim")
        assert len(claims) == 2           # both preserved
        assert result.conclusion.startswith("insufficient evidence")


def test_e2e_deterministic_across_runs():
    loop = CognitiveLoop()
    a = loop.run("evaluate X", claims=[
        {"statement": "X works", "subject": "X", "predicate": "status",
         "object": "works", "source": "s"}], hypotheses=["X works"],
        hypothesis_initial={"X works": 0.6})
    b = loop.run("evaluate Y", claims=[
        {"statement": "Y works", "subject": "Y", "predicate": "status",
         "object": "works", "source": "s"}], hypotheses=["Y works"],
        hypothesis_initial={"Y works": 0.6})
    # same evidence pattern -> same pipeline behavior
    assert a.stage_names() == b.stage_names()
    assert a.epistemic_state == b.epistemic_state


def test_e2e_two_conclusions_two_runs_two_explanations():
    loop = CognitiveLoop()
    r1 = loop.run("alpha", claims=[{"statement": "a ok", "subject": "a",
                                    "predicate": "status", "object": "ok",
                                    "source": "s"}], hypotheses=["alpha"],
                  hypothesis_initial={"alpha": 0.6})
    r2 = loop.run("beta", claims=[{"statement": "b ok", "subject": "b",
                                   "predicate": "status", "object": "ok",
                                   "source": "s"}], hypotheses=["beta"],
                  hypothesis_initial={"beta": 0.6})
    e = Explainer(loop)
    exp1 = e.explain_conclusion(r1.conclusion_id)
    exp2 = e.explain_conclusion(r2.conclusion_id)
    assert exp1.source_run == "run001"
    assert exp2.source_run == "run002"
    assert exp1.steps != exp2.steps or exp1.conclusion != exp2.conclusion