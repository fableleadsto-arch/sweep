"""Phase 16 — Unified Cognitive Loop integration tests."""

import pytest

from cognition.loop import CognitiveLoop, _default_hypotheses


FULL_PIPELINE = [
    "perception", "observation", "working_memory", "retrieval",
    "hypothesis", "neural_reasoning", "logic_reasoning",
    "evidence_check", "contradiction_check", "verification",
    "uncertainty", "conclusion", "feedback", "long_term_memory",
]


def test_full_pipeline_all_stages_executed():
    loop = CognitiveLoop()
    result = loop.run("The gate is open")
    assert result.stage_names() == FULL_PIPELINE


def test_conclusion_reached_with_strong_evidence():
    loop = CognitiveLoop()
    result = loop.run(
        "gate is open",
        claims=[
            {"statement": "sensor shows open", "subject": "gate",
             "predicate": "status", "object": "open", "source": "sensor-a", "weight": 1.0, "strength": 1.0},
            {"statement": "visual check shows open", "subject": "gate",
             "predicate": "status", "object": "open", "source": "camera-b", "weight": 1.0, "strength": 1.0},
        ],
        hypotheses=["gate is open"],
        hypothesis_initial={"gate is open": 0.5},
    )
    assert result.conclusion == "gate is open"
    assert result.conclusion_confidence >= 0.6
    assert result.verified is True or result.epistemic_state == "VERIFIED"


def test_insufficient_evidence_produces_explicit_uncertainty():
    loop = CognitiveLoop()
    result = loop.run("what caused the lights to flicker?")
    assert result.conclusion.startswith("insufficient evidence")
    assert result.epistemic_state in ("INSUFFICIENT_EVIDENCE", "POSSIBLE", "UNRESOLVED")


def test_contradiction_detected_in_pipeline():
    loop = CognitiveLoop()
    result = loop.run(
        "location of drone",
        claims=[
            {"statement": "drone is at A", "subject": "drone", "predicate": "located-at",
             "object": "A", "source": "gps"},
            {"statement": "drone is at B", "subject": "drone", "predicate": "located-at",
             "object": "B", "source": "radar"},
        ],
        hypotheses=["drone is at A", "drone is at B"],
    )
    assert result.contradiction_count == 1
    assert "contradiction" in result.stage_names()[8]
    # both claims preserved -> store still holds both
    claims = loop.store.by_type("Claim")
    assert len(claims) == 2


def test_registered_verifier_used_in_pipeline():
    loop = CognitiveLoop()
    loop.register_verifier("logic", lambda cid: True)
    loop.register_verifier("evidence", lambda cid: True)
    result = loop.run("check status", claims=[
        {"statement": "gate open", "subject": "gate", "predicate": "status",
         "object": "open", "source": "sensor"}])
    # verification stage ran through UncertaintyEngine -> VERIFIED via evidence
    assert "verification" in result.stage_names()
    assert loop.verification.result.__name__  # runner wired
    assert result.epistemic_state in ("VERIFIED", "PROBABLE", "POSSIBLE")


def test_neural_router_runs_custom_handler():
    calls = []

    def handler(task, ctx):
        calls.append(task)
        return {"understood": task}

    loop = CognitiveLoop(register_router_handler=False)
    loop.register_neural("complex_reasoning", handler)
    result = loop.run("something unusual occurred")
    assert calls
    assert result.stage_names()[5] == "neural_reasoning"
    # no degradation: handler succeeded
    assert result.degraded is False


def test_result_serializes():
    loop = CognitiveLoop()
    result = loop.run("simple check")
    d = result.to_dict()
    assert d["run_id"] == "run001"
    assert len(d["stages"]) == 14
    assert d["input_text"] == "simple check"


def test_memory_persists_conclusion_to_ltm():
    loop = CognitiveLoop()
    before = loop.memory.longterm.count()
    loop.run("gate is open", claims=[
        {"statement": "open", "subject": "gate", "predicate": "status",
         "object": "open", "source": "s"}])
    assert loop.memory.longterm.count() > before


def test_default_hypotheses_deterministic():
    a = _default_hypotheses("X vs Y")
    b = _default_hypotheses("X vs Y")
    assert a == b == ["X", "Y"]