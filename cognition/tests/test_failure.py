"""Phase 18 — Failure-path tests.

The system must fail gracefully: explicit errors on bad input, never
synthetic truth, always an auditable outcome.
"""

import pytest

from cognition.loop import CognitiveLoop
from cognition.bridge import NeuralBridge, NeuralOutput
from cognition.verification import VerificationEngine, VerificationStatus
from cognition.explain import Explainer


def test_loop_accepts_empty_and_never_invents_truth():
    loop = CognitiveLoop()
    result = loop.run("")
    assert result.conclusion
    assert result.epistemic_state in ("INSUFFICIENT_EVIDENCE", "POSSIBLE", "UNRESOLVED")
    # it must not claim VERIFIED on no input
    assert result.verified is False


def test_loop_three_runs_each_explainable_after_failure_free_flow():
    loop = CognitiveLoop()
    loop.run("one")
    loop.run("two")
    loop.run("three")
    # hundreds of runs remain explainable (no explosion)
    e = Explainer(loop)
    for r in loop.results:
        assert len(e.explain_conclusion(r.conclusion_id).steps) == 14


def test_neural_bridge_rejects_garbage_output():
    bridge = NeuralBridge(min_confidence=0.7)
    res = bridge.validate(NeuralOutput("", {"statement": "x"}, 0.9))
    assert res.accepted is False
    assert "empty" in res.reason.lower()


def test_neural_bridge_rejects_low_confidence():
    bridge = NeuralBridge(min_confidence=0.7)
    res = bridge.validate(NeuralOutput("gate open", {"statement": "gate open"}, 0.2))
    assert res.accepted is False
    assert "confidence" in res.reason.lower()


def test_verifier_exception_degrades_to_unresolved_not_failure():
    eng = VerificationEngine()
    eng.register_check("neural", lambda cid: True)
    def explode(cid):
        raise RuntimeError("llm call failed")
    eng.register_check("evidence", explode)
    result = eng.run("c-7")
    assert result.status is VerificationStatus.UNRESOLVED
    assert "evidence" in result.unrun


def test_explain_unknown_raises_clear_error():
    loop = CognitiveLoop()
    loop.run("x")
    with pytest.raises(KeyError):
        Explainer(loop).explain_conclusion("nope-123")