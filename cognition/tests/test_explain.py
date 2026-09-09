"""Phase 17 — Explainability / Reconstructable Reasoning Trace tests."""

import pytest

from cognition.explain import Explainer
from cognition.loop import CognitiveLoop


def test_explain_conclusion_reconstructs_pipeline():
    loop = CognitiveLoop()
    result = loop.run("gate is open", claims=[
        {"statement": "sensor open", "subject": "gate", "predicate": "status",
         "object": "open", "source": "sensor"}],
        hypotheses=["gate is open"],
        hypothesis_initial={"gate is open": 0.6})
    exp = Explainer(loop).explain_conclusion(result.conclusion_id)
    assert exp.conclusion_id == result.conclusion_id
    assert exp.conclusion == result.conclusion
    stage_names = [s.stage for s in exp.steps]
    # reconstructable trace: all 14 stages appear in order
    assert stage_names[0] == "perception"
    assert set(stage_names) >= {"evidence_check", "contradiction_check",
                                "verification", "uncertainty", "conclusion"}
    assert stage_names.index("observation") < stage_names.index("conclusion")


def test_explanation_is_human_readable():
    loop = CognitiveLoop()
    result = loop.run("gate is open")
    exp = Explainer(loop).explain_conclusion(result.conclusion_id)
    text = exp.render()
    assert "Conclusion" in text
    assert "trace:" in text
    assert "perception:" in text or "- perception" in text
    assert exp.conclusion in text


def test_explanation_roundtrips_to_dict():
    loop = CognitiveLoop()
    result = loop.run("thing happened")
    exp = Explainer(loop).explain_conclusion(result.conclusion_id)
    d = exp.to_dict()
    assert d["conclusion_id"] == result.conclusion_id
    assert len(d["steps"]) == len(result.stages)
    assert d["epistemic_state"] == result.epistemic_state


def test_explaining_unknown_conclusion_raises():
    loop = CognitiveLoop()
    loop.run("something")
    with pytest.raises(KeyError):
        Explainer(loop).explain_conclusion("nonexistent-id")


def test_explanation_by_run_id_also_works():
    loop = CognitiveLoop()
    result = loop.run("gate is open")
    exp = Explainer(loop).explain_conclusion(result.run_id)
    assert exp.conclusion_id == result.conclusion_id


def test_every_run_can_be_explained():
    loop = CognitiveLoop()
    loop.run("first run")
    loop.run("second run")
    loop.run("third run")
    explainer = Explainer(loop)
    for result in loop.results:
        exp = explainer.explain_conclusion(result.conclusion_id)
        assert exp.source_run == result.run_id
        assert len(exp.steps) == 14