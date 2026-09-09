"""Phase 5 — Inspectable Rule-Based Reasoning tests."""

import pytest

from cognition.rules import Rule, RuleEngine, RuleTrace, _counts_greater, default_compact_ruleset


def test_rule_fires_and_records_trace():
    eng = default_compact_ruleset()
    result = eng.run({"pieces_at_goal": 4, "time_remaining": "yes", "pressure": 0.5})
    assert result["verdict"] == "victory"
    traces = eng.trace()
    assert len(traces) == 1
    assert traces[0].rule_id == "victory_if_at_goal"
    assert traces[0].rule_version == "1.0.0"
    assert traces[0].output["verdict"] == "victory"


def test_identify_exact_rule_for_derived_result():
    eng = default_compact_ruleset()
    eng.run({"pieces_at_goal": 4, "time_remaining": "yes", "pressure": 0.5})
    rule = eng.explain("verdict")
    assert rule is not None
    assert rule.rule_id == "victory_if_at_goal"
    assert rule.version == "1.0.0"


def test_nonmatching_rules_do_not_fire():
    eng = default_compact_ruleset()
    result = eng.run({"pieces_at_goal": 0, "time_remaining": "yes"})
    assert "verdict" not in result
    assert eng.trace() == []


def test_rule_versioning_replaceable():
    eng = RuleEngine()
    def c1(f): return True
    def a1(f): return {"x": 1}
    eng.add_rule(Rule.make("r1", "1.0.0", "first", [c1], a1))
    eng.run({})
    assert eng.trace()[0].rule_version == "1.0.0"
    # replace with a new version of the same rule
    def a2(f): return {"x": 2}
    eng.add_rule(Rule.make("r1", "2.0.0", "second", [c1], a2))
    eng.run({})
    assert eng.trace()[0].rule_version == "2.0.0"
    assert eng.trace()[0].output["x"] == 2


def test_rule_version_and_id_traceable_after_run():
    eng = default_compact_ruleset()
    eng.run({"pieces_at_goal": 4, "time_remaining": "yes", "pressure": 2.5})
    # both rules fired: victory + alert
    ids = {(t.rule_id, t.rule_version) for t in eng.trace()}
    assert ("victory_if_at_goal", "1.0.0") in ids
    assert ("alert_on_high_pressure", "2.1.0") in ids
    # trace identifies rule for each derived key
    assert eng.explain("alert").rule_id == "alert_on_high_pressure"
    assert eng.explain("verdict").rule_id == "victory_if_at_goal"


def test_milestone_flow():
    from cognition.rules import default_compact_ruleset
    facts = {"pieces_at_goal": 4, "time_remaining": "yes", "pressure": 0.2}
    eng = default_compact_ruleset()
    # Input facts -> rule matching -> rule execution -> derived result -> reasoning trace
    result = eng.run(facts)
    assert result["verdict"] == "victory"
    trace = eng.trace()[0].to_dict()
    assert trace["rule_id"] == "victory_if_at_goal"
    assert trace["output"]["verdict"] == "victory"
    # the exact rule responsible is identified afterward
    responsible = eng.explain("verdict")
    assert responsible.rule_id == "victory_if_at_goal"