"""Phase 14 — Feedback / Learning Loop tests."""

import pytest

from cognition.feedback import (ErrorCategory, FeedbackLoop, LearningAction,
                                FeedbackRecord)


def test_memory_gap_error_strengthens_memory():
    loop = FeedbackLoop()
    fired = []
    loop.register_handler(LearningAction.STRENGTHEN_MEMORY, lambda r: fired.append(r.target))
    rec = loop.process("could not recall token format", hints=["memory_gap"],
                       target="token-schema")
    assert rec.category is ErrorCategory.MEMORY_GAP
    assert rec.action is LearningAction.STRENGTHEN_MEMORY
    assert fired == ["token-schema"]


def test_model_bias_error_retrains_model():
    loop = FeedbackLoop()
    fired = []
    loop.register_handler(LearningAction.RETRAIN_MODEL, lambda r: fired.append("retrain"))
    loop.process("overpredicts category A", hints=["model_bias"])
    assert fired == ["retrain"]


def test_rule_gap_adds_rule():
    loop = FeedbackLoop()
    fired = []
    loop.register_handler(LearningAction.ADD_RULE, lambda r: fired.append("rule_added"))
    loop.process("missing rule for X", hints=["rule_gap"])
    assert fired == ["rule_added"]


def test_verification_failure_raises_uncertainty():
    loop = FeedbackLoop()
    rec = loop.process("verification disagreement", hints=["verification_failure"])
    assert rec.action is LearningAction.RAISE_UNCERTAINTY


def test_unclassified_still_raised_uncertainty():
    loop = FeedbackLoop()
    rec = loop.process("bizarre error")
    assert rec.category is ErrorCategory.UNCLASSIFIED
    assert rec.action is LearningAction.RAISE_UNCERTAINTY


def test_explicit_category_overrides_hints():
    loop = FeedbackLoop()
    rec = loop.process("looks like a strategy issue", hints=["model_bias"],
                       category=ErrorCategory.STRATEGY_ERROR)
    assert rec.category is ErrorCategory.STRATEGY_ERROR
    assert rec.action is LearningAction.ADJUST_STRATEGY


def test_action_log_is_inspectable_and_explicit():
    loop = FeedbackLoop()
    loop.process("gap", hints=["memory_gap"])
    log = loop.action_log()
    assert log[0]["category"] == "MEMORY_GAP"
    assert log[0]["action"] == "STRENGTHEN_MEMORY"
    assert loop.count_by_category(ErrorCategory.MEMORY_GAP) == 1


def test_record_roundtrip():
    rec = FeedbackRecord("fb9", ErrorCategory.RULE_GAP, "missing precondition",
                         LearningAction.ADD_RULE, "preconditions")
    d = rec.to_dict()
    assert d["resolved"] is False
    assert d["target"] == "preconditions"