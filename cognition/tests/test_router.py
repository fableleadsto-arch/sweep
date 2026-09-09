"""Phase 7 — Task-Aware Cognitive Resource Selection tests."""

import pytest

from cognition.router import Router, RouterDecision


def _cal(task, ctx): return 42
def _calc(task, ctx): return "calc"
def _ocr(task, ctx): return "text-in-image"
def _vision(task, ctx): return "detected-objects"
def _logic(task, ctx): return "derived"
def _reason(task, ctx): return "reasoned"
def _boom(task, ctx): raise RuntimeError("model failed")


@pytest.fixture
def router():
    r = Router()
    r.register("arithmetic", _cal)
    r.register("calculator", _calc)
    r.register("ocr", _ocr)
    r.register("vision", _vision)
    r.register("logic", _logic)
    r.register("complex_reasoning", _reason)
    return r


def test_arithmetic_routes_to_calculator():
    r = Router()
    d = r.route("what is 3 + 4*2?")
    assert d.selected in ("arithmetic", "calculator")
    assert r.last_decision() is d


def test_ocr_task_routes_to_ocr():
    r = Router()
    d = r.route("run OCR on this image")
    assert d.selected == "ocr"


def test_simple_logic_routes_to_logic_or_complex_reasoning():
    r = Router()
    d = r.route("A is before B and B is before C")
    assert d.selected.startswith(("logic", "simple_logic", "complex_reasoning"))


def test_execute_milestone_mapping(router):
    # arithmetic -> calculator
    sel, res = router.execute("compute 2+2")
    assert sel in ("calculator", "arithmetic")
    # OCR -> OCR
    sel2, res2 = router.execute("run OCR on the photo")
    assert sel2 == "ocr"
    # vision -> vision
    sel3, res3 = router.execute("what is in this picture")
    assert sel3 == "vision"
    # simple logic -> logic engine
    sel4, res4 = router.execute("A before B, B before C, what about A and C?")
    assert sel4 in ("logic", "simple_logic", "complex_reasoning")
    # complex reasoning -> reasoning model
    sel5, res5 = router.execute("explain the economic implications of the trade deal")
    assert sel5 == "complex_reasoning"


def test_every_routing_decision_logged(router):
    import pytest
    for _ in range(5):
        router.execute("compute 1+1")
    assert len(router.decisions("compute 1+1")) == 5


def test_failed_model_triggers_fallback():
    r = Router()
    r.register("arithmetic", _boom)      # fails
    r.register("calculator", _calc)      # fallback succeeds
    sel, res = r.execute("compute 2+2")
    assert sel == "calculator"
    assert res == "calc"
    d = r.last_decision()
    assert "calculator" in d.fallback_chain
    assert "fallback" in d.reason


def test_all_routes_fail_raises():
    r = Router()
    r.register("arithmetic", _boom)
    r.register("calculator", _boom)
    with pytest.raises(RuntimeError):
        r.execute("compute 2+2")


def test_stats_reports_fallbacks():
    r = Router()
    r.register("arithmetic", _boom)
    r.register("calculator", _calc)
    r.execute("compute 1+1")
    stats = r.stats()
    ent = stats.get("arithmetic") or stats.get("calculator") or stats.get("compute")
    assert ent and (ent["routes"] >= 1)