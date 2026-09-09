"""
Phase 18 LIVE DEMO — the full spec-22/24 learning loop with real
executive output (no hand-injected fixtures):

1. An executive controller executes 80 tasks (two content families:
   safe "collect/arrange..." work that succeeds, destructive
   "wipe/clobber..." work whose tools fail).
2. Experience records accumulate in a persistent JSONL store.
3. The BEHAVIORAL pipeline retrains — and is honestly REFUSED, because
   for these goals the hand-built features are identical between the
   classes (same word count, zero markers). This is the Phase 17
   ceiling, demonstrated live.
4. The TEXT pipeline retrains on the same records — the content
   weighting separates the families, beats ALL baselines on the
   holdout, and is promoted through its gate.
5. The promoted estimator drives AdaptiveRouter and PredictionEngine
   (blended path), with the learned reason visible in the decision.

Run with:  pytest sweep_cognitive/tests/test_phase18_demo.py -s -q
"""

import pytest

from sweep_cognitive.learning import (
    ExperienceStore,
    features_from_query,
    LearningPipeline,
    TextLearningPipeline,
    BlendedEstimator,
    TextEstimator,
)
from sweep_cognitive.executive import (
    ExecutiveController,
    ToolRegistry,
    TaskStatus,
)
from sweep_cognitive.routing import AdaptiveRouter
from sweep_cognitive.prediction import PredictionEngine


GOOD_VERBS = ["collect", "arrange", "summarize", "polish"]
BAD_VERBS = ["wipe", "clobber", "corrupt", "detach"]
NOUNS = ["ledger", "bucket", "queue", "cache", "table",
         "buffer", "socket", "stream", "folder", "catalog"]


def _make_tools(succeed: bool) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register("retrieve_context", lambda p: {"found": True})
    if succeed:
        tools.register("perform_task", lambda p: {"done": True})
        tools.register("verify_outcome", lambda p: {"verified": True})
    else:
        def boom(p):
            raise RuntimeError("database locked")
        tools.register("perform_task", boom)
    return tools


def test_live_accumulation_demo(tmp_path, capsys):
    import random

    store = ExperienceStore(str(tmp_path / "experience.jsonl"))
    print("\n=== PHASE 18 LIVE ACCUMULATION DEMO ===")

    # ---- 1. The executive completes a real task stream -----------------
    rng = random.Random(100)
    goals = []
    for k in range(40):
        for verbs, succeed in ((GOOD_VERBS, True), (BAD_VERBS, False)):
            verb = verbs[k % len(verbs)]
            n1 = rng.choice(NOUNS)
            n2 = rng.choice(NOUNS)
            goals.append((f"{verb} the {n1} and sync the {n2}", succeed))

    completed = handoffs = 0
    for goal, succeed in goals:
        ctrl = ExecutiveController(
            tools=_make_tools(succeed), experience_store=store,
        )
        task = ctrl.plan_task(goal)
        ctrl.run(task)
        if task.status == TaskStatus.COMPLETED:
            completed += 1
        elif task.status == TaskStatus.NEEDS_USER:
            handoffs += 1
        else:
            pytest.fail(f"unexpected terminal status {task.status} for {goal!r}")

    stats = store.stats()
    print(f"\n[1] executive ran {len(goals)} tasks: "
          f"{completed} completed, {handoffs} honest user handoffs")
    print(f"    store: {stats['records']} records at {stats['path']} "
          f"({stats['successes']} success / {stats['failures']} failure)")
    assert stats["records"] == 80
    assert stats["successes"] == 40 and stats["failures"] == 40

    # ---- 2. Behavioral pipeline: the honest refusal ---------------------
    behav_report = LearningPipeline(store, seed=5).train()
    print(f"\n[2] behavioral pipeline: accepted={behav_report.accepted}")
    print(f"    {behav_report.rejected_reason or '(promoted)'}")
    # These goals have identical hand-built features across classes
    # (same length, zero markers) — the gate MUST refuse. This is the
    # Phase 17 feature ceiling, demonstrated on live data.
    assert not behav_report.accepted

    # ---- 3. Text pipeline: learned content weighting ---------------------
    text_pipe = TextLearningPipeline(store, seed=11)
    text_report = text_pipe.train()
    print(f"\n[3] text pipeline: accepted={text_report.accepted}")
    print(f"    test_accuracy={text_report.test_accuracy} "
          f"(brier={text_report.test_brier})")
    print(f"    baselines on SAME holdout: "
          f"behavioral={text_report.behavioral_baseline_accuracy} "
          f"lexical={text_report.lexical_baseline_accuracy} "
          f"majority={text_report.majority_class_accuracy}")
    if not text_report.accepted:
        print(f"    rejected: {text_report.rejected_reason}")
    assert text_report.accepted, text_report.to_dict()
    assert text_report.beats_behavioral
    assert text_report.beats_lexical
    assert text_report.beats_majority

    # Promotion through the gate
    model_path = str(tmp_path / "text_model.json")
    assert text_pipe.promote(text_report, model_path)
    promoted = TextLearningPipeline.load_promoted(model_path)
    assert promoted is not None
    p_safe = promoted.predict_proba_text("collect the ledger and sync the queue")
    p_destructive = promoted.predict_proba_text("wipe the ledger and sync the queue")
    print(f"    promoted model: p(success | 'collect...')={p_safe:.3f} "
          f"p(success | 'wipe...')={p_destructive:.3f}")
    assert p_safe > p_destructive

    # ---- 4. The learned model drives routing + prediction ----------------
    # Behavioral gate refused -> the blend is text-only. Honest fallback.
    estimator = BlendedEstimator(text=promoted)
    router = AdaptiveRouter(learned_estimator=estimator)
    safe_decision = router.route("collect the ledger and sync the queue")
    destr_decision = router.route("wipe the ledger and sync the queue")
    print(f"\n[4] routing with promoted estimator:")
    print(f"    safe task  -> {safe_decision.mode.value}")
    for r in safe_decision.reasons:
        if "learned" in r:
            print(f"      reason: {r}")
    print(f"    destructive task -> {destr_decision.mode.value}")
    for r in destr_decision.reasons:
        if "learned" in r:
            print(f"      reason: {r}")
    assert any("learned estimator" in r for r in safe_decision.reasons)
    assert any("learned estimator" in r for r in destr_decision.reasons)
    # The learned difficulty must differ between content families
    assert (1.0 - p_safe) != (1.0 - p_destructive)

    engine = PredictionEngine(learned_estimator=estimator)
    pred = engine.predict_task_complexity("collect the ledger")
    assert "learned(blended)" in pred.metadata.get("learned", "")
    print(f"    prediction metadata: {pred.metadata.get('learned')}")

    print("\n=== DEMO COMPLETE: gates refused what deserved refusal, "
          "promoted what earned promotion ===\n")
