"""
SWEEP Cognitive Benchmark Suite — Phase 20.

Spec section 25: do not measure only final answer accuracy. This suite
measures the capabilities the new architecture actually implements:

1. UNDERSTANDING — semantic entity/relationship extraction quality
2. EVIDENCE QUALITY — quality decomposition behavior
3. SOURCE INDEPENDENCE — lineage collapse (the "ten copies ≠ ten
   confirmations" requirement)
4. CONFIDENCE CALIBRATION — prediction engine calibration report
5. ADAPTIVE ROUTING — mode distribution + compute-tiering behavior
6. PLANNING/REPLANNING — executive controller end-to-end success,
   retry, and honest-failure behavior
7. VERIFICATION INTEGRITY — fake-success must be impossible

Principles (spec section 65):
- Reproducible: fixed fixtures, deterministic modules, seeded where any
  randomness could appear (none currently does).
- Honest: benchmarks report what IS, including scaffold limitations.
  A module is only credited for behavior its tests demonstrate.
- Failure-visible: benchmark failures are reported, never hidden.

Usage:
    python -m sweep_cognitive.benchmarks            # run + print report
    python -m sweep_cognitive.benchmarks --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .semantic import SemanticUnderstandingEngine
from .representation import TextRepresentation
from .evidence2 import SourceRegistry, EvidenceEngine2
from .prediction import PredictionEngine, PredictionCategory
from .routing import AdaptiveRouter, HardwareProfile, HardwareTier, ComputationMode
from .executive import (
    ExecutiveController,
    TaskStatus,
    ToolRegistry,
    ErrorClass,
)
from .world import WorldModel, RelationshipType


# ======================================================================
# Harness
# ======================================================================

@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    score: float                 # 0..1 where meaningful, else 1.0/0.0
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
            **({"error": self.error} if self.error else {}),
        }


def run_benchmark(name: str, fn: Callable[[], tuple[bool, float, dict]]) -> BenchmarkResult:
    t0 = time.perf_counter()
    try:
        passed, score, details = fn()
    except Exception as exc:  # noqa: BLE001 — a benchmark crash IS a failure
        return BenchmarkResult(
            name=name, passed=False, score=0.0,
            details={}, error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
    return BenchmarkResult(
        name=name, passed=passed, score=score, details=details,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


# ======================================================================
# Fixtures (fixed, reproducible)
# ======================================================================

WIRE_STORY = (
    "Researchers at a major university announced a breakthrough in "
    "solid-state battery technology on Monday, claiming doubled energy "
    "density and tripled charge cycles in independent lab tests."
)

UNDERSTANDING_CASES = [
    # (text, min_entities_expected)
    ("Alice met Bob in Paris to discuss the merger.", 2),
    ("Bob presented the quarterly results to Alice and Carol.", 2),
    ("The engineers reviewed the outage timeline in Berlin.", 1),
]


# ======================================================================
# Benchmarks
# ======================================================================

def bench_understanding() -> tuple[bool, float, dict]:
    """Semantic extraction quality on fixed cases."""
    engine = SemanticUnderstandingEngine()
    hits = 0
    per_case = []
    for text, min_entities in UNDERSTANDING_CASES:
        out = engine.understand(TextRepresentation(content=text))
        n = out["entity_count"]
        ok = n >= min_entities
        hits += ok
        per_case.append({"text": text[:40], "entities": n, "needed": min_entities, "ok": ok})
    score = hits / len(UNDERSTANDING_CASES)
    return score == 1.0, score, {"cases": per_case}


def bench_source_independence() -> tuple[bool, float, dict]:
    """THE core requirement: copied sources must not stack."""
    # Case A: 5 declared-syndicated copies → must collapse to 1 lineage
    reg_a = SourceRegistry()
    reg_a.register_source("wire", reliability=0.9)
    for i in range(4):
        reg_a.register_source(f"outlet_{i}", reliability=0.6)
        reg_a.add_declared_dependence("wire", f"outlet_{i}")
    ids_a = [reg_a.add_item("wire", WIRE_STORY, stance=0.9).item_id]
    for i in range(4):
        ids_a.append(reg_a.add_item(f"outlet_{i}", WIRE_STORY, stance=0.9).item_id)

    # Case B: 5 independent sources quoting the same press release at
    # the same time → must stay 5 lineages (no over-merging)
    reg_b = SourceRegistry()
    ids_b = []
    t0 = time.time()
    for i in range(5):
        reg_b.register_source(f"src_{i}", reliability=0.6)
        ids_b.append(reg_b.add_item(f"src_{i}", WIRE_STORY, stance=0.9, timestamp=t0).item_id)

    # Case C: genuine copying via observed near-duplicate + time gap
    reg_c = SourceRegistry()
    reg_c.register_source("origin")
    reg_c.register_source("copier")
    i1 = reg_c.add_item("origin", WIRE_STORY, stance=0.9, timestamp=t0)
    i2 = reg_c.add_item("copier", WIRE_STORY.replace("Monday", "Tuesday"),
                        stance=0.9, timestamp=t0 + 7200)
    reg_c.observe_dependence_among([i1.item_id, i2.item_id])

    engine_a, engine_b = EvidenceEngine2(reg_a), EvidenceEngine2(reg_b)
    r_a = engine_a.consensus(ids_a, claim="A")
    r_b = EvidenceEngine2(reg_b).consensus(ids_b, claim="B")

    collapse_ok = r_a["effective_sources"] == 1 and r_a["verdict"] == "SINGLE_LINEAGE_ONLY"
    no_overmerge_ok = r_b["effective_sources"] == 5
    observed_ok = reg_c.effective_independent_sources(["origin", "copier"]) == 1
    # Copied case must weigh less than the independent case
    weight_ok = r_a["support_weight"] < r_b["support_weight"]

    score = sum([collapse_ok, no_overmerge_ok, observed_ok, weight_ok]) / 4
    return score == 1.0, score, {
        "syndicated_collapse": collapse_ok,
        "independent_not_merged": no_overmerge_ok,
        "observed_copy_detected": observed_ok,
        "copied_weighs_less": weight_ok,
        "case_a_effective_sources": r_a["effective_sources"],
        "case_b_effective_sources": r_b["effective_sources"],
    }


def bench_evidence_quality() -> tuple[bool, float, dict]:
    """Quality decomposition: reliability and recency must matter."""
    reg = SourceRegistry()
    reg.register_source("good", reliability=0.95)
    reg.register_source("bad", reliability=0.3)
    now = time.time()
    i_good = reg.add_item("good", "confirms the finding with data", stance=0.8, timestamp=now - 60)
    i_bad = reg.add_item("bad", "confirms the finding with data", stance=0.8, timestamp=now - 60)
    i_old = reg.add_item("bad", "old confirmation", stance=0.8, timestamp=now - 60 * 86400)
    engine = EvidenceEngine2(reg)
    q_good = engine.score_item(i_good.item_id, now=now)
    q_bad = engine.score_item(i_bad.item_id, now=now)
    q_old = engine.score_item(i_old.item_id, now=now)
    reliability_ok = q_good.overall > q_bad.overall
    recency_ok = q_old.recency_factor < 0.01 and q_good.recency_factor > 0.99
    score = (reliability_ok + recency_ok) / 2
    return score == 1.0, score, {
        "reliability_matters": reliability_ok,
        "recency_decays": recency_ok,
        "q_good": q_good.to_dict(),
        "q_bad": q_bad.to_dict(),
    }


def bench_calibration() -> tuple[bool, float, dict]:
    """Calibration machinery: bucketed report must separate well-
    calibrated from miscalibrated confidence on a fixed seed set."""
    engine = PredictionEngine()
    # Simulated track record: high-confidence mostly true, low-confidence coin flips
    fixed = [(0.9, True), (0.88, True), (0.92, True), (0.85, True),   # 4/4 true
             (0.3, True), (0.3, False),                                # 1/2 true
             (0.6, True), (0.6, False)]                                # 1/2 true
    for conf, outcome in fixed:
        p = engine.predict(PredictionCategory.ACTION, "fixture", confidence=conf)
        engine.verify(p.id, outcome)
    report = engine.calibration()
    high = next(b for b in report if b["bucket"] == "0.8-1.0")
    # Calibration error in the high bucket must be small (0.9 stated vs 1.0 actual)
    high_ok = high["calibration_error"] is not None and high["calibration_error"] <= 0.15
    acc = engine.accuracy()
    accuracy_ok = acc["accuracy"] is not None and abs(acc["accuracy"] - 6/8) < 0.01
    score = (high_ok + accuracy_ok) / 2
    return score == 1.0, score, {
        "high_bucket_error": high["calibration_error"],
        "accuracy": acc["accuracy"],
        "calibration_machinery_works": high_ok,
        "accuracy_correct": accuracy_ok,
    }


def bench_adaptive_routing() -> tuple[bool, float, dict]:
    """Trivial work gets little compute; heavy work gets more; weak
    hardware shrinks budgets without zeroing them."""
    strong = AdaptiveRouter(hardware=HardwareProfile(
        tier=HardwareTier.STANDARD, cpu_count=8, total_ram_gb=16.0,
        has_gpu=False, offline_mode=False))
    d_trivial = strong.route("hi")
    d_heavy = strong.route(
        "investigate why the system fails: analyze compare prove design deeply",
        context={"uncertainty": 0.9})
    d_destructive = strong.route("delete the production database now")

    trivial_cheap = d_trivial.mode in (ComputationMode.FAST, ComputationMode.ROUTINE)
    heavy_deep = d_heavy.mode in (ComputationMode.DEEP, ComputationMode.EXPLORATORY)
    destructive_careful = d_destructive.mode in (
        ComputationMode.DELIBERATE, ComputationMode.DEEP, ComputationMode.EXPLORATORY)

    weak = AdaptiveRouter(hardware=HardwareProfile(
        tier=HardwareTier.CONSTRAINED, cpu_count=1, total_ram_gb=2.0,
        has_gpu=False, offline_mode=False))
    d_weak = weak.route(
        "investigate why the system fails: analyze compare prove design deeply",
        context={"uncertainty": 0.9})
    degraded_not_zero = (
        d_weak.budget.max_retrieval_items >= 3
        and d_weak.budget.max_retrieval_items <= d_heavy.budget.max_retrieval_items
    )
    explainable = all(len(d.reasons) > 0 for d in (d_trivial, d_heavy, d_destructive, d_weak))

    checks = [trivial_cheap, heavy_deep, destructive_careful, degraded_not_zero, explainable]
    score = sum(checks) / len(checks)
    return score == 1.0, score, {
        "trivial_cheap": trivial_cheap,
        "heavy_deep": heavy_deep,
        "destructive_careful": destructive_careful,
        "degraded_not_zero": degraded_not_zero,
        "explainable": explainable,
        "modes": [d.mode.value for d in (d_trivial, d_heavy, d_destructive, d_weak)],
    }


def _make_tools(perform_result: dict) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register("retrieve_context", lambda p: {"found": True, "items": []})
    tools.register("perform_task", lambda p: perform_result)
    tools.register("verify_outcome", lambda p: {"verified": True})
    return tools


def bench_planning_execution() -> tuple[bool, float, dict]:
    """End-to-end executive controller: success, retry, honest failure."""
    # A: happy path completes
    ctrl_a = ExecutiveController(tools=_make_tools({"done": True, "output": "ok"}))
    task_a = ctrl_a.plan_task("complete the analysis")
    ctrl_a.run(task_a)
    happy_ok = task_a.status == TaskStatus.COMPLETED

    # B: transient failure retried then completes
    calls = {"n": 0}
    tools_b = _make_tools({"done": True})
    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient timeout")
        return {"done": True}
    tools_b.register("perform_task", flaky)
    ctrl_b = ExecutiveController(tools=tools_b, max_attempts=2)
    task_b = ctrl_b.plan_task("retry the analysis")
    ctrl_b.run(task_b)
    retry_ok = task_b.status == TaskStatus.COMPLETED and calls["n"] == 2

    # C: unmet verification → honest handoff, NOT completion
    ctrl_c = ExecutiveController(tools=_make_tools({"done": False}), max_attempts=1)
    task_c = ctrl_c.plan_task("meet the criteria")
    ctrl_c.run(task_c)
    honest_fail_ok = task_c.status == TaskStatus.NEEDS_USER

    score = (happy_ok + retry_ok + honest_fail_ok) / 3
    return score == 1.0, score, {
        "happy_path": happy_ok,
        "retry_recovers": retry_ok,
        "verification_failure_handoff_not_fake_success": honest_fail_ok,
    }


def bench_verification_integrity() -> tuple[bool, float, dict]:
    """Spec section 65: try to break it. Fake success must be impossible
    through the three most obvious attack paths."""
    # Path 1: no tools at all → must not complete
    ctrl_1 = ExecutiveController(tools=ToolRegistry(), max_attempts=1)
    t1 = ctrl_1.plan_task("do everything with no tools")
    ctrl_1.run(t1)
    no_tools_ok = t1.status != TaskStatus.COMPLETED

    # Path 2: tool returns wrong values → must not complete
    ctrl_2 = ExecutiveController(tools=_make_tools({"done": False}), max_attempts=1)
    t2 = ctrl_2.plan_task("produce wrong values")
    ctrl_2.run(t2)
    wrong_values_ok = t2.status != TaskStatus.COMPLETED

    # Path 3: world model without the fact → logic must say not_provable
    world = WorldModel()
    a = world.add_entity("a")
    c = world.add_entity("c")
    from .reasoning import ReasoningEngine, ReasoningRequest, ReasoningModality
    from cognition.logic import LogicEngine
    engine = ReasoningEngine(logic_engine=LogicEngine(), world_model=world)
    result = engine.reason(ReasoningRequest(
        query="prove a part of c",
        context={"claim": ("part_of", a.id, c.id)},
    ))
    logic_sub = [r for r in result.metadata["individual_results"]
                 if r["modality"] == "logic_deductive"][0]
    not_provable_ok = logic_sub["conclusion"] == "not_provable"

    score = (no_tools_ok + wrong_values_ok + not_provable_ok) / 3
    return score == 1.0, score, {
        "no_tools_never_completes": no_tools_ok,
        "wrong_values_never_completes": wrong_values_ok,
        "unprovable_claim_reported_honestly": not_provable_ok,
    }


# ======================================================================
# PHASE 21 — LEARNING BENCHMARKS
# ======================================================================

def _mk_rec(goal: str, success: bool, i: int) -> "ExperienceRecord":
    from .learning import ExperienceRecord, features_from_query
    return ExperienceRecord(
        task_id=f"t{i}", goal=goal,
        query_features=features_from_query(goal),
        actions=[], outcome="completed" if success else "failed",
        success=success, verification_passed=success,
        errors=[], replans=0, duration_ms=10.0, confidence=1.0,
    )


def bench_learning_generalization() -> tuple[bool, float, dict]:
    """8. LEARNED GENERALIZATION — the trained estimator must beat the
    lexical baseline on held-out data (spec 22/24: keep only if better)
    and produce a usable model. Fails honestly when data is degenerate."""
    import tempfile
    from .learning import (
        ExperienceStore, LearningPipeline, LogisticEstimator,
        features_from_query,
    )

    hard_success = ("investigate why the distributed cache invalidates slowly "
                    "and compare eviction policies across nodes, then verify")
    hard_fail = ("investigate why the production payment pipeline deadlocks "
                 "under concurrent deployment, then analyze and prove root cause")
    easy_success = "list the files"
    easy_fail = "delete the production database irreversible"

    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(str(os.path.join(tmp, "exp.jsonl")))
        i = 0
        for _ in range(40):
            for goal, ok in ((hard_success, True), (easy_success, True),
                             (hard_fail, False), (easy_fail, False)):
                store.append(_mk_rec(goal, ok, i))
                i += 1
        report = LearningPipeline(store, seed=7).train()
        if not report.accepted:
            return False, 0.0, {
                "beats_baseline": False,
                "report": report.to_dict(),
                "note": "estimator did not beat lexical baseline on holdout",
            }
        model = LogisticEstimator(report.weights, report.bias)
        p_easy = model.predict_proba(features_from_query(easy_success))
        p_hard = model.predict_proba(features_from_query(hard_fail))
        separated = p_easy > p_hard
        score = report.test_accuracy
        return separated, score, {
            "test_accuracy": report.test_accuracy,
            "baseline_accuracy": report.baseline_accuracy,
            "test_brier": report.test_brier,
            "easy_p_success": round(p_easy, 3),
            "hard_p_success": round(p_hard, 3),
        }


def bench_experience_persistence() -> tuple[bool, float, dict]:
    """9. EXPERIENCE PERSISTENCE — outcome records survive process
    boundaries (new store instance), corruption is non-fatal, and the
    executive emits records on terminal states (spec 22)."""
    import tempfile
    from .learning import ExperienceStore
    from .executive import (
        ExecutiveController, ToolRegistry, TaskStatus, StepStatus,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = str(os.path.join(tmp, "exp.jsonl"))

        # Executive emits experience records on success AND on handoff
        tools = ToolRegistry()
        tools.register("retrieve_context", lambda p: {"found": True})
        tools.register("perform_task", lambda p: {"done": True})
        tools.register("verify_outcome", lambda p: {"verified": True})
        store = ExperienceStore(path)
        ctrl = ExecutiveController(tools=tools, experience_store=store)
        t = ctrl.plan_task("persist an experience record")
        ctrl.run(t)
        ok_task = t.status == TaskStatus.COMPLETED

        tools2 = ToolRegistry()  # no tools → needs_user path also records
        ctrl2 = ExecutiveController(tools=tools2, experience_store=store)
        t2 = ctrl2.plan_task("fail honestly with no tools")
        ctrl2.run(t2)
        handoff_recorded = t2.status == TaskStatus.NEEDS_USER

        loaded = ExperienceStore(path).load_all()  # fresh instance
        persisted = len(loaded) == 2
        outcomes = {r.outcome for r in loaded}
        both_paths = {"completed", "needs_user"}.issubset(outcomes)
        records_carry_payload = all(
            r.goal and isinstance(r.query_features, dict) and r.actions
            for r in loaded
        )
        score = (ok_task + handoff_recorded + persisted + both_paths
                 + records_carry_payload) / 5
        return score == 1.0, score, {
            "success_task_completed": ok_task,
            "handoff_also_recorded": handoff_recorded,
            "records_persist_across_instances": persisted,
            "both_terminal_paths_in_data": both_paths,
            "records_carry_full_payload": records_carry_payload,
        }


def bench_skill_transfer_gate() -> tuple[bool, float, dict]:
    """10. SKILL ACQUISITION — one success is NOT a skill; acquisition
    requires transfer evidence across distinct contexts (spec 22) and
    decaying skills are retired."""
    from .learning import SkillRegistry, SkillStatus

    reg = SkillRegistry()
    # Single success → candidate, never acquired
    reg.register_observation("once", "ctx", True)
    single_ok = reg.get("once").status != SkillStatus.ACQUIRED

    # Repeated success in ONE context is not transfer
    for _ in range(6):
        reg.register_observation("single-ctx", "only", True)
    single_ctx_ok = reg.get("single-ctx").status != SkillStatus.ACQUIRED

    # Distinct contexts + rate floor → acquired
    for ctx in ("a", "b", "c", "d"):
        reg.register_observation("transfer", ctx, True)
    acquired_ok = reg.get("transfer").status == SkillStatus.ACQUIRED

    # Decay retires a formerly acquired skill: 4/10 = 0.4 does NOT
    # retire (strictly-below semantics); 4/11 ≈ 0.364 < 0.4 does.
    for _ in range(7):
        reg.register_observation("transfer", "a", False)
    retired_ok = reg.get("transfer").status == SkillStatus.RETIRED

    checks = (single_ok, single_ctx_ok, acquired_ok, retired_ok)
    score = sum(checks) / len(checks)
    return score == 1.0, score, {
        "single_success_not_skill": single_ok,
        "repetition_not_transfer": single_ctx_ok,
        "distinct_contexts_acquire": acquired_ok,
        "decay_retires": retired_ok,
    }


def bench_improvement_gate() -> tuple[bool, float, dict]:
    """11. SELF-IMPROVEMENT GATE — candidates are kept only when they
    beat the baseline by the required margin; the decision is applied
    and recorded (spec 24)."""
    from .learning import SelfImprovementLoop, ImprovementCandidate

    loop = SelfImprovementLoop(min_improvement=0.02)

    def base():
        return 0.70, {"method": "lexical baseline"}

    def better():
        return 0.85, {"method": "learned"}

    def marginal():
        return 0.705, {"method": "within noise"}

    applied: list[bool] = []
    o1 = loop.evaluate_candidate(
        ImprovementCandidate(name="learned-model", evaluate=better),
        baseline=base, apply=applied.append,
    )
    o2 = loop.evaluate_candidate(
        ImprovementCandidate(name="marginal-model", evaluate=marginal),
        baseline=base, apply=applied.append,
    )
    kept_only_better = o1.kept and not o2.kept
    applied_correctly = applied == [True, False]
    recorded = len(loop.regression_summary()) == 2

    checks = (kept_only_better, applied_correctly, recorded)
    score = sum(checks) / len(checks)
    return score == 1.0, score, {
        "kept_only_when_better": kept_only_better,
        "apply_matches_decision": applied_correctly,
        "outcomes_recorded": recorded,
    }


def bench_text_learning_gate() -> tuple[bool, float, dict]:
    """12. LEARNED FEATURES (TEXT) — a text estimator over hashed n-gram
    content must (a) beat ALL baselines — behavioral, lexical, majority —
    on the same holdout when the data carries content signal, and
    (b) be REFUSED when it does not (spec 65: no hidden shortcuts)."""
    import random as _random
    import tempfile
    from .learning import (
        ExperienceStore, TextLearningPipeline, TextEstimator,
        ExperienceRecord, features_from_query,
    )

    good_verbs = ["collect", "arrange", "summarize", "polish"]
    bad_verbs = ["wipe", "clobber", "corrupt", "detach"]
    nouns = ["ledger", "bucket", "queue", "cache", "table",
             "buffer", "socket", "stream", "folder", "catalog"]

    def _records(n_pairs: int) -> list:
        out, i = [], 0
        r = _random.Random(100)
        for k in range(n_pairs):
            for verbs, ok in ((good_verbs, True), (bad_verbs, False)):
                verb = verbs[k % len(verbs)]
                n1 = r.choice(nouns)
                n2 = r.choice(nouns)
                goal = f"{verb} the {n1} and sync the {n2}"
                out.append(ExperienceRecord(
                    task_id=f"b{i}", goal=goal,
                    query_features=features_from_query(goal),
                    actions=[], outcome="completed" if ok else "failed",
                    success=ok, verification_passed=ok,
                    errors=[], replans=0, duration_ms=1.0, confidence=1.0,
                ))
                i += 1
        return out

    with tempfile.TemporaryDirectory() as tmp:
        # (a) signal-bearing data: text must beat everything
        store = ExperienceStore(str(os.path.join(tmp, "a.jsonl")))
        for rec in _records(40):
            store.append(rec)
        report = TextLearningPipeline(store, seed=11).train()
        accepted_with_signal = report.accepted
        beats_everything = (
            report.beats_behavioral and report.beats_lexical
            and report.beats_majority
        )

        # (b) degenerate data (class independent of text): must refuse
        store2 = ExperienceStore(str(os.path.join(tmp, "b.jsonl")))
        i = 0
        for k in range(30):
            for parity, ok in ((0, True), (1, False)):
                goal = f"handle item number {i}"
                store2.append(ExperienceRecord(
                    task_id=f"n{i}", goal=goal,
                    query_features=features_from_query(goal),
                    actions=[], outcome="completed" if ok else "failed",
                    success=ok, verification_passed=ok,
                    errors=[], replans=0, duration_ms=1.0, confidence=1.0,
                ))
                i += 1
        noise_report = TextLearningPipeline(store2, seed=3).train()
        refuses_noise = not noise_report.accepted

        # Promotion consistency: accepted model persists, rejected never does
        model_path = str(os.path.join(tmp, "text_model.json"))
        promoted = TextLearningPipeline(store, seed=11).promote(report, model_path)
        loaded = TextLearningPipeline.load_promoted(model_path) if promoted else None
        separation_ok = (
            loaded is not None
            and loaded.predict_proba_text("collect the ledger and sync the queue")
            > loaded.predict_proba_text("wipe the ledger and sync the queue")
        )

        checks = (accepted_with_signal, beats_everything, refuses_noise,
                  separation_ok)
        score = sum(checks) / len(checks)
        return score == 1.0, score, {
            "text_accepted_with_signal": accepted_with_signal,
            "beats_behavioral_lexical_majority": beats_everything,
            "refuses_noise_data": refuses_noise,
            "promoted_model_separates_content": separation_ok,
            "test_accuracy": report.test_accuracy,
            "behavioral_baseline": report.behavioral_baseline_accuracy,
            "lexical_baseline": report.lexical_baseline_accuracy,
        }


# ======================================================================
# Runner
# ======================================================================

ALL_BENCHMARKS: list[tuple[str, Callable[[], tuple[bool, float, dict]]]] = [
    ("understanding", bench_understanding),
    ("source_independence", bench_source_independence),
    ("evidence_quality", bench_evidence_quality),
    ("calibration", bench_calibration),
    ("adaptive_routing", bench_adaptive_routing),
    ("planning_execution", bench_planning_execution),
    ("verification_integrity", bench_verification_integrity),
    ("learning_generalization", bench_learning_generalization),
    ("experience_persistence", bench_experience_persistence),
    ("skill_transfer_gate", bench_skill_transfer_gate),
    ("improvement_gate", bench_improvement_gate),
    ("text_learning_gate", bench_text_learning_gate),
]


def run_all() -> dict[str, Any]:
    results = [run_benchmark(name, fn) for name, fn in ALL_BENCHMARKS]
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    return {
        "suite": "sweep_cognitive_phase20",
        "timestamp": time.time(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "average_score": round(avg_score, 4),
        "results": [r.to_dict() for r in results],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        "SWEEP COGNITIVE BENCHMARK SUITE — Phase 20",
        "=" * 70,
        f"Passed: {report['passed']}/{report['total']}   "
        f"Average score: {report['average_score']:.2f}",
        "-" * 70,
    ]
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"[{status}] {r['name']:<28} score={r['score']:.2f}  ({r['duration_ms']:.1f} ms)")
        if r.get("error"):
            lines.append(f"      ERROR: {r['error']}")
        elif not r["passed"]:
            for k, v in r["details"].items():
                lines.append(f"      {k}: {v}")
    lines.append("=" * 70)
    lines.append(
        "Note: benchmarks measure implemented behavior only. Lexical/"
        "template scaffolding modules are credited only for the "
        "machinery around them (see module docstrings)."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWEEP cognitive benchmarks")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    report = run_all()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    raise SystemExit(0 if report["failed"] == 0 else 1)
