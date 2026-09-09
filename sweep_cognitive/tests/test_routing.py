"""
Tests for adaptive computation routing (Phase 13).

Verifies:
1. Five computation modes exist with monotone budgets
2. Mode decision is explainable (reasons recorded) and deterministic
3. Hard overrides: time pressure → FAST; compute cap → DELIBERATE
4. Exploratory wins over deep when ambiguity/novelty is high
5. Hardware detection via stdlib with forced overrides
6. Graceful degradation: budgets shrink on weak tiers, never to zero
7. Routing integrates with executive Step params non-destructively
8. Scaffolding honesty
"""

import pytest

from sweep_cognitive.routing import (
    ComputationMode,
    ModeBudget,
    MODE_BUDGETS,
    HardwareTier,
    HardwareProfile,
    detect_hardware,
    ExecutionProfile,
    TIER_PROFILES,
    RoutingSignals,
    RoutingDecision,
    extract_signals,
    AdaptiveRouter,
)
from sweep_cognitive.executive import Step


class TestModesAndBudgets:
    def test_five_modes_exist(self):
        assert {m.value for m in ComputationMode} == {
            "fast", "routine", "deliberate", "deep", "exploratory",
        }

    def test_budgets_monotone(self):
        order = [ComputationMode.FAST, ComputationMode.ROUTINE,
                 ComputationMode.DELIBERATE, ComputationMode.DEEP]
        retrievals = [MODE_BUDGETS[m].max_retrieval_items for m in order]
        assert retrievals == sorted(retrievals)
        hypotheses = [MODE_BUDGETS[m].max_hypotheses for m in order]
        assert hypotheses == sorted(hypotheses)

    def test_fast_has_no_hypotheses(self):
        assert MODE_BUDGETS[ComputationMode.FAST].max_hypotheses == 0
        assert MODE_BUDGETS[ComputationMode.FAST].allow_simulation is False

    def test_deep_allows_simulation(self):
        assert MODE_BUDGETS[ComputationMode.DEEP].allow_simulation is True
        assert MODE_BUDGETS[ComputationMode.DEEP].verification_strictness == 0.9


class TestSignalExtraction:
    def test_simple_query_low_complexity(self):
        s = extract_signals("what is 2+2")
        assert s.complexity < 0.3
        assert s.ambiguity < 0.1

    def test_complex_query_high_complexity(self):
        s = extract_signals("investigate why the distributed system fails and analyze the root cause thoroughly")
        assert s.complexity > 0.4

    def test_ambiguity_markers(self):
        s = extract_signals("maybe it is possibly unclear or something")
        assert s.ambiguity >= 0.5

    def test_consequence_markers(self):
        s = extract_signals("delete the production database")
        assert s.consequence >= 0.35

    def test_context_signals_pass_through(self):
        s = extract_signals("anything", context={
            "time_pressure": 0.9, "evidence_conflict": 0.7,
            "uncertainty": 0.5, "compute_available": 0.2,
        })
        assert s.time_pressure == 0.9
        assert s.evidence_conflict == 0.7
        assert s.uncertainty == 0.5
        assert s.compute_available == 0.2

    def test_signals_clamped(self):
        s = extract_signals("x", context={"time_pressure": 5, "compute_available": -3})
        assert s.time_pressure == 1.0
        assert s.compute_available == 0.0


class TestModeDecision:
    def setup_method(self):
        self.router = AdaptiveRouter(
            hardware=HardwareProfile(
                tier=HardwareTier.STANDARD, cpu_count=8,
                total_ram_gb=16.0, has_gpu=False, offline_mode=False,
            )
        )

    def _route_signals(self, signals: RoutingSignals) -> RoutingDecision:
        """Route with pre-built signals via the context passthrough."""
        ctx = {
            "uncertainty": signals.uncertainty,
            "evidence_conflict": signals.evidence_conflict,
            "time_pressure": signals.time_pressure,
            "compute_available": signals.compute_available,
        }
        decision = self.router.route(
            "investigate analyze compare this deeply complex issue",
            context=ctx,
        )
        return decision

    def test_trivial_query_fast(self):
        d = self.router.route("hi")
        assert d.mode == ComputationMode.FAST
        assert d.reasons  # explainable

    def test_simple_query_routine(self):
        d = self.router.route("summarize this article for me")
        assert d.mode in (ComputationMode.FAST, ComputationMode.ROUTINE)

    def test_complex_query_deliberate_or_above(self):
        d = self.router.route(
            "investigate why the system fails: analyze compare prove design",
        )
        assert d.mode in (ComputationMode.DELIBERATE, ComputationMode.DEEP,
                          ComputationMode.EXPLORATORY)

    def test_deep_conjunction_rule(self):
        """High complexity + high uncertainty/conflict reaches DEEP even
        when the weighted score alone would not."""
        d = self.router.route(
            "investigate analyze compare this system failure",
            context={"uncertainty": 0.9},
        )
        assert d.mode in (ComputationMode.DEEP, ComputationMode.EXPLORATORY)
        assert any("DEEP rule" in r or "deep" in r for r in d.reasons)

    def test_time_pressure_override_to_fast(self):
        d = self.router.route(
            "what time is it", context={"time_pressure": 0.95},
        )
        assert d.mode == ComputationMode.FAST
        assert any("time_pressure" in r for r in d.reasons)

    def test_time_pressure_does_not_override_high_consequence(self):
        d = self.router.route(
            "delete the production database now", context={"time_pressure": 0.95},
        )
        # Rule 1 requires low consequence; here consequence ≥ 0.6 floors
        # the mode at DELIBERATE even under time pressure.
        assert d.mode != ComputationMode.FAST
        assert d.mode in (ComputationMode.DELIBERATE, ComputationMode.DEEP,
                          ComputationMode.EXPLORATORY)

    def test_compute_cap(self):
        """Low compute headroom caps the mode at DELIBERATE (graceful
        degradation) — but the consequence floor still applies first for
        destructive work."""
        d = self.router.route(
            "investigate analyze compare prove design everything deeply",
            context={"compute_available": 0.1},
        )
        assert d.mode == ComputationMode.DELIBERATE
        assert any("capped" in r for r in d.reasons)

    def test_consequence_floor_beats_compute_cap(self):
        """Destructive work under compute pressure floors at DELIBERATE,
        never drops to FAST/ROUTINE."""
        d = self.router.route(
            "delete the production database now",
            context={"compute_available": 0.1},
        )
        order = [ComputationMode.FAST, ComputationMode.ROUTINE,
                 ComputationMode.DELIBERATE]
        assert d.mode in order and d.mode == ComputationMode.DELIBERATE

    def test_decision_deterministic(self):
        d1 = self.router.route("investigate the anomaly and analyze causes")
        d2 = self.router.route("investigate the anomaly and analyze causes")
        assert d1.mode == d2.mode

    def test_every_decision_has_reasons(self):
        for q in ("hi", "summarize this", "investigate analyze compare prove"):
            d = self.router.route(q)
            assert d.reasons, f"no reasons for {q!r}"

    def test_budget_attached_to_decision(self):
        d = self.router.route("investigate deeply and analyze")
        assert d.budget.mode == d.mode
        assert d.budget.max_retrieval_items > 0


class TestHardwareDetection:
    def test_detect_runs_without_error(self):
        profile = detect_hardware()
        assert profile.cpu_count >= 1
        assert isinstance(profile.has_gpu, bool)

    def test_overrides_win(self):
        profile = detect_hardware(overrides={
            "cpu_count": 2, "total_ram_gb": 2.0, "has_gpu": False,
        })
        assert profile.cpu_count == 2
        assert profile.total_ram_gb == 2.0
        assert profile.tier == HardwareTier.CONSTRAINED

    def test_gpu_and_ram_full(self):
        profile = detect_hardware(overrides={"has_gpu": True, "total_ram_gb": 32.0})
        assert profile.tier == HardwareTier.FULL

    def test_offline_forced(self):
        profile = detect_hardware(overrides={"offline_mode": True, "has_gpu": True})
        assert profile.tier == HardwareTier.OFFLINE
        assert profile.offline_mode is True

    def test_unknown_ram_does_not_crash(self):
        profile = detect_hardware(overrides={"cpu_count": 4, "total_ram_gb": None, "has_gpu": False})
        assert profile.cpu_count == 4


class TestGracefulDegradation:
    def test_weak_hardware_shrinks_budgets_never_zero(self):
        weak = HardwareProfile(
            tier=HardwareTier.CONSTRAINED, cpu_count=1,
            total_ram_gb=2.0, has_gpu=False, offline_mode=False,
        )
        router = AdaptiveRouter(hardware=weak)
        d = router.route("investigate the complex issue deeply")
        assert d.budget.max_retrieval_items >= 3
        # Hypotheses shrink but do not vanish for a mode that allows them
        base = MODE_BUDGETS[ComputationMode.DEEP].max_hypotheses
        if d.mode == ComputationMode.DEEP:
            assert 1 <= d.budget.max_hypotheses < base

    def test_tier_profiles_monotone(self):
        """Fuller tiers get ≥ depth; constrained/offline at the bottom."""
        assert TIER_PROFILES[HardwareTier.FULL].retrieval_depth_factor \
            >= TIER_PROFILES[HardwareTier.STANDARD].retrieval_depth_factor \
            >= TIER_PROFILES[HardwareTier.CONSTRAINED].retrieval_depth_factor
        assert TIER_PROFILES[HardwareTier.OFFLINE].retrieval_depth_factor \
            <= TIER_PROFILES[HardwareTier.STANDARD].retrieval_depth_factor

    def test_offline_disables_async(self):
        router = AdaptiveRouter(hardware=HardwareProfile(
            tier=HardwareTier.OFFLINE, cpu_count=4,
            total_ram_gb=8.0, has_gpu=False, offline_mode=True,
        ))
        d = router.route("investigate this thoroughly and analyze")
        assert d.budget.allow_async_tools is False  # parallelism == 1

    def test_execution_profile_attached(self):
        router = AdaptiveRouter()
        d = router.route("hello")
        assert d.execution is not None
        assert d.execution.tier == router.hardware.tier


class TestExecutiveIntegration:
    def test_apply_to_step(self):
        router = AdaptiveRouter(hardware=HardwareProfile(
            tier=HardwareTier.STANDARD, cpu_count=8, total_ram_gb=16.0,
            has_gpu=False, offline_mode=False,
        ))
        decision = router.route("investigate the issue deeply")
        step = Step(description="gather", action_type="search", params={"custom": 1})
        router.apply_to_step(step, decision)
        assert "max_retrieval_items" in step.params
        assert step.params["custom"] == 1  # non-destructive
        assert step.metadata_routing_mode == decision.mode.value

    def test_router_log_and_stats(self):
        router = AdaptiveRouter()
        router.route("a")
        router.route("investigate analyze compare prove design this")
        stats = router.stats()
        assert stats["decisions"] == 2
        assert stats["by_mode"].get("fast", 0) >= 1


class TestScaffoldingHonesty:
    def test_signal_extraction_marked(self):
        import sweep_cognitive.routing as mod
        doc = mod.__doc__ or ""
        assert "TEMPORARY SCAFFOLDING" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
