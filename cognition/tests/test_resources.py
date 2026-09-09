"""Phase 15 — Resource-Aware Execution tests."""

import pytest

from cognition.resources import ExecutionPlan, ResourceBudget, ResourceMonitor


def test_under_budget_no_degradation():
    from cognition.resources import ResourceState
    mon = ResourceMonitor(budget=ResourceBudget(80.0, 1024.0),
                          sampler=lambda: ResourceState(30.0, 512.0))
    m = mon.sample("probe")
    assert m.within_budget is True
    assert mon.quality_level == 3


def test_over_cpu_budget_degrades_gracefully():
    from cognition.resources import ResourceState
    mon = ResourceMonitor(budget=ResourceBudget(50.0, 2048.0),
                          sampler=lambda: ResourceState(cpu_percent=95.0, ram_mb=512.0))
    m = mon.sample("infer")
    assert m.within_budget is False
    assert "cpu" in m.detail
    assert mon.quality_level == 2      # degraded once, not failed


def test_over_ram_budget_degrades():
    from cognition.resources import ResourceState
    mon = ResourceMonitor(budget=ResourceBudget(100.0, 1024.0),
                          sampler=lambda: ResourceState(cpu_percent=10.0, ram_mb=2000.0))
    mon.sample("mem")
    assert "ram" in mon.last_measurement().detail
    assert mon.over_budget() is True


def test_audit_log_records_every_measurement():
    from cognition.resources import ResourceState
    mon = ResourceMonitor(budget=ResourceBudget(100.0, 2048.0),
                          sampler=lambda: ResourceState(cpu_percent=12.0, ram_mb=400.0))
    mon.sample("a")
    mon.sample("b")
    mon.sample("c")
    log = mon.audit()
    assert len(log) == 3
    assert [x["stage"] for x in log] == ["a", "b", "c"]
    assert all(x["within_budget"] for x in log)


def test_budget_requires_positive_limits():
    with pytest.raises(ValueError):
        ResourceBudget(0, 100)
    with pytest.raises(ValueError):
        ResourceBudget(100, -5)


def test_execution_plan_halves_steps_when_degraded():
    plan = ExecutionPlan("synthesis", max_steps=10, memory_budget_mb=512.0, degraded=True)
    assert plan.steps(baseline=8) == 4
    plan2 = ExecutionPlan("synthesis", max_steps=10, memory_budget_mb=512.0, degraded=False)
    assert plan2.steps(baseline=8) == 8


def test_checkpoint_signals_not_failed():
    from cognition.resources import ResourceState
    mon = ResourceMonitor(budget=ResourceBudget(100.0, 2048.0),
                          sampler=lambda: ResourceState(20.0, 300.0))
    assert mon.checkpoint("run") is True
    still_ok = mon.checkpoint("run2")
    assert still_ok is True