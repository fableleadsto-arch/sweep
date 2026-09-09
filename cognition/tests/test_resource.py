"""Phase 18 — Resource tests.

Budget enforcement and graceful degradation under synthetic load.
"""

import pytest

from cognition.resources import (ExecutionPlan, ResourceBudget, ResourceMonitor,
                                 ResourceState)
from cognition.loop import CognitiveLoop


def healthy():
    return ResourceState(cpu_percent=15.0, ram_mb=350.0)


def exhausted_cpu():
    return ResourceState(cpu_percent=97.0, ram_mb=1800.0)


def exhausted_ram():
    return ResourceState(cpu_percent=30.0, ram_mb=4096.0)


def test_monitor_tolerates_sustained_load_by_degrading_not_failing():
    mon = ResourceMonitor(budget=ResourceBudget(95.0, 2048.0), sampler=exhausted_cpu)
    mon.sample("stage-1")
    assert mon.over_budget() is True
    assert mon.quality_level == 2
    mon.sample("stage-2")       # stays degraded, does not crash
    assert mon.quality_level == 1
    assert len(mon.measurements) == 2


def test_monitor_recovers_when_load_drops():
    mon = ResourceMonitor(budget=ResourceBudget(95.0, 2048.0), sampler=exhausted_cpu)
    mon.sample("busy")
    mon.sampler = healthy
    m = mon.sample("idle")
    assert m.within_budget is True


def test_loop_records_resource_measurements_per_stage():
    loop = CognitiveLoop()
    loop.run("measure me")
    assert len(loop.monitor.measurements) == 14      # one per pipeline stage
    log = loop.monitor.audit()
    assert [x["stage"] for x in log] == loop.PIPELINE


def test_degraded_plan_reduces_work():
    full = ExecutionPlan("synth", max_steps=100, memory_budget_mb=1024.0)
    degraded = ExecutionPlan("synth", max_steps=100, memory_budget_mb=1024.0, degraded=True)
    assert full.steps(100) == 100
    assert degraded.steps(100) == 50


def test_loop_degradation_flag_follows_budget():
    loop = CognitiveLoop()
    # force the monitor over budget on first sample
    loop.monitor.budget = ResourceBudget(cpu_limit=0.5, ram_limit_mb=1.0)
    loop.monitor.sampler = lambda: ResourceState(cpu_percent=60.0, ram_mb=256.0)
    result = loop.run("tight budget")
    assert result.degraded is True
    assert result.stages[0].name == "perception"     # still completed the run