"""Phase 15 — Resource-Aware Execution.

Sweep operates within an explicit resource budget. When CPU or RAM limits
are approached or exceeded, the system degrades gracefully (simpler model,
smaller batches, caching, skipped non-essential checks) instead of failing.
Every measurement is written to an audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ResourceState:
    """Sampled CPU / RAM usage."""

    cpu_percent: float
    ram_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {"cpu_percent": round(self.cpu_percent, 2),
                "ram_mb": round(self.ram_mb, 2)}


@dataclass
class ResourceBudget:
    cpu_limit: float
    ram_limit_mb: float

    def __post_init__(self) -> None:
        if self.cpu_limit <= 0 or self.ram_limit_mb <= 0:
            raise ValueError("budget limits must be positive")


@dataclass
class ResourceMeasurement:
    """Recorded measurement in the audit trail."""

    stage: str
    state: ResourceState
    within_budget: bool
    degraded: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "cpu_percent": self.state.cpu_percent,
            "ram_mb": self.state.ram_mb,
            "within_budget": self.within_budget,
            "degraded": self.degraded,
            "detail": self.detail,
        }


class ResourceMonitor:
    """Enforces a CPU/RAM budget with graceful degradation."""

    def __init__(self, budget: Optional[ResourceBudget] = None, sampler: Optional[Callable[[], ResourceState]] = None):
        self.budget = budget or ResourceBudget(cpu_limit=100.0, ram_limit_mb=2048.0)
        self.sampler = sampler or self._os_sampler
        self.measurements: List[ResourceMeasurement] = []
        self.degraded: bool = False
        self.quality_level: int = 3   # 3 full, 2 reduced, 1 minimal, 0 stalled-safe
        # Episode-scoped (per-run) overload tracking. Instantaneous
        # system-wide CPU sampled at millisecond-scale stage boundaries
        # measures the whole machine (other processes, OS/AV/GC jitter),
        # not this pipeline — so CPU spikes alone must not mark a short
        # run degraded. RAM (process RSS) is real pressure at any
        # timescale. Counted CPU overloads stay available for
        # introspection and the audit log keeps every measurement.
        self._episode_ram_over: bool = False
        self._episode_cpu_over_count: int = 0

    def reset_episode(self) -> None:
        """Start a new measurement episode (one pipeline run). Clears the
        episode-scoped verdicts; the audit log and the quality latch are
        preserved. Prevents run N's transient overload from being
        attributed to run N+1."""
        self._episode_ram_over = False
        self._episode_cpu_over_count = 0

    def episode_over_budget(self) -> bool:
        """Run-scoped degradation verdict: RAM pressure at any sample
        during the current episode. CPU overloads are counted for
        introspection but excluded from the verdict — at millisecond
        sampling scale they measure the machine, not this code."""
        return self._episode_ram_over

    def episode_cpu_overloads(self) -> int:
        return self._episode_cpu_over_count

    def _os_sampler(self) -> ResourceState:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None) or 0.0
            ram = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            cpu, ram = 0.0, 0.0
        return ResourceState(cpu_percent=cpu, ram_mb=ram)

    def sample(self, stage: str = "default") -> ResourceMeasurement:
        state = self.sampler()
        within = (state.cpu_percent <= self.budget.cpu_limit
                  and state.ram_mb <= self.budget.ram_limit_mb)
        self.degraded = self.degraded or (not within)
        detail = ""
        if not within:
            detail = "over budget"
            if state.cpu_percent > self.budget.cpu_limit:
                detail += " cpu"
                self._episode_cpu_over_count += 1
            if state.ram_mb > self.budget.ram_limit_mb:
                detail += " ram"
                self._episode_ram_over = True
            self.quality_level = max(1, self.quality_level - 1)  # degrade per overload
        measurement = ResourceMeasurement(stage, state, within, self.degraded, detail)
        self.measurements.append(measurement)
        return measurement

    def degrade_more(self) -> None:
        self.quality_level = max(1, self.quality_level - 1)

    def over_budget(self) -> bool:
        return self.degraded

    def checkpoint(self, stage: str) -> bool:
        """Run a checkpoint; returns True if proceeding is within budget."""
        m = self.sample(stage)
        return m.within_budget

    def audit(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.measurements]

    def last_measurement(self) -> Optional[ResourceMeasurement]:
        return self.measurements[-1] if self.measurements else None


@dataclass
class ExecutionPlan:
    """A resource-bounded plan that degrades gracefully quantity of work."""

    task: str
    max_steps: int
    memory_budget_mb: float
    degraded: bool = False

    def steps(self, baseline: int) -> int:
        if self.degraded:
            return max(1, baseline // 2)
        return baseline

    def step_ram(self) -> float:
        if self.degraded:
            return self.memory_budget_mb * 0.5
        return self.memory_budget_mb