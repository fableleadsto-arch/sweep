"""
Adaptive Computation Routing — Phase 13.

Spec module 14: five computation modes (FAST / ROUTINE / DELIBERATE /
DEEP / EXPLORATORY). Mode selection depends on ambiguity, consequence,
uncertainty, task complexity, time constraints, available compute,
novelty, and evidence conflict. Never spend maximum computation on
every request.

Spec section 12 (hardware-aware execution): detect hardware, then
dynamically choose retrieval depth, reasoning depth, parallelism, and
processing limits. The system must degrade gracefully: FULL → STANDARD
→ CONSTRAINED → CPU → OFFLINE. Never remove intelligence mechanisms
because hardware is weak — route them efficiently.

IMPLEMENTATION STATUS (honest):
- REAL: mode decision function over explicit signals (scored, weighted,
  deterministic), budget allocation per mode, hardware profile
  detection via stdlib (no psutil dependency), execution profile
  derivation with graceful degradation, routing decision log with
  replayability.
- TEMPORARY SCAFFOLDING: signal extraction (complexity/ambiguity/
  novelty/consequence) is lexical and count-based — the Phase 8
  prediction engine's scaffolding with the same caveat. The decision
  function itself is real and stays; signals get replaced by learned
  estimators.
"""

from __future__ import annotations

# Optional learned estimator (Phase 15). Routing works without it —
# the lexical scaffolding is the explicit fallback.
try:
    from ..learning import features_from_query, LogisticEstimator
    _HAS_LEARNING = True
except ImportError:  # pragma: no cover
    _HAS_LEARNING = False

import itertools
import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


# ======================================================================
# COMPUTATION MODES (spec module 14)
# ======================================================================

class ComputationMode(str, Enum):
    FAST = "fast"                # minimal: perception + lookup, no hypotheses
    ROUTINE = "routine"          # normal: retrieval + light reasoning
    DELIBERATE = "deliberate"    # careful: hypotheses + evidence weighing
    DEEP = "deep"                # exhaustive: simulation + counterfactuals
    EXPLORATORY = "exploratory"  # open-ended: broad retrieval, hypothesis gen


# Budgets per mode. These are RELATIVE weights (sum-normalized per task),
# not wall-clock promises — hardware profiles scale them.
@dataclass
class ModeBudget:
    mode: ComputationMode
    max_retrieval_items: int       # associative memory top_k
    max_hypotheses: int            # hypothesis engine competitors
    allow_simulation: bool         # counterfactual/world-model simulation
    allow_async_tools: bool
    verification_strictness: float # 0..1 — verification gate threshold
    time_scale: float              # multiplier on per-step budgets

FAST_BUDGET = ModeBudget(ComputationMode.FAST, 5, 0, False, False, 0.4, 0.5)
ROUTINE_BUDGET = ModeBudget(ComputationMode.ROUTINE, 20, 2, False, False, 0.6, 1.0)
DELIBERATE_BUDGET = ModeBudget(ComputationMode.DELIBERATE, 50, 4, True, True, 0.75, 2.0)
DEEP_BUDGET = ModeBudget(ComputationMode.DEEP, 100, 6, True, True, 0.9, 4.0)
EXPLORATORY_BUDGET = ModeBudget(ComputationMode.EXPLORATORY, 80, 8, True, True, 0.7, 3.0)

MODE_BUDGETS: dict[ComputationMode, ModeBudget] = {
    ComputationMode.FAST: FAST_BUDGET,
    ComputationMode.ROUTINE: ROUTINE_BUDGET,
    ComputationMode.DELIBERATE: DELIBERATE_BUDGET,
    ComputationMode.DEEP: DEEP_BUDGET,
    ComputationMode.EXPLORATORY: EXPLORATORY_BUDGET,
}


# ======================================================================
# HARDWARE PROFILES (spec section 12)
# ======================================================================

class HardwareTier(str, Enum):
    FULL = "full"              # GPU + ample RAM
    STANDARD = "standard"      # decent CPU + RAM
    CONSTRAINED = "constrained"  # limited RAM
    CPU = "cpu"                # CPU-only optimization path
    OFFLINE = "offline"        # local-only tools/models


@dataclass
class HardwareProfile:
    tier: HardwareTier
    cpu_count: int
    total_ram_gb: Optional[float]     # None when undetectable
    has_gpu: bool
    offline_mode: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "cpu_count": self.cpu_count,
            "total_ram_gb": self.total_ram_gb,
            "has_gpu": self.has_gpu,
            "offline_mode": self.offline_mode,
            "notes": self.notes,
        }


def detect_hardware(overrides: Optional[dict[str, Any]] = None) -> HardwareProfile:
    """
    Detect hardware via stdlib only. Unknown values stay None/False —
    we do not guess. `overrides` forces values (for tests and for the
    user explicitly declaring their environment, which wins).
    """
    notes: list[str] = []
    cpu_count = os.cpu_count() or 1

    total_ram_gb: Optional[float] = None
    try:
        # /proc/meminfo on Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_ram_gb = kb / (1024 * 1024)
                    break
    except (OSError, ValueError, IndexError):
        notes.append("ram_unknown")

    has_gpu = False
    # stdlib-visible GPU hints are unreliable; treat presence of common
    # env markers as hints, otherwise unknown = False with a note.
    if os.path.exists("/proc/driver/nvidia"):
        has_gpu = True

    offline_mode = False
    # Offline is an explicit operator choice, not auto-detected
    if overrides and overrides.get("offline_mode") is not None:
        offline_mode = bool(overrides["offline_mode"])
        notes.append("offline_forced_by_config")

    if overrides:
        if overrides.get("cpu_count") is not None:
            cpu_count = int(overrides["cpu_count"])
        if overrides.get("total_ram_gb") is not None:
            total_ram_gb = float(overrides["total_ram_gb"])
        if overrides.get("has_gpu") is not None:
            has_gpu = bool(overrides["has_gpu"])

    # Tier assignment (graceful degradation, spec section 12)
    if offline_mode:
        tier = HardwareTier.OFFLINE
    elif has_gpu and (total_ram_gb is None or total_ram_gb >= 16):
        tier = HardwareTier.FULL
    elif total_ram_gb is not None and total_ram_gb < 4:
        tier = HardwareTier.CONSTRAINED
    elif has_gpu or cpu_count >= 8:
        tier = HardwareTier.STANDARD
    else:
        tier = HardwareTier.CPU

    return HardwareProfile(
        tier=tier, cpu_count=cpu_count, total_ram_gb=total_ram_gb,
        has_gpu=has_gpu, offline_mode=offline_mode, notes=notes,
    )


# Per-tier execution scaling. Intelligence mechanisms are NEVER removed
# for weak hardware — budgets shrink, they don't disappear (spec 12).
@dataclass
class ExecutionProfile:
    tier: HardwareTier
    retrieval_depth_factor: float
    hypothesis_factor: float
    parallelism: int
    max_context_items: int
    prefer_local_models: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "retrieval_depth_factor": self.retrieval_depth_factor,
            "hypothesis_factor": self.hypothesis_factor,
            "parallelism": self.parallelism,
            "max_context_items": self.max_context_items,
            "prefer_local_models": self.prefer_local_models,
        }


TIER_PROFILES: dict[HardwareTier, ExecutionProfile] = {
    HardwareTier.FULL: ExecutionProfile(HardwareTier.FULL, 1.0, 1.0, 4, 200, False),
    HardwareTier.STANDARD: ExecutionProfile(HardwareTier.STANDARD, 0.8, 0.8, 2, 120, False),
    HardwareTier.CONSTRAINED: ExecutionProfile(HardwareTier.CONSTRAINED, 0.5, 0.5, 1, 60, True),
    HardwareTier.CPU: ExecutionProfile(HardwareTier.CPU, 0.6, 0.6, 1, 80, True),
    HardwareTier.OFFLINE: ExecutionProfile(HardwareTier.OFFLINE, 0.5, 0.4, 1, 50, True),
}


# ======================================================================
# ROUTING SIGNALS
# ======================================================================

@dataclass
class RoutingSignals:
    """Inputs to the mode decision. All 0..1 unless noted."""
    complexity: float = 0.3
    ambiguity: float = 0.1
    uncertainty: float = 0.2
    consequence: float = 0.2          # cost of being wrong
    novelty: float = 0.1
    evidence_conflict: float = 0.0
    time_pressure: float = 0.0        # 1 = must answer instantly
    compute_available: float = 1.0    # 0..1 resource headroom

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity,
            "ambiguity": self.ambiguity,
            "uncertainty": self.uncertainty,
            "consequence": self.consequence,
            "novelty": self.novelty,
            "evidence_conflict": self.evidence_conflict,
            "time_pressure": self.time_pressure,
            "compute_available": self.compute_available,
        }


COMPLEXITY_MARKERS = (
    "investigate", "analyze", "compare", "why", "how", "explain",
    "research", "verify", "prove", "design",
)
AMBIGUITY_MARKERS = (
    "maybe", "possibly", "might", "could be", "unclear", "not sure",
    "perhaps", "or something",
)
CONSEQUENCE_MARKERS = (
    "production", "deploy", "delete", "irreversible", "critical",
    "urgent", "security", "payment", "legal",
)
NOVELTY_MARKERS = (
    "first time", "never seen", "new kind", "unknown", "unusual",
)


def extract_signals(query: str, context: Optional[dict[str, Any]] = None) -> RoutingSignals:
    """Lexical signal extraction. SCAFFOLDING — learned estimators
    replace this; the RoutingSignals contract stays.

    Complexity is driven by marker DENSITY (markers per word) plus
    length, so terse marker-dense queries still register as complex.
    """
    context = context or {}
    lower = query.lower()
    words = query.split()
    n_words = max(1, len(words))

    markers = sum(1 for m in COMPLEXITY_MARKERS if m in lower)
    complexity = 0.15
    # Density component (terse marker-dense queries still register)…
    complexity += 0.40 * min(1.0, (markers / n_words) * 1.5)
    # …plus absolute marker count (a query with several analytical verbs
    # is complex even when phrased verbosely).
    complexity += 0.08 * min(markers, 4)
    if len(words) > 40:
        complexity += 0.30
    elif len(words) > 20:
        complexity += 0.20
    elif len(words) > 12:
        complexity += 0.10
    complexity = min(1.0, complexity)

    ambiguity = min(1.0, 0.05 + 0.15 * sum(1 for m in AMBIGUITY_MARKERS if m in lower))
    consequence = min(1.0, 0.05 + 0.30 * sum(1 for m in CONSEQUENCE_MARKERS if m in lower))
    novelty = min(1.0, 0.05 + 0.25 * sum(1 for m in NOVELTY_MARKERS if m in lower))

    return RoutingSignals(
        complexity=round(complexity, 3),
        ambiguity=round(ambiguity, 3),
        uncertainty=round(min(1.0, float(context.get("uncertainty", 0.2))), 3),
        consequence=round(consequence, 3),
        novelty=round(novelty, 3),
        evidence_conflict=round(min(1.0, float(context.get("evidence_conflict", 0.0))), 3),
        time_pressure=round(min(1.0, float(context.get("time_pressure", 0.0))), 3),
        compute_available=round(min(1.0, max(0.0, float(context.get("compute_available", 1.0)))), 3),
    )


# ======================================================================
# ROUTER
# ======================================================================

@dataclass
class RoutingDecision:
    mode: ComputationMode
    budget: ModeBudget
    signals: RoutingSignals
    reasons: list[str]
    decision_id: str = field(default_factory=lambda: _uid("route"))
    created_at: float = field(default_factory=time.time)
    # Execution profile is attached after hardware-aware scaling
    execution: Optional[ExecutionProfile] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "mode": self.mode.value,
            "reasons": self.reasons,
            "signals": self.signals.to_dict(),
            "budget": {
                "max_retrieval_items": self.budget.max_retrieval_items,
                "max_hypotheses": self.budget.max_hypotheses,
                "allow_simulation": self.budget.allow_simulation,
                "verification_strictness": self.budget.verification_strictness,
            },
            "execution": self.execution.to_dict() if self.execution else None,
        }


class AdaptiveRouter:
    """
    Decides HOW MUCH computation a task deserves (spec module 14) and
    scales it to the hardware (spec section 12). Every decision is
    logged with its reasons — routing is auditable and replayable.
    """

    # Decision thresholds. Weights sum to 1.0 in _decide.
    THRESHOLDS = {
        ComputationMode.FAST: 0.25,
        ComputationMode.ROUTINE: 0.45,
        ComputationMode.DELIBERATE: 0.65,
        ComputationMode.DEEP: 0.80,
        # above DEEP threshold → EXPLORATORY if open-ended, else DEEP
    }

    def __init__(self, hardware: Optional[HardwareProfile] = None,
                 learned_estimator: Optional["LogisticEstimator"] = None):
        self.hardware = hardware or detect_hardware()
        self.learned_estimator = learned_estimator
        self._log: list[RoutingDecision] = []

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def route(self, query: str, context: Optional[dict[str, Any]] = None) -> RoutingDecision:
        signals = extract_signals(query, context)
        learn_reasons = self._apply_learned_signals(query, signals)
        mode, reasons = self._decide(signals)
        budget = self._scale_budget(MODE_BUDGETS[mode])
        decision = RoutingDecision(
            mode=mode, budget=budget, signals=signals,
            reasons=learn_reasons + reasons,
            execution=TIER_PROFILES[self.hardware.tier],
        )
        self._log.append(decision)
        return decision

    def _apply_learned_signals(
        self, query: str, signals: RoutingSignals
    ) -> list[str]:
        """When a learned estimator (Phase 15) is attached, use its
        predicted outcome to refine the complexity signal.

        HONEST MAPPING: the estimator is trained/validated to predict
        task SUCCESS from behavioral features. Routing consumes
        difficulty = 1 - p(success) as a difficulty signal. This is a
        documented approximation — it conflates 'hard' with 'likely to
        fail', which is correct when failures correlate with task
        difficulty (the data the gate requires) but would misrank
        easy tasks with broken tools. The lexical fallback remains
        authoritative whenever the estimator is absent or errors.
        """
        if self.learned_estimator is None or not _HAS_LEARNING:
            return []
        try:
            features = features_from_query(query)
            # BlendedEstimator (Phase 18) exposes predict_proba_blended:
            # behavioral + text features combined. Plain LogisticEstimator
            # only has predict_proba over behavioral features.
            blend = getattr(self.learned_estimator, "predict_proba_blended", None)
            if blend is not None:
                p_success = blend(features, query)
            else:
                p_success = self.learned_estimator.predict_proba(features)
        except Exception:  # noqa: BLE001 — learned path must never break routing
            return ["learned estimator errored; lexical signals used"]
        difficulty = 1.0 - p_success
        learned = 0.6 * difficulty + 0.4 * signals.complexity
        signals.complexity = round(min(1.0, max(0.0, learned)), 3)
        return [
            f"learned estimator: p(success)={p_success:.3f} -> "
            f"difficulty={difficulty:.3f} blended into complexity"
        ]

    def _decide(self, s: RoutingSignals) -> tuple[ComputationMode, list[str]]:
        """Explainable decision table (spec module 14). Deterministic;
        every mode choice carries the reasons that produced it.

        Decision rules, applied in order:
        1. time_pressure ≥ 0.8 with consequence < 0.5 → FAST
        2. consequence floor: ≥ 0.6 → at least DELIBERATE,
           ≥ 0.8 → at least DEEP (destructive work is never rushed)
        3. score bands via weighted sum → FAST/ROUTINE/DELIBERATE
        4. DEEP rule: score ≥ deep threshold OR (complexity ≥ 0.55
           AND (uncertainty ≥ 0.6 OR conflict ≥ 0.5 OR consequence ≥ 0.7))
        5. EXPLORATORY: DEEP reached AND (ambiguity ≥ 0.5 OR novelty ≥ 0.5)
        6. compute ceiling: compute_available < 0.3 caps at DELIBERATE
           (graceful degradation, spec section 12)
        """
        reasons: list[str] = []

        # Rule 1: rush only low-consequence work
        if s.time_pressure >= 0.8 and s.consequence < 0.5:
            reasons.append(
                f"time_pressure={s.time_pressure} with low consequence={s.consequence} → FAST"
            )
            return ComputationMode.FAST, reasons

        score = (
            0.40 * s.complexity
            + 0.15 * s.ambiguity
            + 0.15 * s.uncertainty
            + 0.15 * s.consequence
            + 0.08 * s.novelty
            + 0.07 * s.evidence_conflict
        )

        # Rule 2: consequence floor and complexity floor (whichever is
        # higher wins). Complexity floor: genuinely complex tasks are
        # never routed to shallow modes even when other signals are quiet.
        mode_floor: Optional[ComputationMode] = None
        if s.consequence >= 0.8 or s.complexity >= 0.85:
            mode_floor = ComputationMode.DEEP
            if s.consequence >= 0.8:
                reasons.append(f"consequence={s.consequence} ≥ 0.8 → floor DEEP (destructive work)")
            else:
                reasons.append(f"complexity={s.complexity} ≥ 0.85 → floor DEEP")
        elif s.consequence >= 0.6 or s.complexity >= 0.70:
            mode_floor = ComputationMode.DELIBERATE
            if s.consequence >= 0.6:
                reasons.append(f"consequence={s.consequence} ≥ 0.6 → floor DELIBERATE")
            else:
                reasons.append(f"complexity={s.complexity} ≥ 0.70 → floor DELIBERATE")

        # Rules 3+4: score bands with the DEEP conjunction rule
        deep_rule = (
            s.complexity >= 0.55
            and (s.uncertainty >= 0.6 or s.evidence_conflict >= 0.5 or s.consequence >= 0.7)
        )
        if score >= self.THRESHOLDS[ComputationMode.DEEP] or deep_rule:
            if deep_rule and score < self.THRESHOLDS[ComputationMode.DEEP]:
                reasons.append(
                    f"DEEP rule: complexity={s.complexity} ≥ 0.55 with "
                    f"uncertainty={s.uncertainty}/conflict={s.evidence_conflict}/"
                    f"consequence={s.consequence} high"
                )
            else:
                reasons.append(f"score={score:.2f} ≥ {self.THRESHOLDS[ComputationMode.DEEP]} → deep analysis")
            # Rule 5: exploratory split
            if s.ambiguity >= 0.5 or s.novelty >= 0.5:
                mode = ComputationMode.EXPLORATORY
                reasons.append(
                    f"ambiguity={s.ambiguity}/novelty={s.novelty} ≥ 0.5 → open-ended exploration"
                )
            else:
                mode = ComputationMode.DEEP
        elif score >= self.THRESHOLDS[ComputationMode.DELIBERATE]:
            mode = ComputationMode.DELIBERATE
            reasons.append(f"score={score:.2f} → deliberate (hypotheses + evidence weighing)")
        elif score >= self.THRESHOLDS[ComputationMode.ROUTINE]:
            mode = ComputationMode.ROUTINE
            reasons.append(f"score={score:.2f} → routine")
        else:
            mode = ComputationMode.FAST
            reasons.append(f"score={score:.2f} → fast path")

        # Apply consequence floor (never downgrade below it)
        if mode_floor is not None:
            order = [ComputationMode.FAST, ComputationMode.ROUTINE,
                     ComputationMode.DELIBERATE, ComputationMode.DEEP,
                     ComputationMode.EXPLORATORY]
            if order.index(mode) < order.index(mode_floor):
                mode = mode_floor
                reasons.append(f"raised to {mode_floor.value} by consequence floor")

        # Rule 6: compute ceiling (graceful degradation)
        if s.compute_available < 0.3:
            order = [ComputationMode.FAST, ComputationMode.ROUTINE,
                     ComputationMode.DELIBERATE, ComputationMode.DEEP,
                     ComputationMode.EXPLORATORY]
            if order.index(mode) > order.index(ComputationMode.DELIBERATE):
                mode = ComputationMode.DELIBERATE
                reasons.append(
                    f"compute_available={s.compute_available} < 0.3 → capped at DELIBERATE"
                )

        return mode, reasons

    def _scale_budget(self, budget: ModeBudget) -> ModeBudget:
        """Scale a mode budget by the hardware tier profile. Intelligence
        mechanisms shrink, never vanish (spec section 12)."""
        prof = TIER_PROFILES[self.hardware.tier]
        return ModeBudget(
            mode=budget.mode,
            max_retrieval_items=max(3, int(budget.max_retrieval_items * prof.retrieval_depth_factor)),
            max_hypotheses=(
                0 if budget.max_hypotheses == 0
                else max(1, int(budget.max_hypotheses * prof.hypothesis_factor))
            ),
            allow_simulation=budget.allow_simulation,
            allow_async_tools=budget.allow_async_tools and prof.parallelism > 1,
            verification_strictness=budget.verification_strictness,
            time_scale=budget.time_scale * (2.0 if prof.tier == HardwareTier.CONSTRAINED else 1.0),
        )

    # ------------------------------------------------------------------
    # Integration points
    # ------------------------------------------------------------------

    def apply_to_step(self, step, decision: RoutingDecision) -> None:
        """Attach routing-derived limits to an executive Step (Phase 12
        contract): retrieval depth as param override, verification
        strictness recorded. Non-destructive: step params win."""
        budget = decision.budget
        step.params.setdefault("max_retrieval_items", budget.max_retrieval_items)
        step.params.setdefault("max_hypotheses", budget.max_hypotheses)
        step.params.setdefault("verification_strictness", budget.verification_strictness)
        step.metadata_routing_mode = decision.mode.value

    def decisions(self) -> list[RoutingDecision]:
        return list(self._log)

    def stats(self) -> dict[str, Any]:
        by_mode: dict[str, int] = {}
        for d in self._log:
            by_mode[d.mode.value] = by_mode.get(d.mode.value, 0) + 1
        return {
            "decisions": len(self._log),
            "by_mode": by_mode,
            "hardware_tier": self.hardware.tier.value,
        }


__all__ = [
    "ComputationMode",
    "ModeBudget",
    "MODE_BUDGETS",
    "HardwareTier",
    "HardwareProfile",
    "detect_hardware",
    "ExecutionProfile",
    "TIER_PROFILES",
    "RoutingSignals",
    "extract_signals",
    "RoutingDecision",
    "AdaptiveRouter",
]
