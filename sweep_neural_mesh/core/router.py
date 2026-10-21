"""
ModelRouter — selects the best computational path through the Mesh.

Given a task, input characteristics, and resource constraints, the
router ranks candidate nodes and selects the optimal execution strategy.

    candidates = registry.find_capability("visual_embedding")
    selected = router.rank(candidates, context)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from ..core.node import Framework, NeuralNode, NodeStatus
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class RoutingContext:
    """All information the router needs to make a decision."""
    task: str = ""
    required_capability: str = ""
    input_quality: float = 1.0
    latency_budget_ms: float = 5000.0
    memory_budget_mb: float = 4096.0
    require_gpu: bool = False
    preferred_frameworks: list[Framework] = field(default_factory=list)
    exclude_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Output of the routing decision."""
    selected_node: NeuralNode | None = None
    alternatives: list[NeuralNode] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    fallback_used: bool = False


class ModelRouter:
    """
    Selects the best node(s) for a given task and context.

    The router uses a weighted scoring system:
      - capability_match (required)
      - latency_fit
      - memory_fit
      - framework_preference
      - historical_performance
      - cost_efficiency
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._weights = {
            "capability_match": 10.0,
            "latency_fit": 3.0,
            "memory_fit": 2.0,
            "framework_preference": 1.5,
            "performance": 2.0,
            "cost_efficiency": 1.0,
        }

    def rank(
        self,
        candidates: list[NeuralNode],
        context: RoutingContext,
    ) -> RoutingResult:
        """Rank candidates and select the best one."""
        if not candidates:
            return RoutingResult(reason="no candidates provided")

        budgets = (context.memory_budget_mb, context.latency_budget_ms)
        if any(not math.isfinite(value) or value <= 0 for value in budgets):
            return RoutingResult(reason="memory and latency budgets must be finite and positive")

        # Filter by status
        viable = [
            n for n in candidates
            if n.status not in (NodeStatus.FAILED, NodeStatus.UNLOADED)
            and n.node_id not in context.exclude_node_ids
            and (not context.required_capability or context.required_capability in n.capabilities)
        ]
        if not viable:
            return RoutingResult(reason="all candidates filtered out")

        # Budgets are admission limits, not preferences. In particular, the
        # least over-budget candidate must not win when none can fit. These
        # estimates do not replace OS worker limits on actual resource use.
        viable = [node for node in viable if self._fits_budgets(node, context)]
        if not viable:
            return RoutingResult(reason="no candidates fit the memory, latency or GPU requirements")

        # Score each candidate
        scores: dict[str, float] = {}
        for node in viable:
            score = self._score_node(node, context)
            scores[node.node_id] = score

        # Sort by score descending
        ranked = sorted(viable, key=lambda n: scores[n.node_id], reverse=True)
        best = ranked[0]

        return RoutingResult(
            selected_node=best,
            alternatives=ranked[1:],
            scores=scores,
            reason=f"selected {best.name} (score={scores[best.node_id]:.2f})",
        )

    def rank_multi(
        self,
        capability: str,
        context: RoutingContext,
        top_k: int = 1,
    ) -> RoutingResult:
        """Find candidates by capability and rank them."""
        candidates = self.registry.find_capability(capability)
        result = self.rank(candidates, context)
        if top_k > 1 and result.selected_node:
            result.alternatives = result.alternatives[: top_k - 1]
        return result

    @staticmethod
    def _fits_budgets(node: NeuralNode, context: RoutingContext) -> bool:
        cost = node.cost
        estimates = (cost.memory_mb, cost.min_ram_mb, cost.avg_latency_ms,
                     cost.p95_latency_ms, cost.startup_time_ms)
        if any(not math.isfinite(value) or value < 0 for value in estimates):
            return False
        latency = max(cost.avg_latency_ms, cost.p95_latency_ms)
        if node.status == NodeStatus.IDLE:
            latency += cost.startup_time_ms
        return (max(cost.memory_mb, cost.min_ram_mb) <= context.memory_budget_mb
                and latency <= context.latency_budget_ms
                and (not context.require_gpu or cost.gpu_required))

    def _score_node(self, node: NeuralNode, ctx: RoutingContext) -> float:
        """Compute a weighted score for a node."""
        score = 0.0

        # Capability match (always 1 if we got here)
        score += self._weights["capability_match"]

        # Latency fit
        if ctx.latency_budget_ms > 0 and node.cost.avg_latency_ms > 0:
            ratio = node.cost.avg_latency_ms / ctx.latency_budget_ms
            if ratio <= 1.0:
                score += self._weights["latency_fit"] * (1.0 - ratio)
            else:
                score -= self._weights["latency_fit"] * (ratio - 1.0) * 2

        # Memory fit
        if ctx.memory_budget_mb > 0 and node.cost.memory_mb > 0:
            ratio = node.cost.memory_mb / ctx.memory_budget_mb
            if ratio <= 1.0:
                score += self._weights["memory_fit"] * (1.0 - ratio)
            else:
                score -= self._weights["memory_fit"] * (ratio - 1.0) * 2

        # GPU requirement
        if ctx.require_gpu and not node.cost.gpu_required:
            score -= 5.0
        if not ctx.require_gpu and node.cost.gpu_required:
            score -= 1.0  # mild penalty, not disqualifying

        # Framework preference
        if ctx.preferred_frameworks and node.framework in ctx.preferred_frameworks:
            score += self._weights["framework_preference"]

        # Historical performance
        if node.history:
            avg_conf = sum(r.confidence for r in node.history) / len(node.history)
            success_rate = 1.0 - node.failure_rate
            score += self._weights["performance"] * avg_conf * success_rate

        # Cost efficiency (prefer cheaper nodes when scores are close)
        if node.cost.memory_mb > 0:
            score += self._weights["cost_efficiency"] * (
                1.0 / (1.0 + node.cost.memory_mb / 1024.0)
            )

        return score

    def __repr__(self) -> str:
        return f"ModelRouter(registry={self.registry})"
