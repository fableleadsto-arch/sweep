"""
Cortex Loop — BG-Thalamus action selection loop.

Extracted from cortex.py to reduce its size.
The cortex proposes actions, the basal ganglia decides Go/NoGo,
and the thalamus relays selected actions back for execution.
"""
from __future__ import annotations

from typing import Any

from .signal import Signal, SignalType
from .basal_ganglia import BasalGanglia, Thalamus, ActionProposal, ActionType
from .trace import ReasoningTrace


def cortex_propose(
    integrated: Signal,
    all_signals: list[Signal],
    ctx: dict[str, Any],
    memory_recalls: list[Any],
) -> list[ActionProposal]:
    """Cortex proposes actions for the BG-Thalamus loop.

    Based on integration confidence, evidence count, and memory recalls,
    the cortex proposes escalation, confidence adjustment, or continuation.
    """
    proposals: list[ActionProposal] = []
    conf = integrated.confidence
    ev_count = ctx.get("evidence_count", 0)

    if 0.3 < conf < 0.7:
        proposals.append(ActionProposal(
            action_type=ActionType.ESCALATE_CREDIBILITY, confidence=0.6,
            reasoning="moderate confidence suggests credibility could help",
            evidence_ids=[], metadata=ctx))
    if ev_count > 5:
        proposals.append(ActionProposal(
            action_type=ActionType.ESCALATE_CONTRADICTION, confidence=0.5,
            reasoning=f"{ev_count} items warrant contradiction check",
            evidence_ids=[], metadata=ctx))
    if memory_recalls:
        avg_mc = sum(r.confidence for r in memory_recalls) / len(memory_recalls)
        if avg_mc > 0.6:
            proposals.append(ActionProposal(
                action_type=ActionType.INCREASE_CONFIDENCE, confidence=avg_mc,
                reasoning=f"{len(memory_recalls)} similar past episodes support this",
                evidence_ids=[], metadata=ctx))
    if conf > 0.5:
        proposals.append(ActionProposal(
            action_type=ActionType.PROCEED_TO_CONSENSUS, confidence=conf,
            reasoning="sufficient integration for decision",
            evidence_ids=[], metadata=ctx))
    if ev_count < 3:
        proposals.append(ActionProposal(
            action_type=ActionType.REQUEST_MORE_EVIDENCE, confidence=0.4,
            reasoning=f"only {ev_count} evidence items available",
            evidence_ids=[], metadata=ctx))
    return proposals


def execute_actions(
    selected_actions: list[Any],
    integrated: Signal,
) -> Signal:
    """Execute selected BG actions by adjusting confidence."""
    conf = integrated.confidence
    for action in selected_actions:
        p = action.proposal
        if p.action_type == ActionType.INCREASE_CONFIDENCE:
            conf = min(1.0, conf + p.confidence * 0.1)
        elif p.action_type == ActionType.DECREASE_CONFIDENCE:
            conf = max(0.0, conf - p.confidence * 0.1)
    return Signal(
        data=integrated.data, signal_type=integrated.signal_type,
        confidence=conf, source_center=integrated.source_center,
        metadata={**integrated.metadata, "bg_adjusted": True},
        history=list(integrated.history),
    )


def short_circuit(
    query: str,
    evidence: list,
    reason: str,
    t0: float,
    traces: list[ReasoningTrace],
) -> Any:
    """Handle hindbrain rejection or reflexive shortcuts."""
    import time as _time
    from .trace import ReasoningResult
    from .thought_chain import build_fast_path_chain

    lat = (_time.perf_counter() - t0) * 1000
    chain = build_fast_path_chain(
        query, "insufficient", 0.0, "hindbrain_gate",
        route_note=("hindbrain stopped the pass before full processing"),
        ruled_out=[reason],
        latency_ms=lat,
    )
    trace = ReasoningTrace(
        query=query, input_evidence_count=len(evidence),
        center_outputs={}, integration_confidence=0.0,
        decision="insufficient", decision_confidence=0.0,
        reasoning=f"hindbrain rejection: {reason}", total_latency_ms=lat,
        thought_chain=chain,
    )
    traces.append(trace)
    return ReasoningResult(
        query=query, decision="insufficient", confidence=0.0,
        reasoning=f"hindbrain rejection: {reason}",
        explanation_data={}, trace=trace, factors=[],
        memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
    )
