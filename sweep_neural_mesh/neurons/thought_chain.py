"""Structured thought chains — a detailed, human-readable record of the mesh's
thought process for a single reasoning pass.

Every reasoning path (fast paths, logic engines, the full pipeline) now emits
a ``ThoughtChain``: an ordered list of ``ThoughtStep`` objects that records
*what the mesh noticed*, *what it ruled out*, and *why it decided what it
decided*. Chains are attached to ``ReasoningTrace.thought_chain`` so they:

  - never interfere with the scoring contract (the ``reasoning`` first line
    stays byte-identical), and
  - can be displayed by the chat CLI (``/why``) and stored in traces.

A chain is deliberately more detailed than the one-line ``reasoning`` string:
per-evidence verdicts, center activations, cross-reference adjustments,
basal-ganglia proposals, metacognitive flags and the final decision rationale
all appear as separate steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtStep:
    """One stage of the mesh's thought process.

    stage: pipeline stage tag, e.g. ``"input"``, ``"route"``, ``"evidence"``,
           ``"hindbrain"``, ``"forebrain"``, ``"consensus"``, ``"decision"``.
    summary: single human-readable line ("what happened").
    detail: structured payload — evidence verdicts, counts, scores.
    latency_ms: wall time consumed by this stage (0.0 when not measured).
    """

    stage: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "summary": self.summary,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class ThoughtChain:
    """Ordered thought steps for one reasoning pass."""

    query: str
    decision: str
    confidence: float
    steps: list[ThoughtStep] = field(default_factory=list)

    def add(self, stage: str, summary: str,
            detail: dict[str, Any] | None = None,
            latency_ms: float = 0.0) -> None:
        self.steps.append(ThoughtStep(stage, summary, detail or {}, latency_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_text(self) -> str:
        """Render as an indented, human-readable block (for the CLI)."""
        lines = [f"thought chain — {len(self.steps)} steps -> {self.decision} "
                 f"(conf {self.confidence:.2f})"]
        for i, s in enumerate(self.steps, 1):
            lat = f" [{s.latency_ms:.1f}ms]" if s.latency_ms else ""
            lines.append(f"  {i:2d}. [{s.stage}] {s.summary}{lat}")
            for k, v in s.detail.items():
                lines.append(f"        {k}: {v}")
        return "\n".join(lines)


def _ev_text(e: Any) -> str:
    if isinstance(e, str):
        return e
    if isinstance(e, dict):
        return str(e.get("text", ""))
    return str(e)


def _roundf(x: Any, nd: int = 4) -> Any:
    return round(x, nd) if isinstance(x, (int, float)) else x


# ──────────────────────────────────────────────────────────────────────
# Fast-path chain builder
# ──────────────────────────────────────────────────────────────────────
def build_fast_path_chain(
    query: str,
    decision: str,
    confidence: float,
    engine: str,
    *,
    route_note: str = "",
    evidence_texts: list[str] | None = None,
    per_evidence: list[dict[str, Any]] | None = None,
    candidates: dict[str, Any] | None = None,
    ruled_out: list[str] | None = None,
    latency_ms: float = 0.0,
) -> ThoughtChain:
    """Build a standard chain for fast-path decisions.

    route_note:      which fast path fired and why.
    evidence_texts:  the raw evidence considered.
    per_evidence:    one dict per evidence item (verdict / label / conf...).
    candidates:      alternate verdicts considered (e.g. label scores).
    ruled_out:       explicit rejections ("hedged evidence -> abstain").
    """
    chain = ThoughtChain(query=query, decision=decision, confidence=confidence)
    chain.add("input", f"query received ({len(query)} chars)",
              {"query": query[:160]})
    if route_note:
        chain.add("route", f"{engine} fast path selected", {"reason": route_note})
    evs = evidence_texts or []
    if evs:
        chain.add("evidence", f"considering {len(evs)} evidence item(s)",
                  {"items": [t[:120] for t in evs[:8]]})
    for pe in per_evidence or []:
        idx = pe.get("index", "?")
        chain.add("evidence", f"evidence[{idx}] -> {pe.get('verdict', 'n/a')}",
                  {k: _roundf(v) for k, v in pe.items() if k != "index"})
    for out in ruled_out or []:
        chain.add("ruled_out", out)
    if candidates:
        chain.add("alternatives", "scores considered",
                  {k: _roundf(v) for k, v in candidates.items()})
    chain.add("decision", f"decided {decision}",
              {"engine": engine, "confidence": _roundf(confidence)},
              latency_ms=latency_ms)
    return chain


# ──────────────────────────────────────────────────────────────────────
# Full-pipeline chain builder
# ──────────────────────────────────────────────────────────────────────
def build_full_chain(
    query: str,
    decision: str,
    confidence: float,
    *,
    evidence_count: int,
    filtered_count: int,
    salience: float,
    energy_state: str,
    sanity_passed: bool,
    prediction_confidence: float,
    reflexive: bool,
    hindbrain_ms: float,
    midbrain: dict[str, Any],
    forebrain_ms: float,
    center_outputs: dict[str, int],
    xref_boosted: int,
    xref_suppressed: int,
    integration_confidence: float,
    bg_proposals: int,
    bg_decisions: int,
    consensus: dict[str, Any],
    proof_mesh_applied: bool,
    bayesian_applied: bool,
    metacognition: dict[str, Any],
    human_reasoning: dict[str, Any],
    complexity: str,
    active_modules: list[str],
    entropy_bits: float,
    memory_recalls: int,
    semantic_knowledge: int,
    total_latency_ms: float,
) -> ThoughtChain:
    """Build the detailed chain for a full-pipeline reasoning pass.

    Every argument is already computed by ``Cortex.reason``; this function
    only arranges it into a narrative of what the mesh thought.
    """
    chain = ThoughtChain(query=query, decision=decision, confidence=confidence)

    # 1. Input
    chain.add("input", f"query received; {evidence_count} raw evidence item(s)",
              {"query": query[:160], "raw_evidence": evidence_count})

    # 2. Fast-path scan
    chain.add(
        "route",
        "no fast path applied — full pipeline engaged"
        if not reflexive else "reflexive shortcut detected",
        {"fast_paths_checked": [
            "claim_evidence", "neural_evidence", "contradiction",
            "uncertainty", "general_intelligence", "logic_engines",
            "task_router", "live_knowledge",
        ]},
    )

    # 3. Hindbrain
    dropped = max(evidence_count - filtered_count, 0)
    chain.add(
        "hindbrain",
        f"salience {salience:.2f}; kept {filtered_count}/{evidence_count} "
        f"evidence after filtering",
        {
            "salience": _roundf(salience),
            "filtered_out": dropped,
            "sanity_passed": sanity_passed,
            "energy_state": energy_state,
            "prediction_confidence": _roundf(prediction_confidence),
        },
        latency_ms=hindbrain_ms,
    )

    # 4. Midbrain
    chain.add(
        "midbrain",
        "dopaminergic value/salience modulation computed",
        {k: _roundf(v) for k, v in (midbrain or {}).items()},
    )

    # 5. Forebrain centers
    active = {name: n for name, n in (center_outputs or {}).items() if n}
    chain.add(
        "forebrain",
        f"{len(active)} specialized center(s) produced signals",
        {"activations": active, "workspace_and_wm": (
            "global workspace ignited" if active else "no ignition")},
        latency_ms=forebrain_ms,
    )

    # 6. Cross-referencing
    chain.add(
        "evidence",
        f"cross-reference: {xref_boosted} boosted, {xref_suppressed} suppressed",
        {"boosted": xref_boosted, "suppressed": xref_suppressed},
    )

    # 7. Basal ganglia
    chain.add(
        "action",
        f"basal ganglia: {bg_proposals} proposal(s), {bg_decisions} action(s) relayed",
        {"proposals": bg_proposals, "executed_actions": bg_decisions},
    )

    # 8. Consensus
    cs = consensus or {}
    chain.add(
        "consensus",
        f"consensus engine -> {cs.get('decision', decision)} "
        f"({cs.get('supporting_evidence', '?')} supporting / "
        f"{cs.get('refuting_evidence', '?')} refuting / "
        f"{cs.get('contradicting_evidence', '?')} contradicting)",
        {
            "supporting": cs.get("supporting_evidence"),
            "refuting": cs.get("refuting_evidence"),
            "neutral": cs.get("neutral_evidence"),
            "contradicting": cs.get("contradicting_evidence"),
            "direction_balance": _roundf(cs.get("direction_balance", 0.0)),
            "raw_confidence": _roundf(cs.get("raw_confidence", confidence)),
            "proof_mesh_override": proof_mesh_applied,
            "bayesian_adjustment": bayesian_applied,
        },
    )

    # 9. Metacognition
    chain.add(
        "metacognition",
        "confidence self-monitored"
        + (" — escalation recommended" if (metacognition or {}).get("escalation_recommended")
           else ""),
        {
            "awareness": _roundf((metacognition or {}).get("awareness", 0.0)),
            "uncertainty_signals": (metacognition or {}).get("uncertainty_signals", 0),
            "escalation_recommended": (metacognition or {}).get("escalation_recommended", False),
            "confidence_adjusted": (metacognition or {}).get("confidence_adjusted", False),
        },
    )

    # 10. Human-style reasoning modules
    hr = human_reasoning or {}
    used = [m for m, on in hr.items() if on]
    chain.add(
        "reasoning",
        f"complexity={complexity}; {len(active_modules)} reasoning module(s) active",
        {
            "complexity": complexity,
            "active_modules": active_modules,
            "human_capabilities_used": used,
            "evidence_entropy_bits": _roundf(entropy_bits),
        },
    )

    # 11. Memory
    chain.add(
        "memory",
        f"{memory_recalls} episodic recall(s), {semantic_knowledge} semantic hit(s)",
        {"episodic_recalls": memory_recalls, "semantic_knowledge": semantic_knowledge},
    )

    # 12. Decision
    chain.add(
        "decision",
        f"decided {decision}",
        {"confidence": _roundf(confidence), "latency_ms": _roundf(total_latency_ms, 2)},
        latency_ms=total_latency_ms,
    )
    return chain
