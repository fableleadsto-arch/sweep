"""
Reasoning data types — shared data classes for traces and results.

These were extracted from cortex.py to reduce its size and make
the data types independently importable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .thought_chain import ThoughtChain


@dataclass
class ReasoningTrace:
    """Complete trace of a single reasoning pass through the cortex.

    Records every measurable aspect of a reasoning pass:
    timings, mechanism activations, grades, and ML outputs.
    """
    query: str
    input_evidence_count: int
    center_outputs: dict[str, int]
    integration_confidence: float
    decision: str
    decision_confidence: float
    reasoning: str
    total_latency_ms: float
    factors: list[dict[str, Any]] = field(default_factory=list)

    # Brain division timings
    hindbrain_ms: float = 0.0
    midbrain_ms: float = 0.0
    forebrain_ms: float = 0.0
    bg_decisions: int = 0
    salience_score: float = 0.0
    memory_recall_count: int = 0

    # Learning phase
    mastery_phase: str = "novice"

    # Multi-dimensional grade
    grade: dict[str, Any] = field(default_factory=dict)

    # Hindbrain: Predictive coding
    prediction_accuracy: float = 0.0
    reflexive_shortcut: bool = False

    # Hindbrain: Energy gating
    energy_state: str = "fresh"

    # Midbrain: Dopaminergic reward
    avg_value_prediction: float = 0.0

    # Midbrain: Salience modulation
    avg_salience_modulation: float = 0.0

    # Midbrain: Inhibitory gating
    avg_inhibition: float = 0.0

    # Forebrain: Global Workspace
    workspace_ignitions: int = 0
    workspace_entries: int = 0

    # Forebrain: Working Memory
    working_memory_size: int = 0

    # Forebrain: Metacognition
    metacognition_awareness: float = 0.0
    uncertainty_signals: int = 0
    escalation_recommended: bool = False

    # Human Reasoning Capabilities
    analogical_mappings: int = 0
    causal_nodes: int = 0
    counterfactual_scenarios: int = 0
    common_sense_plausibility: float = 0.5
    theory_of_mind_trust: float = 0.5
    abductive_hypotheses: int = 0
    narrative_coherence: float = 0.0

    # Adaptive pipeline depth
    query_complexity: str = "moderate"
    active_modules: list[str] = field(default_factory=list)

    # Advanced Math Modules
    evidence_entropy_bits: float = 0.0
    evidence_pagerank: dict[str, float] = field(default_factory=dict)

    # ML Engine Outputs
    query_sentiment: str = "neutral"
    query_sentiment_valence: float = 0.0
    evidence_sentiments: list[str] = field(default_factory=list)
    extracted_entities: list[dict[str, str]] = field(default_factory=list)
    query_embedding_backend: str = "none"

    # Structured thought process (detailed step-by-step chain; optional so
    # existing code that constructs ReasoningTrace keeps working)
    thought_chain: "ThoughtChain | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "input_evidence_count": self.input_evidence_count,
            "center_outputs": self.center_outputs,
            "integration_confidence": round(self.integration_confidence, 4),
            "decision": self.decision,
            "decision_confidence": round(self.decision_confidence, 4),
            "reasoning": self.reasoning,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "factors": self.factors,
            "brain_divisions": {
                "hindbrain_ms": round(self.hindbrain_ms, 2),
                "midbrain_ms": round(self.midbrain_ms, 2),
                "forebrain_ms": round(self.forebrain_ms, 2),
                "salience_score": round(self.salience_score, 4),
                "bg_decisions": self.bg_decisions,
                "memory_recall_count": self.memory_recall_count,
            },
            "biological_mechanisms": {
                "prediction_accuracy": round(self.prediction_accuracy, 4),
                "reflexive_shortcut": self.reflexive_shortcut,
                "energy_state": self.energy_state,
                "avg_value_prediction": round(self.avg_value_prediction, 4),
                "avg_salience_modulation": round(self.avg_salience_modulation, 4),
                "avg_inhibition": round(self.avg_inhibition, 4),
                "workspace_ignitions": self.workspace_ignitions,
                "workspace_entries": self.workspace_entries,
                "working_memory_size": self.working_memory_size,
                "metacognition_awareness": round(self.metacognition_awareness, 4),
                "uncertainty_signals": self.uncertainty_signals,
                "escalation_recommended": self.escalation_recommended,
            },
            "human_reasoning": {
                "analogical_mappings": self.analogical_mappings,
                "causal_nodes": self.causal_nodes,
                "counterfactual_scenarios": self.counterfactual_scenarios,
                "common_sense_plausibility": round(self.common_sense_plausibility, 4),
                "theory_of_mind_trust": round(self.theory_of_mind_trust, 4),
                "abductive_hypotheses": self.abductive_hypotheses,
                "narrative_coherence": round(self.narrative_coherence, 4),
            },
            "adaptive_pipeline": {
                "query_complexity": self.query_complexity,
                "active_modules": self.active_modules,
                "skipped_modules": [
                    m for m in [
                        "common_sense", "abductive", "theory_of_mind",
                        "causal", "narrative", "analogical", "counterfactual",
                    ]
                    if m not in self.active_modules
                ],
            },
            "math_modules": {
                "evidence_entropy_bits": round(self.evidence_entropy_bits, 4),
                "evidence_pagerank": {
                    k: round(v, 4) for k, v in self.evidence_pagerank.items()
                },
            },
            "ml_engines": {
                "query_sentiment": self.query_sentiment,
                "query_sentiment_valence": round(self.query_sentiment_valence, 4),
                "evidence_sentiments": self.evidence_sentiments,
                "extracted_entities": self.extracted_entities,
                "query_embedding_backend": self.query_embedding_backend,
            },
            "mastery_phase": self.mastery_phase,
            "grade": self.grade,
            "thought_chain": (
                self.thought_chain.to_dict()
                if self.thought_chain is not None else None
            ),
        }


@dataclass
class ReasoningResult:
    """The complete output of a reasoning pass.

    Contains the final decision, confidence, reasoning trace,
    and supporting metadata.
    """
    query: str
    decision: str
    confidence: float
    reasoning: str
    explanation_data: dict[str, Any]
    trace: ReasoningTrace
    factors: list[dict[str, Any]]
    memory_context: dict[str, Any] = field(default_factory=dict)
    grade: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "factors": self.factors,
            "explanation": self.explanation_data,
            "trace": self.trace.to_dict(),
            "memory_context": self.memory_context,
            "grade": self.grade,
        }
