"""
Reasoning Integration Layer — Harmonization of reasoning systems.

This module unifies:
1. Rule-based reasoning (cognition/rules.py — RuleEngine)
2. Logic-based reasoning (cognition/logic.py — LogicEngine, 
   sweep_neural_mesh logical_inference.py, proof_mesh.py)
3. Representation-based understanding (sweep_cognitive/representation, semantic)
4. Evidence-based reasoning (cognition/evidence_graph.py, contradiction.py)
5. Hypothesis-based reasoning (cognition/hypotheses.py)
6. Belief revision (cognition/beliefs.py)

Key principle: Explicit logic sits ABOVE OR ALONGSIDE learned representations
where it provides a real benefit. The integration layer provides a unified
interface for all reasoning modalities.

This is the "reasoning orchestration" layer that allows SWEEP to:
- Use fast deterministic paths when appropriate
- Use deep reasoning when needed
- Combine multiple reasoning modalities
- Track which reasoning method produced which conclusion
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..representation import (
    Representation,
    TextRepresentation,
    RepresentationQuality,
    RepresentationType,
)
from ..semantic import (
    SemanticUnderstandingEngine,
    SemanticEntity,
    SemanticRelationship,
    SemanticEvent,
)


# ════════════════════════════════════════════════════════════════════
# REASONING MODALITY
# ════════════════════════════════════════════════════════════════════

class ReasoningModality(str, Enum):
    """What kind of reasoning produced a conclusion."""
    PERCEPTUAL = "perceptual"           # Direct perception/sensing
    REPRESENTATIONAL = "representational"  # From learned/augmented representations
    RULE_BASED = "rule_based"           # From explicit rules (cognition/rules.py)
    LOGIC_DEDUCTIVE = "logic_deductive" # From formal logic (cognition/logic.py)
    LOGIC_SYLLOGISTIC = "logic_syllogistic"  # From syllogistic reasoning
    LOGIC_PROOF_MESH = "logic_proof_mesh"  # From proof mesh
    EVIDENCE_AGGREGATION = "evidence_aggregation"  # From evidence consensus
    HYPOTHESIS_BASED = "hypothesis_based"  # From hypothesis evaluation
    BELIEF_REVISION = "belief_revision"  # From belief updating
    SEMANTIC_EXTRACTION = "semantic_extraction"  # From semantic understanding
    PREDICTIVE = "predictive"           # From prediction
    MIXED = "mixed"                     # Combined multiple modalities


# ════════════════════════════════════════════════════════════════════
# REASONING RESULT
# ════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningResult:
    """
    Unified result from any reasoning modality.
    
    This is the canonical output format for all reasoning in SWEEP.
    Every reasoning module should produce a ReasoningResult.
    """
    conclusion: str = ""                  # supported, refuted, mixed, insufficient, etc.
    confidence: float = 0.0
    reasoning: str = ""                   # Human-readable reasoning summary
    modality: ReasoningModality = ReasoningModality.REPRESENTATIONAL
    evidence_count: int = 0
    contradiction_count: int = 0
    entities_identified: int = 0
    relationships_identified: int = 0
    processing_latency_ms: float = 0.0
    used_modalities: list[ReasoningModality] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "modality": self.modality.value,
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "entities_identified": self.entities_identified,
            "relationships_identified": self.relationships_identified,
            "processing_latency_ms": self.processing_latency_ms,
            "used_modalities": [m.value for m in self.used_modalities],
            "metadata": self.metadata,
        }


# ════════════════════════════════════════════════════════════════════
# REASONING REQUEST
# ════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningRequest:
    """
    Request to reason about something.
    
    Can include:
    - Query text
    - Evidence (list of text or representations)
    - Context
    - Preferred reasoning modalities
    - Constraints (time, compute, etc.)
    """
    query: str = ""
    evidence: list[str | Representation] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    preferred_modalities: list[ReasoningModality] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query_preview": self.query[:100] if self.query else "",
            "evidence_count": len(self.evidence),
            "context_keys": list(self.context.keys()),
            "preferred_modalities": [m.value for m in self.preferred_modalities],
            "constraints": self.constraints,
            "source": self.source,
        }


# ════════════════════════════════════════════════════════════════════
# REASONING ENGINE (INTEGRATION)
# ════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """
    Unified reasoning engine that orchestrates multiple reasoning modalities.
    
    This is the harmonization layer that brings together:
    - Perceptual reasoning (from representations)
    - Semantic reasoning (from semantic understanding)
    - Rule-based reasoning (from cognition/rules.py RuleEngine)
    - Logic-based reasoning (from cognition/logic.py LogicEngine)
    - Evidence-based reasoning (from evidence consensus)
    - Hypothesis-based reasoning (from sweep_cognitive.hypotheses)
    - Predictive reasoning (from sweep_cognitive.prediction)
    
    The engine:
    1. Receives a ReasoningRequest
    2. Determines appropriate reasoning modalities
    3. Applies reasoning in priority order
    4. Combines results into a unified ReasoningResult
    5. Tracks which modalities were used
    
    Integration contract: modalities whose backing engines were not
    provided report conclusion="modality_unavailable" with confidence 0
    — they NEVER fabricate results. Rule and logic engines from the
    legacy cognition/ package are the real implementations.
    """
    
    def __init__(self,
                 rule_engine=None,
                 logic_engine=None,
                 world_model=None,
                 hypothesis_engine=None,
                 prediction_engine=None):
        self._semantic_engine = SemanticUnderstandingEngine()
        self._results_cache: dict[str, ReasoningResult] = {}
        # Optional real engines (harmonization with legacy + new modules)
        self._rule_engine = rule_engine
        self._logic_engine = logic_engine
        self._world_model = world_model
        self._hypothesis_engine = hypothesis_engine
        self._prediction_engine = prediction_engine
    
    def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """
        Reason about a query using appropriate modalities.
        
        Args:
            request: The reasoning request
            
        Returns:
            Unified ReasoningResult.
        """
        t0 = time.perf_counter()
        
        # Step 1: Perceptual/semantic understanding (always run first)
        semantic_result = self._apply_semantic_reasoning(request)
        
        # Step 2: Determine additional modalities needed
        modalities_to_apply = self._select_modalities(request, semantic_result)
        
        # Step 3: Apply selected modalities
        results = [semantic_result]
        for modality in modalities_to_apply:
            result = self._apply_modality(request, modality, semantic_result)
            if result:
                results.append(result)
        
        # Step 4: Combine results
        combined = self._combine_results(request, results)
        combined.processing_latency_ms = (time.perf_counter() - t0) * 1000
        
        return combined
    
    def _apply_semantic_reasoning(self, request: ReasoningRequest) -> ReasoningResult:
        """Apply semantic understanding to the query."""
        if not request.query:
            return ReasoningResult(
                conclusion="insufficient",
                confidence=0.0,
                reasoning="No query provided",
                modality=ReasoningModality.SEMANTIC_EXTRACTION,
            )
        
        # Create text representation
        text_rep = TextRepresentation(
            content=request.query,
            source=request.source,
        )
        
        # Add evidence as entities if available
        if request.evidence:
            for ev in request.evidence:
                if isinstance(ev, str):
                    text_rep.entities.append({
                        "text": ev[:50],
                        "type": "evidence",
                    })
        
        # Apply semantic understanding
        semantic_output = self._semantic_engine.understand(text_rep)
        
        # Build result
        result = ReasoningResult(
            conclusion="needs_further_reasoning",
            confidence=0.3,
            reasoning=f"Semantic analysis: identified {semantic_output['entity_count']} entities, "
                     f"{semantic_output['relationship_count']} relationships",
            modality=ReasoningModality.SEMANTIC_EXTRACTION,
            entities_identified=semantic_output["entity_count"],
            relationships_identified=semantic_output["relationship_count"],
            metadata={
                "semantic_output": semantic_output,
                "semantic_quality": semantic_output["semantic_quality"].value,
            },
        )
        
        return result
    
    def _select_modalities(self, request: ReasoningRequest,
                          semantic_result: ReasoningResult) -> list[ReasoningModality]:
        """
        Select additional reasoning modalities based on the request and semantic analysis.
        
        This is the adaptive reasoning controller.
        """
        modalities = []
        
        # Always consider evidence aggregation if evidence provided
        if request.evidence:
            modalities.append(ReasoningModality.EVIDENCE_AGGREGATION)
        
        # An explicit claim to prove always routes to deductive logic
        if request.context.get("claim") is not None:
            modalities.append(ReasoningModality.LOGIC_DEDUCTIVE)
        
        # Consider rule-based reasoning for structured queries
        if self._looks_like_rule_query(request.query):
            modalities.append(ReasoningModality.RULE_BASED)
        
        # Consider logic-based reasoning for logical structures
        if self._looks_like_logical_query(request.query):
            modalities.append(ReasoningModality.LOGIC_DEDUCTIVE)
        
        # If user specified preferred modalities, prioritize those
        if request.preferred_modalities:
            for modality in request.preferred_modalities:
                if modality not in modalities:
                    modalities.append(modality)
        
        return modalities
    
    def _looks_like_rule_query(self, query: str) -> bool:
        """Check if query looks like it needs rule-based reasoning."""
        rule_indicators = [
            "if", "then", "otherwise", "unless", "except",
            "when", "while", "because", "therefore",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in rule_indicators)
    
    def _looks_like_logical_query(self, query: str) -> bool:
        """Check if query looks like it needs logical reasoning."""
        logical_indicators = [
            "all", "no", "none", "some", "every",
            "if and only if", "implies", "entails",
            "therefore", "thus", "hence",
            "greater than", "less than", "equal to",
            "before", "after", "during",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in logical_indicators)
    
    def _apply_modality(self, request: ReasoningRequest,
                        modality: ReasoningModality,
                        semantic_result: ReasoningResult) -> Optional[ReasoningResult]:
        """Apply a specific reasoning modality."""
        
        if modality == ReasoningModality.EVIDENCE_AGGREGATION:
            return self._apply_evidence_aggregation(request, semantic_result)
        
        elif modality == ReasoningModality.RULE_BASED:
            return self._apply_rule_reasoning(request, semantic_result)
        
        elif modality == ReasoningModality.LOGIC_DEDUCTIVE:
            return self._apply_logic_reasoning(request, semantic_result)
        
        elif modality == ReasoningModality.HYPOTHESIS_BASED:
            return self._apply_hypothesis_reasoning(request, semantic_result)
        
        elif modality == ReasoningModality.PREDICTIVE:
            return self._apply_predictive_reasoning(request, semantic_result)
        
        elif modality == ReasoningModality.SEMANTIC_EXTRACTION:
            return semantic_result
        
        return None
    
    def _apply_evidence_aggregation(self, request: ReasoningRequest,
                                    semantic_result: ReasoningResult) -> ReasoningResult:
        """Aggregate evidence to form a conclusion."""
        # Simple evidence aggregation (scaffolding)
        # In real implementation, this would use the evidence graph
        
        evidence_texts = []
        for ev in request.evidence:
            if isinstance(ev, str):
                evidence_texts.append(ev)
            elif isinstance(ev, Representation):
                evidence_texts.append(ev.content)
        
        if not evidence_texts:
            return ReasoningResult(
                conclusion="insufficient",
                confidence=0.0,
                reasoning="No evidence provided",
                modality=ReasoningModality.EVIDENCE_AGGREGATION,
            )
        
        # Count supporting/refuting indicators (simple scaffolding)
        support_count = 0
        refute_count = 0
        
        for text in evidence_texts:
            text_lower = text.lower()
            if any(word in text_lower for word in ["support", "confirm", "prove", "show", "demonstrate"]):
                support_count += 1
            if any(word in text_lower for word in ["refute", "contradict", "false", "not", "deny"]):
                refute_count += 1
        
        if support_count > refute_count:
            conclusion = "supported"
            confidence = min(0.9, 0.5 + 0.1 * support_count)
        elif refute_count > support_count:
            conclusion = "refuted"
            confidence = min(0.9, 0.5 + 0.1 * refute_count)
        else:
            conclusion = "mixed"
            confidence = 0.5
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=confidence,
            reasoning=f"Evidence aggregation: {support_count} supporting, {refute_count} refuting",
            modality=ReasoningModality.EVIDENCE_AGGREGATION,
            evidence_count=len(evidence_texts),
        )
    
    def _apply_rule_reasoning(self, request: ReasoningRequest,
                              semantic_result: ReasoningResult) -> ReasoningResult:
        """Apply the REAL RuleEngine from cognition/rules.py.
        
        Facts are assembled from request.context plus evidence counts,
        run through the rule engine (which produces a full trace), and
        the derived conclusion/verdict is returned. If no rule engine
        was provided, report honestly that the modality is unavailable.
        """
        if self._rule_engine is None:
            return ReasoningResult(
                conclusion="modality_unavailable",
                confidence=0.0,
                reasoning="No RuleEngine provided; rule-based reasoning skipped",
                modality=ReasoningModality.RULE_BASED,
                metadata={"note": "Provide rule_engine=RuleEngine() to enable"},
            )
        
        # Assemble facts: context values + evidence statistics
        facts: dict[str, Any] = dict(request.context.get("facts", {}))
        facts["evidence_count"] = len(request.evidence)
        
        working = self._rule_engine.run(facts)
        trace = self._rule_engine.trace()
        
        conclusion = working.get("conclusion", "no_rule_fired")
        verdict = working.get("verdict")
        fired = len(trace)
        
        return ReasoningResult(
            conclusion=str(conclusion),
            confidence=0.85 if fired else 0.3,
            reasoning=(
                f"Rule engine: {fired} rule(s) fired; "
                f"verdict={verdict!r}" if fired else "Rule engine: no rules matched"
            ),
            modality=ReasoningModality.RULE_BASED,
            evidence_count=len(request.evidence),
            metadata={
                "rules_fired": fired,
                "verdict": verdict,
                "trace": [t.to_dict() for t in trace],
            },
        )
    
    def _apply_logic_reasoning(self, request: ReasoningRequest,
                               semantic_result: ReasoningResult) -> ReasoningResult:
        """Apply the REAL LogicEngine from cognition/logic.py.
        
        If a world model is available, its relationships are loaded as
        logic facts (predicate strings map directly since both use
        lowercase snake_case). The request.context may specify a claim
        {"claim": (relation, subject, object)} to prove; otherwise the
        result reports the size of the derivable closure.
        """
        if self._logic_engine is None:
            return ReasoningResult(
                conclusion="modality_unavailable",
                confidence=0.0,
                reasoning="No LogicEngine provided; logic reasoning skipped",
                modality=ReasoningModality.LOGIC_DEDUCTIVE,
                metadata={"note": "Provide logic_engine=LogicEngine() to enable"},
            )
        
        # Load world model relationships as logic facts
        loaded = 0
        if self._world_model is not None:
            for rel in self._world_model.state.query_relationships():
                self._logic_engine.add_fact(
                    rel.predicate.value, rel.subject_id, rel.object_id
                )
                loaded += 1
        
        claim = request.context.get("claim")
        if claim is not None:
            relation, subject, obj = claim
            proof = self._logic_engine.derive(relation, subject, obj)
            if proof is not None:
                return ReasoningResult(
                    conclusion="proven",
                    confidence=0.9,
                    reasoning=f"Derived {relation}({subject}, {obj}) in {len(proof)} step(s)",
                    modality=ReasoningModality.LOGIC_DEDUCTIVE,
                    metadata={
                        "proof": [s.to_dict() for s in proof],
                        "facts_loaded": loaded,
                    },
                )
            return ReasoningResult(
                conclusion="not_provable",
                confidence=0.4,
                reasoning=f"{relation}({subject}, {obj}) is not derivable from known facts",
                modality=ReasoningModality.LOGIC_DEDUCTIVE,
                metadata={"facts_loaded": loaded},
            )
        
        closure_size = len(self._logic_engine.closure())
        return ReasoningResult(
            conclusion="closure_computed",
            confidence=0.6,
            reasoning=f"Logic closure: {closure_size} derivable facts ({loaded} loaded from world model)",
            modality=ReasoningModality.LOGIC_DEDUCTIVE,
            metadata={"closure_size": closure_size, "facts_loaded": loaded},
        )
    
    def _apply_hypothesis_reasoning(self, request: ReasoningRequest,
                                    semantic_result: ReasoningResult) -> ReasoningResult:
        """Apply the Phase 9 HypothesisEngine: abductive selection with
        ambiguity refusal."""
        if self._hypothesis_engine is None:
            return ReasoningResult(
                conclusion="modality_unavailable",
                confidence=0.0,
                reasoning="No HypothesisEngine provided",
                modality=ReasoningModality.HYPOTHESIS_BASED,
            )
        
        best = self._hypothesis_engine.best_explanation()
        viable = self._hypothesis_engine.viable()
        if best is None:
            return ReasoningResult(
                conclusion="ambiguous" if viable else "no_hypotheses",
                confidence=0.3,
                reasoning=(
                    f"{len(viable)} viable hypotheses, none dominant — "
                    "refusing to pick without more evidence"
                    if viable else "No viable hypotheses registered"
                ),
                modality=ReasoningModality.HYPOTHESIS_BASED,
                metadata={"viable_count": len(viable)},
            )
        return ReasoningResult(
            conclusion=best.statement,
            confidence=best.confidence,
            reasoning=f"Best explanation of {len(viable)} viable hypotheses",
            modality=ReasoningModality.HYPOTHESIS_BASED,
            metadata={"hypothesis_id": best.hypothesis_id},
        )
    
    def _apply_predictive_reasoning(self, request: ReasoningRequest,
                                    semantic_result: ReasoningResult) -> ReasoningResult:
        """Apply the Phase 8 PredictionEngine for task-complexity-driven
        computation mode selection."""
        if self._prediction_engine is None:
            return ReasoningResult(
                conclusion="modality_unavailable",
                confidence=0.0,
                reasoning="No PredictionEngine provided",
                modality=ReasoningModality.PREDICTIVE,
            )
        pred = self._prediction_engine.predict_task_complexity(request.query)
        return ReasoningResult(
            conclusion=f"recommended_mode_{pred.metadata['mode']}",
            confidence=pred.confidence,
            reasoning=f"Predicted task complexity: mode={pred.metadata['mode']}, "
                     f"uncertainty={pred.uncertainty}",
            modality=ReasoningModality.PREDICTIVE,
            metadata={"prediction_id": pred.id, "mode": pred.metadata["mode"]},
        )
    
    def _combine_results(self, request: ReasoningRequest,
                         results: list[ReasoningResult]) -> ReasoningResult:
        """Combine multiple reasoning results into one."""
        if not results:
            return ReasoningResult(
                conclusion="insufficient",
                confidence=0.0,
                reasoning="No reasoning results to combine",
            )
        
        # Find the strongest conclusion
        best_result = max(results, key=lambda r: r.confidence)
        
        # Track all used modalities
        used_modalities = list(set(
            r.modality for r in results if r.modality != ReasoningModality.SEMANTIC_EXTRACTION
        ))
        
        # Combine evidence counts
        total_evidence = sum(r.evidence_count for r in results)
        total_contradictions = sum(r.contradiction_count for r in results)
        
        # Build combined reasoning
        reasoning_parts = []
        for r in results:
            if r.reasoning:
                reasoning_parts.append(f"[{r.modality.value}] {r.reasoning}")
        
        combined = ReasoningResult(
            conclusion=best_result.conclusion,
            confidence=best_result.confidence,
            reasoning=" | ".join(reasoning_parts),
            modality=best_result.modality,
            evidence_count=total_evidence,
            contradiction_count=total_contradictions,
            entities_identified=best_result.entities_identified,
            relationships_identified=best_result.relationships_identified,
            used_modalities=used_modalities,
            metadata={
                "result_count": len(results),
                "individual_results": [r.to_dict() for r in results],
            },
        )
        
        return combined


# ════════════════════════════════════════════════════════════════════
# REASONING REGISTRY
# ════════════════════════════════════════════════════════════════════

class ReasoningRegistry:
    """
    Registry of available reasoning modalities and their capabilities.
    
    This allows the system to know what reasoning methods are available
    and when to use each one.
    """
    
    def __init__(self):
        self._modalities: dict[ReasoningModality, dict[str, Any]] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default reasoning modalities."""
        defaults = {
            ReasoningModality.PERCEPTUAL: {
                "description": "Direct perceptual reasoning from sensory input",
                "latency": "low",
                "confidence_reliability": "medium",
                "use_for": ["simple perception", "direct observation"],
                "requires": ["perception_engine"],
            },
            ReasoningModality.REPRESENTATIONAL: {
                "description": "Reasoning from learned representations",
                "latency": "medium",
                "confidence_reliability": "medium",
                "use_for": ["semantic understanding", "pattern recognition"],
                "requires": ["representation_layer", "embedding_models"],
            },
            ReasoningModality.RULE_BASED: {
                "description": "Explicit rule-based reasoning",
                "latency": "low",
                "confidence_reliability": "high",
                "use_for": ["structured rules", "deterministic logic"],
                "requires": ["rule_engine"],
            },
            ReasoningModality.LOGIC_DEDUCTIVE: {
                "description": "Formal deductive reasoning",
                "latency": "medium",
                "confidence_reliability": "high",
                "use_for": ["logical inference", "proof derivation"],
                "requires": ["logic_engine"],
            },
            ReasoningModality.LOGIC_SYLLOGISTIC: {
                "description": "Syllogistic reasoning",
                "latency": "low",
                "confidence_reliability": "high",
                "use_for": ["categorical reasoning", "set membership"],
                "requires": ["syllogism_engine"],
            },
            ReasoningModality.LOGIC_PROOF_MESH: {
                "description": "Proof mesh reasoning with confidence propagation",
                "latency": "medium",
                "confidence_reliability": "medium-high",
                "use_for": ["complex logical structures", "confidence-aware inference"],
                "requires": ["proof_mesh"],
            },
            ReasoningModality.EVIDENCE_AGGREGATION: {
                "description": "Evidence consensus and aggregation",
                "latency": "medium",
                "confidence_reliability": "medium",
                "use_for": ["claim verification", "source comparison"],
                "requires": ["evidence_engine", "source_analysis"],
            },
            ReasoningModality.HYPOTHESIS_BASED: {
                "description": "Hypothesis generation and evaluation",
                "latency": "medium-high",
                "confidence_reliability": "medium",
                "use_for": ["ambiguous problems", "multiple explanations"],
                "requires": ["hypothesis_engine"],
            },
            ReasoningModality.BELIEF_REVISION: {
                "description": "Belief updating based on new evidence",
                "latency": "low",
                "confidence_reliability": "medium",
                "use_for": ["belief updating", "revision tracking"],
                "requires": ["belief_engine"],
            },
            ReasoningModality.SEMANTIC_EXTRACTION: {
                "description": "Semantic understanding from text",
                "latency": "low-medium",
                "confidence_reliability": "medium",
                "use_for": ["entity extraction", "relationship detection"],
                "requires": ["semantic_engine"],
            },
            ReasoningModality.PREDICTIVE: {
                "description": "Predictive reasoning about likely outcomes",
                "latency": "medium",
                "confidence_reliability": "medium",
                "use_for": ["prediction", "anticipation"],
                "requires": ["prediction_engine"],
            },
            ReasoningModality.MIXED: {
                "description": "Combined multiple reasoning modalities",
                "latency": "variable",
                "confidence_reliability": "variable",
                "use_for": ["complex problems requiring multiple approaches"],
                "requires": ["reasoning_orchestration"],
            },
        }
        
        for modality, info in defaults.items():
            self._modalities[modality] = info
    
    def get_modality_info(self, modality: ReasoningModality) -> dict[str, Any]:
        """Get information about a reasoning modality."""
        return self._modalities.get(modality, {})
    
    def list_modalities(self) -> list[dict[str, Any]]:
        """List all available reasoning modalities."""
        return [
            {
                "modality": m.value,
                "description": info["description"],
                "latency": info["latency"],
                "confidence_reliability": info["confidence_reliability"],
                "use_for": info["use_for"],
                "requires": info["requires"],
            }
            for m, info in self._modalities.items()
        ]


# ════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════

__all__ = [
    "ReasoningModality",
    "ReasoningResult",
    "ReasoningRequest",
    "ReasoningEngine",
    "ReasoningRegistry",
]
