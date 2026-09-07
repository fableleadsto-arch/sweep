"""
ReasoningCortex — the orchestrator that runs the full neuronal pipeline.

Implements the three-division brain architecture:
  1. Hindbrain: fast filtering and salience detection
  2. Midbrain:  signal routing and attention gating
  3. Forebrain: processing centres + memory + action selection

Also implements the cortex–basal ganglia–thalamus loop:
the cortex proposes actions, the basal ganglia decides Go/NoGo,
and the thalamus relays selected actions back for execution.

Signal flow::

    Raw Input
        ↓
    Hindbrain: Predict → Reflex → Energy → Filter → Salience
        ↓
    Midbrain: Value Predict → Salience Modulate → Inhibit → Route
        ↓
    Forebrain: Workspace Broadcast → Working Memory → Process
        ↓
    Cortex-BG-Thalamus Loop → Metacognition Check → Output
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .signal import Signal, SignalType, Synapse, SynapseType
from .centers import (
    EvidenceGatherer,
    CredibilityAssessor,
    TemporalSequencer,
    CausalLinker,
    ContradictionDetector,
    ExplanationBuilder,
    ProcessingCenter,
)
from .integration import IntegrationHub, ConsensusEngine
from .brain import Hindbrain, Midbrain, Forebrain
from .basal_ganglia import BasalGanglia, Thalamus, ActionProposal, ActionType
from .plasticity import SynapticPlasticity
from .grading import EvidenceGrader
from .semantic_embeddings import EmbeddingResult
from .sentiment_engine import SentimentEngine, SentimentResult, SentimentLabel
from .ner_engine import NEREngine
from .text_summarizer import TextSummarizer
from .amygdala import Amygdala
from .bayesian import BayesianReasoner
from .proof_mesh import NeuralProofMesh
from .logical_inference import LogicalInferenceEngine
from .world_knowledge import WorldKnowledge
from .general_intelligence import GeneralIntelligence
from .live_knowledge import LiveKnowledgeRetriever
from .web_scraper import WebScraper, WebResearcher

# ── Extracted modules (reduced from original cortex.py) ────────
from .trace import ReasoningTrace, ReasoningResult
from .fast_path import try_fast_path
from .evidence_pipeline import cross_reference_evidence, apply_xref_adjustments
from .task_handlers import TaskRouter, TaskClassification
from .complexity import classify_query_complexity, select_reasoning_modules
from .human_reasoning import run_human_reasoning
from .cortex_loop import cortex_propose, execute_actions, short_circuit
from .synapse_ops import (
    build_default_synapses, apply_synaptic_input,
    hebbian_update, update_synaptic_plasticity, apply_myelination,
)
from .cortex_math import compute_math_modules, midbrain_metrics

logger = logging.getLogger(__name__)


class ReasoningCortex:
    """Master orchestrator of Sweep's neuronal reasoning system.

    Usage::

        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is Python a good language for ML?",
            evidence=["Python has extensive ML libraries", ...],
            sources=["wikipedia", "github"],
        )
        print(result.explanation)
    """

    def __init__(self, enable_ml: bool = True) -> None:
        self._enable_ml = enable_ml

        # ── Brain divisions ──────────────────────────────────
        self._hindbrain = Hindbrain()
        self._midbrain = Midbrain()
        self._forebrain = Forebrain()

        # ── Processing centres ───────────────────────────────
        self._centers: dict[str, ProcessingCenter] = {
            "evidence_gatherer": EvidenceGatherer(),
            "credibility_assessor": CredibilityAssessor(),
            "temporal_sequencer": TemporalSequencer(),
            "causal_linker": CausalLinker(),
            "contradiction_detector": ContradictionDetector(),
            "explanation_builder": ExplanationBuilder(),
        }

        # ── Integration & consensus ──────────────────────────
        self._integration_hub = IntegrationHub()
        self._consensus_engine = ConsensusEngine()

        # ── BG-Thalamus loop ─────────────────────────────────
        self._basal_ganglia = BasalGanglia()
        self._thalamus = Thalamus()

        # ── Learning & grading ───────────────────────────────
        self._plasticity = SynapticPlasticity()
        self._grader = EvidenceGrader()

        # ── Emotion & reasoning engines ──────────────────────
        self._amygdala = Amygdala()
        self._bayesian = BayesianReasoner()
        self._proof_mesh = NeuralProofMesh()
        self._logical_engine = LogicalInferenceEngine()

        # ── Knowledge ────────────────────────────────────────
        self._world_knowledge = WorldKnowledge()
        self._general_intelligence = GeneralIntelligence()
        try:
            self._world_knowledge.load_training_knowledge()
        except Exception:
            pass

        # ── Live knowledge (lazy) ────────────────────────────
        self._live_knowledge = None
        self._web_scraper = None
        self._web_researcher = None

        # ── Multi-core (5 cores) ─────────────────────────────
        from .cores import MultiCoreCoordinator
        self._multi_core = MultiCoreCoordinator(num_cores=5)

        # ── Task router (fast-path for structured tasks) ──────
        self._task_router = TaskRouter()

        # ── ML engines (lazy) ────────────────────────────────
        self._embedder = None
        self._ner_engine = None
        self._sentiment_engine = None
        self._summarizer = None
        self._ml_loaded = False

        # ── Synapses & history ───────────────────────────────
        self._synapses: dict[str, Synapse] = build_default_synapses()
        self._traces: list[ReasoningTrace] = []

        logger.info("ReasoningCortex initialized (6 centres, BG-Thalamus, "
                     "plasticity, grading, amygdala, bayesian, proof_mesh)")

    # ════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════

    def reason(
        self,
        query: str,
        evidence: list[str | dict[str, Any]],
        sources: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """Run the full neuronal reasoning pipeline."""
        t0 = time.perf_counter()
        logger.info(f"Reasoning: '{query[:80]}...' ({len(evidence)} evidence)")

        # ── Fast paths ───────────────────────────────────────

        # Claim-evidence fast path: deterministic verification of declarative
        # claims against evidence (supports / refutes / unknown). Runs before
        # the neural classifiers so degenerate/hedged/conflicting evidence is
        # handled correctly and the cortex abstains when it cannot decide.
        claim_result = self._try_claim_evidence_fast_path(query, evidence, t0)
        if claim_result is not None:
            return claim_result

        # Neural fast-path: use pretrained BERT models for evidence/contradiction
        neural_result = self._try_neural_fast_path(query, evidence, t0)
        if neural_result is not None:
            return neural_result

        # NEW: Contradiction-aware fast-path (before logic engines intercept)
        contra_result = self._try_contradiction_fast_path(query, evidence, t0)
        if contra_result is not None:
            return contra_result

        # NEW: Uncertainty-aware fast-path (before live_knowledge overrides)
        unc_result = self._try_uncertainty_fast_path(query, evidence, t0)
        if unc_result is not None:
            return unc_result

        result = self._try_gi_fast_path(query, evidence, t0)
        if result is not None:
            return result

        # Logic engines: proof_mesh + logical_inference (PRIMARY reasoning path)
        logic_result = self._try_logic_engines(query, evidence, t0)
        if logic_result is not None:
            return logic_result

        # Task router: structured logic/math/evidence/temporal/causal (fallback)
        task_result = self._try_task_router(query, evidence, t0)
        if task_result is not None:
            return task_result

        result = self._try_live_knowledge(query, evidence, t0)
        if result is not None:
            return result

        # ── Hindbrain ────────────────────────────────────────
        hb = self._hindbrain.process(query, evidence, sources)
        hindbrain_ms = (time.perf_counter() - t0) * 1000

        if not hb.sanity_passed:
            return short_circuit(query, evidence, hb.rejection_reason, t0, self._traces)
        if hb.reflexive_match:
            return short_circuit(query, evidence,
                                       f"reflexive shortcut: {hb.reflexive_response}", t0, self._traces)

        filtered_evidence = hb.filtered_evidence
        salience = hb.salience_score

        # ── Fast path: unanimous evidence direction ──────────
        fast = try_fast_path(query, filtered_evidence, self._world_knowledge, t0, self._traces)
        if fast is not None:
            return fast

        # ── ML preprocessing ─────────────────────────────────
        query_sent, evidence_sents, all_entities, query_emb = (
            self._ml_preprocess(query, filtered_evidence)
        )

        # ── Memory recall ────────────────────────────────────
        memory_recalls, semantic_knowledge = self._recall_memory(query)

        # ── Midbrain ─────────────────────────────────────────
        midbrain_result = self._midbrain.process(
            filtered_evidence, query, salience, prediction=hb.prediction,
        )
        midbrain_ms = (time.perf_counter() - t0) * 1000 - hindbrain_ms

        # ── Forebrain: workspace, working memory, centres ────
        forebrain_start = time.perf_counter()
        input_signal, center_outputs, all_signals, center_times = (
            self._run_forebrain(query, evidence, sources, context,
                                filtered_evidence, salience, hb, midbrain_result,
                                semantic_knowledge, memory_recalls)
        )

        # ── Amygdala: emotional valence tagging ──────────────
        self._tag_emotional_valence(center_outputs)

        # ── Evidence cross-referencing ───────────────────────
        ev_xref = center_outputs.get("evidence_gatherer", [])
        boosted, suppressed = cross_reference_evidence(
            ev_xref,
            center_outputs.get("credibility_assessor", []),
            center_outputs.get("causal_linker", []),
            center_outputs.get("contradiction_detector", []),
        )
        if boosted or suppressed:
            center_outputs["evidence_gatherer"] = apply_xref_adjustments(ev_xref, boosted, suppressed)
            all_signals = [s for s in all_signals if s.signal_type != SignalType.EVIDENCE]
            all_signals.extend(center_outputs["evidence_gatherer"])

        # ── Integration ──────────────────────────────────────
        processed = [s for s in all_signals if s.signal_type != SignalType.RAW]
        integrated = self._integration_hub.integrate(processed)

        # ── BG-Thalamus loop ─────────────────────────────────
        ctx = {"confidence": integrated.confidence,
               "evidence_count": len(filtered_evidence), "salience": salience}
        proposals = cortex_propose(integrated, all_signals, ctx, memory_recalls)
        bg_decisions = self._basal_ganglia.decide(proposals, ctx)
        thalamus_relay = self._thalamus.relay(bg_decisions)
        integrated = execute_actions(thalamus_relay.selected_actions, integrated)

        # ── Consensus + Proof Mesh + Bayesian ────────────────
        consensus = self._consensus_engine.decide(integrated, all_signals)
        cd = consensus.data if isinstance(consensus.data, dict) else {}
        cd = self._apply_proof_mesh(query, filtered_evidence, cd)
        cd = self._apply_bayesian(center_outputs.get("evidence_gatherer", []), cd)

        # ── Metacognition + memory recording ─────────────────
        explanation_signals = self._centers["explanation_builder"].process(all_signals)
        meta = self._forebrain.metacognition.monitor_reasoning(
            confidence=cd.get("confidence", 0.0),
            evidence_count=len(filtered_evidence),
            center_outputs={n: len(s) for n, s in center_outputs.items()},
            contradictions=len(center_outputs.get("contradiction_detector", [])),
            processing_phase=self._plasticity.get_metrics().overall_phase.value,
        )
        final_conf = cd.get("confidence", 0.0)
        if meta.should_adjust_confidence:
            final_conf = max(0.0, min(1.0, final_conf + meta.confidence_adjustment))

        self._record_working_memory(cd, final_conf)
        update_synaptic_plasticity(final_conf, cd, self._synapses)
        self._record_episode(query, cd, final_conf, filtered_evidence,
                             center_outputs.get("evidence_gatherer", []))

        # ── Human reasoning + complexity ─────────────────────
        evidence_texts = [e.get("text", "") for e in filtered_evidence if e.get("text")]
        complexity = classify_query_complexity(query, len(filtered_evidence))
        active_modules = select_reasoning_modules(complexity, len(filtered_evidence))

        hr = run_human_reasoning(
            active_modules, query, evidence_texts, sources,
            final_conf, cd.get("decision", "unknown"), self._forebrain,
        )

        # ── Information theory + graph algorithms ────────────
        ev_entropy, ev_pagerank = compute_math_modules(
            center_outputs.get("evidence_gatherer", []), self._forebrain
        )

        # ── Build trace ──────────────────────────────────────
        forebrain_ms = (time.perf_counter() - forebrain_start) * 1000
        total_latency = (time.perf_counter() - t0) * 1000
        apply_myelination(center_times, self._plasticity)
        learning = self._plasticity.get_metrics()
        grade = self._grader.grade(
            evidence_signals=center_outputs.get("evidence_gatherer", []),
            credibility_signals=center_outputs.get("credibility_assessor", []),
            temporal_signals=center_outputs.get("temporal_sequencer", []),
            causal_signals=center_outputs.get("causal_linker", []),
            contradiction_signals=center_outputs.get("contradiction_detector", []),
            integrated_confidence=integrated.confidence,
            processing_phase=learning.overall_phase.value,
        )

        hb_pred_acc = hb.prediction.confidence if hb.prediction else 0.0
        avg_val, avg_sal, avg_inh = midbrain_metrics(midbrain_result)
        ws_stats = self._forebrain.workspace.stats
        wm_stats = self._forebrain.working_memory.stats
        center_counts = {n: len(s) for n, s in center_outputs.items()}
        expl_data = (explanation_signals[0].data
                     if explanation_signals and isinstance(explanation_signals[0].data, dict)
                     else {})

        trace = ReasoningTrace(
            query=query, input_evidence_count=len(evidence),
            center_outputs=center_counts,
            integration_confidence=integrated.confidence,
            decision=cd.get("decision", "unknown"),
            decision_confidence=final_conf,
            reasoning=cd.get("reasoning", ""),
            total_latency_ms=total_latency,
            factors=cd.get("factors", []),
            hindbrain_ms=hindbrain_ms, midbrain_ms=midbrain_ms,
            forebrain_ms=forebrain_ms,
            bg_decisions=len(bg_decisions), salience_score=salience,
            memory_recall_count=len(memory_recalls),
            mastery_phase=learning.overall_phase.value,
            grade=grade.to_dict(),
            prediction_accuracy=hb_pred_acc,
            reflexive_shortcut=hb.reflexive_match,
            energy_state=hb.energy_state,
            avg_value_prediction=avg_val,
            avg_salience_modulation=avg_sal,
            avg_inhibition=avg_inh,
            workspace_ignitions=ws_stats.get("ignition_count", 0),
            workspace_entries=ws_stats.get("active_entries", 0),
            working_memory_size=wm_stats.get("size", 0),
            metacognition_awareness=meta.awareness_score,
            uncertainty_signals=len(meta.uncertainty_signals),
            escalation_recommended=meta.escalation_recommended,
            analogical_mappings=hr.analogical_mappings,
            causal_nodes=hr.causal_nodes,
            counterfactual_scenarios=hr.counterfactual_scenarios,
            common_sense_plausibility=hr.common_sense_plausibility,
            theory_of_mind_trust=hr.theory_of_mind_trust,
            abductive_hypotheses=hr.abductive_hypotheses,
            narrative_coherence=hr.narrative_coherence,
            query_complexity=complexity, active_modules=active_modules,
            evidence_entropy_bits=ev_entropy,
            evidence_pagerank=ev_pagerank,
            query_sentiment=query_sent.label.value,
            query_sentiment_valence=query_sent.valence,
            evidence_sentiments=evidence_sents,
            extracted_entities=[{"text": e.text, "label": e.label}
                                for e in all_entities[:10]],
            query_embedding_backend=query_emb.backend,
        )
        self._traces.append(trace)

        logger.info(f"Reasoning complete: {cd.get('decision', 'unknown')} "
                     f"(conf={final_conf:.3f}, grade={grade.overall_grade}, "
                     f"latency={total_latency:.1f}ms)")

        return ReasoningResult(
            query=query,
            decision=cd.get("decision", "unknown"),
            confidence=final_conf,
            reasoning=cd.get("reasoning", ""),
            explanation_data=expl_data,
            trace=trace,
            factors=cd.get("factors", []),
            memory_context={"episodic_recalls": len(memory_recalls),
                            "semantic_knowledge": len(semantic_knowledge)},
            grade=grade.to_dict(),
        )

    def multi_core_reason(self, query: str, evidence: list[str] | None = None) -> ReasoningResult:
        """Use multi-core neural processing for fast reasoning."""
        t0 = time.perf_counter()
        ev = evidence or []
        consensus = self._multi_core.process(query, ev, parallel=True)
        decision = "supported" if consensus.confidence > 0.5 else "insufficient"
        if consensus.answer and consensus.confidence > 0.3:
            decision = "supported"

        trace = ReasoningTrace(
            query=query, input_evidence_count=len(ev),
            center_outputs={"multi_core": len(consensus.core_results)},
            integration_confidence=consensus.agreement_score,
            decision=decision, decision_confidence=consensus.confidence,
            reasoning=f"Multi-core ({consensus.method}): {consensus.reasoning}",
            total_latency_ms=consensus.latency_ms,
            factors=[{"name": "multi_core", "score": consensus.confidence,
                      "detail": consensus.answer}],
        )
        self._traces.append(trace)
        return ReasoningResult(
            query=query, decision=decision, confidence=consensus.confidence,
            reasoning=f"Multi-core ({consensus.method}): {consensus.reasoning}",
            explanation_data={"multi_core_answer": consensus.answer},
            trace=trace,
            factors=[{"name": "multi_core", "score": consensus.confidence}],
            memory_context={"core_results": len(consensus.core_results)},
        )

    def retrieve_live_knowledge(self, query: str) -> str | None:
        """Retrieve live knowledge from external APIs.

        Primary path: the multi-tier GeneralKnowledge engine (mega KB →
        Wikipedia RAG → Wikidata office-holders → local LLM), with the
        legacy LiveKnowledgeRetriever as a secondary fallback.
        """
        # Tier 1: multi-tier GeneralKnowledge engine
        try:
            from .general_knowledge import get_general_knowledge
            gk = get_general_knowledge()
            answer = gk.answer(query, timeout_live=4.0)
            if answer and answer.confidence >= 0.5 and answer.answer:
                return answer.answer
        except Exception:
            pass
        # Tier 2: legacy live retriever
        if self._live_knowledge is None:
            try:
                self._live_knowledge = LiveKnowledgeRetriever()
            except Exception:
                return None
        try:
            result = self._live_knowledge.retrieve(query)
            if result and result.success and result.answer:
                return result.answer
        except Exception:
            pass
        return None

    def web_research(
        self,
        query: str,
        max_results: int = 10,
        sources: list[str] | None = None,
    ):
        """Conduct web research using multiple sources.

        Fetches from Wikipedia, arXiv, OpenAlex, and other sources.
        Returns a ResearchReport with findings, key facts, and entities.

        Usage::

            report = cortex.web_research("quantum computing applications")
            for finding in report.findings:
                print(f"[{finding.source}] {finding.title}")
        """
        if self._web_researcher is None:
            try:
                self._web_researcher = WebResearcher()
            except Exception:
                return None
        try:
            return self._web_researcher.research(
                query=query, max_results=max_results, sources=sources,
            )
        except Exception:
            return None

    def fetch_web_page(self, url: str) -> str | None:
        """Fetch and extract content from a specific URL.

        Returns clean text content from the page.
        """
        if self._web_scraper is None:
            try:
                self._web_scraper = WebScraper()
            except Exception:
                return None
        try:
            page = self._web_scraper.fetch(url)
            if page.success and page.text:
                return page.text
        except Exception:
            pass
        return None

    def gather_intelligence(
        self,
        query: str,
        documents: list[str] | None = None,
        evidence: list[str] | None = None,
        max_items: int = 20,
    ):
        """Gather, organize, and analyze intelligence about a topic.

        Returns an IntelligenceReport with gathered items, organized
        clusters, insights, and key findings.

        Usage::

            report = cortex.gather_intelligence(
                query="quantum computing",
                documents=["Quantum computing uses qubits..."],
            )
            print(report.analyzed.actionable_summary)
        """
        from .intelligence import IntelligencePipeline

        pipeline = IntelligencePipeline()
        return pipeline.run(
            query=query,
            documents=documents,
            evidence=evidence,
            world_knowledge=self._world_knowledge,
            live_retriever=self._live_knowledge,
            max_items=max_items,
        )

    # ════════════════════════════════════════════════════════════
    # INTERNAL — FAST PATHS
    # ════════════════════════════════════════════════════════════

    def _try_gi_fast_path(self, query, evidence, t0):
        gi = self._general_intelligence.answer(
            query,
            [e if isinstance(e, str) else e.get("text", "") for e in evidence],
        )
        if gi is not None and gi.confidence >= 0.85:
            lat = (time.perf_counter() - t0) * 1000
            decision = ("refuted" if gi.answer.lower() in ("no", "false")
                        else "supported")
            trace = ReasoningTrace(
                query=query, input_evidence_count=len(evidence),
                center_outputs={"general_intelligence": 1},
                integration_confidence=gi.confidence,
                decision=decision, decision_confidence=gi.confidence,
                reasoning=f"General intelligence ({gi.method}): {gi.reasoning}",
                total_latency_ms=lat,
                factors=[{"name": "general_intelligence", "score": gi.confidence,
                          "detail": gi.reasoning}],
            )
            self._traces.append(trace)
            return ReasoningResult(
                query=query, decision=decision, confidence=gi.confidence,
                reasoning=f"General intelligence ({gi.method}): {gi.reasoning}",
                explanation_data={}, trace=trace,
                factors=[{"name": "general_intelligence", "score": gi.confidence}],
                memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
            )
        return None

    # ════════════════════════════════════════════════════════════
    # CLAIM-EVIDENCE FAST PATH: deterministic claim verification
    # ════════════════════════════════════════════════════════════

    def _try_claim_evidence_fast_path(self, query, evidence, t0):
        """Verify a declarative claim against evidence (supports/refutes/unknown).

        Handles the cases the neural evidence classifier gets wrong:
          - refuting evidence (negation, "it is false that ...")
          - double negation ("not not" -> supports)
          - hedged evidence ("might be", "around 1992") -> unknown
          - irrelevant evidence (different predicate) -> unknown
          - conflicting sources -> unknown
        Returns None when the query is not a verifiable claim.
        """
        try:
            from sweep_neural_mesh.neurons.claim_evidence import (
                analyze_claim_evidence, looks_like_claim,
            )
            if not looks_like_claim(query) or not evidence:
                return None
            ev_texts = []
            for e in evidence:
                if isinstance(e, str):
                    ev_texts.append(e)
                elif isinstance(e, dict):
                    ev_texts.append(e.get("text", str(e)))
            if not ev_texts:
                return None
            verdict = analyze_claim_evidence(query, ev_texts)
            if verdict.method == "claim_evidence_analyzer:no_template":
                return None
            lat = (time.perf_counter() - t0) * 1000
            decision_map = {"supported": "supported", "refuted": "refuted",
                            "unknown": "insufficient"}
            decision = decision_map[verdict.decision]
            reasoning = (f"Claim evidence analyzer ({verdict.decision}, "
                         f"conf={verdict.confidence:.2f}):\n{verdict.reasoning}")
            trace = ReasoningTrace(
                query=query, input_evidence_count=len(ev_texts),
                center_outputs={"claim_evidence_analyzer": 1},
                integration_confidence=verdict.confidence,
                decision=decision, decision_confidence=verdict.confidence,
                reasoning=reasoning,
                total_latency_ms=lat,
                factors=[{"name": "claim_evidence_analyzer",
                          "score": verdict.confidence,
                          "detail": verdict.reasoning}],
            )
            self._traces.append(trace)
            return ReasoningResult(
                query=query, decision=decision, confidence=verdict.confidence,
                reasoning=reasoning,
                explanation_data={"verdict": verdict.decision,
                                 "evidence_pieces": [
                                     {"verdict": p.verdict, "detail": p.detail}
                                     for p in verdict.pieces
                                 ]},
                trace=trace,
                factors=[{"name": "claim_evidence_analyzer",
                          "score": verdict.confidence}],
                memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
            )
        except Exception as e:
            logger.debug(f"Claim evidence fast path failed: {e}")
        return None

    # ════════════════════════════════════════════════════════════
    # NEURAL FAST PATH: Pretrained BERT models
    # ════════════════════════════════════════════════════════════

    def _try_neural_fast_path(self, query, evidence, t0):
        """Use pretrained neural models for evidence classification and contradiction detection.
        
        This is the first fast-path tried, using fine-tuned BERT models:
        - Intent classification for routing
        - Evidence classification (supports/refutes/neutral)
        - Contradiction detection (contradiction/consistent/partial)
        """
        try:
            from sweep_neural_mesh.neurons.neural_engine import NeuralEngine
            engine = NeuralEngine()
            engine.wait_until_ready(timeout=30.0)
            if not engine.ready:
                return None
            
            ev_texts = []
            for e in evidence:
                if isinstance(e, str):
                    ev_texts.append(e)
                elif isinstance(e, dict):
                    ev_texts.append(e.get("text", ""))
            
            # Bare-claim mode: the query itself is the claim to assess
            # (used to keep verdicts honest below).
            query_is_claim = not query.strip().endswith("?")
            # The neural contradiction branch compares two evidence items to
            # EACH OTHER. That answers "do these sources agree?" — it does not
            # answer arbitrary questions. Only short-circuit when the query is
            # actually an evidence-assessment query (bare claim, or asks about
            # consistency/contrast); otherwise let the full pipeline run.
            q_lower = query.lower()
            query_is_assessment = query_is_claim or any(
                k in q_lower for k in self._CONTRA_QUERY_KEYWORDS
            )
            
            # Try contradiction detection if 2+ evidence items
            if len(ev_texts) >= 2 and query_is_assessment:
                contra_result = engine.detect_contradiction(ev_texts[0], ev_texts[1])
                if contra_result.ready and contra_result.confidence > 0.6:
                    lat = (time.perf_counter() - t0) * 1000
                    # Map neural labels to cortex decisions
                    label = contra_result.label
                    if label == "contradiction":
                        decision = "refuted"
                    elif label == "consistent":
                        decision = "supported"
                    elif label == "partial":
                        decision = "mixed"
                    else:
                        decision = "unknown"
                    # Contradicting evidence on a bare claim means the claim
                    # is UNDETERMINED (mixed), not refuted/supported — never
                    # convert disagreement between sources into a verdict on
                    # the claim itself.
                    if decision in ("refuted", "supported") and query_is_claim:
                        decision = "mixed"
                    trace = ReasoningTrace(
                        query=query, input_evidence_count=len(ev_texts),
                        center_outputs={"neural_contradiction": 1},
                        integration_confidence=contra_result.confidence,
                        decision=decision, decision_confidence=contra_result.confidence,
                        reasoning=f"Neural contradiction detector ({contra_result.label}, conf={contra_result.confidence:.2f})",
                        total_latency_ms=lat,
                        factors=[{"name": "neural_contradiction", "score": contra_result.confidence}],
                    )
                    self._traces.append(trace)
                    return ReasoningResult(
                        query=query, decision=decision, confidence=contra_result.confidence,
                        reasoning=f"Neural contradiction detector ({contra_result.label}, conf={contra_result.confidence:.2f})",
                        explanation_data={}, trace=trace,
                        factors=[{"name": "neural_contradiction", "score": contra_result.confidence}],
                        memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
                    )
            
            # Evidence classification over ALL provided evidence items.
            # The evidence classifier is pair-trained: tokenizer(evidence, claim)
            # — see training/train_evidence_pairs.py and metadata input_format.
            # Same assessment gate as the contradiction branch: neural votes on
            # (evidence, query) only answer "does the evidence bear on this
            # claim?" — arbitrary questions belong to the full pipeline.
            if len(ev_texts) >= 1 and query_is_assessment:
                votes: list[tuple[str, float]] = []
                for ev_text in ev_texts:
                    r = engine.classify_evidence(ev_text, query)
                    if r.ready and r.confidence > 0.5:
                        votes.append((r.label, r.confidence))
                if votes:
                    lat = (time.perf_counter() - t0) * 1000
                    n_supports = sum(1 for lbl, _ in votes if lbl == "supports")
                    n_refutes = sum(1 for lbl, _ in votes if lbl == "refutes")
                    n_neutral = len(votes) - n_supports - n_refutes
                    mean_conf = sum(c for _, c in votes) / len(votes)
                    if n_supports and n_refutes:
                        # Evidence disagrees -> the claim is undetermined.
                        decision = "mixed"
                    elif n_supports:
                        decision = "supported"
                    elif n_refutes:
                        decision = "refuted"
                    else:
                        # No directional evidence (neutral-only or below gate):
                        # abstain — never convert "no evidence" into a verdict.
                        decision = "insufficient"
                    # NOTE: reasoning intentionally does NOT start with
                    # "Neural evidence classifier (supports|refutes|neutral"
                    # — that legacy format is regex-matched by
                    # sweep_benchmark.scoring.model_claim for the OLD
                    # single-vote output and would mis-map the aggregate.
                    trace = ReasoningTrace(
                        query=query, input_evidence_count=len(ev_texts),
                        center_outputs={"neural_evidence": len(votes)},
                        integration_confidence=mean_conf,
                        decision=decision, decision_confidence=mean_conf,
                        reasoning=(
                            f"Neural evidence vote: sup={n_supports}, "
                            f"ref={n_refutes}, neu={n_neutral}, "
                            f"conf={mean_conf:.2f} -> {decision}"
                        ),
                        total_latency_ms=lat,
                        factors=[{"name": "neural_evidence", "score": mean_conf}],
                    )
                    self._traces.append(trace)
                    return ReasoningResult(
                        query=query, decision=decision, confidence=mean_conf,
                        reasoning=trace.reasoning,
                        explanation_data={
                            "neural_votes": {"supports": n_supports,
                                             "refutes": n_refutes,
                                             "neutral": n_neutral},
                        },
                        trace=trace,
                        factors=[{"name": "neural_evidence", "score": mean_conf}],
                        memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
                    )
        except Exception as e:
            logger.debug(f"Neural fast path failed: {e}")
        return None

    # ════════════════════════════════════════════════════════════
    # NEW FAST PATHS: Contradiction & Uncertainty
    # ════════════════════════════════════════════════════════════

    # Contradiction keywords that signal the user wants conflict detection
    _CONTRA_QUERY_KEYWORDS = [
        "contradict", "conflict", "inconsistent", "disagree",
        "opposite", "different", "versus", "vs",
        "compare", "consistent", "compatible",
    ]

    # Negation patterns for detecting contradictory statements
    _NEGATION_PATTERNS = [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bneither\b",
        r"\bdidn't\b", r"\bdoesn't\b", r"\bwasn't\b", r"\bweren't\b",
        r"\bcan't\b", r"\bwon't\b", r"\bshouldn't\b",
        r"\bincreased\b.*\bdecreased\b", r"\bdecreased\b.*\bincreased\b",
        r"\beffective\b.*\bineffective\b", r"\bineffective\b.*\beffective\b",
        r"\bright\b.*\bwrong\b", r"\bwrong\b.*\bright\b",
        r"\btrue\b.*\bfalse\b", r"\bfalse\b.*\btrue\b",
    ]

    # Future/uncertain keywords
    _UNCERTAIN_KEYWORDS = [
        "tomorrow", "next week", "next month", "next year",
        "future", "predict", "forecast", "will happen",
        "going to happen", "supposed to", "might happen",
        "could happen", "uncertain", "unclear", "unknown",
        "no one knows", "impossible to know", "can't predict",
    ]

    def _try_contradiction_fast_path(self, query, evidence, t0):
        """
        Fast-path for contradiction/conflict detection.
        
        Detects when:
        1. Query asks about contradictions ("Do these contradict?", "Are these consistent?")
        2. Evidence contains contradictory statements
        
        Returns refuted/mixed if contradictions found, supported if consistent.
        """
        q_lower = query.lower()
        
        # Check if query is about contradictions
        is_contra_query = any(kw in q_lower for kw in self._CONTRA_QUERY_KEYWORDS)
        
        if not is_contra_query or len(evidence) < 2:
            return None
        
        # Extract evidence texts
        ev_texts = []
        for e in evidence:
            if isinstance(e, str):
                ev_texts.append(e)
            elif isinstance(e, dict):
                ev_texts.append(e.get("text", str(e)))
        
        if len(ev_texts) < 2:
            return None
        
        lat = (time.perf_counter() - t0) * 1000
        
        # Simple contradiction detection: check for negation patterns
        # between evidence statements
        contradictions_found = 0
        consistency_found = 0
        
        for i in range(len(ev_texts)):
            for j in range(i + 1, len(ev_texts)):
                ev_i = ev_texts[i].lower()
                ev_j = ev_texts[j].lower()
                
                # Check word overlap
                words_i = set(ev_i.split())
                words_j = set(ev_j.split())
                overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                
                if overlap < 0.3:
                    continue
                
                # Check for negation differences
                has_neg_i = any(re.search(pat, ev_i) for pat in self._NEGATION_PATTERNS)
                has_neg_j = any(re.search(pat, ev_j) for pat in self._NEGATION_PATTERNS)
                
                # Check for direction opposites (increased vs decreased, etc.)
                direction_opposites = [
                    ("increased", "decreased"), ("higher", "lower"),
                    ("more", "less"), ("better", "worse"),
                    ("effective", "ineffective"), ("support", "refute"),
                    ("true", "false"), ("yes", "no"),
                    ("round", "flat"), ("faster", "slower"),
                    ("hotter", "colder"), ("bigger", "smaller"),
                    ("profitable", "losses"), ("profit", "loss"),
                    ("sunny", "raining"), ("raining", "dry"),
                ]
                
                has_opposite = False
                for pos, neg in direction_opposites:
                    if (pos in ev_i and neg in ev_j) or (neg in ev_i and pos in ev_j):
                        has_opposite = True
                        break
                
                if has_opposite:
                    contradictions_found += 1
                elif has_neg_i != has_neg_j and overlap > 0.5:
                    contradictions_found += 1
                else:
                    # Check for same structure, different specific values
                    # (e.g., "at 3 PM" vs "at 4 PM", "in Delhi" vs "in London")
                    import re as _re
                    nums_i = set(_re.findall(r'\b\d+\b', ev_i))
                    nums_j = set(_re.findall(r'\b\d+\b', ev_j))
                    # Extract capitalized words (proper nouns: names, locations, etc.)
                    caps_i = set(_re.findall(r'\b[A-Z][a-z]+\b', ev_texts[i]))
                    caps_j = set(_re.findall(r'\b[A-Z][a-z]+\b', ev_texts[j]))
                    # If same words but different numbers → contradiction
                    if overlap > 0.6 and nums_i and nums_j and nums_i != nums_j:
                        contradictions_found += 1
                    # If same structure but different locations/entities in similar positions → contradiction
                    # Only check for location-like contradictions ("in X" vs "in Y", "at X" vs "at Y")
                    loc_pattern = r'\b(?:in|at|from|to)\s+(\w+)\b'
                    locs_i = set(_re.findall(loc_pattern, ev_i))
                    locs_j = set(_re.findall(loc_pattern, ev_j))
                    if overlap > 0.6 and locs_i and locs_j and locs_i != locs_j:
                        contradictions_found += 1
                    elif overlap > 0.6 and not has_neg_i and not has_neg_j:
                        consistency_found += 1
        
        # Decision
        if contradictions_found > 0:
            decision = "refuted" if contradictions_found >= consistency_found else "mixed"
            conf = min(0.95, 0.7 + contradictions_found * 0.05)
            reasoning = f"Found {contradictions_found} contradiction(s) between {len(ev_texts)} evidence items"
        elif consistency_found > 0:
            decision = "supported"
            conf = min(0.9, 0.6 + consistency_found * 0.05)
            reasoning = f"Evidence is consistent ({consistency_found} matching pairs)"
        else:
            decision = "insufficient"
            conf = 0.3
            reasoning = f"Cannot determine contradiction from {len(ev_texts)} evidence items"
        
        trace = ReasoningTrace(
            query=query, input_evidence_count=len(ev_texts),
            center_outputs={"contradiction_fast_path": 1},
            integration_confidence=conf,
            decision=decision, decision_confidence=conf,
            reasoning=reasoning,
            total_latency_ms=lat,
            factors=[{"name": "contradiction_detection", "score": conf,
                      "detail": reasoning}],
        )
        self._traces.append(trace)
        
        return ReasoningResult(
            query=query, decision=decision, confidence=conf,
            reasoning=reasoning,
            explanation_data={"contradictions": contradictions_found, "consistent": consistency_found},
            trace=trace,
            factors=[{"name": "contradiction_detection", "score": conf}],
            memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
        )

    def _try_uncertainty_fast_path(self, query, evidence, t0):
        """
        Fast-path for inherently uncertain queries.
        
        Detects queries that are fundamentally uncertain:
        - Future predictions ("What will happen tomorrow?")
        - Opinion-based ("Is X the best?")
        - Insufficient evidence scenarios
        
        Returns low-confidence result instead of hallucinating certainty.
        """
        q_lower = query.lower()
        
        # Check for inherently uncertain patterns
        is_uncertain = any(kw in q_lower for kw in self._UNCERTAIN_KEYWORDS)
        
        # Check for opinion/best/worst queries
        opinion_patterns = [
            r"\bbest\b.*\bin\b", r"\bworst\b.*\bin\b",
            r"\bmost\b.*\bimportant\b", r"\bleast\b.*\bimportant\b",
            r"\bshould\b.*\b\?\b", r"\bwill\b.*\b\?\b",
            r"\bcould\b.*\b\?\b", r"\bmight\b.*\b\?\b",
        ]
        is_opinion = any(re.search(pat, q_lower) for pat in opinion_patterns)
        
        if not (is_uncertain or is_opinion):
            return None
        
        # If there IS evidence, let the full pipeline handle it
        if evidence:
            return None
        
        lat = (time.perf_counter() - t0) * 1000
        
        # Determine uncertainty level
        # NOTE: confidence here means how confident we are in the ANSWER,
        # not how confident we are that we can't answer.
        # For uncertain queries, confidence should be LOW.
        if any(kw in q_lower for kw in ["tomorrow", "next week", "next month", "future"]):
            uncertainty = 0.15
            level = "uncertain"
            reasoning = "Future predictions cannot be made with certainty"
        elif any(kw in q_lower for kw in ["best", "worst", "most important"]):
            uncertainty = 0.25
            level = "uncertain"
            reasoning = "Opinions and rankings are subjective and context-dependent"
        else:
            uncertainty = 0.20
            level = "uncertain"
            reasoning = "Insufficient evidence to provide a definitive answer"
        
        trace = ReasoningTrace(
            query=query, input_evidence_count=0,
            center_outputs={"uncertainty_fast_path": 1},
            integration_confidence=uncertainty,
            decision="insufficient", decision_confidence=uncertainty,
            reasoning=reasoning,
            total_latency_ms=lat,
            factors=[{"name": "uncertainty_detection", "score": uncertainty,
                      "detail": reasoning}],
        )
        self._traces.append(trace)
        
        return ReasoningResult(
            query=query, decision="insufficient", confidence=uncertainty,
            reasoning=reasoning,
            explanation_data={"uncertainty_level": level},
            trace=trace,
            factors=[{"name": "uncertainty_detection", "score": uncertainty}],
            memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
        )

    def _try_task_router(self, query, evidence, t0):
        """Use the task router for structured logic/math/evidence/temporal/causal tasks."""
        try:
            classification = self._task_router.route(query, evidence)
            if classification.confidence >= 0.7 and classification.answer:
                lat = (time.perf_counter() - t0) * 1000
                # Map task answer to reasoning decision
                ans = classification.answer.lower()
                if ans in ("unknown", "insufficient", "cannot determine", "paradox"):
                    decision = "insufficient"
                elif ans in ("no", "false", "refuted"):
                    decision = "refuted"
                elif ans.startswith("not "):
                    decision = "refuted"
                else:
                    decision = "supported"
                trace = ReasoningTrace(
                    query=query, input_evidence_count=len(evidence),
                    center_outputs={f"task_{classification.category}": 1},
                    integration_confidence=classification.confidence,
                    decision=decision, decision_confidence=classification.confidence,
                    reasoning=f"Task router ({classification.category}/{classification.subcategory}): {classification.answer[:200]}",
                    total_latency_ms=lat,
                    factors=[{"name": f"task_{classification.category}",
                              "score": classification.confidence,
                              "detail": classification.answer[:200]}],
                )
                self._traces.append(trace)
                return ReasoningResult(
                    query=query, decision=decision, confidence=classification.confidence,
                    reasoning=f"Task router ({classification.category}/{classification.subcategory}): {classification.answer[:200]}",
                    explanation_data={"task_classification": classification.category,
                                     "task_method": classification.method,
                                     **classification.details},
                    trace=trace,
                    factors=[{"name": f"task_{classification.category}",
                              "score": classification.confidence}],
                    memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
                )
        except Exception:
            pass
        return None

    def _try_logic_engines(self, query, evidence, t0):
        """Use formal logic engines (proof_mesh + logical_inference) as primary reasoning."""
        ev_texts = []
        for e in evidence:
            if isinstance(e, str):
                ev_texts.append(e)
            elif isinstance(e, dict):
                ev_texts.append(e.get("text", str(e)))

        # ── Try Proof Mesh first (atom/bond grounding + propagation) ──
        try:
            pr = self._proof_mesh.solve(query, ev_texts)
            # "insufficient" is an honest verdict too: the goal is recognised
            # but underdetermined by the evidence (e.g. a transitivity gap), so
            # short-circuit to unknown instead of letting later tiers guess.
            take_pr = (pr.conclusion in ("supported", "refuted", "mixed") and pr.confidence >= 0.60) \
                or (pr.conclusion == "insufficient" and pr.confidence >= 0.20
                    and pr.reasoning and any("not derivable" in line for line in pr.reasoning))
            if take_pr:
                lat = (time.perf_counter() - t0) * 1000
                chain_str = " -> ".join(pr.proof_chain[:5]) if pr.proof_chain else (pr.reasoning[0] if pr.reasoning else "formal logic")
                trace = ReasoningTrace(
                    query=query, input_evidence_count=len(ev_texts),
                    center_outputs={"proof_mesh": 1},
                    integration_confidence=pr.confidence,
                    decision=pr.conclusion, decision_confidence=pr.confidence,
                    reasoning=f"Proof mesh ({pr.conclusion}): {chain_str}",
                    total_latency_ms=lat,
                    factors=[{"name": "proof_mesh", "score": pr.confidence, "detail": chain_str}],
                )
                self._traces.append(trace)
                return ReasoningResult(
                    query=query, decision=pr.conclusion, confidence=pr.confidence,
                    reasoning=f"Proof mesh ({pr.conclusion}): {chain_str}",
                    explanation_data={"proof_chain": pr.proof_chain, "atoms": len(pr.atoms), "bonds": len(pr.bonds)},
                    trace=trace,
                    factors=[{"name": "proof_mesh", "score": pr.confidence}],
                    memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
                )
        except Exception:
            pass

        # ── Try Logical Inference Engine (modus ponens/tollens, transitivity, syllogisms) ──
        try:
            lr = self._logical_engine.analyze(query, ev_texts)
            # Accept "insufficient" when it is a genuine underdetermination
            # verdict (converse error, unreachable target) rather than a parse
            # failure — the engine abstained on purpose.
            take_lr = (lr.conclusion in ("supported", "refuted", "mixed") and lr.confidence >= 0.60) \
                or (lr.conclusion == "insufficient" and lr.confidence >= 0.5
                    and any(k in lr.reasoning for k in
                            ("Converse", "not derivable", "cannot derive", "does not imply")))
            if take_lr:
                lat = (time.perf_counter() - t0) * 1000
                chain_str = " -> ".join(lr.inference_chain[:5]) if lr.inference_chain else lr.reasoning[:200]
                trace = ReasoningTrace(
                    query=query, input_evidence_count=len(ev_texts),
                    center_outputs={"logical_inference": 1},
                    integration_confidence=lr.confidence,
                    decision=lr.conclusion, decision_confidence=lr.confidence,
                    reasoning=f"Logical inference ({lr.conclusion}): {chain_str}",
                    total_latency_ms=lat,
                    factors=[{"name": "logical_inference", "score": lr.confidence, "detail": chain_str}],
                )
                self._traces.append(trace)
                return ReasoningResult(
                    query=query, decision=lr.conclusion, confidence=lr.confidence,
                    reasoning=f"Logical inference ({lr.conclusion}): {chain_str}",
                    explanation_data={"inference_chain": lr.inference_chain},
                    trace=trace,
                    factors=[{"name": "logical_inference", "score": lr.confidence}],
                    memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
                )
        except Exception:
            pass

        return None

    def _try_live_knowledge(self, query, evidence, t0):
        gi_has = (self._general_intelligence.answer(
            query,
            [e if isinstance(e, str) else e.get("text", "") for e in evidence],
        ) is not None)
        if not evidence and not gi_has:
            try:
                live = self.retrieve_live_knowledge(query)
                if live:
                    lat = (time.perf_counter() - t0) * 1000
                    trace = ReasoningTrace(
                        query=query, input_evidence_count=0,
                        center_outputs={"live_knowledge": 1},
                        integration_confidence=0.80,
                        decision="supported", decision_confidence=0.80,
                        reasoning=f"Live knowledge retrieval: {live[:200]}",
                        total_latency_ms=lat,
                        factors=[{"name": "live_knowledge", "score": 0.80,
                                  "detail": live[:200]}],
                    )
                    self._traces.append(trace)
                    return ReasoningResult(
                        query=query, decision="supported", confidence=0.80,
                        reasoning=f"Live knowledge retrieval: {live[:200]}",
                        explanation_data={"live_answer": live},
                        trace=trace,
                        factors=[{"name": "live_knowledge", "score": 0.80}],
                        memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
                    )
            except Exception:
                pass
        return None

    # ════════════════════════════════════════════════════════════
    # INTERNAL — ML PREPROCESSING
    # ════════════════════════════════════════════════════════════

    def _ml_preprocess(self, query, filtered_evidence):
        q_sent = SentimentResult(text=query, label=SentimentLabel.NEUTRAL,
                                 score=0.5, valence=0.0, confidence=0.0, backend="none")
        ev_sents: list[str] = []
        entities: list = []
        q_emb = EmbeddingResult(text=query, vector=None, dim=0, backend="none")

        if self._enable_ml:
            self._ensure_ml_engines()
            if self._sentiment_engine:
                q_sent = self._sentiment_engine.analyze(query)
                for ev in filtered_evidence[:5]:
                    t = ev.get("text", "")
                    if t:
                        ev_sents.append(self._sentiment_engine.analyze(t).label.value)
            if self._ner_engine:
                entities.extend(self._ner_engine.extract(query).entities)
                for ev in filtered_evidence[:2]:
                    t = ev.get("text", "")
                    if t:
                        entities.extend(self._ner_engine.extract(t).entities)
            if self._embedder:
                q_emb = self._embedder.embed(query)

        return q_sent, ev_sents, entities, q_emb

    def _ensure_ml_engines(self):
        if self._ml_loaded:
            return
        self._ml_loaded = True
        try:
            self._embedder = __import__(".".join([".", "semantic_embeddings"]), fromlist=["SemanticEmbedder"]).SemanticEmbedder()
            self._ner_engine = __import__(".".join([".", "ner_engine"]), fromlist=["NEREngine"]).NEREngine()
            self._sentiment_engine = __import__(".".join([".", "sentiment_engine"]), fromlist=["SentimentEngine"]).SentimentEngine()
            self._summarizer = __import__(".".join([".", "text_summarizer"]), fromlist=["TextSummarizer"]).TextSummarizer()
        except Exception as e:
            logger.warning(f"Failed to load ML engines: {e}")

    # ════════════════════════════════════════════════════════════
    # INTERNAL — MEMORY
    # ════════════════════════════════════════════════════════════

    def _recall_memory(self, query):
        recalls = []
        semantic = []
        if self._forebrain.episodic_count > 0:
            recalls = self._forebrain.recall_similar(query)
        if self._forebrain.semantic_count > 0:
            semantic = self._forebrain.get_semantic_knowledge(query)
        return recalls, semantic

    def _record_working_memory(self, cd, final_conf):
        self._forebrain.working_memory.insert(
            slot_type=self._forebrain._MemorySlot.FINDING,
            content={"center": "consensus", "decision": cd.get("decision", "unknown"),
                     "confidence": final_conf},
            priority=0.9,
        )

    def _record_episode(self, query, cd, final_conf, filtered_evidence, evidence_signals):
        key_ev = [s.data.get("evidence_text", "") for s in evidence_signals[:5]
                  if s.data.get("evidence_text")]
        self._forebrain.record_episode(
            query=query, decision=cd.get("decision", "unknown"),
            confidence=final_conf, evidence_count=len(filtered_evidence),
            key_evidence=key_ev,
        )

    # ════════════════════════════════════════════════════════════
    # INTERNAL — FOREBRAIN PIPELINE
    # ════════════════════════════════════════════════════════════

    def _run_forebrain(self, query, evidence, sources, context,
                       filtered_evidence, salience, hb, midbrain_result,
                       semantic_knowledge, memory_recalls):
        """Set up working memory, workspace, run centres, return results."""
        # Working memory
        self._forebrain.working_memory.insert(
            slot_type=self._forebrain._MemorySlot.QUERY,
            content={"query": query, "sources": sources or []}, priority=0.9,
        )
        if semantic_knowledge:
            self._forebrain.working_memory.insert(
                slot_type=self._forebrain._MemorySlot.CONTEXT,
                content={"semantic_knowledge": [s.topic for s in semantic_knowledge[:3]]},
                priority=0.6,
            )
        if memory_recalls:
            self._forebrain.working_memory.insert(
                slot_type=self._forebrain._MemorySlot.CONTEXT,
                content={"episodic_recalls": [r.query for r in memory_recalls[:3]]},
                priority=0.5,
            )

        # Workspace
        self._forebrain.workspace.publish(
            source_center="hindbrain",
            content={"salience": salience, "evidence_count": len(filtered_evidence)},
            salience=salience,
        )
        if midbrain_result.value_predictions:
            avg_v = sum(v["value"] for v in midbrain_result.value_predictions) / len(midbrain_result.value_predictions)
            self._forebrain.workspace.publish(
                source_center="midbrain",
                content={"avg_value_prediction": avg_v}, salience=avg_v,
            )

        # Input signal
        raw_data = {
            "query": query,
            "evidence": [{"text": e.get("text", ""),
                          "source": e.get("source", e.get("_hindbrain_source", ""))}
                         for e in filtered_evidence],
            "sources": sources or [], "context": context or {},
            "_hindbrain_salience": salience,
            "_midbrain_attention": midbrain_result.attention_weights,
            "_semantic_knowledge": [{"topic": s.topic, "pattern": s.pattern}
                                    for s in semantic_knowledge],
            "_episodic_recalls": [{"query": r.query, "decision": r.decision,
                                   "confidence": r.confidence}
                                  for r in memory_recalls],
        }
        input_signal = Signal(data=raw_data, signal_type=SignalType.RAW,
                              confidence=1.0, source_center="sensory_input")

        # Evidence gatherer
        center_outputs: dict[str, list[Signal]] = {}
        all_signals: list[Signal] = [input_signal]
        ev_sigs = self._centers["evidence_gatherer"].process([input_signal])
        center_outputs["evidence_gatherer"] = ev_sigs
        all_signals.extend(ev_sigs)

        if ev_sigs:
            top = max(ev_sigs, key=lambda s: s.confidence)
            self._forebrain.workspace.publish(
                source_center="evidence_gatherer",
                content={"top_evidence": top.data.get("evidence_text", "")[:200],
                         "evidence_count": len(ev_sigs)},
                salience=top.confidence,
            )

        # Parallel centres
        processing = ["credibility_assessor", "temporal_sequencer",
                      "causal_linker", "contradiction_detector"]
        times: dict[str, float] = {}

        def _run(name):
            center = self._centers[name]
            start = time.perf_counter()
            mod = apply_synaptic_input(name, all_signals, self._synapses)
            out = center.process(mod)
            return name, out, (time.perf_counter() - start) * 1000

        with ThreadPoolExecutor(max_workers=min(4, len(processing))) as pool:
            futures = {pool.submit(_run, n): n for n in processing}
            for fut in as_completed(futures):
                name, out, elapsed = fut.result()
                times[name] = elapsed
                center_outputs[name] = out
                all_signals.extend(out)

        # Post-parallel: working memory + plasticity
        for name in processing:
            out = center_outputs.get(name, [])
            if out:
                avg_c = sum(s.confidence for s in out) / len(out)
                self._forebrain.working_memory.insert(
                    slot_type=self._forebrain._MemorySlot.FINDING,
                    content={"center": name, "signal_count": len(out),
                             "avg_confidence": avg_c},
                    priority=avg_c * 0.8,
                )
                self._forebrain.workspace.publish(
                    source_center=name,
                    content={"signal_count": len(out), "avg_confidence": avg_c},
                    salience=avg_c,
                )
                self._plasticity.record_activation(
                    from_center="evidence_gatherer", to_center=name,
                    output_quality=avg_c, processing_time_ms=times.get(name, 0.0),
                )
                hebbian_update(name, out, self._synapses)

        return input_signal, center_outputs, all_signals, times

    # ════════════════════════════════════════════════════════════
    # INTERNAL — AMYGDALA, PROOF MESH, BAYESIAN
    # ════════════════════════════════════════════════════════════

    def _tag_emotional_valence(self, center_outputs):
        for es in center_outputs.get("evidence_gatherer", []):
            ev_text = es.data.get("evidence_text", "")
            if ev_text:
                val = self._amygdala.evaluate(ev_text)
                if val.arousal > 0.5:
                    es.confidence = min(1.0, es.confidence + min(0.15, val.arousal * 0.2))
                    es.data["_amygdala_valence"] = val.valence
                    es.data["_amygdala_arousal"] = val.arousal
                    es.data["_amygdala_category"] = val.category.value
                if val.arousal > 0.6:
                    self._amygdala.encode_emotional_memory(ev_text, val)

    def _apply_proof_mesh(self, query, filtered_evidence, cd):
        ev_texts = [e.get("text", "") for e in filtered_evidence if e.get("text")]
        if not ev_texts:
            return cd
        try:
            pr = self._proof_mesh.solve(query, ev_texts)
            if pr.conclusion in ("supported", "refuted", "mixed"):
                if pr.confidence >= 0.75:
                    cd["decision"] = pr.conclusion
                    old = cd.get("confidence", 0.0)
                    cd["confidence"] = round(0.7 * pr.confidence + 0.3 * old, 4)
                    chain = " -> ".join(pr.proof_chain[:3]) if pr.proof_chain else (
                        pr.reasoning[0] if pr.reasoning else "formal logic")
                    cd["reasoning"] = f"Proof mesh ({pr.conclusion}): {chain}"
                elif pr.confidence >= 0.60 and cd.get("decision") == "insufficient":
                    cd["decision"] = pr.conclusion
                    cd["confidence"] = round(0.6 * pr.confidence + 0.4 * 0.3, 4)
                    cd["reasoning"] = f"Proof mesh rescue: {pr.reasoning[0] if pr.reasoning else pr.conclusion}"
        except Exception:
            pass
        return cd

    def _apply_bayesian(self, evidence_signals, cd):
        if not evidence_signals:
            return cd
        try:
            sup = sum(1 for s in evidence_signals if s.data.get("support_direction") == "supports")
            ref = sum(1 for s in evidence_signals if s.data.get("support_direction") == "refutes")
            total = len(evidence_signals)
            if total > 0 and sup + ref > 0:
                p_s = (sup + 0.1) / (total + 0.2)
                p_r = (ref + 0.1) / (total + 0.2)
                bf = p_s / p_r
                prior = 0.5
                post = (bf * prior) / (bf * prior + (1 - prior))
                old = cd.get("confidence", 0.5)
                cd["confidence"] = round(0.5 * old + 0.5 * post, 4)
        except Exception:
            pass
        return cd

    # ════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════
    # PROPERTIES
    # ════════════════════════════════════════════════════════════

    @property
    def traces(self) -> list[ReasoningTrace]:
        return list(self._traces)

    @property
    def synapse_state(self) -> dict[str, dict[str, Any]]:
        return {k: s.to_dict() for k, s in self._synapses.items()}

    @property
    def brain_stats(self) -> dict[str, Any]:
        learning = self._plasticity.get_metrics()
        return {
            "hindbrain": {"salience_history": len(getattr(self._hindbrain, "_process_history", []))},
            "midbrain": {"attention_history": len(self._midbrain._attention_history)},
            "forebrain": {"episodic_memories": self._forebrain.episodic_count,
                          "semantic_memories": self._forebrain.semantic_count},
            "basal_ganglia": self._basal_ganglia.stats,
            "thalamus": {"relay_count": self._thalamus.relay_count},
            "plasticity": {
                "mastery_phase": learning.overall_phase.value,
                "total_activations": learning.total_activations,
                "total_ltp": learning.total_ltp_events,
                "total_ltd": learning.total_ltd_events,
                "avg_myelination": round(learning.avg_myelination, 4),
                "shortcut_count": learning.shortcut_count,
                "efficiency_gain": round(learning.efficiency_gain, 4),
            },
        }

    def stats(self) -> dict[str, Any]:
        if not self._traces:
            return {"reasoning_passes": 0}
        learning = self._plasticity.get_metrics()
        n = len(self._traces)
        return {
            "reasoning_passes": n,
            "avg_latency_ms": sum(t.total_latency_ms for t in self._traces) / n,
            "decision_breakdown": {
                d: sum(1 for t in self._traces if t.decision == d)
                for d in {t.decision for t in self._traces}
            },
            "synapse_count": len(self._synapses),
            "avg_hindbrain_ms": sum(t.hindbrain_ms for t in self._traces) / n,
            "avg_midbrain_ms": sum(t.midbrain_ms for t in self._traces) / n,
            "avg_forebrain_ms": sum(t.forebrain_ms for t in self._traces) / n,
            "avg_salience": sum(t.salience_score for t in self._traces) / n,
            "total_bg_decisions": sum(t.bg_decisions for t in self._traces),
            "total_memory_recalls": sum(t.memory_recall_count for t in self._traces),
            "mastery_phase": learning.overall_phase.value,
            "total_ltp_events": learning.total_ltp_events,
            "total_ltd_events": learning.total_ltd_events,
            "avg_myelination": round(learning.avg_myelination, 4),
            "efficiency_gain": round(learning.efficiency_gain, 4),
        }
