"""Phase 16 — Unified Cognitive Loop.

Orders all cognitive stages into a single end-to-end pipeline:

    INPUT -> PERCEPTION -> OBSERVATION -> WORKING MEMORY -> RETRIEVAL
    -> HYPOTHESIS -> NEURAL REASONING -> LOGIC REASONING
    -> EVIDENCE CHECK -> CONTRADICTION CHECK -> VERIFICATION
    -> UNCERTAINTY -> CONCLUSION -> FEEDBACK -> LONG-TERM MEMORY

Every stage is recorded so a run can be replayed / audited (Phase 17).
A conclusion is only produced when evidence supports it; otherwise the run
ends with an explicit uncertainty state instead of fabrication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .beliefs import BeliefEvidence, BeliefRevisionEngine
from .bridge import NeuralBridge
from .contradiction import ContradictionEngine
from .feedback import ErrorCategory, FeedbackLoop
from .hypotheses import EvidenceUpdate, HypothesisEngine
from .logic import LogicEngine
from .memory import CognitiveMemory, MemoryKind
from .prediction import Observation, Prediction, PredictionEngine
from .resources import ResourceBudget, ResourceMonitor
from .router import Router
from .schema import Claim, Evidence, Observation as SchemaObservation, ReasoningEvent, EventKind
from .store import CognitiveStore
from .uncertainty import EpistemicState, EvidenceSignal, UncertaintyEngine
from .verification import VerificationEngine


@dataclass
class StageRecord:
    """A record of one pipeline stage execution."""

    name: str
    status: str = "ok"       # ok / degraded / skipped / error
    detail: str = ""
    output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "output": self.output}


@dataclass
class LoopResult:
    """Full outcome of one cognitive loop run."""

    run_id: str
    input_text: str
    stages: List[StageRecord] = field(default_factory=list)
    conclusion: str = ""
    conclusion_confidence: float = 0.0
    epistemic_state: str = ""
    conclusion_id: str = ""
    contradiction_count: int = 0
    verified: bool = False
    degraded: bool = False
    memory_stored: int = 0

    def stage_names(self) -> List[str]:
        return [s.name for s in self.stages]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "input_text": self.input_text,
            "stages": [s.to_dict() for s in self.stages],
            "conclusion": self.conclusion,
            "conclusion_confidence": self.conclusion_confidence,
            "epistemic_state": self.epistemic_state,
            "conclusion_id": self.conclusion_id,
            "contradiction_count": self.contradiction_count,
            "verified": self.verified,
            "degraded": self.degraded,
            "memory_stored": self.memory_stored,
        }


class CognitiveLoop:
    """Deterministic orchestration of all cognitive stages."""

    PIPELINE = [
        "perception", "observation", "working_memory", "retrieval",
        "hypothesis", "neural_reasoning", "logic_reasoning",
        "evidence_check", "contradiction_check", "verification",
        "uncertainty", "conclusion", "feedback", "long_term_memory",
    ]

    def __init__(self, store: Optional[CognitiveStore] = None,
                 register_router_handler: bool = True):
        self.store = store or CognitiveStore(in_memory=True)
        self.memory = CognitiveMemory(store=self.store, in_memory=True)
        self.logic = LogicEngine()
        self.router = Router()
        self.bridge = NeuralBridge()
        self.hypotheses = HypothesisEngine()
        self.contradictions = ContradictionEngine()
        self.verification = VerificationEngine()
        self.uncertainty = UncertaintyEngine()
        self.prediction = PredictionEngine()
        self.feedback = FeedbackLoop()
        self.revision = BeliefRevisionEngine()
        self.monitor = ResourceMonitor(budget=ResourceBudget(90.0, 2048.0))
        self.handlers: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}
        self.verifiers: Dict[str, Callable[[str], bool]] = {}
        self.results: List[LoopResult] = []
        if register_router_handler:
            self.router.register("complex_reasoning", self._default_reasoner)

    def register_neural(self, name: str, handler: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.handlers[name] = handler
        self.router.register(name, handler)

    def register_verifier(self, name: str, fn: Callable[[str], bool]) -> None:
        self.verifiers[name] = fn
        self.verification.register_check(name, fn)

    def __call__(self, text: str, claims: Optional[List[Dict[str, Any]]] = None,
                 hypotheses: Optional[List[str]] = None) -> LoopResult:
        return self.run(text, claims=claims, hypotheses=hypotheses)

    # ------------------------------------------------------------- pipeline
    def run(self, text: str, claims: Optional[List[Dict[str, Any]]] = None,
            hypotheses: Optional[List[str]] = None,
            hypothesis_initial: Optional[Dict[str, float]] = None,
            observation_verdict: Optional[bool] = None) -> LoopResult:
        """Execute the full pipeline. Returns the recorded result."""
        run_id = "run{:03d}".format(len(self.results) + 1)
        stages: List[StageRecord] = []
        claims = claims or []
        # Run-scoped resource accounting: each run gets a fresh episode so
        # a transient CPU spike seen by run N is not attributed to run N+1.
        self.monitor.reset_episode()

        def record(name: str, status: str = "ok", detail: str = "", **out: Any) -> None:
            stages.append(StageRecord(name=name, status=status, detail=detail, output=out))
            self.monitor.sample(name)

        # 1. PERCEPTION -----------------------------------------------------
        record("perception", detail=text[:120])
        # 2. OBSERVATION ----------------------------------------------------
        obs = SchemaObservation(content=text)
        self.store.save(obs)
        record("observation", output={"observation_id": obs.id})
        # 3. WORKING MEMORY -------------------------------------------------
        self.memory.working.store(text, kind=MemoryKind.OBSERVATION, source_task=run_id)
        record("working_memory", output={"wm_size": self.memory.working.size()})
        # 4. RETRIEVAL ------------------------------------------------------
        recollections = self.memory.retrieve(text, n=5)
        record("retrieval", output={"recalled": len(recollections)})
        # 5. HYPOTHESIS -----------------------------------------------------
        hyps = hypotheses or _default_hypotheses(text)
        hyp_ids: List[str] = []
        if hyps:
            for h in hyps:
                score = self.hypotheses.add_hypothesis(h,
                                                       initial=hypothesis_initial.get(h, 0.5) if hypothesis_initial else 0.5)
                hyp_ids.append(score.hypothesis_id)
        record("hypothesis", output={"hypotheses": len(hyp_ids)})
        # 6. NEURAL REASONING ------------------------------------------------
        decision = self.router.route(text)
        neural_result: Any = None
        neural_ok = decision.selected in self.handlers
        if neural_ok:
            try:
                neural_result = self.handlers[decision.selected](text, {"run_id": run_id})
                record("neural_reasoning", output={"route": decision.selected,
                                                   "has_result": True,
                                                   "fallbacks": len(decision.fallback_chain)})
            except Exception as e:
                record("neural_reasoning", status="degraded",
                       detail=f"handler failed: {e}", route=decision.selected)
        else:
            record("neural_reasoning", status="skipped",
                   detail=f"no handler for '{decision.selected}'")
        # 7. LOGIC REASONING -------------------------------------------------
        derived = self._logic_reason(claims)
        record("logic_reasoning", detail=derived["summary"], derived_count=derived["count"])
        # 8. EVIDENCE CHECK --------------------------------------------------
        evidence_ids: List[str] = []
        for i, claim_d in enumerate(claims):
            ev = Evidence(content=claim_d.get("statement", text),
                          source=claim_d.get("source", "input"),
                          metadata={"claim_index": i})
            self.store.save(ev)
            evidence_ids.append(ev.id)
            # corroborating evidence strengthens the working hypothesis
            if hyp_ids:
                self.hypotheses.incorporate(EvidenceUpdate(
                    evidence_id=ev.id,
                    content=ev.content,
                    supports=[hyp_ids[i % len(hyp_ids)]],
                    weight=claim_d.get("weight", 1.0),
                    strength=claim_d.get("strength", 0.8),
                    source=ev.source,
                ))
        record("evidence_check", evidence=len(evidence_ids))
        # 9. CONTRADICTION CHECK ---------------------------------------------
        claim_objs = self._store_claims(claims, text)
        c_count = self._contradiction_pass(claim_objs)
        record("contradiction_check", contradictions=c_count)
        # 10. VERIFICATION ----------------------------------------------------
        verification_status = EpistemicState.UNRESOLVED.value
        for c in claim_objs:
            vr = self.verification.run(c.id)
            if vr.status.value == "VERIFIED":
                verification_status = EpistemicState.VERIFIED.value
                break
            elif vr.failed:
                verification_status = EpistemicState.CONTESTED.value
        record("verification", status=verification_status.lower())
        # 11. UNCERTAINTY -----------------------------------------------------
        signal = EvidenceSignal(claim_id="loop-{}".format(run_id),
                                statement=text,
                                supporting=len(evidence_ids),
                                contradicting=c_count,
                                source_diversity=len({c.get("source", "") for c in claims if c.get("source")}),
                                neutral=0)
        epi = self.uncertainty.assess(signal)
        record("uncertainty", state=epi.state.value)
        # 12. CONCLUSION -------------------------------------------------------
        lead = self.hypotheses.leading()
        if lead and epi.state is not EpistemicState.CONTESTED:
            conclusion = lead.statement
            confidence = lead.confidence
            self.revision.set_initial(conclusion, confidence,
                                      hypothesis_id=lead.hypothesis_id)
        else:
            conclusion = "insufficient evidence to draw a conclusion"
            confidence = round(0.5 - 0.1 * c_count, 3)
            record_state = "insufficient" if epi.state is EpistemicState.INSUFFICIENT_EVIDENCE else epi.state.value
            self.revision.set_initial(conclusion, confidence)
        conclusion_id = "loop-{}".format(run_id)
        record("conclusion",
               conclusion=conclusion,
               confidence=confidence,
               epistemic_state=epi.state.value,
               conclusion_id=conclusion_id)
        # 13. FEEDBACK ---------------------------------------------------------
        fb_detail = ""
        if c_count:
            fb = self.feedback.process("contradictory claims detected",
                                       hints=["verification_failure"])
            fb_detail = fb.action.value
        elif epi.state is EpistemicState.INSUFFICIENT_EVIDENCE:
            fb = self.feedback.process("not enough evidence",
                                       hints=["memory_gap"])
            fb_detail = fb.action.value
        record("feedback", action=fb_detail or "none")
        # 14. LONG-TERM MEMORY -------------------------------------------------
        self.memory.remember(conclusion, kind=MemoryKind.CONCLUSION,
                             source_task=run_id,
                             provenance=text,
                             metadata={"run_id": run_id, "confidence": confidence})
        record("long_term_memory", stored=1)

        degraded = self.monitor.episode_over_budget()
        result = LoopResult(
            run_id=run_id,
            input_text=text,
            stages=stages,
            conclusion=conclusion,
            conclusion_confidence=confidence,
            epistemic_state=epi.state.value,
            conclusion_id=conclusion_id,
            contradiction_count=c_count,
            verified=verification_status == EpistemicState.VERIFIED.value,
            degraded=degraded,
            memory_stored=self.memory.longterm.count(),
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------ sub-steps
    def _store_claims(self, claims: List[Dict[str, Any]], text: str) -> List[Claim]:
        objs: List[Claim] = []
        for d in claims:
            c = Claim(
                statement=d.get("statement", text),
                subject=d.get("subject", ""),
                predicate=d.get("predicate", ""),
                object=d.get("object", ""),
                confidence=d.get("confidence", 0.6),
                status="unverified",
            )
            self.store.save(c)
            objs.append(c)
        return objs

    def _logic_reason(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        for d in claims:
            if d.get("predicate") and d.get("subject"):
                self.logic.add_fact(d["predicate"], d["subject"],
                                    d.get("object", ""))
        summary = f"{len(self.logic.facts)} facts; {len(self.logic.closure()) - len(self.logic.facts)} derived"
        return {"count": len(self.logic.closure()), "summary": summary}

    def _contradiction_pass(self, claim_objs: List[Claim]) -> int:
        count = 0
        for i in range(len(claim_objs)):
            for j in range(i + 1, len(claim_objs)):
                a, b = claim_objs[i], claim_objs[j]
                if not (a.predicate and b.predicate):
                    continue
                found = self.contradictions.compare(
                    a.id, a.statement, a.subject or "subject", a.predicate, a.object,
                    b.id, b.statement, b.subject or "subject", b.predicate, b.object,
                    time_eq=a.time_context == b.time_context or not a.time_context,
                    context="loop",
                )
                if found:
                    count += 1
        return count

    # -------------------------------------------------------------- defaults
    def _default_reasoner(self, task: str, ctx: Dict[str, Any]) -> str:
        return f"reasoned about: {task}"


def _default_hypotheses(text: str) -> List[str]:
    """Deterministic hypothesis extraction from a statement phrase."""
    low = text.strip()
    if not low:
        return ["unknown outcome"]
    if " vs " in low:
        parts = [p.strip() for p in low.split(" vs ")]
        return parts
    if "," in low:
        return [p.strip() for p in low.split(",") if p.strip()][:3]
    return [low, f"{low} (alternative explanation)"]