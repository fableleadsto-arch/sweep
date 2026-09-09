"""Phase 19 — Reproducible Capability Benchmark.

Runs a deterministic battery of known-answer tasks across all phases and
reports the twelve required measurements:

    logical_accuracy, evidence_attribution, contradiction_detection,
    memory_retrieval, hypothesis_ranking, belief_revision,
    uncertainty_calibration, hallucination_error_rate, router_accuracy,
    latency, cpu_usage, ram_usage

Deterministic: seeded inputs, no LLM dependency, output is a JSON report.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .beliefs import BeliefEvidence, BeliefRevisionEngine
from .contradiction import ContradictionEngine
from .hypotheses import EvidenceUpdate, HypothesisEngine
from .logic import LogicEngine
from .loop import CognitiveLoop
from .memory import CognitiveMemory, MemoryKind
from .prediction import Observation, Prediction, PredictionEngine
from .router import Router
from .uncertainty import EpistemicState, EvidenceSignal, UncertaintyEngine
from .verification import VerificationEngine, VerificationStatus


@dataclass
class BenchReport:
    measurements: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"measurements": self.measurements, "extra": self.extra}


def _pct(hit: int, total: int) -> float:
    return round(100.0 * hit / total, 2) if total else 0.0


class BenchRunner:
    """Deterministic, self-contained benchmark for the cognition package."""

    def __init__(self, trials: int = 50):
        self.trials = trials

    # ------------------------------------------------------------ logic
    def _logic_accuracy(self) -> float:
        logic = LogicEngine()
        logic.add_fact("before", "A", "B")
        logic.add_fact("before", "B", "C")
        logic.add_fact("before", "C", "D")
        known_true = [("before", "A", "C"), ("before", "A", "D"),
                      ("before", "B", "D")]
        known_false = [("before", "D", "A"), ("before", "C", "B"),
                       ("north-of", "A", "A")]
        hits = sum(1 for r, s, o in known_true if logic.holds(r, s, o))
        hits += sum(1 for r, s, o in known_false if not logic.holds(r, s, o))
        return _pct(hits, len(known_true) + len(known_false))

    # ------------------------------------------------ contradiction
    def _contradiction_detection(self) -> float:
        eng = ContradictionEngine()
        pairs = [
            # (claim_a, claim_b, expected_contradiction)
            (("c1", "X at A", "x", "located-at", "A"),
             ("c2", "X at B", "x", "located-at", "B"), True),
            (("c3", "gate open", "gate", "is-open", "open"),
             ("c4", "gate closed", "gate", "is-closed", "closed"), True),
            (("c5", "Y at A", "y", "located-at", "A"),
             ("c6", "Y at A", "y", "located-at", "A"), False),
            (("c7", "Z at A 10am", "z", "located-at", "A"),
             ("c8", "Z at B 11am", "z", "located-at", "B"), False),  # diff time
        ]
        hits = 0
        for (a, b, expect) in pairs:
            found = eng.compare(a[0], a[1], a[2], a[3], a[4],
                                b[0], b[1], b[2], b[3], b[4],
                                time_eq=b[0] != "c8")
            hits += 1 if (found is not None) == expect else 0
        return _pct(hits, len(pairs))

    # ------------------------------------------------- memory retrieval
    def _memory_retrieval(self) -> float:
        mem = CognitiveMemory(in_memory=True)
        mem.remember("apollo 11 landed on the moon in 1969", kind=MemoryKind.FACT)
        mem.remember("eleanor roosevelt was first lady", kind=MemoryKind.FACT)
        mem.remember("python is a programming language", kind=MemoryKind.FACT)
        qs = [("moon", "apollo 11 landed"), ("first lady", "eleanor"),
              ("programming", "python")]
        hits = sum(1 for q, expect in qs
                   if any(expect in r.content for r in mem.retrieve(q, n=1)))
        return _pct(hits, len(qs))

    # --------------------------------------------- hypothesis ranking
    def _hypothesis_ranking(self) -> float:
        eng = HypothesisEngine()
        h1 = eng.add_hypothesis("cats are warm-blooded", initial=0.4)
        h2 = eng.add_hypothesis("cats are fish", initial=0.5)
        eng.incorporate(EvidenceUpdate("e1", "mammal evidence", supports=[h1.hypothesis_id],
                                       weight=1.0, strength=1.0))
        eng.incorporate(EvidenceUpdate("e2", "no gills", supports=[h1.hypothesis_id],
                                       weight=1.0, strength=1.0))
        eng.incorporate(EvidenceUpdate("e3", "has scales", opposes=[h2.hypothesis_id],
                                       weight=1.0, strength=1.0))
        top = eng.rank()[0].statement
        return 100.0 if top == "cats are warm-blooded" else 0.0

    # ---------------------------------------------- belief revision
    def _belief_revision(self) -> float:
        eng = BeliefRevisionEngine()
        eng.set_initial("alice did it", 0.9, "a")
        eng.register_candidate("glitch did it", 0.5, "b")
        eng.consider(BeliefEvidence("alibi", supports=False, strength=1.0, hypothesis_target="a"))
        eng.consider(BeliefEvidence("log shows glitch", supports=True, strength=1.0, hypothesis_target="b"))
        ok = (eng.leading_hypothesis() == "glitch did it"
              and eng.count_revisions() >= 1
              and eng.history[0].previous["conclusion"] == "alice did it")
        return 100.0 if ok else 0.0

    # -------------------------------------------- uncertainty calibration
    def _uncertainty_calibration(self) -> float:
        eng = UncertaintyEngine()
        cases = [
            (EvidenceSignal("a", "verified", 3, 0, 3), EpistemicState.VERIFIED),
            (EvidenceSignal("b", "probable", 1, 0, 1), EpistemicState.PROBABLE),
            (EvidenceSignal("c", "possible", 0, 0, 0, 2), EpistemicState.POSSIBLE),
            (EvidenceSignal("d", "contested", 2, 2, 2), EpistemicState.CONTESTED),
            (EvidenceSignal("e", "unresolved", 0, 2), EpistemicState.UNRESOLVED),
            (EvidenceSignal("f", "none", 0, 0, 0), EpistemicState.INSUFFICIENT_EVIDENCE),
        ]
        hits = sum(1 for sig, exp in cases if eng.assess(sig).state is exp)
        return _pct(hits, len(cases))

    # -------------------------------------------------- hallucination
    def _hallucination_rate(self) -> float:
        """What % of empty/ambiguous inputs get wrongly declared VERIFIED?"""
        loop = CognitiveLoop()
        fabrications = 0
        for i in range(self.trials):
            r = loop.run(f"random ambiguous statement {i}")
            if r.verified and r.epistemic_state == "VERIFIED":
                fabrications += 1
        return round(100.0 * fabrications / self.trials, 2)

    # ------------------------------------------------- router accuracy
    def _router_accuracy(self) -> float:
        r = Router()
        cases = [
            ("compute 5 + 3", "arithmetic"),
            ("sum this list", "arithmetic"),
            ("extract text from image", "ocr"),
            ("look at the photo", "vision"),
            ("if A before B and B before C then?", "logic"),
            ("what is the deeper meaning", "complex_reasoning"),
        ]
        hits = sum(1 for task, expect in cases if r.classify(task) == expect)
        return _pct(hits, len(cases))

    # ------------------------------------------- latency + resources
    def _latency(self) -> float:
        loop = CognitiveLoop()
        # warm up
        loop.run("warmup")
        start = time.perf_counter()
        for _ in range(self.trials):
            loop.run("benchmark task")
        elapsed = time.perf_counter() - start
        return round(1000.0 * elapsed / self.trials, 4)   # ms / loop

    def _resources(self) -> Dict[str, float]:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.3) or 0.0
            ram = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            cpu, ram = 0.0, 0.0
        return {"cpu": round(cpu, 1), "ram_mb": round(ram, 1)}

    # ------------------------------------------------ evidence attribution
    def _evidence_attribution(self) -> float:
        """Claim -> Evidence -> Source attribution reconstructs 100%."""
        from cognition.schema import Claim, Evidence, EvidentialRelation
        from cognition.evidence_graph import EvidenceGraph
        from cognition.store import CognitiveStore
        store = CognitiveStore(in_memory=True)
        graph = EvidenceGraph(store=store)
        ev = graph.add_evidence(Evidence(content="sensor reads high", source="sensor-42",
                                         retrieval_time="2026-09-09T10:00:00Z"))
        clm = graph.add_claim(Claim(statement="pressure is high", subject="tank",
                                    predicate="is", object="high"))
        graph.link(clm.id, ev.id, EvidentialRelation.SUPPORTS, confidence=0.95, reason="direct")
        chain = graph.evidence_with_provenance(clm.id)
        if not chain:
            return 0.0
        c = chain[0]
        correct = (c["source"] == "sensor-42"
                   and c["relation"] == "supports"
                   and c["evidence"].content == "sensor reads high"
                   and c["provenance"] == "")
        return 100.0 if correct else 0.0

    # ----------------------------------------------------------- run all
    def run(self) -> BenchReport:
        rep = BenchReport()
        rep.measurements["logical_accuracy"] = self._logic_accuracy()
        rep.measurements["evidence_attribution"] = self._evidence_attribution()
        rep.measurements["contradiction_detection"] = self._contradiction_detection()
        rep.measurements["memory_retrieval"] = self._memory_retrieval()
        rep.measurements["hypothesis_ranking"] = self._hypothesis_ranking()
        rep.measurements["belief_revision"] = self._belief_revision()
        rep.measurements["uncertainty_calibration"] = self._uncertainty_calibration()
        rep.measurements["hallucination_error_rate"] = self._hallucination_rate()
        rep.measurements["router_accuracy"] = self._router_accuracy()
        rep.measurements["latency_ms"] = self._latency()
        res = self._resources()
        rep.measurements["cpu_usage"] = res["cpu"]
        rep.measurements["ram_usage_mb"] = res["ram_mb"]
        rep.extra = {
            "trials": self.trials,
            "python": sys.version.split()[0],
            "platform": sys.platform,
        }
        return rep


def run_benchmark(trials: int = 50) -> BenchReport:
    return BenchRunner(trials=trials).run()


def write_report(report: BenchReport, path: str) -> str:
    """Persist a benchmark report; returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(report.to_dict(), indent=2))
    return path


if __name__ == "__main__":
    rep = run_benchmark()
    print(json.dumps(rep.to_dict(), indent=2))
    write_report(rep, "benchmark/cognition_benchmark.json")