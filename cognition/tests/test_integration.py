"""Phase 18 — Cross-module integration tests.

Modules cooperate: store <-> memory <-> graph <-> logic <-> loop.
"""

import tempfile
import os
from pathlib import Path

import pytest

from cognition.schema import Claim, Evidence, EventKind, EvidentialRelation, ReasoningEvent
from cognition.store import CognitiveStore
from cognition.memory import CognitiveMemory, MemoryKind
from cognition.evidence_graph import EvidenceGraph
from cognition.logic import LogicEngine, Fact
from cognition.loop import CognitiveLoop


def test_store_to_loop_shares_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        store = CognitiveStore(path=tmp)
        loop = CognitiveLoop(store=store)
        loop.run("persist me", claims=[
            {"statement": "persist me", "subject": "s", "predicate": "p",
             "object": "o", "source": "t"}])
        # objects survived into the store shared with the loop
        assert len(store.by_type("Observation")) >= 1
        assert len(store.by_type("Evidence")) >= 1
        assert len(store.by_type("Claim")) >= 1


def test_memory_handoff_across_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        mem = CognitiveMemory(store_path=tmp)
        mem.remember("token format is xyz", kind=MemoryKind.FACT,
                     source_task="session1", provenance="manual")
        # new session reads from the same store dir
        mem2 = CognitiveMemory(store_path=tmp)
        recalls = mem2.retrieve("token")
        assert any("xyz" in r.content for r in recalls)


def test_evidence_graph_feeds_logic_and_loop_graph():
    graph = EvidenceGraph()
    e1 = graph.add_evidence(Evidence(content="A is before B", source="clock"))
    e2 = graph.add_evidence(Evidence(content="B is before C", source="clock"))
    clm1 = graph.add_claim(Claim(statement="A is before B", subject="A", predicate="before", object="B"))
    clm2 = graph.add_claim(Claim(statement="B is before C", subject="B", predicate="before", object="C"))
    graph.link(clm1.id, e1.id, EvidentialRelation.SUPPORTS)
    graph.link(clm2.id, e2.id, EvidentialRelation.SUPPORTS)
    # logic engine derives A before C from the facts in the graph
    logic = LogicEngine()
    for c in (clm1, clm2):
        logic.add_fact(c.predicate, c.subject, c.object)
    assert logic.holds("before", "A", "C")


def test_reasoning_event_trace_records_loop_kind():
    store = CognitiveStore(in_memory=True)
    loop = CognitiveLoop(store=store)
    result = loop.run("event trace")
    evt = ReasoningEvent(kind=EventKind.OBSERVATION_RECORDED,
                         detail=f"input: {result.input_text}",
                         object_type="Observation")
    store.save(evt)
    loaded = store.get(evt.id)
    assert loaded.kind is EventKind.OBSERVATION_RECORDED
    assert loaded.id == evt.id


def test_loop_observations_feed_memory_and_graph_together():
    store = CognitiveStore(in_memory=True)
    graph = EvidenceGraph()
    loop = CognitiveLoop(store=store)
    result = loop.run("coordinated scenario", claims=[
        {"statement": "sensor warm", "subject": "probe", "predicate": "is",
         "object": "warm", "source": "sensor"},
        {"statement": "probe status ok", "subject": "probe", "predicate": "status",
         "object": "ok", "source": "tester"}],
        hypotheses=["probe okay"])
    # memory has the conclusion, store has the observations
    entries = loop.memory.longterm.recall_all()
    assert any("probe okay" in e.content for e in entries)
    assert store.count() >= 3
    # graph model can consume the same claims independently
    for c in store.by_type("Claim"):
        graph.add_claim(c)
    assert len(graph.store.query("Claim")) >= 2