"""Phase 2 — Traceable Evidence Network tests."""

import tempfile
from pathlib import Path

import pytest

from cognition.evidence_graph import EvidenceGraph, Relation, Relationship
from cognition.schema import Claim, Evidence, EvidentialRelation, Entity
from cognition.store import CognitiveStore


def _build_graph(session_store=None):
    store = session_store or CognitiveStore(in_memory=True)
    g = EvidenceGraph(store=store)
    ent = Entity(name="alice", kind="person")
    ent2 = Entity(name="bob", kind="person")
    g.add_entity(ent)
    g.add_entity(ent2)

    ev1 = Evidence(
        content="Badge 4481 used alarm-door at 14:00 UTC (access log)",
        source="access-log",
        retrieval_time="2026-09-08T14:05:00Z",
        entity_ids=[ent.id],
        metadata={"provenance": "camera-2 feed + badge reader, both logged by DC01"},
    )
    ev2 = Evidence(
        content="Badge 4481 is issued to alice per HR directory",
        source="hr-directory",
        entity_ids=[ent.id],
    )
    ev3 = Evidence(
        content="Witness states bob was actually in the server room",
        source="witness-report",
        entity_ids=[ent2.id],
    )
    g.add_evidence(ev1)
    g.add_evidence(ev2)
    g.add_evidence(ev3)

    clm = Claim(
        statement="alice was in the server room at 14:00 UTC",
        subject="alice",
        predicate="located-at",
        object="server room",
        time_context="2026-09-08T14:00:00Z",
        evidence_ids=[ev1.id, ev2.id, ev3.id],
        entity_ids=[ent.id],
    )
    g.add_claim(clm)
    g.link(clm.id, ev1.id, EvidentialRelation.SUPPORTS, 0.9, "badge + camera agree")
    g.link(clm.id, ev2.id, EvidentialRelation.SUPPORTS, 0.8, "HR confirms badge owner")
    g.link(clm.id, ev3.id, EvidentialRelation.REFUTES, 0.6, "eyewitness contradicts")

    g.add_relationship(Relationship(subject_id=ent.id, predicate="associated-with", object_id="server-room-7",
                                    source="access-log"))
    g.add_relationship(Relationship(subject_id=ent.id, predicate="supervises", object_id=ent2.id,
                                    source="hr-directory"))
    return g, ent, ent2, clm, ev1, ev2, ev3


def test_entity_to_claim_to_evidence_traversal():
    g, ent, ent2, clm, ev1, ev2, ev3 = _build_graph()
    result = g.reconstruct_from_entity(ent.id)
    assert len(result["claims"]) == 1
    claim_blob = result["claims"][0]
    assert claim_blob["claim"]["id"] == clm.id
    assert len(claim_blob["supporting"]) == 2
    assert len(claim_blob["contradicting"]) == 1
    assert claim_blob["contradicting"][0]["content"] == ev3.content


def test_provenance_preserved_in_chain():
    g, ent, ent2, clm, ev1, ev2, ev3 = _build_graph()
    chain = g.evidence_with_provenance(clm.id)
    sources = {c["source"] for c in chain}
    assert sources == {"access-log", "hr-directory", "witness-report"}
    access = [c for c in chain if c["source"] == "access-log"][0]
    assert access["content_hash"]
    assert "camera-2 feed" in access["provenance"]
    assert access["relation"] == "supports"


def test_supporting_and_contradicting_split():
    g, ent, ent2, clm, ev1, ev2, ev3 = _build_graph()
    sup = g.supporting_evidence(clm.id)
    con = g.contradicting_evidence(clm.id)
    assert {e.id for e in sup} == {ev1.id, ev2.id}
    assert {e.id for e in con} == {ev3.id}


def test_relationship_traversal():
    g, ent, ent2, clm, ev1, ev2, ev3 = _build_graph()
    rels = g.relationship_chain(ent.id)
    assert any(r.predicate == "associated-with" for r in rels)
    assert any(r.predicate == "supervises" for r in rels)


def test_persistence_across_graph_instances():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "store"
        store1 = CognitiveStore(path=path)
        g1, ent, ent2, clm, ev1, ev2, ev3 = _build_graph(store1)

        store2 = CognitiveStore(path=path)
        g2 = EvidenceGraph(store=store2)
        # reloaded relations
        rels = g2.relations_for(clm.id)
        assert len(rels) == 3
        # entity/claim/evidence objects reloaded
        assert g2.store.get(clm.id).statement == clm.statement
        reconstructed = g2.reconstruct_from_entity(ent.id)
        assert len(reconstructed["claims"]) == 1
        assert len(reconstructed["claims"][0]["supporting"]) == 2


def test_relation_serialization_roundtrip():
    r = Relation("c1", "e1", EvidentialRelation.SUPPORTS, 0.84, "reason text")
    d = r.to_dict()
    r2 = Relation.from_dict(d)
    assert r2.relation == EvidentialRelation.SUPPORTS
    assert r2.confidence == 0.84
    assert r2.reason == "reason text"