"""Phase 4 — Deterministic Reasoning Core tests."""

import pytest

from cognition.logic import Fact, LogicEngine, ProofStep


def test_transitivity_before():
    le = LogicEngine()
    le.add_many([("before", "A", "B"), ("before", "B", "C")])
    assert le.holds("before", "A", "C")
    assert le.holds("before", "A", "B")
    assert not le.holds("before", "C", "A")


def test_milestone_diagram_executes():
    le = LogicEngine()
    le.add(Fact("before", "A", "B"))
    le.add(Fact("before", "B", "C"))
    #   A before B
    #   B before C
    #   ------------
    #   A before C
    proof = le.derive("before", "A", "C")
    assert proof is not None
    assert any("transitivity" in step.rule for step in proof)


def test_repeat_execution_deterministic():
    le = LogicEngine()
    le.add_many([("before", "A", "B"), ("before", "B", "C"), ("before", "C", "D"),
                 ("after", "D", "C"), ("after", "C", "B")])
    r1 = sorted(le.closure())
    for _ in range(20):
        le2 = LogicEngine()
        le2.add_many([("before", "A", "B"), ("before", "B", "C"), ("before", "C", "D"),
                      ("after", "D", "C"), ("after", "C", "B")])
        assert sorted(le2.closure()) == r1


def test_symmetric_relations_infer_both_directions():
    le = LogicEngine()
    le.add(Fact("adjacent-to", "room-1", "room-2"))
    assert le.holds("adjacent-to", "room-2", "room-1")


def test_nontransitive_relations_do_not_close_incorrectly():
    le = LogicEngine(transitive_relations=set(), symmetric_relations=set())
    le.add_many([("works-at", "alice", "acme"), ("works-at", "acme", "build-4")])
    assert not le.holds("works-at", "alice", "build-4")


def test_derivation_for_given_deep_chain():
    le = LogicEngine()
    le.add_many([("before", "P1", "P2"), ("before", "P2", "P3"), ("before", "P3", "P4")])
    proof = le.derive("before", "P1", "P4")
    assert proof
    # the proof is a chain of pairwise transitivity steps
    assert any(p.conclusion == "before(P1, P3)" for p in proof)
    assert any(p.conclusion == "before(P1, P4)" for p in proof)


def test_proof_step_serialization():
    ps = ProofStep(conclusion="before(A, C)", premises=["before(A, B)", "before(B, C)"], rule="transitivity")
    d = ps.to_dict()
    assert d["conclusion"] == "before(A, C)"
    assert d["rule"] == "transitivity"