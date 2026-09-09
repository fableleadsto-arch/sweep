"""Phase 3 — Persistent Cognitive Memory tests."""

import tempfile
from pathlib import Path

import pytest

from cognition.memory import CognitiveMemory, LongTermMemory, MemoryEntry, MemoryKind, WorkingMemory
from cognition.store import CognitiveStore


def test_working_memory_store_and_retrieve():
    wm = WorkingMemory(capacity=4)
    wm.store("alice badge 4481", key="badge")
    wm.store("alice in server room", key="location")
    e = wm.retrieve("badge")
    assert e.content == "alice badge 4481"
    assert e.retrievals == 1
    assert len(wm.recent(10)) == 2


def test_working_memory_capacity_evicts_oldest():
    wm = WorkingMemory(capacity=2)
    wm.store("one", key="1")
    wm.store("two", key="2")
    wm.store("three", key="3")
    assert wm.size() == 2
    assert wm.retrieve("1") is None
    assert wm.retrieve("3") is not None


def test_longterm_persists_across_sessions():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memstore"
        s1 = CognitiveStore(path=path)
        ltm1 = LongTermMemory(store=s1)
        ltm1.remember("fan blade 7 detached at 14:00", kind=MemoryKind.EVENT, source_task="inspection",
                      provenance="video-feed")
        ltm1.remember("fan RPM normal all morning", kind=MemoryKind.OBSERVATION, source_task="inspection")

        # simulate session end + new session
        s2 = CognitiveStore(path=path)
        ltm2 = LongTermMemory(store=s2)
        assert ltm2.count() == 2
        results = ltm2.retrieve("badge? fan blade", n=5)
        assert any("fan blade 7" in r.content for r in results)


def test_retrieval_retains_provenance_and_history():
    ltm = LongTermMemory(in_memory=True)
    ltm.remember("the suspect wore a red jacket", kind=MemoryKind.FACT, source_task="investigation-42",
                 provenance="witness-statement-03")
    results = ltm.retrieve("red jacket", n=10)
    assert len(results) == 1
    e = results[0]
    assert e.source_task == "investigation-42"
    assert e.provenance == "witness-statement-03"
    assert e.retrievals == 1
    assert e.last_retrieved_at is not None


def test_memory_independent_of_context_window():
    """Store a memory, end session, new session retrieves and uses it."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "store"
        mem1 = CognitiveMemory(store=CognitiveStore(path=path))
        mem1.observe("container B-3 unsealed at 22:10", provenance="camera-17")
        # Session ends (new object, same store path)
        mem2 = CognitiveMemory(store=CognitiveStore(path=path))
        recalled = mem2.retrieve("unsealed", n=5)
        assert any("B-3" in r.content for r in recalled)
        recalled_entry = [r for r in recalled if "B-3" in r.content][0]
        assert recalled_entry.provenance == "camera-17"


def test_kind_filtered_retrieval():
    ltm = LongTermMemory(in_memory=True)
    ltm.remember("conclusion: door lock failing", kind=MemoryKind.CONCLUSION, source_task="t")
    ltm.remember("door opened slowly", kind=MemoryKind.OBSERVATION, source_task="t")
    facts = ltm.retrieve("", kind=MemoryKind.CONCLUSION)
    assert len(facts) == 1
    assert facts[0].content == "conclusion: door lock failing"
    obs = ltm.retrieve("", kind=MemoryKind.OBSERVATION)
    assert len(obs) == 1


def test_memory_entry_serialization_roundtrip():
    e = MemoryEntry(content="x", kind=MemoryKind.EVENT, source_task="t", provenance="p", evidence_ids=["ev1"])
    d = e.to_dict()
    e2 = MemoryEntry.from_dict(d)
    assert e2.id == e.id
    assert e2.kind == MemoryKind.EVENT
    assert e2.evidence_ids == ["ev1"]