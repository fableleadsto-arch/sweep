"""Phase 18 — Regression (known-answer) tests.

These lock in behaviors agreed in earlier phases so later work never
silently changes them.
"""

import pytest

from cognition.hypotheses import EvidenceUpdate, HypothesisEngine
from cognition.logic import LogicEngine
from cognition.memory import CognitiveMemory
from cognition.uncertainty import EpistemicState, EvidenceSignal, UncertaintyEngine


def test_regression_leading_hypothesis_gate():
    # Phase 8 gate: confidence >= 0.6 AND uncertainty <= 0.5 required
    eng = HypothesisEngine()
    h = eng.add_hypothesis("slow and weak hypothesis", initial=0.5)
    eng.incorporate(EvidenceUpdate("e1", "small support", supports=[h.hypothesis_id],
                                   weight=0.2, strength=0.5))
    # confidence 0.5 -> 0.52 < 0.6 so no conclusion forced
    assert eng.leading() is None
    assert eng.no_conclusion() is True


def test_regression_logic_transitivity_known_answer():
    logic = LogicEngine()
    logic.add_fact("before", "A", "B")
    logic.add_fact("before", "B", "C")
    assert logic.holds("before", "A", "C")          # deterministic closure


def test_regression_epistemic_state_known_mapping():
    eng = UncertaintyEngine()
    # supporting>=2 across >=2 sources and no opposition -> VERIFIED
    assert eng.assess(EvidenceSignal("x", "x", 3, 0, 3)).state is EpistemicState.VERIFIED
    # nothing -> INSUFFICIENT_EVIDENCE
    assert eng.assess(EvidenceSignal("y", "y", 0, 0, 0)).state is EpistemicState.INSUFFICIENT_EVIDENCE


def test_regression_memory_handoff_zero_loss():
    from cognition.store import CognitiveStore
    from cognition.memory import MemoryKind
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m1 = CognitiveMemory(store= CognitiveStore(path=tmp))
        m1.remember("known constant pie=3.14159", kind=MemoryKind.FACT)
        m2 = CognitiveMemory(store_path=tmp)
        recalled = m2.retrieve("pie")
        assert any("3.14159" in r.content for r in recalled)


def test_regression_wm_capacity_eviction():
    from cognition.memory import WorkingMemory
    wm = WorkingMemory(capacity=3)
    for i in range(5):
        wm.store(f"item {i}")
    assert wm.size() == 3
    assert wm.recent()[-1].content == "item 4"