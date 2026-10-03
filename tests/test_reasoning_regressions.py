import time

from sweep_cognitive.perception.document import StructuredDataPerceptionProcessor
from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.neurons.multi_core import CausalCore, TemporalCore


def test_empty_structured_input_returns_low_confidence():
    result = StructuredDataPerceptionProcessor().process("")
    assert result.confidence.overall == 0.0
    assert not result.representation.content


def test_pattern_based_cores_return_explanations():
    temporal = TemporalCore().process("When was the first moon landing?", [])
    causal = CausalCore().process("Why does it rain?", [])
    assert temporal.answer == "1969" and temporal.reasoning
    assert causal.answer and causal.reasoning


def test_uncertain_fast_path_abstains_without_loading_models():
    # Exercise the deterministic route directly, without model initialization.
    cortex = ReasoningCortex.__new__(ReasoningCortex)
    cortex._traces = []
    result = cortex._try_uncertainty_fast_path("What will happen tomorrow?", [], time.perf_counter())
    assert result.decision == "insufficient"
    assert result.confidence < 0.5
    assert cortex._traces
