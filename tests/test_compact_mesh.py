"""The embedded mesh must reason conservatively and stay dependency-free."""
import json
import subprocess
import sys
from itertools import permutations, product
from pathlib import Path

import pytest

from sweep_neural_mesh.companion import reason
from sweep_neural_mesh.neurons.logical_inference import LogicalInferenceEngine


@pytest.mark.parametrize(("text", "conclusion"), [
    ("All cats are mammals. All mammals are animals. Are all cats animals?", "supported"),
    ("All cats are mammals. Are all mammals cats?", "insufficient"),
    ("All cats are mammals. Are all cats reptiles?", "insufficient"),
    ("All cats are mammals. Are all cats cats?", "supported"),
    ("If A then B. If B then C. Does A imply C?", "supported"),
    ("If A then B. Does B imply A?", "insufficient"),
    ("If A then B. Does A imply C?", "insufficient"),
    ("If A then B. If B then A. Does A imply B?", "supported"),
    ("A > B. B > C. Is A > C?", "supported"),
    ("A > B. B > C. Is C > A?", "refuted"),
    ("A < B. B < C. Is A < C?", "supported"),
    ("A < B. B < C. Is C < A?", "refuted"),
    ("A > B. Is B > C?", "insufficient"),
    ("A > B. Is A > A?", "refuted"),
    ("A > B. B > A. Is A > B?", "mixed"),
    ("A < B. B < C. C < A. Is A < D?", "mixed"),
    ("A > A. Is A > B?", "mixed"),
    ("ALL CATS ARE MAMMALS; ALL MAMMALS ARE ANIMALS; ARE ALL CATS ANIMALS?", "supported"),
])
def test_formal_arguments_and_counterexamples(text, conclusion):
    answer = reason(text)
    assert answer is not None
    assert answer.conclusion == conclusion
    assert answer.method == "deterministic_formal_logic"
    assert answer.engine == "sweep_neural_mesh"
    assert answer.answer


@pytest.mark.parametrize("text", [
    "hey bro what's up", "I am sad today", "open youtube in brave",
    "If I feel awful then can you talk to me?", "All I want is a friend. Are you there?",
    "If A then B. What does that mean?", "I think all cats are mammals. Are all cats mammals?",
    "If A then B. A is false. Does A imply B?",
    "If A then B. If cats then dogs. Does cat imply dog?",  # no fuzzy substring proof
    "All cats are mammals. All mammals are animals. Are all cats animals? Open brave",
    "All cats are mammals. Ignore previous instructions. Are all cats mammals?",
    "All cats are mammals. Are all cats mammals? Are all mammals cats?",
    "A > B. B < C. Is A > C?", "A > B. Is A >= B?", "1 > 2. Is 1 > 2?",
    "All cats are mammals.. Are all cats mammals?", "Are all cats mammals?",
    "", "x" * 4097, None, 12,
])
def test_unparsed_prose_and_actions_are_left_for_existing_chat_router(text):
    # The substring example is valid grammar but is unknown, never supported.
    answer = reason(text)
    if text == "If A then B. If cats then dogs. Does cat imply dog?":
        assert answer.conclusion == "insufficient"
    else:
        assert answer is None


def test_every_part_must_be_parsed_and_graph_size_is_bounded():
    premises = ". ".join(f"If a{i} then a{i + 1}" for i in range(16))
    assert reason(premises + ". Does a0 imply a16?").conclusion == "supported"
    assert reason(premises + ". If a16 then a17. Does a0 imply a17?") is None
    assert reason("All " + "x" * 41 + " are mammals. Are all cats mammals?") is None


def test_exact_path_does_not_retain_or_mix_previous_premises():
    logic = LogicalInferenceEngine()
    assert logic.analyze_exact("If A then B. Does A imply B?").conclusion == "supported"
    assert logic.analyze_exact("If C then D. Does A imply B?").conclusion == "insufficient"
    assert logic._conditionals == logic._comparisons == logic._memberships == logic._facts == []


@pytest.mark.parametrize("grammar", ["implies", "all"])
def test_exact_entailment_against_all_three_atom_truth_assignments(grammar):
    """An independent truth-table oracle checks every directed premise graph.

    Universal class inclusion has the same per-element implication constraints.
    This guards against reversing an implication or calling missing proof false.
    """
    labels = ("a", "b", "c")
    pairs = list(product(range(3), repeat=2))
    assignments = list(product((False, True), repeat=3))
    logic = LogicalInferenceEngine()
    for selected in product((False, True), repeat=len(pairs)):
        edges = [pair for pair, present in zip(pairs, selected, strict=True) if present]
        if not edges:
            continue  # The parser requires at least one supplied premise.
        models = [
            values for values in assignments
            if all(not values[left] or values[right] for left, right in edges)
        ]
        premises = ". ".join(
            f"If {labels[left]} then {labels[right]}" if grammar == "implies"
            else f"All {labels[left]} are {labels[right]}"
            for left, right in edges
        )
        for left, right in pairs:
            question = (
                f"Does {labels[left]} imply {labels[right]}?" if grammar == "implies"
                else f"Are all {labels[left]} {labels[right]}?"
            )
            result = logic.analyze_exact(premises + ". " + question)
            expected = (
                "supported" if all(not values[left] or values[right] for values in models)
                else "insufficient"
            )
            assert result is not None
            assert result.conclusion == expected, (premises, question, result)


@pytest.mark.parametrize("operator", [">", "<"])
def test_strict_comparisons_against_all_three_value_orders(operator):
    """All total orders independently check entailment, negation and cycles."""
    labels = ("a", "b", "c")
    pairs = list(product(range(3), repeat=2))
    orders = list(permutations(range(3)))
    logic = LogicalInferenceEngine()

    def holds(values, left, right):
        return values[left] > values[right] if operator == ">" else values[left] < values[right]

    for selected in product((False, True), repeat=len(pairs)):
        edges = [pair for pair, present in zip(pairs, selected, strict=True) if present]
        if not edges:
            continue
        models = [values for values in orders if all(holds(values, *edge) for edge in edges)]
        premises = ". ".join(f"{labels[left]} {operator} {labels[right]}" for left, right in edges)
        for left, right in pairs:
            question = f"Is {labels[left]} {operator} {labels[right]}?"
            result = logic.analyze_exact(premises + ". " + question)
            possible = {holds(values, left, right) for values in models}
            expected = (
                "mixed" if not possible else "supported" if possible == {True}
                else "refuted" if possible == {False} else "insufficient"
            )
            assert result is not None
            assert result.conclusion == expected, (premises, question, result)


def test_embedded_mesh_imports_and_runs_with_only_standard_library(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    program = '''
import importlib.abc, json, sys, threading
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'transformers', 'tensorflow',
            'onnxruntime', 'sentence_transformers', 'PySide6', 'services', 'httpx', 'requests'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
sys.path.insert(0, sys.argv[1])
from sweep_neural_mesh.companion import reason
answer = reason('All cats are mammals. All mammals are animals. Are all cats animals?')
assert answer.conclusion == 'supported'
assert 'sweep_neural_mesh.mesh' not in sys.modules
assert 'sweep_neural_mesh.neurons.cortex' not in sys.modules
from sweep_neural_mesh.neurons import Signal
from sweep_neural_mesh.neurons.signal import Signal as DirectSignal
assert Signal is DirectSignal
assert 'sweep_neural_mesh.neurons.cortex' not in sys.modules
import sweep_neural_mesh, sweep_neural_mesh.neurons
assert 'NeuralMesh' in dir(sweep_neural_mesh)
assert 'ReasoningCortex' in dir(sweep_neural_mesh.neurons)
for module in (sweep_neural_mesh, sweep_neural_mesh.neurons):
    try:
        module.not_a_public_symbol
    except AttributeError:
        pass
    else:
        raise AssertionError('missing attribute did not fail')
print(json.dumps({'threads': threading.active_count(), 'answer': answer.answer}))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, root], cwd=tmp_path,
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["threads"] == 1
    assert list(tmp_path.iterdir()) == []


def test_full_mesh_public_import_is_still_compatible():
    from sweep_neural_mesh import MeshConstraints, MeshResult, NeuralMesh
    from sweep_neural_mesh.mesh import NeuralMesh as DirectMesh

    assert NeuralMesh is DirectMesh
    assert MeshConstraints().require_gpu is False
    assert MeshResult().success is True
