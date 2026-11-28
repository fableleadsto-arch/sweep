"""Small embedded desktop reasoning slice of Sweep's existing neural mesh.

This is deterministic formal reasoning, not trained language generation. It uses
the existing graph, node and execution engine with retention disabled. General
conversation must continue to the configured language model. Importing or running
this module never downloads models, starts threads, or executes computer actions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core.engine import ExecutionEngine
from .core.graph import MeshGraph
from .core.node import NeuralNode
from .core.packet import NeuralPacket
from .neurons.logical_inference import InferenceResult, LogicalInferenceEngine


@dataclass(frozen=True, slots=True)
class CompanionReasoning:
    answer: str
    conclusion: str
    method: str = "deterministic_formal_logic"
    engine: str = "sweep_neural_mesh"


def reason(text: str) -> CompanionReasoning | None:
    """Answer a completely parsed formal argument; otherwise defer to chat.

    No history, files, action permissions or external data are accepted here.
    The caller must never treat an answer as authorization for a tool action.
    """
    if not isinstance(text, str) or not 1 <= len(text) <= 4096:
        return None
    logic = LogicalInferenceEngine()
    node = NeuralNode(
        node_id="companion-formal-logic", name="Bounded formal logic",
        capabilities=["formal_logic"], history_limit=0,
        execute_fn=lambda value, **kwargs: logic.analyze_exact(value),
    )
    graph = MeshGraph(name="companion-reasoning")
    graph.add_node(node)
    result = ExecutionEngine(history_limit=0).execute(
        graph, {node.node_id: NeuralPacket(data=text)},
    )
    if not result.success or len(result.output_packets) != 1:
        return None
    deduction = result.output_packets[0].data
    if not isinstance(deduction, InferenceResult):
        return None
    return CompanionReasoning(answer=deduction.reasoning, conclusion=deduction.conclusion)
