"""
ExecutionEngine — topologically sorts and runs a MeshGraph.

The engine walks the graph in dependency order, executing each node
and threading NeuralPackets through edges. It records failures and skips
dependent nodes rather than passing incomplete inputs to them.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any

from .graph import MeshGraph
from .node import NodeResult
from .packet import NeuralPacket

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Aggregated result of executing a full mesh graph."""

    def __init__(self, graph_id: str = "") -> None:
        self.graph_id = graph_id
        self.success = True
        self.node_results: dict[str, NodeResult] = {}
        self.output_packets: list[NeuralPacket] = []
        self.total_latency_ms: float = 0.0
        self.total_nodes_executed: int = 0
        self.total_nodes_failed: int = 0
        self.total_nodes_skipped: int = 0
        self.warnings: list[str] = []
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "success": self.success,
            "total_latency_ms": self.total_latency_ms,
            "nodes_executed": self.total_nodes_executed,
            "nodes_failed": self.total_nodes_failed,
            "nodes_skipped": self.total_nodes_skipped,
            "output_packets": len(self.output_packets),
            "warnings": self.warnings,
        }


class ExecutionEngine:
    """
    Executes a MeshGraph by walking it in topological order.

    Each node receives NeuralPackets from its predecessors,
    executes its computation, and produces output packets for
    its successors.
    """

    def __init__(self, history_limit: int = 64) -> None:
        if type(history_limit) is not int or history_limit < 0:
            raise ValueError("history_limit must be a nonnegative integer")
        self._execution_history: deque[ExecutionResult] = deque(maxlen=history_limit)

    def execute(
        self,
        graph: MeshGraph,
        initial_inputs: dict[str, NeuralPacket] | None = None,
    ) -> ExecutionResult:
        """
        Execute the full graph.

        Args:
            graph: The MeshGraph to execute.
            initial_inputs: Mapping of root node_id → input NeuralPacket.

        Returns:
            ExecutionResult with all node results and output packets.
        """
        t0 = time.perf_counter()
        result = ExecutionResult(graph_id=graph.graph_id)
        initial_inputs = initial_inputs or {}
        # Inputs are not outputs: a failed root must never return its input as
        # if it had successfully processed it.
        packet_map: dict[str, NeuralPacket] = {}
        nodes = {node.node_id: node for node in graph.nodes}

        try:
            order = graph.execution_order
        except ValueError as exc:
            result.success = False
            result.warnings.append(str(exc))
            result.total_latency_ms = (time.perf_counter() - t0) * 1000
            self._execution_history.append(result)
            return result

        for node_id in order:
            node = nodes[node_id]

            # Gather input packets from predecessors
            predecessors = graph.predecessors(node_id)
            missing = [p.name for p in predecessors if p.node_id not in packet_map]
            if missing:
                failure = NodeResult(success=False, error="Dependencies failed: " + ", ".join(missing))
                graph.store_result(node_id, failure)
                result.node_results[node_id] = failure
                result.total_nodes_skipped += 1
                result.warnings.append(f"Node {node.name} skipped: {failure.error}")
                continue
            if predecessors:
                input_packets = [
                    packet_map[p.node_id]
                    for p in predecessors
                    if p.node_id in packet_map
                ]
            else:
                input_packets = [
                    initial_inputs[node_id]
                ] if node_id in initial_inputs else []

            # Custom nodes can override execute(), so contain failures here as
            # well as in the base node wrapper.
            try:
                if input_packets:
                    primary = input_packets[0]
                    node_result = node.execute(
                        primary.data,
                        packets=input_packets,
                        metadata=primary.metadata,
                    )
                else:
                    node_result = node.execute()
                if not isinstance(node_result, NodeResult):
                    raise TypeError("Node execution must return a NodeResult")
                if node_result.success and (
                    not math.isfinite(node_result.confidence)
                    or not 0 <= node_result.confidence <= 1
                ):
                    raise ValueError("Node confidence must be finite and between 0 and 1")
            except Exception as exc:
                node_result = NodeResult(success=False, error=str(exc))

            graph.store_result(node_id, node_result)
            result.node_results[node_id] = node_result
            result.total_nodes_executed += 1

            if node_result.success:
                # Successful execution does not establish certainty about the
                # result; preserve the node's actual (possibly zero) confidence.
                out_packet = NeuralPacket(
                    data=node_result.output,
                    confidence=node_result.confidence,
                    source_node_id=node_id,
                    source_node_name=node.name,
                    metadata={
                        **node_result.metadata,
                        "latency_ms": node_result.latency_ms,
                    },
                )
                packet_map[node_id] = out_packet
            else:
                result.total_nodes_failed += 1
                result.warnings.append(
                    f"Node {node.name} ({node_id}) failed: {node_result.error}"
                )
                logger.warning(
                    "Node %s failed: %s", node.name, node_result.error
                )

        # Collect output packets from leaf nodes
        for leaf in graph.leaves():
            if leaf.node_id in packet_map:
                result.output_packets.append(packet_map[leaf.node_id])

        result.success = result.total_nodes_failed == 0 and result.total_nodes_skipped == 0
        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        self._execution_history.append(result)
        return result

    @property
    def history(self) -> list[ExecutionResult]:
        return list(self._execution_history)

    @property
    def stats(self) -> dict[str, Any]:
        """Statistics over the retained recent execution window."""
        if not self._execution_history:
            return {"executions": 0}
        total = len(self._execution_history)
        success = sum(1 for r in self._execution_history if r.success)
        avg_latency = (
            sum(r.total_latency_ms for r in self._execution_history) / total
        )
        return {
            "executions": total,
            "success_rate": success / total,
            "avg_latency_ms": avg_latency,
        }

    def __repr__(self) -> str:
        return f"ExecutionEngine(history={len(self._execution_history)})"
