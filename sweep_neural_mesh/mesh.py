"""
NeuralMesh — the top-level API for the Sweep Neural Mesh.

The Mesh is the computational fabric between Sweep's intent layer and
its specialized ML models. It provides:

    mesh.analyze(data, task, constraints) -> MeshResult

The rest of Sweep should never need to know whether the result came
from PyTorch, TensorFlow, ONNX, scikit-learn, CPU, GPU, one model,
or five models.

Usage:

    from sweep_neural_mesh import NeuralMesh

    mesh = NeuralMesh()
    mesh.register_node(vision_node)
    mesh.register_node(audio_node)

    result = mesh.analyze(
        data=image_bytes,
        task="face_detection",
        modalities=["image"],
    )
    print(result.confidence, result.output)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from .adapters.base import BaseAdapter
from .core.engine import ExecutionEngine
from .core.graph import MeshGraph
from .core.node import (
    Framework,
    Modality,
    NeuralNode,
    NodeSchema,
)
from .core.packet import NeuralPacket, Provenance, SpatialContext
from .core.router import ModelRouter, RoutingContext
from .fusion import FusionEngine
from .fusion.confidence import ConfidenceEngine, ConfidenceReport
from .fusion.verification import VerificationAction, VerificationEngine, VerificationResult
from .memory import FeatureCache
from .registry import ModelRegistry
from .registry.capability_registry import CapabilityRegistry
from .resources import ResourceManager
from .telemetry import Telemetry

logger = logging.getLogger(__name__)
MAX_CACHE_VALUE_BYTES = 1_048_576


@dataclass
class MeshConstraints:
    """Resource and quality constraints for an analysis request."""
    latency_budget_ms: float = 5000.0
    memory_budget_mb: float = 4096.0
    require_gpu: bool = False
    preferred_frameworks: list[Framework] = field(default_factory=list)
    quality_threshold: float = 0.5
    fusion_strategy: str = "confidence_weighted"
    verification_enabled: bool = False


@dataclass
class MeshResult:
    """The result of a mesh.analyze() call."""
    success: bool = True
    output: Any = None
    confidence: float = 0.0
    confidence_report: ConfidenceReport | None = None
    verification: VerificationResult | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    nodes_used: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    resource_usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    raw_packets: list[NeuralPacket] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "nodes_used": self.nodes_used,
            "models_used": self.models_used,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
            "output_type": type(self.output).__name__,
            "evidence_count": len(self.evidence),
        }


class NeuralMesh:
    """
    The top-level Neural Mesh API.

    Orchestrates node registration, capability discovery, routing,
    execution, fusion, verification, and telemetry. This is the only
    class the rest of Sweep needs to import.
    """

    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.capabilities = CapabilityRegistry()
        self.router = ModelRouter(self.registry)
        self.engine = ExecutionEngine()
        self.fusion = FusionEngine()
        self.confidence = ConfidenceEngine()
        self.verification = VerificationEngine()
        self.resources = ResourceManager()
        self.telemetry = Telemetry()
        # Cache only small observations. Model tensors/handles must not be
        # copied or retained by this general-purpose result cache.
        self.cache = FeatureCache(max_entries=32)

    # -- Registration --

    def register_node(self, node: NeuralNode) -> None:
        """Register a node with the Mesh."""
        self.registry.register(node)
        self.cache.clear()
        self.telemetry.inc_counter("nodes_registered")
        logger.info("Registered node: %s (%s)", node.name, node.node_id)

    def register_nodes(self, nodes: list[NeuralNode]) -> None:
        for n in nodes:
            self.register_node(n)

    def register_adapter(
        self,
        adapter: BaseAdapter,
        model: Any,
        name: str,
        capabilities: list[str],
        schema: NodeSchema | None = None,
    ) -> NeuralNode:
        """Register a model via an adapter — wraps it as a NeuralNode."""
        node = adapter.wrap_as_node(model, name, capabilities, schema)
        self.register_node(node)
        return node

    def register_capability(
        self,
        name: str,
        modality: Modality,
        description: str = "",
        prerequisites: list[str] | None = None,
    ) -> None:
        self.capabilities.register(name, modality, description, prerequisites)

    # -- Analysis --

    def analyze(
        self,
        data: Any,
        task: str,
        modalities: list[str] | None = None,
        constraints: MeshConstraints | None = None,
    ) -> MeshResult:
        """
        The primary Mesh API.

        Analyzes input data for a given task, using the registered
        nodes, capabilities, and constraints.

        Returns a MeshResult with output, confidence, provenance,
        and telemetry.
        """
        t0 = time.perf_counter()
        constraints = constraints or MeshConstraints()
        result = MeshResult(success=False)
        modality_enums = [Modality(m) for m in (modalities or ["tensor"])]

        # Find capable nodes
        context = RoutingContext(
            task=task,
            required_capability=task,
            latency_budget_ms=constraints.latency_budget_ms,
            memory_budget_mb=constraints.memory_budget_mb,
            require_gpu=constraints.require_gpu,
            preferred_frameworks=constraints.preferred_frameworks,
        )
        routing = self.router.rank_multi(task, context, top_k=3)

        if routing.selected_node is None:
            result.success = False
            result.warnings.append(f"no node found for capability: {task} ({routing.reason})")
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Check the current route and constraints before reusing an observation.
        # A cache hit preserves the original confidence/evidence, and callers
        # receive a copy so editing one result cannot corrupt later answers.
        cache_key = self._cache_key(data, task, modality_enums, constraints, routing.selected_node)
        cached = self.cache.get(cache_key) if cache_key is not None else None
        if isinstance(cached, MeshResult) and cached.success:
            result = deepcopy(cached)
            self.telemetry.inc_counter("cache_hits")
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.warnings.append("served from cache")
            return result

        # Execute
        packet = NeuralPacket(
            data=data,
            modality=modality_enums[0] if modality_enums else Modality.TENSOR,
            source_node_name="mesh_input",
        )

        self.telemetry.record("routing", node_name=routing.selected_node.name)

        exec_result = self.engine.execute(
            self._build_single_node_graph(routing.selected_node),
            initial_inputs={routing.selected_node.node_id: packet},
        )
        result.success = exec_result.success
        result.warnings.extend(exec_result.warnings)
        if not exec_result.success:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.resource_usage = self.resources.profile.to_dict()
            self.telemetry.record("analysis", node_name=task,
                                  duration_ms=result.latency_ms, success=False)
            return result

        # Collect results
        node_confidences: dict[str, float] = {}
        packets: list[NeuralPacket] = []

        if exec_result.output_packets:
            packets = exec_result.output_packets
            for p in packets:
                node_confidences[p.provenance.source_node_name] = p.confidence

        # Confidence
        conf_report = self.confidence.evaluate(node_confidences)
        result.confidence_report = conf_report
        result.confidence = conf_report.score

        # Output
        if packets:
            primary = packets[0]
            result.output = primary.data
            result.raw_packets = packets
            result.provenance = primary.provenance_chain

        # Verification
        if constraints.verification_enabled and len(packets) > 1:
            outputs = [p.data for p in packets]
            sources = [p.provenance.source_node_name for p in packets]
            result.verification = self.verification.verify(outputs, sources)

        # Evidence
        for p in packets:
            result.evidence.append({
                "source": p.provenance.source_node_name,
                "confidence": p.confidence,
                "modality": p.modality.value,
                "provenance_depth": len(p.provenance.chain),
            })

        # Nodes/models used
        result.nodes_used = list(set(
            p.provenance.source_node_id for p in packets if p.provenance.source_node_id
        ))
        result.models_used = list(set(
            p.provenance.source_node_name for p in packets
        ))

        # Telemetry
        result.latency_ms = (time.perf_counter() - t0) * 1000
        self.telemetry.record(
            "analysis",
            node_name=task,
            duration_ms=result.latency_ms,
            success=result.success,
            confidence=result.confidence,
        )

        # Resource snapshot
        result.resource_usage = self.resources.profile.to_dict()

        if cache_key is not None and self._cacheable_value(result):
            try:
                self.cache.put(cache_key, deepcopy(result))
            except (TypeError, ValueError, RuntimeError):
                # Some framework outputs cannot be copied. The computation
                # remains usable; it simply does not participate in caching.
                logger.debug("Result cannot be copied into the feature cache", exc_info=True)

        return result

    # -- Helpers --

    @staticmethod
    def _cacheable_value(value: Any) -> bool:
        """Bound traversal and Python payload size before any copying/encoding.

        Exact types prevent opaque framework objects from invoking serializers
        or expensive __deepcopy__ methods. This estimates retained Python memory;
        it is not an OS memory limit for execution itself.
        """
        records = {MeshResult, ConfidenceReport, VerificationResult, NeuralPacket,
                   Provenance, SpatialContext}
        seen: set[int] = set()
        remaining = MAX_CACHE_VALUE_BYTES
        visits = 0

        def visit(item: Any, depth: int = 0) -> bool:
            nonlocal remaining, visits
            visits += 1
            if depth > 32 or visits > 10_000:
                return False
            kind = type(item)
            if kind not in {str, bytes, bool, int, float, type(None), list, tuple, dict,
                            Modality, VerificationAction} | records:
                return False
            if id(item) in seen:
                return True
            seen.add(id(item))
            remaining -= sys.getsizeof(item)
            if remaining < 0:
                return False
            if kind is dict:
                return all(visit(key, depth + 1) and visit(child, depth + 1) for key, child in item.items())
            if kind in {list, tuple}:
                return all(visit(child, depth + 1) for child in item)
            if kind in records:
                return visit(vars(item), depth + 1)
            return True

        return visit(value)

    @staticmethod
    def _cache_key(data: Any, task: str, modalities: list[Modality],
                   constraints: MeshConstraints, node: NeuralNode) -> str | None:
        if not NeuralMesh._cacheable_value(data):
            return None
        def json_data(value: Any, depth: int = 0) -> bool:
            if depth > 32:
                return False
            if type(value) in (str, bool, int, float, type(None)):
                return True
            if type(value) is list:
                return all(json_data(item, depth + 1) for item in value)
            if type(value) is dict:
                return all(type(key) is str and json_data(item, depth + 1)
                           for key, item in value.items())
            return False

        # Arbitrary objects/arrays can have identical truncated reprs. Cache
        # only content whose full identity we can represent unambiguously.
        if type(data) is bytes:
            content = {"bytes_sha256": hashlib.sha256(data).hexdigest()}
        elif json_data(data):
            content = {"json": data}
        else:
            return None
        try:
            policy = asdict(constraints)
            policy["preferred_frameworks"] = [item.value for item in constraints.preferred_frameworks]
            raw = json.dumps({"task": task, "data": content,
                              "modalities": [item.value for item in modalities],
                              "constraints": policy, "node": node.node_id,
                              "version": asdict(node.version), "cost": asdict(node.cost)},
                             sort_keys=True, allow_nan=False, ensure_ascii=False)
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()
        except (TypeError, ValueError, RecursionError):
            return None

    def _build_single_node_graph(self, node: NeuralNode) -> MeshGraph:
        graph = MeshGraph(name=f"single:{node.name}")
        graph.add_node(node)
        return graph

    # -- Stats --

    def summary(self) -> dict[str, Any]:
        return {
            "registry": self.registry.summary(),
            "capabilities": self.capabilities.summary(),
            "engine": self.engine.stats,
            "telemetry": self.telemetry.summary(),
            "resources": self.resources.profile.to_dict(),
            "workload_state": self.resources.state.value,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralMesh(nodes={self.registry.size}, "
            f"caps={len(self.capabilities.all_names)}, "
            f"state={self.resources.state.value})"
        )
