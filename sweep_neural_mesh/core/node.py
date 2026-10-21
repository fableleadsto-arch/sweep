"""
NeuralNode — the fundamental unit of computation in the Mesh.

Every model, encoder, classifier, fusion module, or custom computation
is represented as a NeuralNode. Nodes are framework-agnostic: they declare
what they accept, what they produce, and what they cost — but not how
they compute.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Modality(Enum):
    """Supported input/output modalities."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    DOCUMENT = "document"
    TENSOR = "tensor"
    EMBEDDING = "embedding"
    STRUCTURED = "structured"
    DEPTH = "depth"
    SENSOR = "sensor"
    TIMESERIES = "timeseries"
    NETWORK = "network"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class Framework(Enum):
    """Supported ML frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TFLITE = "tflite"
    SKLEARN = "sklearn"
    JAX = "jax"
    CUSTOM = "custom"
    PURE_PYTHON = "pure_python"


class NodeStatus(Enum):
    """Runtime status of a node."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    EXECUTING = "executing"
    FAILED = "failed"
    UNLOADED = "unloaded"


@dataclass
class NodeCostProfile:
    """Resource cost profile for a node."""
    memory_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    startup_time_ms: float = 0.0
    flop_estimate: float = 0.0
    gpu_required: bool = False
    min_ram_mb: float = 0.0


@dataclass
class NodeVersion:
    """Version metadata for a node."""
    model_id: str = ""
    weights_version: str = "1.0.0"
    architecture_version: str = "1.0.0"
    preprocessing_version: str = "1.0.0"
    embedding_schema_version: str = "1.0.0"


@dataclass
class NodeSchema:
    """Describes what a node accepts and produces."""
    input_modalities: list[Modality] = field(default_factory=list)
    output_modalities: list[Modality] = field(default_factory=list)
    input_shape: list[int | str] | None = None
    output_shape: list[int | str] | None = None
    input_dtype: str = "float32"
    output_dtype: str = "float32"


@dataclass
class NodeResult:
    """Result of executing a node."""
    success: bool
    output: Any = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralNode:
    """
    A computational unit in the Neural Mesh.

    Nodes wrap models, encoders, classifiers, or any computation.
    They are identified by capability rather than implementation,
    allowing the Mesh to swap them without architectural changes.
    """

    def __init__(
        self,
        node_id: str | None = None,
        name: str = "unnamed",
        version: NodeVersion | None = None,
        schema: NodeSchema | None = None,
        cost: NodeCostProfile | None = None,
        framework: Framework = Framework.PURE_PYTHON,
        execute_fn: Callable[..., Any] | None = None,
        validate_fn: Callable[..., bool] | None = None,
        capabilities: list[str] | None = None,
        tags: dict[str, str] | None = None,
        history_limit: int = 64,
    ):
        self.node_id = node_id or str(uuid.uuid4())[:12]
        self.name = name
        self.version = version or NodeVersion()
        self.schema = schema or NodeSchema()
        self.cost = cost or NodeCostProfile()
        self.framework = framework
        self.capabilities = capabilities or []
        self.tags = tags or {}
        self.status = NodeStatus.IDLE
        self._execute_fn = execute_fn
        self._validate_fn = validate_fn
        self._model: Any = None
        if type(history_limit) is not int or history_limit < 0:
            raise ValueError("history_limit must be a nonnegative integer")
        self._history: deque[NodeResult] = deque(maxlen=history_limit)

    # -- Execution --

    def execute(self, *args: Any, **kwargs: Any) -> NodeResult:
        """Execute this node's computation."""
        if self._execute_fn is None:
            return NodeResult(
                success=False,
                error=f"Node {self.name} has no execute function",
            )
        self.status = NodeStatus.EXECUTING
        t0 = time.perf_counter()
        try:
            output = self._execute_fn(*args, **kwargs)
            latency = (time.perf_counter() - t0) * 1000
            result = NodeResult(success=True, output=output, latency_ms=latency)
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            result = NodeResult(
                success=False,
                error=str(exc),
                latency_ms=latency,
            )
            self.status = NodeStatus.FAILED
        finally:
            if self.status != NodeStatus.FAILED:
                self.status = NodeStatus.READY
        self._history.append(result)
        return result

    def validate_input(self, data: Any) -> bool:
        """Validate input against this node's schema."""
        if self._validate_fn is not None:
            return self._validate_fn(data)
        return True

    # -- Model lifecycle --

    def load_model(self, model: Any) -> None:
        """Attach a loaded model to this node."""
        self._model = model
        self.status = NodeStatus.READY

    def unload_model(self) -> None:
        """Detach the model."""
        self._model = None
        self.status = NodeStatus.UNLOADED

    @property
    def model(self) -> Any:
        return self._model

    @property
    def is_ready(self) -> bool:
        return self.status == NodeStatus.READY

    # -- History --

    @property
    def history(self) -> list[NodeResult]:
        """Recent executions only; set history_limit=0 to retain no results."""
        return list(self._history)

    @property
    def avg_latency_ms(self) -> float:
        if not self._history:
            return 0.0
        return sum(r.latency_ms for r in self._history) / len(self._history)

    @property
    def failure_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if not r.success) / len(self._history)

    # -- Serialisation --

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "status": self.status.value,
            "framework": self.framework.value,
            "capabilities": self.capabilities,
            "cost": {
                "memory_mb": self.cost.memory_mb,
                "avg_latency_ms": self.cost.avg_latency_ms,
                "gpu_required": self.cost.gpu_required,
            },
            "version": {
                "model_id": self.version.model_id,
                "weights_version": self.version.weights_version,
            },
            "history_size": len(self._history),
            "avg_latency_ms": self.avg_latency_ms,
            "failure_rate": self.failure_rate,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralNode(id={self.node_id}, name={self.name}, "
            f"status={self.status.value}, fw={self.framework.value})"
        )
