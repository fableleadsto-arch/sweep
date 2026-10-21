"""Execution failures, admission limits and cached evidence must stay truthful."""

import gc
import weakref
from copy import deepcopy

import pytest

from sweep_neural_mesh.core.engine import ExecutionEngine
from sweep_neural_mesh.core.graph import MeshGraph
from sweep_neural_mesh.core.node import NeuralNode, NodeCostProfile, NodeResult, NodeStatus
from sweep_neural_mesh.core.packet import NeuralPacket
from sweep_neural_mesh.core.router import ModelRouter, RoutingContext
from sweep_neural_mesh.mesh import MeshConstraints, NeuralMesh
from sweep_neural_mesh.registry import ModelRegistry


def test_small_result_cache_evicts_old_observations():
    mesh = NeuralMesh()
    node = ScoredNode()
    mesh.register_node(node)
    for index in range(33):
        mesh.analyze(str(index), task="inspect")
    assert mesh.cache.size == 32
    mesh.analyze("0", task="inspect")
    assert node.calls == 34


def test_oversized_and_opaque_results_are_not_copied_into_cache():
    class Opaque:
        def __deepcopy__(self, memo):
            pytest.fail("Opaque model outputs must not be copied")

    for output in [Opaque(), "x" * 1_048_577]:
        mesh = NeuralMesh()
        mesh.register_node(NeuralNode(capabilities=["inspect"], execute_fn=lambda *a, **kw: output))
        assert mesh.analyze("input", task="inspect").success
        assert mesh.cache.size == 0


def test_recent_history_bounds_release_evicted_result_references():
    node = NeuralNode(execute_fn=lambda data, **kw: data, history_limit=2)
    engine = ExecutionEngine(history_limit=2)
    graph = MeshGraph()
    graph.add_node(node)
    first = engine.execute(graph, {node.node_id: NeuralPacket(data="old")})
    first_ref = weakref.ref(first)
    node_ref = weakref.ref(first.node_results[node.node_id])
    del first
    for value in ("middle", "latest"):
        engine.execute(graph, {node.node_id: NeuralPacket(data=value)})
    gc.collect()
    assert first_ref() is None and node_ref() is None
    assert len(node.history) == len(engine.history) == 2
    assert node.history[-1].output == "latest"
    assert engine.stats["executions"] == 2


def test_history_can_be_disabled_and_rejects_invalid_limits():
    node = NeuralNode(execute_fn=lambda *a, **kw: "value", history_limit=0)
    engine = ExecutionEngine(history_limit=0)
    graph = MeshGraph()
    graph.add_node(node)
    assert engine.execute(graph).success
    assert node.history == engine.history == []
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            NeuralNode(history_limit=value)
        with pytest.raises(ValueError):
            ExecutionEngine(history_limit=value)


class ScoredNode(NeuralNode):
    """A model adapter that supplies its own explicit confidence."""

    def __init__(self, confidence=0.42, **kwargs):
        super().__init__(name="scored", capabilities=["inspect"], **kwargs)
        self.confidence = confidence
        self.calls = 0

    def execute(self, data, **kwargs):
        self.calls += 1
        self.status = NodeStatus.READY
        return NodeResult(
            success=True, output={"observations": [data]}, confidence=self.confidence,
            metadata={"method": "test observation"},
        )


def test_failed_node_does_not_return_original_input_as_success():
    def fail(data, **kwargs):
        raise RuntimeError("decoder unavailable")

    mesh = NeuralMesh()
    mesh.register_node(NeuralNode(name="decoder", capabilities=["decode"], execute_fn=fail))
    result = mesh.analyze({"unprocessed": "input"}, task="decode")

    assert not result.success
    assert result.output is None
    assert not result.raw_packets
    assert not result.evidence
    assert not result.models_used
    assert any("decoder unavailable" in warning for warning in result.warnings)
    assert mesh.cache.size == 0


def test_failed_dependency_skips_join_but_independent_branch_still_runs():
    def fail(data, **kwargs):
        raise RuntimeError("missing model")

    broken = NeuralNode(node_id="a", name="broken", execute_fn=fail)
    good = NeuralNode(node_id="b", name="good", execute_fn=lambda data, **kw: data * 2)
    joined = NeuralNode(node_id="c", name="join", execute_fn=lambda data, **kw: "incorrect")
    descendant = NeuralNode(node_id="d", name="descendant", execute_fn=lambda *a, **kw: "bad")
    graph = MeshGraph()
    for node in (broken, good, joined, descendant):
        graph.add_node(node)
    graph.add_edge(broken, joined)
    graph.add_edge(good, joined)
    graph.add_edge(joined, descendant)
    result = ExecutionEngine().execute(graph, {
        broken.node_id: NeuralPacket(data="unprocessed"),
        good.node_id: NeuralPacket(data=3),
        # An initial packet cannot override a failed dependency.
        joined.node_id: NeuralPacket(data="bypass"),
    })

    assert not result.success
    assert result.total_nodes_executed == 2
    assert result.total_nodes_failed == 1
    assert result.total_nodes_skipped == 2
    assert result.node_results[good.node_id].output == 6
    assert not joined.history and not descendant.history
    assert not result.output_packets
    assert "broken" in graph.get_result(joined.node_id).error
    assert result.to_dict()["nodes_skipped"] == 2


def test_graph_rerun_does_not_reuse_a_previous_success_after_failure():
    calls = 0

    def process(data, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second attempt failed")
        return "processed"

    node = NeuralNode(execute_fn=process)
    graph = MeshGraph()
    graph.add_node(node)
    engine = ExecutionEngine()
    inputs = {node.node_id: NeuralPacket(data="input")}
    assert engine.execute(graph, inputs).output_packets[0].data == "processed"
    second = engine.execute(graph, inputs)
    assert not second.success
    assert not second.output_packets


def test_custom_execute_exception_is_a_failed_result():
    class BrokenNode(NeuralNode):
        def execute(self, *args, **kwargs):
            raise RuntimeError("adapter failure")

    mesh = NeuralMesh()
    mesh.register_node(BrokenNode(capabilities=["inspect"]))
    result = mesh.analyze("input", task="inspect")
    assert not result.success
    assert any("adapter failure" in warning for warning in result.warnings)


@pytest.mark.parametrize("confidence", [0.0, 0.42, 1.0])
def test_cache_preserves_confidence_models_evidence_and_provenance(confidence):
    mesh = NeuralMesh()
    node = ScoredNode(confidence)
    mesh.register_node(node)
    original = mesh.analyze("label", task="inspect")
    saved = deepcopy(original)

    # A caller mutating their response must not mutate the cached observation.
    original.output["observations"].append("tampered")
    original.evidence[0]["source"] = "tampered"
    original.provenance.clear()
    original.models_used.clear()
    original.raw_packets[0].metadata["method"] = "tampered"

    cached = mesh.analyze("label", task="inspect")
    assert cached.success
    assert node.calls == 1
    assert cached.confidence == pytest.approx(confidence)
    assert cached.confidence_report == saved.confidence_report
    assert cached.output == saved.output
    assert cached.evidence == saved.evidence
    assert cached.provenance == saved.provenance
    assert cached.nodes_used == saved.nodes_used
    assert cached.models_used == saved.models_used
    assert cached.raw_packets[0].metadata == saved.raw_packets[0].metadata
    assert cached.warnings == [*saved.warnings, "served from cache"]

    cached.evidence.clear()
    third = mesh.analyze("label", task="inspect")
    assert third.evidence == saved.evidence
    assert third.warnings.count("served from cache") == 1


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_model_confidence_cannot_be_reported_or_cached(confidence):
    mesh = NeuralMesh()
    mesh.register_node(ScoredNode(confidence))
    result = mesh.analyze("label", task="inspect")
    assert not result.success
    assert result.output is None
    assert mesh.cache.size == 0
    assert any("confidence" in warning for warning in result.warnings)


@pytest.mark.parametrize("cost", [
    NodeCostProfile(memory_mb=101),
    NodeCostProfile(min_ram_mb=101),
    NodeCostProfile(avg_latency_ms=101),
    NodeCostProfile(p95_latency_ms=101),
    NodeCostProfile(avg_latency_ms=40, startup_time_ms=61),
    NodeCostProfile(memory_mb=float("inf")),
    NodeCostProfile(p95_latency_ms=float("nan")),
    NodeCostProfile(startup_time_ms=-1),
])
def test_over_budget_or_invalid_estimate_is_rejected_before_execution(cost):
    mesh = NeuralMesh()
    node = ScoredNode(cost=cost)
    mesh.register_node(node)
    result = mesh.analyze("input", task="inspect", constraints=MeshConstraints(
        memory_budget_mb=100, latency_budget_ms=100,
    ))
    assert not result.success
    assert node.calls == 0
    assert result.output is None
    assert any("no candidates fit" in warning for warning in result.warnings)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_request_budget_is_rejected(budget):
    mesh = NeuralMesh()
    node = ScoredNode()
    mesh.register_node(node)
    result = mesh.analyze("input", task="inspect", constraints=MeshConstraints(
        memory_budget_mb=budget,
    ))
    assert not result.success and node.calls == 0


def test_cache_cannot_bypass_a_tighter_budget_or_failed_node_status():
    mesh = NeuralMesh()
    node = ScoredNode(cost=NodeCostProfile(memory_mb=100))
    mesh.register_node(node)
    assert mesh.analyze("input", task="inspect").success
    result = mesh.analyze("input", task="inspect", constraints=MeshConstraints(memory_budget_mb=99))
    assert not result.success
    assert node.calls == 1
    node.status = NodeStatus.FAILED
    assert not mesh.analyze("input", task="inspect").success
    assert node.calls == 1


def test_budget_boundary_and_loaded_model_startup_cost():
    mesh = NeuralMesh()
    node = ScoredNode(cost=NodeCostProfile(memory_mb=100, avg_latency_ms=40, startup_time_ms=60))
    mesh.register_node(node)
    assert mesh.analyze("input", task="inspect", constraints=MeshConstraints(
        memory_budget_mb=100, latency_budget_ms=100,
    )).success
    # The already loaded model no longer needs startup time.
    assert mesh.analyze("different", task="inspect", constraints=MeshConstraints(
        memory_budget_mb=100, latency_budget_ms=40,
    )).success
    assert node.calls == 2


def test_router_checks_required_capability_and_gpu_requirement():
    router = ModelRouter(ModelRegistry())
    cpu = ScoredNode()
    gpu = ScoredNode(cost=NodeCostProfile(gpu_required=True))
    assert router.rank([cpu], RoutingContext(require_gpu=True)).selected_node is None
    assert router.rank([cpu, gpu], RoutingContext(require_gpu=True)).selected_node is gpu
    assert router.rank([cpu], RoutingContext(required_capability="other")).selected_node is None


def test_cache_tracks_version_constraints_modality_and_registered_replacements():
    mesh = NeuralMesh()
    node = ScoredNode()
    mesh.register_node(node)
    mesh.analyze("input", task="inspect")
    node.version.weights_version = "2"
    assert "served from cache" not in mesh.analyze("input", task="inspect").warnings
    assert "served from cache" not in mesh.analyze("input", task="inspect", modalities=["text"]).warnings
    assert "served from cache" not in mesh.analyze("input", task="inspect", constraints=MeshConstraints(
        verification_enabled=True,
    )).warnings
    assert node.calls == 4

    replacement = ScoredNode(0.7, node_id=node.node_id)
    replacement.version = deepcopy(node.version)
    mesh.register_node(replacement)
    result = mesh.analyze("input", task="inspect")
    assert result.confidence == pytest.approx(0.7)
    assert replacement.calls == 1


def test_opaque_objects_with_identical_repr_do_not_collide_in_cache():
    class Opaque:
        def __repr__(self):
            return "same truncated preview"

    mesh = NeuralMesh()
    node = ScoredNode()
    mesh.register_node(node)
    first, second = Opaque(), Opaque()
    assert mesh.analyze(first, task="inspect").output["observations"] == [first]
    assert mesh.analyze(second, task="inspect").output["observations"] == [second]
    assert node.calls == 2 and mesh.cache.size == 0


def test_successful_none_output_can_be_cached_without_fabricating_confidence():
    mesh = NeuralMesh()
    node = NeuralNode(capabilities=["inspect"], execute_fn=lambda *a, **kw: None)
    mesh.register_node(node)
    first = mesh.analyze("input", task="inspect")
    cached = mesh.analyze("input", task="inspect")
    assert first.success and cached.success
    assert first.output is None and cached.output is None
    assert first.confidence == cached.confidence == 0
    assert len(node.history) == 1


def test_unsupported_copyable_output_does_not_turn_success_into_an_error():
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise TypeError("framework handle cannot be copied")

    output = CannotCopy()
    mesh = NeuralMesh()
    node = NeuralNode(capabilities=["inspect"], execute_fn=lambda *a, **kw: output)
    mesh.register_node(node)
    result = mesh.analyze("input", task="inspect")
    assert result.success and result.output is output
    assert mesh.cache.size == 0
