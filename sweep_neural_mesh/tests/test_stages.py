"""Tests for fusion, confidence, verification, adapters, resources, telemetry, mesh (Stages 2-5)."""
from __future__ import annotations

import pytest

from sweep_neural_mesh.fusion import (
    ConfidenceWeightedFusion,
    EarlyFusion,
    FusionEngine,
    GatedFusion,
    LateFusion,
)
from sweep_neural_mesh.fusion.confidence import ConfidenceEngine, ConfidenceReport
from sweep_neural_mesh.fusion.verification import (
    VerificationAction,
    VerificationEngine,
    VerificationResult,
)
from sweep_neural_mesh.adapters.base import BaseAdapter
from sweep_neural_mesh.adapters.sklearn_adapter import SklearnAdapter
from sweep_neural_mesh.core.node import Framework, Modality, NeuralNode, NodeSchema
from sweep_neural_mesh.core.packet import NeuralPacket
from sweep_neural_mesh.resources import ResourceManager, WorkloadState
from sweep_neural_mesh.telemetry import Telemetry
from sweep_neural_mesh.memory import FeatureCache
from sweep_neural_mesh.mesh import MeshConstraints, MeshResult, NeuralMesh


# ── Fusion ──

class TestFusion:
    def test_late_fusion_average(self):
        f = LateFusion()
        result = f.fuse([2.0, 4.0, 6.0])
        assert result == 4.0

    def test_confidence_weighted(self):
        f = ConfidenceWeightedFusion()
        result = f.fuse([10.0, 20.0], confidences=[0.8, 0.2])
        expected = (10.0 * 0.8 + 20.0 * 0.2) / 1.0
        assert abs(result - expected) < 1e-6

    def test_early_fusion_concat(self):
        f = EarlyFusion()
        result = f.fuse([[1, 2], [3, 4], [5]])
        assert result == [1, 2, 3, 4, 5]

    def test_gated_fusion(self):
        f = GatedFusion(temperature=0.1)
        result = f.fuse([10.0, 20.0], confidences=[10.0, 0.1])
        assert result >= 10.0  # high confidence source dominates

    def test_fusion_engine_available_strategies(self):
        engine = FusionEngine()
        strategies = FusionEngine.available_strategies()
        assert "late" in strategies
        assert "confidence_weighted" in strategies

    def test_fusion_engine_unknown_strategy(self):
        engine = FusionEngine()
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            engine.fuse([1, 2], strategy="nonexistent")


# ── Confidence ──

class TestConfidence:
    def test_high_confidence(self):
        ce = ConfidenceEngine()
        report = ce.evaluate({"node_a": 0.95, "node_b": 0.90})
        assert report.score > 0.85
        assert report.quality_tier == "high"

    def test_low_confidence(self):
        ce = ConfidenceEngine()
        report = ce.evaluate({"node_a": 0.3, "node_b": 0.2})
        assert report.score < 0.5
        assert report.quality_tier in ("low", "very_low")

    def test_agreement(self):
        ce = ConfidenceEngine()
        report = ce.evaluate(
            {"a": 0.9, "b": 0.88},
            agreement_scores=[0.95],
        )
        assert report.agreement_score == 0.95

    def test_empty_input(self):
        ce = ConfidenceEngine()
        report = ce.evaluate({})
        assert report.quality_label == "no_data"

    def test_calibration(self):
        ce = ConfidenceEngine()
        raw = 0.6
        calibrate = ce.calibrate(raw, temperature=2.0)
        assert 0 <= calibrate <= 1


# ── Verification ──

class TestVerification:
    def test_exact_agreement(self):
        ve = VerificationEngine()
        result = ve.verify(["cat", "cat", "cat"], metric="classification")
        assert result.agreement == 1.0
        assert result.recommended_action == VerificationAction.ACCEPT

    def test_disagreement(self):
        ve = VerificationEngine()
        result = ve.verify(["cat", "dog", "bird"], metric="classification")
        assert result.agreement < 0.5
        assert result.recommended_action == VerificationAction.REJECT

    def test_numerical_close(self):
        ve = VerificationEngine()
        result = ve.verify([1.0, 1.1, 1.05], metric="numerical")
        assert result.agreement > 0.8

    def test_cosine_similarity(self):
        ve = VerificationEngine()
        result = ve.verify(
            [[1, 0, 0], [1, 0.1, 0]],
            metric="cosine",
        )
        assert result.agreement > 0.9

    def test_single_output(self):
        ve = VerificationEngine()
        result = ve.verify(["only_one"])
        assert result.agreement == 1.0

    def test_auto_detect_metric(self):
        ve = VerificationEngine()
        result = ve.verify(["a", "b"])  # auto → classification
        assert result.agreement == 0.5  # "a" matches itself, "b" doesn't


# ── Adapters ──

class TestSklearnAdapter:
    def test_wrap_as_node(self):
        adapter = SklearnAdapter()
        node = adapter.wrap_as_node(
            model=None,
            name="test_sklearn",
            capabilities=["classification"],
        )
        assert node.framework == Framework.SKLEARN
        assert "classification" in node.capabilities
        assert node.name == "test_sklearn"


# ── Resources ──

class TestResourceManager:
    def test_snapshot(self):
        rm = ResourceManager()
        p = rm.profile
        assert p.cpu_count >= 1
        assert p.platform in ("Windows", "Linux", "Darwin")

    def test_state(self):
        rm = ResourceManager()
        assert rm.state in WorkloadState

    def test_can_fit(self):
        rm = ResourceManager()
        assert rm.can_fit(1)  # 1MB always fits


# ── Telemetry ──

class TestTelemetry:
    def test_record(self):
        t = Telemetry()
        t.record("test_event", node_name="x", duration_ms=1.5, success=True)
        assert t.get_counter("test_event") == 1
        assert t.avg_latency("test_event") == 1.5

    def test_disabled(self):
        t = Telemetry(enabled=False)
        t.record("test")
        assert t.get_counter("test") == 0

    def test_summary(self):
        t = Telemetry()
        t.record("a", success=True)
        t.record("a", success=False)
        s = t.summary()
        assert s["total_events"] == 2
        assert s["failures"] == 1


# ── Cache ──

class TestFeatureCache:
    def test_put_get(self):
        c = FeatureCache()
        c.put("k1", [1, 2, 3])
        assert c.get("k1") == [1, 2, 3]
        assert c.hit_rate > 0

    def test_miss(self):
        c = FeatureCache()
        assert c.get("nonexistent") is None
        assert c.size == 0

    def test_eviction(self):
        c = FeatureCache(max_entries=2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        assert c.size == 2


# ── NeuralMesh (Integration) ──

class TestNeuralMesh:
    def test_register_and_analyze(self):
        mesh = NeuralMesh()
        node = NeuralNode(
            name="identity",
            capabilities=["echo"],
            execute_fn=lambda data, **kw: data,
        )
        mesh.register_node(node)
        result = mesh.analyze(data="hello", task="echo")
        assert result.success
        assert result.output == "hello"
        # Executing successfully does not supply a model confidence score.
        assert result.confidence == 0
        assert len(result.nodes_used) > 0

    def test_analyze_no_matching_node(self):
        mesh = NeuralMesh()
        result = mesh.analyze(data="x", task="nonexistent_task")
        assert not result.success

    def test_mesh_summary(self):
        mesh = NeuralMesh()
        mesh.register_node(NeuralNode(name="a", capabilities=["c1"]))
        s = mesh.summary()
        assert s["registry"]["total_nodes"] == 1

    def test_cache_hit(self):
        mesh = NeuralMesh()
        node = NeuralNode(name="fast", capabilities=["cached_task"],
                          execute_fn=lambda data, **kw: data * 2)
        mesh.register_node(node)
        r1 = mesh.analyze(data=5, task="cached_task")
        r2 = mesh.analyze(data=5, task="cached_task")
        assert "served from cache" in r2.warnings

    def test_multimodal_register(self):
        mesh = NeuralMesh()
        mesh.register_capability("image_encode", Modality.IMAGE)
        mesh.register_capability("audio_encode", Modality.AUDIO)
        assert len(mesh.capabilities.all_names) == 2
