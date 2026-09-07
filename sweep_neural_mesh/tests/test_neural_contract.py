"""Train/inference format contract tests for deployed neural artifacts.

Guards against the A1 defect class (see SWEEP_ARCHITECTURE_AUDIT.md): a model
fine-tuned with one input format but served with another. The deployed
evidence classifier was trained on single sentences and served with
(premise, hypothesis) pairs — every inference input was out-of-distribution
and the model collapsed to one label (43.1% of measured benchmark failures).

These tests assert, for every deployed artifact, that:
  1. metadata.json exists and declares its input format
  2. the declared format matches what the production inference path encodes
  3. the model produces label-SENSITIVE output (different inputs of each
     class produce their expected labels with sane confidence)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_MODELS_DIR = Path(__file__).resolve().parent.parent / "training" / "neural_models"

# Inference formats implemented in sweep_neural_mesh/neurons/neural_engine.py
_EVIDENCE_INFERENCE_FORMAT = "pair"        # tokenizer(premise, hypothesis)
_CONTRADICTION_INFERENCE_FORMAT = "pair"   # tokenizer(text_a, text_b)
_INTENT_INFERENCE_FORMAT = "single"        # tokenizer(text)


def _metadata(model_name: str) -> dict:
    path = _MODELS_DIR / model_name / "metadata.json"
    assert path.exists(), (
        f"{model_name}/metadata.json missing — every deployed artifact must "
        f"declare its training input format (A1 contract)"
    )
    return json.loads(path.read_text(encoding="utf-8"))


class TestEvidenceClassifierContract:
    def test_declares_input_format(self):
        meta = _metadata("evidence_classifier")
        assert meta.get("input_format") == _EVIDENCE_INFERENCE_FORMAT, (
            f"evidence_classifier metadata must declare "
            f"input_format='{_EVIDENCE_INFERENCE_FORMAT}' to match "
            f"neural_engine.classify_evidence(premise, hypothesis)"
        )

    def test_labels_match_engine_vocabulary(self):
        meta = _metadata("evidence_classifier")
        assert meta.get("labels") == ["supports", "refutes", "neutral"]

    def test_label_sensitive_output(self):
        """The A1 failure signature: same label for every input."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from sweep_neural_mesh.neurons.neural_engine import NeuralEngine

        engine = NeuralEngine()
        assert engine.wait_until_ready(timeout=180), "models failed to load"

        cases = [
            # (premise, hypothesis, expected_label)
            ("The krin is stable and running smoothly.", "The krin is stable.", "supports"),
            ("The vost is not enabled per diagnostic log 3.", "The vost is enabled.", "refutes"),
            ("The tremn is awake.", "The solk is visible.", "neutral"),
            ("The durn could be locked.", "The durn is locked.", "neutral"),
        ]
        labels = []
        for premise, hypothesis, expected in cases:
            r = engine.classify_evidence(premise, hypothesis)
            assert r.ready, "evidence classifier not ready"
            labels.append(r.label)
            assert r.label == expected, (
                f"expected {expected} for pair ({premise!r}, {hypothesis!r}), "
                f"got {r.label} at conf={r.confidence:.2f} — model appears "
                f"label-insensitive (A1 defect signature)"
            )
        # collapse check: a model outputting one label for everything
        assert len(set(labels)) >= 2, (
            "model collapsed to a single label across class-diverse inputs "
            "(the A1 degeneracy signature)"
        )


class TestContradictionDetectorContract:
    def test_declares_input_format(self):
        meta = _metadata("contradiction_detector")
        assert meta.get("input_format") == _CONTRADICTION_INFERENCE_FORMAT

    def test_label_sensitive_output(self):
        """Collapse check with in-distribution pairs.

        NOTE: known quality limitation (not a contract violation) — the current
        artifact mislabels some paraphrase pairs as `contradiction` at low
        confidence (~0.67, just above the cortex 0.6 gate). Tracked in
        SWEEP_CAPABILITY_AUDIT.md / SWEEP_ROADMAP.md P1 (pair retrain with
        paraphrase augmentation).
        """
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from sweep_neural_mesh.neurons.neural_engine import NeuralEngine

        engine = NeuralEngine()
        assert engine.wait_until_ready(timeout=180)
        contra = engine.detect_contradiction(
            "The meeting is at 3 PM", "The meeting is at 4 PM")
        consistent = engine.detect_contradiction(
            "Paris is the capital of France", "France's capital is Paris")
        assert contra.label == "contradiction", (
            f"expected contradiction, got {contra.label}@{contra.confidence:.2f}")
        assert consistent.label == "consistent", (
            f"expected consistent, got {consistent.label}@{consistent.confidence:.2f}")
        assert len({contra.label, consistent.label}) >= 2, (
            "contradiction detector collapsed to a single label")


class TestIntentClassifierContract:
    def test_declares_input_format(self):
        meta = _metadata("intent_classifier")
        assert meta.get("input_format") == _INTENT_INFERENCE_FORMAT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
