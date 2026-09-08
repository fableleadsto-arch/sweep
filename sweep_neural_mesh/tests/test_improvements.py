"""Tests for the 10 improvement modules: embeddings, amygdala, forgetting, STDP, homeostatic, adaptive pipeline, RPE, per-evidence grading, signal enhancements, semantic recall."""
from __future__ import annotations

import time
import unittest

from sweep_neural_mesh.neurons.embeddings import EmbeddingEngine
from sweep_neural_mesh.neurons.amygdala import Amygdala, ValenceCategory
from sweep_neural_mesh.neurons.forgetting import ForgettingCurve, MemoryTrace
from sweep_neural_mesh.neurons.signal import Signal, SignalType
from sweep_neural_mesh.neurons.plasticity import SynapticPlasticity, MasteryPhase
from sweep_neural_mesh.neurons.basal_ganglia import BasalGanglia, ActionProposal, ActionType
from sweep_neural_mesh.neurons.grading import EvidenceGrader, EvidenceGrade
from sweep_neural_mesh.neurons.cortex import ReasoningCortex, ReasoningTrace
from sweep_neural_mesh.neurons.complexity import (
    classify_query_complexity,
    select_reasoning_modules,
)


class TestEmbeddingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EmbeddingEngine()

    def test_fingerprint_returns_nonzero(self):
        fp = self.engine.fingerprint("hello world")
        self.assertNotEqual(fp.bits, 0)

    def test_identical_text_high_similarity(self):
        a = self.engine.fingerprint("machine learning algorithms")
        b = self.engine.fingerprint("machine learning algorithms")
        self.assertEqual(self.engine.similarity(a, b), 1.0)

    def test_similar_text_high_similarity(self):
        a = self.engine.fingerprint("machine learning algorithms")
        b = self.engine.fingerprint("deep learning algorithms")
        sim = self.engine.similarity(a, b)
        self.assertGreater(sim, 0.3, f"Expected >0.3 similarity for similar texts, got {sim}")

    def test_different_text_lower_similarity(self):
        a = self.engine.fingerprint("machine learning algorithms")
        b = self.engine.fingerprint("cooking recipes for dinner")
        sim = self.engine.similarity(a, b)
        self.assertLess(sim, 0.7, f"Expected <0.7 similarity for different texts, got {sim}")

    def test_text_similarity_direct(self):
        sim = self.engine.text_similarity("hello world", "hello world")
        self.assertEqual(sim, 1.0)

    def test_batch_fingerprint(self):
        fps = self.engine.batch_fingerprint(["hello", "world", "test"])
        self.assertEqual(len(fps), 3)
        for fp in fps:
            self.assertIsNotNone(fp.bits)

    def test_empty_text(self):
        fp = self.engine.fingerprint("")
        self.assertEqual(fp.bits, 0)


class TestAmygdala(unittest.TestCase):
    def setUp(self):
        self.amygdala = Amygdala()

    def test_evaluate_threat(self):
        result = self.amygdala.evaluate("this is dangerous and harmful")
        self.assertEqual(result.category, ValenceCategory.THREAT)
        self.assertLess(result.valence, 0)

    def test_evaluate_reward(self):
        result = self.amygdala.evaluate("great success and reward achieved")
        self.assertEqual(result.category, ValenceCategory.REWARD)
        self.assertGreater(result.valence, 0)

    def test_evaluate_neutral(self):
        result = self.amygdala.evaluate("the weather is normal today")
        self.assertEqual(result.category, ValenceCategory.NEUTRAL)

    def test_evaluate_novelty(self):
        result = self.amygdala.evaluate("this is completely new and innovative")
        self.assertEqual(result.category, ValenceCategory.NOVELTY)

    def test_priority_multiplier(self):
        threat = self.amygdala.evaluate("dangerous threat risk")
        neutral = self.amygdala.evaluate("regular information")
        self.assertGreater(threat.priority_multiplier, neutral.priority_multiplier)

    def test_stats_property(self):
        self.amygdala.evaluate("good great reward")
        self.amygdala.evaluate("bad dangerous threat")
        stats = self.amygdala.stats
        self.assertGreaterEqual(stats["total_valuations"], 2)


class TestForgettingCurve(unittest.TestCase):
    def setUp(self):
        self.curve = ForgettingCurve()

    def test_encode_memory(self):
        trace = self.curve.encode(content="test content")
        self.assertEqual(trace.content, "test content")
        self.assertGreater(trace.stability, 0)

    def test_retention_decays(self):
        trace = self.curve.encode(content="test")
        self.curve.update_retention(trace.memory_id)
        # Should be near 1.0 since we just encoded it
        self.assertLessEqual(trace.retention, 1.0)
        self.assertGreater(trace.retention, 0.0)

    def test_review_boosts_stability(self):
        trace = self.curve.encode(content="test")
        old_stability = trace.stability
        self.curve.review(trace.memory_id)
        self.assertGreater(trace.stability, old_stability)

    def test_is_forgotten_property(self):
        trace = self.curve.encode(content="test")
        trace.retention = 0.01
        self.assertTrue(trace.is_forgotten)

    def test_emotional_memory_decays_slower(self):
        emotional = self.curve.encode(content="emotional", emotional_boost=0.9)
        neutral = self.curve.encode(content="neutral", emotional_boost=0.0)
        # Manually set both to same last_reviewed time in the past
        past = time.time() - 3600
        emotional.last_reviewed = past
        neutral.last_reviewed = past
        self.curve.update_retention(emotional.memory_id)
        self.curve.update_retention(neutral.memory_id)
        self.assertGreater(emotional.retention, neutral.retention)

    def test_stats_property(self):
        self.curve.encode(content="a")
        self.curve.encode(content="b")
        stats = self.curve.stats
        self.assertGreaterEqual(stats["total_memories"], 2)


class TestSignalEnhancements(unittest.TestCase):
    def test_signal_has_embedding_bits(self):
        sig = Signal(data={"text": "hello"}, embedding_bits=42)
        self.assertEqual(sig.embedding_bits, 42)

    def test_signal_has_urgency(self):
        sig = Signal(data={"text": "urgent"}, urgency=0.9)
        self.assertEqual(sig.urgency, 0.9)

    def test_signal_has_ttl(self):
        sig = Signal(data={"text": "test"}, ttl=60.0)
        self.assertEqual(sig.ttl, 60.0)

    def test_signal_has_emotional_valence(self):
        sig = Signal(data={"text": "threat"}, emotional_valence=-0.5)
        self.assertEqual(sig.emotional_valence, -0.5)

    def test_amplify_preserves_embedding(self):
        sig = Signal(data={"text": "hello"}, embedding_bits=42, urgency=0.5)
        amplified = sig.amplify(1.5)
        self.assertEqual(amplified.embedding_bits, 42)
        self.assertAlmostEqual(amplified.urgency, 0.75, places=2)

    def test_stamp_preserves_embedding(self):
        sig = Signal(data={"text": "hello"}, embedding_bits=42)
        stamped = sig.stamp("center1")
        self.assertEqual(stamped.embedding_bits, 42)

    def test_to_dict_shows_embedding_flag(self):
        sig = Signal(data={"text": "hello"}, embedding_bits=42)
        d = sig.to_dict()
        self.assertTrue(d["has_embedding"])

    def test_to_dict_shows_urgency(self):
        sig = Signal(data={"text": "hello"}, urgency=0.75)
        d = sig.to_dict()
        self.assertEqual(d["urgency"], 0.75)


class TestSTDPlasticity(unittest.TestCase):
    def setUp(self):
        self.plasticity = SynapticPlasticity()

    def test_stdp_ltp(self):
        weight_change = self.plasticity.record_stdp_event(
            "center_a", "center_b",
            pre_time=0.0, post_time=0.01,
        )
        self.assertGreater(weight_change, 0, "Pre before post should produce LTP")

    def test_stdp_ltd(self):
        weight_change = self.plasticity.record_stdp_event(
            "center_a", "center_b",
            pre_time=0.01, post_time=0.0,
        )
        self.assertLess(weight_change, 0, "Post before pre should produce LTD")

    def test_stdp_outside_window(self):
        weight_change = self.plasticity.record_stdp_event(
            "center_a", "center_b",
            pre_time=0.0, post_time=1.0,
        )
        self.assertEqual(weight_change, 0.0, "Outside timing window should produce no change")

    def test_stdp_stats(self):
        self.plasticity.record_stdp_event("a", "b", 0.0, 0.01)
        self.plasticity.record_stdp_event("a", "b", 0.01, 0.0)
        stats = self.plasticity.get_stdp_stats()
        key = "a->b"
        self.assertIn(key, stats)
        self.assertEqual(stats[key]["pre_spikes"], 2)
        self.assertEqual(stats[key]["post_spikes"], 2)

    def test_stdp_metaplasticity_increases(self):
        for i in range(5):
            self.plasticity.record_stdp_event(
                "a", "b", i * 0.01, i * 0.01 + 0.005,
            )
        stats = self.plasticity.get_stdp_stats()
        self.assertGreater(stats["a->b"]["metaplasticity"], 0)


class TestHomeostaticPlasticity(unittest.TestCase):
    def setUp(self):
        self.plasticity = SynapticPlasticity()

    def test_homeostatic_scaling_no_data(self):
        scaled = self.plasticity.homeostatic_scaling()
        self.assertEqual(scaled, 0)

    def test_homeostatic_scaling(self):
        for i in range(10):
            self.plasticity.record_activation(
                "evidence_gatherer", "credibility_assessor",
                output_quality=0.9, processing_time_ms=10.0,
            )
        self.plasticity.record_activation(
            "evidence_gatherer", "credibility_assessor",
            output_quality=0.1, processing_time_ms=10.0,
        )
        scaled = self.plasticity.homeostatic_scaling()
        self.assertIsInstance(scaled, int)

    def test_homeostatic_stats(self):
        stats = self.plasticity.get_homeostatic_stats()
        self.assertIn("total_synapses_scaled", stats)
        self.assertIn("avg_homeostatic_offset", stats)


class TestRewardPredictionError(unittest.TestCase):
    def setUp(self):
        self.bg = BasalGanglia()

    def test_rpe_in_decisions(self):
        proposals = [
            ActionProposal(
                action_type=ActionType.PROCEED_TO_CONSENSUS,
                confidence=0.7,
                reasoning="test",
                evidence_ids=[],
            ),
        ]
        decisions = self.bg.decide(proposals, {"confidence": 0.7, "evidence_count": 5})
        for d in decisions:
            self.assertEqual(d.expected_value, 0.0)
            self.assertEqual(d.prediction_error, 0.0)

    def test_rpe_after_learning(self):
        proposals = [
            ActionProposal(
                action_type=ActionType.PROCEED_TO_CONSENSUS,
                confidence=0.7,
                reasoning="test",
                evidence_ids=[],
            ),
        ]
        decisions = self.bg.decide(proposals, {"confidence": 0.7, "evidence_count": 5})
        rpes = self.bg.learn(decisions, reward=0.8)
        self.assertEqual(len(rpes), 1)
        self.assertAlmostEqual(rpes[0], 0.8, places=1)

    def test_rpe_stats(self):
        proposals = [
            ActionProposal(
                action_type=ActionType.PROCEED_TO_CONSENSUS,
                confidence=0.7, reasoning="test", evidence_ids=[],
            ),
        ]
        decisions = self.bg.decide(proposals, {"confidence": 0.7, "evidence_count": 5})
        self.bg.learn(decisions, reward=0.8)
        stats = self.bg.get_rpe_stats()
        self.assertEqual(stats["total_rpe_events"], 1)
        self.assertGreater(stats["avg_rpe"], 0)


class TestPerEvidenceGrading(unittest.TestCase):
    def setUp(self):
        self.grader = EvidenceGrader()

    def test_per_evidence_grading(self):
        sig1 = Signal(data={"text": "evidence one is very detailed and comprehensive"}, confidence=0.9)
        sig2 = Signal(data={"text": "short"}, confidence=0.3)
        grades = self.grader.grade_per_evidence(
            evidence_signals=[sig1, sig2],
            credibility_signals=[],
            decision_outcome="approved",
        )
        self.assertEqual(len(grades), 2)
        self.assertIn("depth_score", grades[0])
        self.assertIn("reliability_score", grades[0])
        self.assertIn("feedback_useful", grades[0])

    def test_feedback_computation(self):
        sig1 = Signal(data={"text": "detailed evidence"}, confidence=0.9)
        grades = self.grader.grade_per_evidence(
            evidence_signals=[sig1],
            credibility_signals=[],
            decision_outcome="approved",
        )
        feedback = self.grader.compute_feedback_from_grades(grades)
        self.assertIn("ltp_strength", feedback)
        self.assertIn("ltd_strength", feedback)
        self.assertIn("avg_quality", feedback)
        self.assertGreater(feedback["ltp_strength"], 0)

    def test_empty_grades_feedback(self):
        feedback = self.grader.compute_feedback_from_grades([])
        self.assertEqual(feedback["ltp_strength"], 0.0)
        self.assertEqual(feedback["avg_quality"], 0.5)


class TestAdaptivePipelineDepth(unittest.TestCase):
    def test_trivial_query(self):
        complexity = classify_query_complexity("hello", 0)
        self.assertEqual(complexity, "trivial")

    def test_simple_query(self):
        complexity = classify_query_complexity("what is machine learning", 0)
        self.assertEqual(complexity, "simple")

    def test_complex_query(self):
        complexity = classify_query_complexity(
            "analyze the causal relationships between these 10 evidence items",
            10,
        )
        self.assertIn(complexity, ("complex", "deep"))

    def test_deep_query_with_counterfactual(self):
        complexity = classify_query_complexity(
            "what if the evidence were different", 5,
        )
        self.assertIn(complexity, ("moderate", "complex", "deep"))

    def test_select_modules_trivial(self):
        modules = select_reasoning_modules("trivial", 0)
        self.assertEqual(modules, [])

    def test_select_modules_simple(self):
        modules = select_reasoning_modules("simple", 0)
        self.assertEqual(modules, ["common_sense"])

    def test_select_modules_deep(self):
        modules = select_reasoning_modules("deep", 10)
        self.assertIn("common_sense", modules)
        self.assertIn("abductive", modules)
        self.assertIn("theory_of_mind", modules)
        self.assertIn("causal", modules)
        self.assertIn("narrative", modules)
        self.assertIn("analogical", modules)
        self.assertIn("counterfactual", modules)

    def test_trace_includes_adaptive_fields(self):
        trace = ReasoningTrace(
            query="test",
            input_evidence_count=0,
            center_outputs={},
            integration_confidence=0.5,
            decision="test",
            decision_confidence=0.5,
            reasoning="test",
            total_latency_ms=1.0,
            query_complexity="deep",
            active_modules=["common_sense", "abductive"],
        )
        d = trace.to_dict()
        self.assertIn("adaptive_pipeline", d)
        self.assertEqual(d["adaptive_pipeline"]["query_complexity"], "deep")
        self.assertIn("common_sense", d["adaptive_pipeline"]["active_modules"])


if __name__ == "__main__":
    unittest.main()
