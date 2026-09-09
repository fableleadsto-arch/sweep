"""
Tests for the hypothesis & counterfactual engine (Phase 9).

Verifies:
1. Hypotheses are explanations, never conclusions (status lifecycle)
2. Evidence updates confidence monotonically via a real update rule
3. Elimination on strong contradiction
4. Abductive best_explanation refuses to pick when ambiguous
5. Discriminating test recommendation separates top-2 hypotheses
6. Counterfactual consistency checks against the world model
7. Scaffolding honesty
"""

import pytest

from sweep_cognitive.world import WorldModel, RelationshipType
from sweep_cognitive.hypotheses import (
    Hypothesis,
    HypothesisStatus,
    HypothesisEngine,
    EvidenceItem,
    CounterfactualEngine,
)


class TestHypothesisBasics:
    def test_basic_hypothesis(self):
        hyp = Hypothesis(statement="The server is down due to a network fault")
        assert hyp.status == HypothesisStatus.OPEN
        assert hyp.confidence == 0.5

    def test_supporting_and_contradicting_split(self):
        hyp = Hypothesis(statement="H")
        hyp.evidence.append(EvidenceItem(description="a", weight=0.8))
        hyp.evidence.append(EvidenceItem(description="b", weight=-0.6))
        assert len(hyp.supporting) == 1
        assert len(hyp.contradicting) == 1

    def test_reliability_scales_weight(self):
        e = EvidenceItem(description="a", weight=0.8, reliability=0.5)
        assert e.effective_weight == pytest.approx(0.4)

    def test_ids_unique(self):
        assert Hypothesis(statement="a").hypothesis_id != Hypothesis(statement="b").hypothesis_id

    def test_to_dict(self):
        d = Hypothesis(statement="x").to_dict()
        assert d["status"] == "open"
        assert d["confidence"] == 0.5


class TestEvidenceUpdates:
    def setup_method(self):
        self.engine = HypothesisEngine()

    def test_supporting_evidence_raises_confidence(self):
        hyp = self.engine.generate("Network fault", prior=0.5)
        c0 = hyp.confidence
        self.engine.add_evidence(hyp.hypothesis_id, "ping fails", weight=0.8)
        assert hyp.confidence > c0

    def test_contradicting_evidence_lowers_confidence(self):
        hyp = self.engine.generate("Network fault", prior=0.5)
        c0 = hyp.confidence
        self.engine.add_evidence(hyp.hypothesis_id, "ping succeeds", weight=-0.6)
        assert hyp.confidence < c0

    def test_more_evidence_monotone(self):
        hyp = self.engine.generate("H", prior=0.5)
        confs = [hyp.confidence]
        for i in range(4):
            self.engine.add_evidence(hyp.hypothesis_id, f"ev{i}", weight=0.5)
            confs.append(hyp.confidence)
        assert confs == sorted(confs)
        assert all(c < 1.0 for c in confs)

    def test_strong_contradiction_eliminates(self):
        hyp = self.engine.generate("H", prior=0.8)
        self.engine.add_evidence(hyp.hypothesis_id, "fatal", weight=-0.9, reliability=1.0)
        assert hyp.status == HypothesisStatus.ELIMINATED
        assert hyp.confidence == 0.0

    def test_weak_contradiction_weakens(self):
        hyp = self.engine.generate("H", prior=0.8)
        self.engine.add_evidence(hyp.hypothesis_id, "oddity", weight=-0.3)
        assert hyp.status == HypothesisStatus.WEAKENED

    def test_no_evidence_on_eliminated(self):
        hyp = self.engine.generate("H", prior=0.5)
        self.engine.add_evidence(hyp.hypothesis_id, "fatal", weight=-0.9, reliability=1.0)
        with pytest.raises(ValueError):
            self.engine.add_evidence(hyp.hypothesis_id, "more", weight=0.5)

    def test_unknown_hypothesis_raises(self):
        with pytest.raises(KeyError):
            self.engine.add_evidence("hyp_nope", "x", weight=0.5)


class TestGeneration:
    def setup_method(self):
        self.engine = HypothesisEngine()

    def test_generate_registers(self):
        hyp = self.engine.generate("H1", prior=0.6)
        assert self.engine.get(hyp.hypothesis_id) is hyp
        assert hyp.confidence == 0.6  # starts at prior

    def test_generate_from_observations_creates_alternative(self):
        hyps = self.engine.generate_from_observations(
            ["The CPU is at 100%.", "Disk I/O is saturated."], context="server alert"
        )
        assert len(hyps) == 3  # 2 per-observation + 1 alternative
        assert any(h.metadata.get("generated_from") == "alternative_template" for h in hyps)

    def test_empty_observations(self):
        assert self.engine.generate_from_observations([]) == []

    def test_viable_excludes_eliminated(self):
        h1 = self.engine.generate("H1", prior=0.7)
        h2 = self.engine.generate("H2", prior=0.3)
        self.engine.add_evidence(h1.hypothesis_id, "fatal", weight=-0.9, reliability=1.0)
        viable = self.engine.viable()
        assert h1 not in viable
        assert h2 in viable

    def test_viable_ranks_by_confidence(self):
        h_low = self.engine.generate("low", prior=0.3)
        h_high = self.engine.generate("high", prior=0.8)
        ranked = self.engine.viable()
        assert ranked[0].hypothesis_id == h_high.hypothesis_id
        # Leader is marked SUPPORTED
        assert ranked[0].status == HypothesisStatus.SUPPORTED


class TestAbduction:
    def setup_method(self):
        self.engine = HypothesisEngine()

    def test_clear_winner_selected(self):
        h1 = self.engine.generate("clear cause", prior=0.85)
        h2 = self.engine.generate("weak alternative", prior=0.20)
        best = self.engine.best_explanation()
        assert best is not None
        assert best.hypothesis_id == h1.hypothesis_id

    def test_ambiguous_refuses_to_pick(self):
        """Hallucination control: near-tied hypotheses → no pick."""
        self.engine.generate("H1", prior=0.55)
        self.engine.generate("H2", prior=0.50)
        assert self.engine.best_explanation() is None

    def test_empty_engine(self):
        assert self.engine.best_explanation() is None


class TestDiscriminatingTests:
    def setup_method(self):
        self.engine = HypothesisEngine()

    def test_recommends_separating_observation(self):
        h1 = self.engine.generate(
            "network fault", prior=0.6,
            predictions_if_true=["Ping to gateway fails"],
        )
        h2 = self.engine.generate(
            "disk failure", prior=0.4,
            predictions_if_true=["Disk SMART errors present"],
        )
        test = self.engine.recommend_discriminating_test()
        assert test is not None
        assert "Ping to gateway fails" in test.description
        assert h1.hypothesis_id in test.expectations

    def test_no_discriminator_when_predictions_overlap(self):
        self.engine.generate("H1", prior=0.6, predictions_if_true=["same thing"])
        self.engine.generate("H2", prior=0.5, predictions_if_true=["same thing"])
        assert self.engine.recommend_discriminating_test() is None

    def test_single_hypothesis_no_test(self):
        self.engine.generate("only H", prior=0.6, predictions_if_true=["x"])
        assert self.engine.recommend_discriminating_test() is None


class TestCounterfactual:
    def setup_method(self):
        self.world = WorldModel()
        self.engine = CounterfactualEngine(self.world)

    def test_consistent_premise(self):
        cat = self.world.add_entity("cat")
        mat = self.world.add_entity("mat")
        result = self.engine.evaluate_premise(
            cat.id, RelationshipType.LOCATED_IN, mat.id,
            question="Would the cat be on the mat?",
        )
        assert result.is_consistent is True
        assert result.blocking_contradictions == []

    def test_inconsistent_premise(self):
        """Premise contradicts an existing DIFFERENT_FROM relationship."""
        a = self.world.add_entity("a")
        b = self.world.add_entity("b")
        self.world.add_relationship(a.id, RelationshipType.DIFFERENT_FROM, b.id)
        result = self.engine.evaluate_premise(a.id, RelationshipType.PART_OF, b.id)
        assert result.is_consistent is False
        assert len(result.blocking_contradictions) == 1

    def test_reverse_direction_inconsistent(self):
        """Existing PART_OF blocks a hypothetical DIFFERENT_FROM."""
        a = self.world.add_entity("wheel")
        b = self.world.add_entity("car")
        self.world.add_relationship(a.id, RelationshipType.PART_OF, b.id)
        result = self.engine.evaluate_premise(a.id, RelationshipType.DIFFERENT_FROM, b.id)
        assert result.is_consistent is False

    def test_unrelated_pair_consistent(self):
        a = self.world.add_entity("a")
        b = self.world.add_entity("b")
        c = self.world.add_entity("c")
        self.world.add_relationship(a.id, RelationshipType.DIFFERENT_FROM, c.id)
        result = self.engine.evaluate_premise(a.id, RelationshipType.PART_OF, b.id)
        assert result.is_consistent is True

    def test_observations_if_false(self):
        hyp = Hypothesis(
            statement="H",
            predictions_if_true=["log entry appears", "light turns on"],
        )
        engine = CounterfactualEngine(WorldModel())
        missing = engine.observations_if_false(hyp)
        assert missing == ["log entry appears", "light turns on"]

    def test_what_would_change(self):
        a = self.world.add_entity("a")
        b = self.world.add_entity("b")
        c = self.world.add_entity("c")
        self.world.add_relationship(b.id, RelationshipType.RELATED_TO, c.id)
        result = self.engine.what_would_change(a.id, RelationshipType.LOCATED_IN, b.id)
        assert result["affected_count"] >= 1
        assert b.id in result["affected_entities"]


class TestScaffoldingHonesty:
    def test_template_generation_is_marked(self):
        import sweep_cognitive.hypotheses as mod
        doc = mod.__doc__ or ""
        assert "TEMPORARY SCAFFOLDING" in doc

    def test_counterfactual_scope_is_marked(self):
        from sweep_cognitive.hypotheses import CounterfactualEngine
        doc = CounterfactualEngine.__doc__ or ""
        assert "NOT a learned world simulator" in doc or "graph operations" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
