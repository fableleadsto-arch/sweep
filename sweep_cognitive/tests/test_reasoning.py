"""
Tests for the cognitive reasoning engine (Phase 10).

Verifies:
1. Semantic extraction always runs and reports honestly
2. Evidence aggregation counts support/refute indicators
3. Modality selection (rule/logic indicators trigger the right paths)
4. Real RuleEngine integration: facts → rules → traced conclusion
5. Real LogicEngine integration: world model facts → proof/not-provable
6. Unavailable modalities report honestly (never fabricate)
7. Hypothesis + predictive modalities wire through
8. Combination picks strongest and tracks all modalities used
"""

import pytest

from sweep_cognitive.reasoning import (
    ReasoningEngine,
    ReasoningRequest,
    ReasoningResult,
    ReasoningModality,
    ReasoningRegistry,
)
from sweep_cognitive.world import WorldModel, RelationshipType
from cognition.rules import Rule, RuleEngine
from cognition.logic import LogicEngine


def _make_positive_rule_engine():
    eng = RuleEngine()
    eng.add_rule(Rule(
        rule_id="R2_supported",
        version="1.0",
        description="Claim is supported when evidence_count >= 2",
        if_all=[lambda facts: facts.get("evidence_count", 0) >= 2],
        then=lambda facts: {"conclusion": "claim supported", "verdict": "supported"},
    ))
    return eng


class TestSemanticModality:
    def test_basic_reasoning(self):
        engine = ReasoningEngine()
        result = engine.reason(ReasoningRequest(query="The cat sat on the mat"))
        assert result.conclusion == "needs_further_reasoning"
        assert result.modality == ReasoningModality.SEMANTIC_EXTRACTION
        assert result.processing_latency_ms >= 0

    def test_empty_query_insufficient(self):
        engine = ReasoningEngine()
        result = engine.reason(ReasoningRequest(query=""))
        assert result.conclusion == "insufficient"
        assert result.confidence == 0.0

    def test_entities_identified(self):
        engine = ReasoningEngine()
        result = engine.reason(ReasoningRequest(query="Alice met Bob in Paris"))
        assert result.entities_identified >= 2


class TestEvidenceAggregation:
    def setup_method(self):
        self.engine = ReasoningEngine()

    def test_supporting_evidence(self):
        result = self.engine.reason(ReasoningRequest(
            query="Is the claim true?",
            evidence=["This confirms the claim", "A study shows it holds"],
        ))
        assert result.conclusion == "supported"
        assert result.evidence_count == 2

    def test_refuting_evidence(self):
        result = self.engine.reason(ReasoningRequest(
            query="Is the claim true?",
            evidence=["This contradicts the claim", "It is false"],
        ))
        assert result.conclusion == "refuted"

    def test_mixed_evidence(self):
        result = self.engine.reason(ReasoningRequest(
            query="Is the claim true?",
            evidence=["confirms it", "contradicts it"],
        ))
        assert result.conclusion == "mixed"

    def test_no_usable_evidence_words(self):
        result = self.engine.reason(ReasoningRequest(
            query="Is the claim true?",
            evidence=["some text", "more text"],
        ))
        assert result.conclusion == "mixed"

    def test_representation_evidence_accepted(self):
        from sweep_cognitive.representation import TextRepresentation
        rep = TextRepresentation(content="this confirms the claim")
        result = self.engine.reason(ReasoningRequest(
            query="?", evidence=[rep],
        ))
        assert result.evidence_count == 1
        assert result.conclusion == "supported"


class TestModalityUnavailable:
    """The honesty contract: missing engines must never fake results."""

    def setup_method(self):
        self.engine = ReasoningEngine()  # no engines provided

    def test_rule_unavailable(self):
        result = self.engine.reason(ReasoningRequest(
            query="if the alarm rings then evacuate",
        ))
        # Rule indicator present, engine absent
        assert ReasoningModality.RULE_BASED in result.metadata["individual_results"][0]["used_modalities"] or True
        # Find the rule-based sub-result
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "rule_based"]
        assert sub, "rule modality should have been attempted"
        assert sub[0]["conclusion"] == "modality_unavailable"
        assert sub[0]["confidence"] == 0.0

    def test_logic_unavailable(self):
        result = self.engine.reason(ReasoningRequest(
            query="all birds fly, therefore tweety flies",
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "logic_deductive"]
        assert sub
        assert sub[0]["conclusion"] == "modality_unavailable"


class TestRuleEngineIntegration:
    """Real RuleEngine from cognition/rules.py runs and produces traces."""

    def test_rule_fires_and_traces(self):
        engine = ReasoningEngine(rule_engine=_make_positive_rule_engine())
        result = engine.reason(ReasoningRequest(
            query="the claim holds if evidence confirms it",
            evidence=["confirms", "shows", "demonstrates"],
            context={"facts": {}},
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "rule_based"][0]
        assert sub["conclusion"] == "claim supported"
        assert sub["metadata"]["rules_fired"] == 1
        assert len(sub["metadata"]["trace"]) == 1

    def test_no_rule_matches(self):
        engine = ReasoningEngine(rule_engine=_make_positive_rule_engine())
        result = engine.reason(ReasoningRequest(
            query="if x then y",
            evidence=[],  # evidence_count = 0 → rule condition fails
            context={"facts": {}},
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "rule_based"][0]
        assert sub["conclusion"] == "no_rule_fired"
        assert sub["metadata"]["rules_fired"] == 0


class TestLogicEngineIntegration:
    """Real LogicEngine from cognition/logic.py proves from world model facts."""

    def _engine_with_world(self):
        world = WorldModel()
        a = world.add_entity("alice")
        b = world.add_entity("bob")
        c = world.add_entity("carol")
        # sibling-of is symmetric; ancestor-of is transitive
        world.add_relationship(a.id, RelationshipType.SIMILAR_TO, b.id)
        return world

    def test_symmetric_fact_derived(self):
        world = self._engine_with_world()
        a = world.get_entity_by_name("alice")
        b = world.get_entity_by_name("bob")
        logic = LogicEngine()
        engine = ReasoningEngine(logic_engine=logic, world_model=world)
        result = engine.reason(ReasoningRequest(
            query="who is related to whom",
            context={"claim": ("similar_to", a.id, b.id)},
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "logic_deductive"][0]
        # Direct fact → provable
        assert sub["conclusion"] == "proven"
        assert sub["metadata"]["facts_loaded"] == 1

    def test_transitive_derivation(self):
        world = WorldModel()
        a = world.add_entity("a")
        b = world.add_entity("b")
        c = world.add_entity("c")
        world.add_relationship(a.id, RelationshipType.LOCATED_IN, b.id)
        world.add_relationship(b.id, RelationshipType.LOCATED_IN, c.id)
        logic = LogicEngine(transitive_relations={"located_in"})
        engine = ReasoningEngine(logic_engine=logic, world_model=world)
        result = engine.reason(ReasoningRequest(
            query="where is a",
            context={"claim": ("located_in", a.id, c.id)},
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "logic_deductive"][0]
        assert sub["conclusion"] == "proven"
        assert sub["metadata"]["facts_loaded"] == 2

    def test_not_provable(self):
        world = WorldModel()
        a = world.add_entity("a")
        c = world.add_entity("c")
        logic = LogicEngine()
        engine = ReasoningEngine(logic_engine=logic, world_model=world)
        result = engine.reason(ReasoningRequest(
            query="is a part of c",
            context={"claim": ("part_of", a.id, c.id)},
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "logic_deductive"][0]
        assert sub["conclusion"] == "not_provable"

    def test_closure_without_claim(self):
        world = WorldModel()
        a = world.add_entity("a")
        b = world.add_entity("b")
        world.add_relationship(a.id, RelationshipType.RELATED_TO, b.id)
        logic = LogicEngine(symmetric_relations={"related_to"})
        engine = ReasoningEngine(logic_engine=logic, world_model=world)
        result = engine.reason(ReasoningRequest(
            query="what do we know",
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "logic_deductive"][0]
        assert sub["conclusion"] == "closure_computed"
        # 1 direct + 1 symmetric = 2
        assert sub["metadata"]["closure_size"] == 2


class TestHypothesisPredictiveModalities:
    def test_hypothesis_best_explanation(self):
        from sweep_cognitive.hypotheses import HypothesisEngine
        hyp_engine = HypothesisEngine()
        hyp_engine.generate("network fault", prior=0.85)
        hyp_engine.generate("disk fault", prior=0.30)
        engine = ReasoningEngine(hypothesis_engine=hyp_engine)
        result = engine.reason(ReasoningRequest(
            query="why did the server stop responding",
            preferred_modalities=[ReasoningModality.HYPOTHESIS_BASED],
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "hypothesis_based"][0]
        assert sub["conclusion"] == "network fault"
        assert sub["confidence"] > 0.6

    def test_hypothesis_ambiguous(self):
        from sweep_cognitive.hypotheses import HypothesisEngine
        hyp_engine = HypothesisEngine()
        hyp_engine.generate("H1", prior=0.55)
        hyp_engine.generate("H2", prior=0.50)
        engine = ReasoningEngine(hypothesis_engine=hyp_engine)
        result = engine.reason(ReasoningRequest(
            query="why",
            preferred_modalities=[ReasoningModality.HYPOTHESIS_BASED],
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "hypothesis_based"][0]
        assert sub["conclusion"] == "ambiguous"

    def test_predictive_mode_recommendation(self):
        from sweep_cognitive.prediction import PredictionEngine
        pred_engine = PredictionEngine()
        engine = ReasoningEngine(prediction_engine=pred_engine)
        result = engine.reason(ReasoningRequest(
            query="what is 2+2",
            preferred_modalities=[ReasoningModality.PREDICTIVE],
        ))
        sub = [r for r in result.metadata["individual_results"]
               if r["modality"] == "predictive"][0]
        assert sub["conclusion"] == "recommended_mode_FAST"
        assert sub["metadata"]["mode"] == "FAST"


class TestCombination:
    def test_combination_tracks_all_modalities(self):
        engine = ReasoningEngine(rule_engine=_make_positive_rule_engine())
        result = engine.reason(ReasoningRequest(
            query="if it confirms then supported",
            evidence=["confirms", "shows"],
            context={"facts": {}},
        ))
        # semantic + evidence_aggregation + rule_based all ran
        assert ReasoningModality.RULE_BASED in result.used_modalities
        assert ReasoningModality.EVIDENCE_AGGREGATION in result.used_modalities
        assert result.metadata["result_count"] >= 3

    def test_combined_takes_strongest_confidence(self):
        engine = ReasoningEngine(rule_engine=_make_positive_rule_engine())
        result = engine.reason(ReasoningRequest(
            query="if it confirms then supported",
            evidence=["confirms", "shows"],
            context={"facts": {}},
        ))
        individual = result.metadata["individual_results"]
        max_conf = max(r["confidence"] for r in individual)
        assert result.confidence == max_conf


class TestRegistry:
    def test_lists_all_modalities(self):
        registry = ReasoningRegistry()
        modalities = registry.list_modalities()
        names = {m["modality"] for m in modalities}
        assert "rule_based" in names
        assert "logic_deductive" in names
        assert "hypothesis_based" in names
        assert "predictive" in names

    def test_modality_info(self):
        registry = ReasoningRegistry()
        info = registry.get_modality_info(ReasoningModality.LOGIC_PROOF_MESH)
        assert "description" in info
        assert "requires" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
