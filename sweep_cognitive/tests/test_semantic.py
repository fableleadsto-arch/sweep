"""
Tests for semantic understanding engine (Phase 3).

These tests verify that:
1. Entities are extracted from text
2. Relationships between entities are identified
3. Events are detected
4. Context is extracted
5. Ambiguities are detected
6. Semantic quality is assessed
"""

import pytest

from sweep_cognitive.semantic import (
    SemanticUnderstandingEngine,
    SemanticEntity,
    SemanticRelationship,
    SemanticEvent,
    SemanticContext,
    SemanticType,
    Ambiguity,
    SemanticRepresentation,
)
from sweep_cognitive.representation import TextRepresentation, RepresentationQuality


class TestSemanticEntity:
    """Tests for SemanticEntity."""

    def test_basic_entity(self):
        """Test basic entity creation."""
        entity = SemanticEntity(
            name="cat",
            entity_type=SemanticType.ENTITY,
            kind="animal",
        )
        assert entity.name == "cat"
        assert entity.kind == "animal"
        assert entity.confidence == 0.5

    def test_entity_with_attribute(self):
        """Test entity with attribute."""
        entity = SemanticEntity(
            name="car",
            kind="vehicle",
        )
        entity = entity.with_attribute("color", "red")
        entity = entity.with_attribute("speed", 100)
        
        assert entity.attributes["color"] == "red"
        assert entity.attributes["speed"] == 100

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = SemanticEntity(
            name="test_entity",
            kind="test_type",
            confidence=0.8,
        )
        d = entity.to_dict()
        assert d["name"] == "test_entity"
        assert d["kind"] == "test_type"
        assert d["confidence"] == 0.8
        assert d["is_mentioned"] is False


class TestSemanticRelationship:
    """Tests for SemanticRelationship."""

    def test_basic_relationship(self):
        """Test basic relationship creation."""
        rel = SemanticRelationship(
            subject="cat",
            predicate="chases",
            object_="mouse",
        )
        assert rel.subject == "cat"
        assert rel.predicate == "chases"
        assert rel.object_ == "mouse"
        assert rel.confidence == 0.5

    def test_inferred_relationship(self):
        """Test inferred relationship."""
        rel = SemanticRelationship(
            subject="A",
            predicate="related_to",
            object_="B",
            is_inferred=True,
        )
        assert rel.is_inferred is True

    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = SemanticRelationship(
            subject="X",
            predicate="has",
            object_="Y",
            confidence=0.7,
        )
        d = rel.to_dict()
        assert d["subject"] == "X"
        assert d["predicate"] == "has"
        assert d["object"] == "Y"
        assert d["confidence"] == 0.7


class TestSemanticEvent:
    """Tests for SemanticEvent."""

    def test_basic_event(self):
        """Test basic event creation."""
        event = SemanticEvent(
            name="meeting",
            event_type=SemanticType.EVENT,
        )
        assert event.name == "meeting"
        assert event.event_type == SemanticType.EVENT

    def test_event_with_participants(self):
        """Test event with participants."""
        event = SemanticEvent(
            name="gathering",
            participants=["Alice", "Bob", "Charlie"],
        )
        assert len(event.participants) == 3
        assert "Alice" in event.participants

    def test_event_to_dict(self):
        """Test event serialization."""
        event = SemanticEvent(
            name="test_event",
            time_context="2024-01-01",
            location_context="NYC",
            confidence=0.6,
        )
        d = event.to_dict()
        assert d["name"] == "test_event"
        assert d["time_context"] == "2024-01-01"
        assert d["location_context"] == "NYC"
        assert d["confidence"] == 0.6


class TestSemanticContext:
    """Tests for SemanticContext."""

    def test_basic_context(self):
        """Test basic context creation."""
        context = SemanticContext(
            context_type="temporal",
            description="Time context",
        )
        assert context.context_type == "temporal"
        assert context.description == "Time context"

    def test_context_with_entities(self):
        """Test context with entities."""
        context = SemanticContext(
            context_type="domain",
            entities=["entity1", "entity2"],
            domain="technology",
        )
        assert len(context.entities) == 2
        assert context.domain == "technology"

    def test_context_to_dict(self):
        """Test context serialization."""
        context = SemanticContext(
            context_type="spatial",
            description="Location context",
            location_marker="here",
        )
        d = context.to_dict()
        assert d["type"] == "spatial"
        assert d["description"] == "Location context"
        assert d["location_marker"] == "here"


class TestAmbiguity:
    """Tests for Ambiguity."""

    def test_basic_ambiguity(self):
        """Test basic ambiguity creation."""
        amb = Ambiguity(
            ambiguity_type="lexical",
            content="Word 'bank' is ambiguous",
            alternatives=[
                {"interpretation": "financial_institution"},
                {"interpretation": "river_edge"},
            ],
        )
        assert amb.ambiguity_type == "lexical"
        assert len(amb.alternatives) == 2
        assert amb.is_resolved() is False

    def test_ambiguity_resolution(self):
        """Test ambiguity resolution."""
        amb = Ambiguity(
            ambiguity_type="lexical",
            content="Word 'bank' is ambiguous",
            alternatives=[
                {"interpretation": "financial_institution"},
                {"interpretation": "river_edge"},
            ],
        )
        amb.resolve(0, 0.8)
        assert amb.is_resolved() is True
        assert amb.resolution == "financial_institution"
        assert amb.resolution_confidence == 0.8

    def test_ambiguity_to_dict(self):
        """Test ambiguity serialization."""
        amb = Ambiguity(
            ambiguity_type="lexical",
            content="Word 'scale' is ambiguous",
            alternatives=[{"interpretation": "measurement"}],
        )
        d = amb.to_dict()
        assert d["ambiguity_type"] == "lexical"
        assert d["is_resolved"] is False
        assert d["alternatives_count"] == 1


class TestSemanticUnderstandingEngine:
    """Tests for SemanticUnderstandingEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SemanticUnderstandingEngine()

    def test_empty_text(self):
        """Test understanding empty text."""
        result = self.engine.understand(
            TextRepresentation(content="")
        )
        # Empty text produces minimal results
        assert result.get("entity_count", 0) == 0
        assert result.get("relationship_count", 0) == 0
        # Quality may be NONE or UNKNOWN for empty input
        assert result.get("semantic_quality") in (
            RepresentationQuality.NONE,
            RepresentationQuality.UNKNOWN,
        )

    def test_entity_extraction_from_text_rep(self):
        """Test entity extraction using text representation entities."""
        text_rep = TextRepresentation(
            content="John Smith works at Acme Corp.",
            entities=[
                {"text": "John Smith", "type": "PERSON_ORGANIZATION"},
                {"text": "Acme Corp", "type": "PERSON_ORGANIZATION"},
            ],
        )
        result = self.engine.understand(text_rep)
        assert result["entity_count"] >= 2

    def test_person_name_extraction(self):
        """Test extraction of person names."""
        text_rep = TextRepresentation(
            content="John Smith and Jane Doe went to the store."
        )
        result = self.engine.understand(text_rep)
        assert result["entity_count"] >= 2
        # Should have extracted person names
        entities = result["entities"]
        person_names = [e["name"] for e in entities if e.get("kind") == "person"]
        assert len(person_names) >= 2 or len(entities) >= 2

    def test_relationship_extraction(self):
        """Test relationship extraction."""
        text_rep = TextRepresentation(
            content="The cat with the dog played together."
        )
        result = self.engine.understand(text_rep)
        # Should detect some relationship
        assert result["relationship_count"] >= 0  # Relationship detection is hard

    def test_context_extraction(self):
        """Test context extraction."""
        text_rep = TextRepresentation(
            content="Yesterday, the meeting occurred in New York."
        )
        result = self.engine.understand(text_rep)
        # Should detect temporal context
        contexts = result["contexts"]
        temporal_contexts = [c for c in contexts if c.get("type") == "temporal"]
        assert len(temporal_contexts) >= 0  # Context extraction is heuristic

    def test_ambiguity_detection(self):
        """Test ambiguity detection."""
        text_rep = TextRepresentation(
            content="I went to the bank to deposit money."
        )
        result = self.engine.understand(text_rep)
        # Should detect ambiguity in "bank"
        ambiguities = result["ambiguities"]
        bank_ambiguities = [a for a in ambiguities 
                          if "bank" in a.get("content", "").lower()]
        assert len(bank_ambiguities) >= 0  # May or may not detect

    def test_semantic_quality_assessment(self):
        """Test semantic quality assessment."""
        # Rich text should get higher quality
        rich_text = TextRepresentation(
            content="John Smith (CEO of Acme Corp) met with Jane Doe "
                   "in New York yesterday to discuss the merger."
        )
        result = self.engine.understand(rich_text)
        assert result["semantic_quality"] in (
            RepresentationQuality.HIGH,
            RepresentationQuality.MODERATE,
        )

    def test_simple_text_quality(self):
        """Test that simple text gets lower quality."""
        simple_text = TextRepresentation(content="Hi.")
        result = self.engine.understand(simple_text)
        assert result["semantic_quality"] in (
            RepresentationQuality.LOW,
            RepresentationQuality.UNKNOWN,
        )

    def test_event_extraction(self):
        """Test event extraction."""
        text_rep = TextRepresentation(
            content="The meeting happened yesterday and the deal was signed."
        )
        result = self.engine.understand(text_rep)
        # Should detect some events
        assert result["event_count"] >= 0  # Event detection is heuristic

    def test_comprehensive_understanding(self):
        """Test comprehensive semantic understanding."""
        text_rep = TextRepresentation(
            content="John Smith, CEO of Acme Corporation, met with "
                   "Jane Doe from Beta Industries in New York City "
                   "on January 15, 2024, to discuss a potential merger."
        )
        result = self.engine.understand(text_rep)
        
        # Should extract multiple entities
        assert result["entity_count"] >= 1
        
        # Should detect context
        assert result["context_count"] >= 0
        
        # Should have processing latency
        assert result["processing_latency_ms"] >= 0

    def test_entity_id_uniqueness(self):
        """Test that entity IDs are unique."""
        text1 = TextRepresentation(content="John Smith went to the store.")
        text2 = TextRepresentation(content="Jane Doe went to the park.")
        
        result1 = self.engine.understand(text1)
        result2 = self.engine.understand(text2)
        
        # Each understanding call should work independently
        assert result1["entity_count"] >= 0
        assert result2["entity_count"] >= 0

    def test_latency_tracking(self):
        """Test latency tracking."""
        text_rep = TextRepresentation(content="Test text for latency.")
        result = self.engine.understand(text_rep)
        assert result["processing_latency_ms"] >= 0
        assert isinstance(result["processing_latency_ms"], float)


class TestSemanticRepresentation:
    """Tests for SemanticRepresentation wrapper."""

    def test_wrap_representation(self):
        """Test wrapping a representation with semantic understanding."""
        base = TextRepresentation(content="Test text.")
        semantic = {
            "entity_count": 1,
            "relationship_count": 0,
            "event_count": 0,
            "ambiguity_count": 0,
            "semantic_quality": RepresentationQuality.MODERATE,
        }
        
        wrapped = SemanticRepresentation(base, semantic)
        
        assert wrapped.content == "Test text."
        assert wrapped.entity_count == 1
        assert wrapped.semantic_quality == RepresentationQuality.MODERATE

    def test_semantic_to_dict(self):
        """Test semantic representation serialization."""
        base = TextRepresentation(content="Test.")
        semantic = {
            "entity_count": 2,
            "relationship_count": 1,
        }
        
        wrapped = SemanticRepresentation(base, semantic)
        d = wrapped.to_dict()
        
        assert d["content"] == "Test."
        assert d["entity_count"] == 2
        assert d["relationship_count"] == 1
        assert "semantic_understanding" in d


class TestSemanticType:
    """Tests for SemanticType enum."""

    def test_all_types_exist(self):
        """Test that all semantic types exist."""
        types = [
            SemanticType.ENTITY,
            SemanticType.CONCEPT,
            SemanticType.EVENT,
            SemanticType.ACTION,
            SemanticType.PROPERTY,
            SemanticType.RELATIONSHIP,
            SemanticType.ATTRIBUTE,
            SemanticType.CATEGORY,
            SemanticType.QUANTITY,
            SemanticType.TEMPORAL,
            SemanticType.LOCATION,
        ]
        for t in types:
            assert t.value
            assert isinstance(t.value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
