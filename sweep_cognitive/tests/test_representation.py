"""
Tests for the representation layer (Phase 1).

These tests verify that:
1. Representations are created correctly
2. Embeddings work as expected
3. The representation gateway validates properly
4. Confidence and quality are tracked
"""

import pytest
import time

from ..representation import (
    Representation,
    TextRepresentation,
    MultimodalRepresentation,
    EmbeddingVector,
    RepresentationQuality,
    RepresentationType,
    RepresentationGateway,
)


class TestEmbeddingVector:
    """Tests for EmbeddingVector."""

    def test_creation(self):
        """Test embedding vector creation."""
        vec = EmbeddingVector(vector=[0.1, 0.2, 0.3, 0.4])
        assert vec.dimension == 4
        assert len(vec.vector) == 4
        assert vec.model_id == "unknown"

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = EmbeddingVector(vector=[1.0, 0.0, 0.0])
        sim = vec.cosine_similarity(EmbeddingVector(vector=[1.0, 0.0, 0.0]))
        assert sim == 1.0

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec = EmbeddingVector(vector=[1.0, 0.0])
        sim = vec.cosine_similarity(EmbeddingVector(vector=[0.0, 1.0]))
        assert sim == 0.0

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        vec = EmbeddingVector(vector=[1.0, 0.0])
        sim = vec.cosine_similarity(EmbeddingVector(vector=[-1.0, 0.0]))
        assert sim == -1.0

    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        vec1 = EmbeddingVector(vector=[0.0, 0.0])
        vec2 = EmbeddingVector(vector=[3.0, 4.0])
        dist = vec1.euclidean_distance(vec2)
        assert dist == 5.0

    def test_empty_vector_similarity(self):
        """Test similarity with empty vectors."""
        vec = EmbeddingVector(vector=[])
        sim = vec.cosine_similarity(EmbeddingVector(vector=[1.0, 2.0]))
        assert sim == 0.0


class TestRepresentation:
    """Tests for base Representation."""

    def test_basic_representation(self):
        """Test basic representation creation."""
        rep = Representation(
            content="test content",
            representation_type=RepresentationType.SEMANTIC,
            confidence=0.75,
            source="test_source",
        )
        assert rep.content == "test content"
        assert rep.representation_type == RepresentationType.SEMANTIC
        assert rep.confidence == 0.75
        assert rep.source == "test_source"
        assert rep.quality == RepresentationQuality.UNKNOWN

    def test_representation_with_embedding(self):
        """Test representation with embedding."""
        embedding = EmbeddingVector(vector=[0.1, 0.2, 0.3])
        rep = Representation(
            content="test",
            embedding=embedding,
        )
        assert rep.embedding is not None
        assert rep.embedding.dimension == 3

    def test_with_confidence(self):
        """Test confidence update."""
        rep = Representation(content="test", confidence=0.5)
        updated = rep.with_confidence(0.8)
        assert updated.confidence == 0.8
        assert updated.content == rep.content  # Other fields preserved

    def test_with_embedding_method(self):
        """Test embedding attachment."""
        rep = Representation(content="test")
        embedding = EmbeddingVector(vector=[0.5, 0.6])
        updated = rep.with_embedding(embedding)
        assert updated.embedding is not None
        assert updated.embedding.vector == [0.5, 0.6]

    def test_to_dict(self):
        """Test serialization."""
        rep = Representation(
            content="test",
            confidence=0.9,
            source="source1",
        )
        d = rep.to_dict()
        assert d["content"] == "test"
        assert d["confidence"] == 0.9
        assert d["source"] == "source1"
        assert d["has_embedding"] is False


class TestTextRepresentation:
    """Tests for TextRepresentation."""

    def test_text_representation(self):
        """Test text representation creation."""
        rep = TextRepresentation(
            content="The cat sat on the mat.",
            entities=[{"text": "cat", "type": "animal"}],
            predicates=["sat on"],
            temporal_markers=[],
            certainty_markers=[""],
            sentiment=0.1,
            is_declarative=True,
            is_question=False,
        )
        assert rep.content == "The cat sat on the mat."
        assert len(rep.entities) == 1
        assert rep.entities[0]["text"] == "cat"
        assert rep.is_declarative is True
        assert rep.is_question is False

    def test_text_to_dict(self):
        """Test text representation serialization."""
        rep = TextRepresentation(
            content="Test sentence.",
            language="English",
        )
        d = rep.to_dict()
        assert d["content"] == "Test sentence."
        assert d["language"] == "English"
        assert d["is_question"] is False


class TestMultimodalRepresentation:
    """Tests for MultimodalRepresentation."""

    def test_multimodal_creation(self):
        """Test multimodal representation."""
        rep = MultimodalRepresentation(
            content="Combined text and image",
            modality="multimodal",
        )
        assert rep.fused is False
        assert rep.alignment_quality == 0.5

    def test_add_modality_source(self):
        """Test adding modality sources."""
        rep = MultimodalRepresentation(content="Combined")
        rep.add_modality_source(
            modality="text",
            content="Some text",
            confidence=0.9,
            metadata={"source": "keyboard"},
        )
        assert len(rep.modality_sources) == 1
        assert rep.modality_sources[0]["modality"] == "text"
        assert rep.modality_sources[0]["confidence"] == 0.9

    def test_multimodal_to_dict(self):
        """Test multimodal serialization."""
        rep = MultimodalRepresentation(content="Test")
        rep.add_modality_source("text", "content", 0.8)
        d = rep.to_dict()
        assert d["fused"] is False
        assert len(d["modality_sources"]) == 1


class TestRepresentationGateway:
    """Tests for the representation gateway (validation)."""

    def test_accept_high_confidence(self):
        """Test accepting high-confidence representation."""
        gateway = RepresentationGateway(min_confidence=0.5)
        rep = Representation(content="test", confidence=0.8)
        assert gateway.accept(rep) is True

    def test_reject_low_confidence(self):
        """Test rejecting low-confidence representation."""
        gateway = RepresentationGateway(min_confidence=0.7)
        rep = Representation(content="test", confidence=0.5)
        assert gateway.accept(rep) is False

    def test_enrich(self):
        """Test enrichment with context."""
        gateway = RepresentationGateway()
        rep = Representation(content="test", confidence=0.8)
        context = {"task": "testing", "source": "unit_test"}
        enriched = gateway.enrich(rep, context)
        assert "input_context" in enriched.metadata
        assert enriched.metadata["input_context"]["task"] == "testing"

    def test_reject_count(self):
        """Test rejection tracking."""
        gateway = RepresentationGateway(min_confidence=0.9)
        gateway.accept(Representation(content="low", confidence=0.5))
        gateway.accept(Representation(content="high", confidence=0.95))
        assert gateway.reject_count() == 1

    def test_clear_rejected(self):
        """Test clearing rejection log."""
        gateway = RepresentationGateway(min_confidence=0.9)
        gateway.accept(Representation(content="low", confidence=0.5))
        assert gateway.reject_count() == 1
        gateway.clear_rejected()
        assert gateway.reject_count() == 0


class TestRepresentationQuality:
    """Tests for representation quality enumeration."""

    def test_quality_values(self):
        """Test that all quality levels exist."""
        assert RepresentationQuality.HIGH.value == "high"
        assert RepresentationQuality.MODERATE.value == "moderate"
        assert RepresentationQuality.LOW.value == "low"
        assert RepresentationQuality.UNKNOWN.value == "unknown"
        assert RepresentationQuality.NONE.value == "none"


class TestRepresentationType:
    """Tests for representation type enumeration."""

    def test_type_values(self):
        """Test that all representation types exist."""
        types = [
            RepresentationType.PERCEPTUAL,
            RepresentationType.SEMANTIC,
            RepresentationType.CONCEPTUAL,
            RepresentationType.RELATIONAL,
            RepresentationType.CONTEXTUAL,
            RepresentationType.PREDICTIVE,
            RepresentationType.HYPOTHETICAL,
            RepresentationType.EVIDENTIAL,
            RepresentationType.PROCEDURAL,
        ]
        for t in types:
            assert t.value
            assert isinstance(t.value, str)


class TestRepresentationIdentity:
    """Tests for representation identity preservation."""

    def test_id_generation(self):
        """Test that IDs are generated."""
        rep1 = Representation(content="test1")
        rep2 = Representation(content="test2")
        assert rep1.id != rep2.id
        assert rep1.id.startswith("rep_")

    def test_content_preserved(self):
        """Test that content is preserved through operations."""
        original = "This is a test of content preservation."
        rep = Representation(content=original)
        updated = rep.with_confidence(0.9)
        assert updated.content == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
