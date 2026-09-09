"""
Tests for the knowledge base (Phase 1).

These tests verify that:
1. Knowledge entries are created and stored correctly
2. Confidence tracking works
3. Evidence and contradictions update confidence
4. Retrieval works correctly
5. Default initialization provides baseline knowledge
"""

import pytest

from ..knowledge import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeType,
    KnowledgeConfidence,
    KnowledgeSource,
    get_knowledge_base,
    reset_knowledge_base,
)


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry."""

    def test_basic_entry(self):
        """Test basic knowledge entry creation."""
        entry = KnowledgeEntry(
            subject="bird",
            predicate="has_property",
            object="has_wings",
            knowledge_type=KnowledgeType.ENTITY_PROPERTY,
            confidence=KnowledgeConfidence.LIKELY,
            confidence_score=0.7,
        )
        assert entry.subject == "bird"
        assert entry.predicate == "has_property"
        assert entry.object == "has_wings"
        assert entry.confidence_score == 0.7
        assert entry.evidence_count == 0
        assert entry.contradiction_count == 0

    def test_entry_with_sources(self):
        """Test entry with sources."""
        source = KnowledgeSource(
            name="wiki",
            source_type="retrieval",
            reliability=0.8,
        )
        entry = KnowledgeEntry(
            subject="earth",
            predicate="shape",
            object="oblate_sphere",
            sources=[source],
        )
        assert len(entry.sources) == 1
        assert entry.sources[0].name == "wiki"

    def test_entry_to_dict(self):
        """Test entry serialization."""
        entry = KnowledgeEntry(
            subject="test",
            predicate="is",
            object="something",
            knowledge_type=KnowledgeType.FACT,
        )
        d = entry.to_dict()
        assert d["subject"] == "test"
        assert d["predicate"] == "is"
        assert d["object"] == "something"
        assert d["type"] == "fact"
        assert d["confidence"] == "unknown"

    def test_updated_confidence(self):
        """Test confidence update."""
        entry = KnowledgeEntry(
            subject="test",
            predicate="is",
            object="something",
            confidence_score=0.5,
        )
        updated = entry.with_updated_confidence(
            KnowledgeConfidence.KNOWN, 0.9, "new evidence"
        )
        assert updated.confidence == KnowledgeConfidence.KNOWN
        assert updated.confidence_score == 0.9
        assert "new evidence" in updated.metadata.get("last_update_reason", "")


class TestKnowledgeBase:
    """Tests for KnowledgeBase."""

    def setup_method(self):
        """Set up fresh knowledge base for each test."""
        reset_knowledge_base()
        self.kb = KnowledgeBase()

    def test_add_entry(self):
        """Test adding entries."""
        entry = KnowledgeEntry(
            subject="cat",
            predicate="is_a",
            object="animal",
        )
        self.kb.add(entry)
        assert self.kb.count() == 1

    def test_add_duplicate_merges(self):
        """Test that adding same entry merges rather than duplicates."""
        entry1 = KnowledgeEntry(
            subject="cat",
            predicate="is_a",
            object="animal",
            evidence_count=1,
        )
        entry2 = KnowledgeEntry(
            subject="cat",
            predicate="is_a",
            object="animal",
            evidence_count=2,
        )
        self.kb.add(entry1)
        self.kb.add(entry2)
        # Should merge, not create duplicate
        assert self.kb.count() == 1
        # Evidence counts should combine
        stored = self.kb.get("cat", "is_a", "animal")[0]
        assert stored.evidence_count == 3

    def test_get_by_subject(self):
        """Test retrieval by subject."""
        self.kb.add(KnowledgeEntry("cat", "has", "fur"))
        self.kb.add(KnowledgeEntry("cat", "can", "meow"))
        self.kb.add(KnowledgeEntry("dog", "has", "fur"))

        cat_entries = self.kb.get_by_subject("cat")
        assert len(cat_entries) == 2

        dog_entries = self.kb.get_by_subject("dog")
        assert len(dog_entries) == 1

    def test_get_by_predicate(self):
        """Test retrieval by predicate."""
        self.kb.add(KnowledgeEntry("bird", "can", "fly"))
        self.kb.add(KnowledgeEntry("human", "can", "speak"))
        
        can_entries = self.kb.get("", "can", "")
        assert len(can_entries) == 2

    def test_get_confident(self):
        """Test retrieval by confidence threshold."""
        self.kb.add(KnowledgeEntry(
            "known_fact", "is", "true",
            confidence_score=0.9,
        ))
        self.kb.add(KnowledgeEntry(
            "unknown_fact", "is", "maybe",
            confidence_score=0.4,
        ))
        
        confident = self.kb.get_confident(0.7)
        assert len(confident) == 1
        assert confident[0].subject == "known_fact"

    def test_get_disputed(self):
        """Test retrieval of disputed entries."""
        self.kb.add(KnowledgeEntry(
            "controversial", "is", "debated",
            contradiction_count=2,
        ))
        self.kb.add(KnowledgeEntry(
            "settled", "is", "fact",
            contradiction_count=0,
        ))
        
        disputed = self.kb.get_disputed()
        assert len(disputed) == 1
        assert disputed[0].subject == "controversial"

    def test_update_confidence(self):
        """Test confidence update."""
        entry = KnowledgeEntry(
            subject="test",
            predicate="is",
            object="something",
            confidence_score=0.5,
        )
        self.kb.add(entry)
        entry_id = entry.id

        updated = self.kb.update_confidence(
            entry_id, KnowledgeConfidence.KNOWN, 0.95, "verified"
        )
        assert updated is not None
        assert updated.confidence_score == 0.95

    def test_add_evidence(self):
        """Test adding evidence boosts confidence."""
        entry = KnowledgeEntry(
            subject="test",
            predicate="is",
            object="verifiable",
            confidence_score=0.5,
        )
        self.kb.add(entry)
        entry_id = entry.id

        source = KnowledgeSource("test_source", "observation", 0.8)
        self.kb.add_evidence(entry_id, source, strength=1.0)

        updated = self.kb.get_by_subject("test")[0]
        assert updated.evidence_count == 1
        assert updated.confidence_score > 0.5

    def test_add_contradiction(self):
        """Test adding contradiction reduces confidence."""
        entry = KnowledgeEntry(
            subject="test",
            predicate="is",
            object="debatable",
            confidence_score=0.7,
        )
        self.kb.add(entry)
        entry_id = entry.id

        source = KnowledgeSource("contradicting_source", "observation", 0.7)
        self.kb.add_contradiction(entry_id, entry, source)

        updated = self.kb.get_by_subject("test")[0]
        assert updated.contradiction_count == 1
        assert updated.confidence_score < 0.7

    def test_remove_entry(self):
        """Test entry removal."""
        entry = KnowledgeEntry(
            subject="removable",
            predicate="is",
            object=" removable",
        )
        self.kb.add(entry)
        entry_id = entry.id

        assert self.kb.remove(entry_id) is True
        assert self.kb.count() == 0
        assert self.kb.remove(entry_id) is False  # Already removed

    def test_clear(self):
        """Test clearing all knowledge."""
        self.kb.add(KnowledgeEntry("a", "is", "1"))
        self.kb.add(KnowledgeEntry("b", "is", "2"))
        assert self.kb.count() == 2

        self.kb.clear()
        assert self.kb.count() == 0

    def test_statistics(self):
        """Test statistics."""
        self.kb.add(KnowledgeEntry(
            "fact1", "is", "true",
            knowledge_type=KnowledgeType.FACT,
            confidence=KnowledgeConfidence.KNOWN,
        ))
        self.kb.add(KnowledgeEntry(
            "prop1", "has", "property",
            knowledge_type=KnowledgeType.ENTITY_PROPERTY,
            confidence=KnowledgeConfidence.LIKELY,
        ))
        self.kb.add(KnowledgeEntry(
            "contr1", "is", "disputed",
            confidence_score=0.3,
            contradiction_count=1,
        ))

        stats = self.kb.stats()
        assert stats["total_entries"] == 3
        # Count by type
        fact_count = sum(1 for v in stats["by_type"].values() if v >= 1)
        assert fact_count >= 2
        assert stats["disputed_count"] == 1


class TestDefaultInitialization:
    """Tests for default knowledge initialization."""

    def setup_method(self):
        """Set up fresh knowledge base."""
        reset_knowledge_base()
        self.kb = get_knowledge_base()

    def test_initialized_with_baseline(self):
        """Test that default initialization adds baseline knowledge."""
        self.kb.initialize_defaults()
        assert self.kb.count() > 0

        # Should have basic facts about birds, cats, etc.
        birds = self.kb.get_by_subject("bird")
        assert len(birds) > 0, "Should have knowledge about birds"

        cats = self.kb.get_by_subject("cat")
        assert len(cats) > 0, "Should have knowledge about cats"

    def test_initialization_idempotent(self):
        """Test that initialization can only happen once."""
        self.kb.initialize_defaults()
        count1 = self.kb.count()
        self.kb.initialize_defaults()
        count2 = self.kb.count()
        assert count1 == count2

    def test_default_entries_have_low_confidence(self):
        """Test that default entries start with low confidence."""
        self.kb.initialize_defaults()
        entry = self.kb.get("bird", "has_property", "has_wings")[0]
        # Default entries should start at POSSIBLE or similar
        assert entry.confidence in (
            KnowledgeConfidence.POSSIBLE,
            KnowledgeConfidence.UNKNOWN,
        )


class TestKnowledgeSource:
    """Tests for KnowledgeSource."""

    def test_source_creation(self):
        """Test source creation."""
        source = KnowledgeSource(
            name="wiki",
            source_type="retrieval",
            reliability=0.8,
        )
        assert source.name == "wiki"
        assert source.source_type == "retrieval"
        assert source.reliability == 0.8

    def test_source_to_dict(self):
        """Test source serialization."""
        source = KnowledgeSource(name="test", reliability=0.9)
        d = source.to_dict()
        assert d["name"] == "test"
        assert d["reliability"] == 0.9


class TestKnowledgeConfidence:
    """Tests for KnowledgeConfidence."""

    def test_confidence_levels(self):
        """Test all confidence levels exist."""
        levels = [
            KnowledgeConfidence.KNOWN,
            KnowledgeConfidence.LIKELY,
            KnowledgeConfidence.POSSIBLE,
            KnowledgeConfidence.DISPUTED,
            KnowledgeConfidence.UNKNOWN,
            KnowledgeConfidence.FALSE,
        ]
        for level in levels:
            assert level.value
            assert isinstance(level.value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
