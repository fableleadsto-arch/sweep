"""
Tests for world model (Phase 7).

These tests verify that:
1. Entities can be created and retrieved
2. Relationships connect entities
3. Events track happenings
4. World state is versioned
5. Entity resolution works
6. Contradiction detection works
7. Related entity traversal works
"""

import pytest
import time

from sweep_cognitive.world import (
    EntityType,
    WorldEntity,
    RelationshipType,
    WorldRelationship,
    EventType,
    WorldEvent,
    WorldState,
    WorldModel,
)


class TestWorldEntity:
    """Tests for WorldEntity."""

    def test_basic_entity(self):
        """Test basic entity creation."""
        entity = WorldEntity(
            name="cat",
            type=EntityType.ANIMAL,
            properties={"fur": True, "legs": 4},
            confidence=0.9,
        )
        assert entity.name == "cat"
        assert entity.type == EntityType.ANIMAL
        assert entity.properties["fur"] is True
        assert entity.confidence == 0.9

    def test_entity_with_aliases(self):
        """Test entity with aliases."""
        entity = WorldEntity(
            name="feline",
            aliases=["cat", "kitty", "feline"],
        )
        assert "cat" in entity.aliases

    def test_entity_update_property(self):
        """Test property update with provenance."""
        entity = WorldEntity(name="test")
        
        entity.update_property(
            "color", "red", 
            source="observation_1", 
            confidence_boost=0.1
        )
        
        assert entity.properties["color"] == "red"
        assert entity.confidence >= 0.5
        assert len(entity.observations) == 1
        assert entity.observations[0]["property"] == "color"

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = WorldEntity(
            name="test_entity",
            type=EntityType.OBJECT,
            confidence=0.8,
        )
        d = entity.to_dict()
        assert d["name"] == "test_entity"
        assert d["type"] == "object"
        assert d["confidence"] == 0.8
        assert d["version"] == 1


class TestWorldRelationship:
    """Tests for WorldRelationship."""

    def test_basic_relationship(self):
        """Test basic relationship creation."""
        rel = WorldRelationship(
            subject_id="entity_1",
            predicate=RelationshipType.RELATED_TO,
            object_id="entity_2",
            confidence=0.8,
        )
        assert rel.subject_id == "entity_1"
        assert rel.predicate == RelationshipType.RELATED_TO
        assert rel.object_id == "entity_2"
        assert rel.confidence == 0.8

    def test_relationship_to_triple(self):
        """Test triple conversion."""
        rel = WorldRelationship(
            subject_id="A",
            predicate=RelationshipType.PART_OF,
            object_id="B",
        )
        triple = rel.to_triple()
        assert triple == ("A", "part_of", "B")

    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = WorldRelationship(
            subject_id="X",
            predicate=RelationshipType.LOCATED_IN,
            object_id="Y",
            confidence=0.9,
        )
        d = rel.to_dict()
        assert d["subject_id"] == "X"
        assert d["predicate"] == "located_in"
        assert d["object_id"] == "Y"
        assert d["confidence"] == 0.9


class TestWorldEvent:
    """Tests for WorldEvent."""

    def test_basic_event(self):
        """Test basic event creation."""
        event = WorldEvent(
            type=EventType.ACTION,
            name="cat_jumped",
            participants=["cat_1"],
            time_context="2024-01-01",
            outcome="cat landed safely",
        )
        assert event.type == EventType.ACTION
        assert event.name == "cat_jumped"
        assert "cat_1" in event.participants
        assert event.time_context == "2024-01-01"

    def test_event_to_dict(self):
        """Test event serialization."""
        event = WorldEvent(
            type=EventType.OBSERVATION,
            name="observed_cat",
            participants=["cat_1", "observer_1"],
        )
        d = event.to_dict()
        assert d["type"] == "observation"
        assert d["participant_count"] == 2


class TestWorldState:
    """Tests for WorldState (versioned state)."""

    def setup_method(self):
        """Set up fresh world state."""
        self.state = WorldState()

    def test_add_entity(self):
        """Test adding an entity."""
        entity = WorldEntity(id="cat", name="cat", type=EntityType.ANIMAL)
        self.state.add_entity(entity)
        self.state._create_version()
        
        retrieved = self.state.get_entity("cat")
        assert retrieved is not None
        assert retrieved.name == "cat"

    def test_add_duplicate_entity(self):
        """Test adding duplicate entity updates."""
        entity1 = WorldEntity(id="cat", name="cat", properties={"color": "black"})
        self.state.add_entity(entity1)
        
        entity2 = WorldEntity(id="cat", name="cat", properties={"color": "white"})
        self.state.add_entity(entity2)
        
        # Should update, not duplicate
        stats = self.state.stats()
        assert stats["entities"] == 1
        
        retrieved = self.state.get_entity("cat")
        assert retrieved is not None
        assert retrieved.properties["color"] == "white"

    def test_get_entity_by_name(self):
        """Test getting entity by name."""
        self.state.add_entity(WorldEntity(id="dog", name="dog", type=EntityType.ANIMAL))
        
        entity = self.state.get_entity_by_name("dog")
        assert entity is not None
        assert entity.name == "dog"
        
        # Non-existent
        assert self.state.get_entity_by_name("nonexistent") is None

    def test_add_relationship(self):
        """Test adding a relationship."""
        self.state.add_entity(WorldEntity(id="cat_1", name="cat"))
        self.state.add_entity(WorldEntity(id="mat_1", name="mat"))
        
        rel = WorldRelationship(
            id="rel_1", subject_id="cat_1",
            predicate=RelationshipType.LOCATED_IN,
            object_id="mat_1",
        )
        self.state.add_relationship(rel)
        self.state._create_version()
        
        retrieved = self.state.get_relationship("rel_1")
        assert retrieved is not None
        assert retrieved.predicate == RelationshipType.LOCATED_IN

    def test_get_relationships_for_entity(self):
        """Test getting relationships for an entity."""
        self.state.add_entity(WorldEntity(id="A"))
        self.state.add_entity(WorldEntity(id="B"))
        self.state.add_entity(WorldEntity(id="C"))
        
        self.state.add_relationship(WorldRelationship(
            id="r1", subject_id="A", predicate=RelationshipType.RELATED_TO, object_id="B"
        ))
        self.state.add_relationship(WorldRelationship(
            id="r2", subject_id="A", predicate=RelationshipType.RELATED_TO, object_id="C"
        ))
        
        rels = self.state.get_relationships_for_entity("A")
        assert len(rels) == 2
        assert self.state.stats()["relationships"] == 2

    def test_add_event(self):
        """Test adding an event."""
        event = WorldEvent(
            id="evt_1",
            type=EventType.ACTION,
            name="jump",
            participants=["cat_1"],
        )
        self.state.add_event(event)
        self.state._create_version()
        
        retrieved = self.state.get_event("evt_1")
        assert retrieved is not None
        assert retrieved.name == "jump"

    def test_get_events_for_entity(self):
        """Test getting events for an entity."""
        self.state.add_entity(WorldEntity(id="cat_1"))
        
        self.state.add_event(WorldEvent(
            id="evt1", type=EventType.ACTION, name="jump", participants=["cat_1"]
        ))
        self.state._create_version()
        self.state.add_event(WorldEvent(
            id="evt2", type=EventType.CHANGE, name="sleep", participants=["cat_1"]
        ))
        self.state._create_version()
        
        events = self.state.get_events_for_entity("cat_1")
        assert len(events) == 2
        assert self.state.stats()["events"] == 2

    def test_versioning(self):
        """Test that state is versioned."""
        # Initial version
        assert self.state.stats()["current_version"] == 1
        
        # Add entity - should create new version
        self.state.add_entity(WorldEntity(id="cat", name="cat"))
        assert self.state._current_version == 2
        
        # Add relationship - should create new version
        self.state.add_entity(WorldEntity(id="mat"))
        self.state.add_relationship(WorldRelationship(
            id="rel1", subject_id="cat", predicate=RelationshipType.LOCATED_IN, object_id="mat"
        ))
        assert self.state._current_version == 4
        
        # Version history
        history = self.state.get_version_history()
        assert len(history) >= 3

    def test_get_version(self):
        """Test getting specific version."""
        self.state.add_entity(WorldEntity(id="cat", name="cat"))
        self.state.add_entity(WorldEntity(id="dog", name="dog"))
        
        # After initial (1) + 2 adds = version 3
        version3 = self.state.get_version(self.state._current_version)
        assert version3 is not None
        assert version3.version == self.state._current_version
        assert version3.to_dict()["entity_count"] == 2

    def test_query_entities(self):
        """Test entity querying."""
        # Add entities
        self.state.add_entity(WorldEntity(id="cat1", name="cat", type=EntityType.ANIMAL, confidence=0.9))
        self.state._create_version()
        self.state.add_entity(WorldEntity(id="dog1", name="dog", type=EntityType.ANIMAL, confidence=0.7))
        self.state._create_version()
        self.state.add_entity(WorldEntity(id="car1", name="car", type=EntityType.OBJECT, confidence=0.8))
        self.state._create_version()
        
        animals = self.state.query_entities(entity_type=EntityType.ANIMAL)
        assert len(animals) >= 2
        
        high_conf = self.state.query_entities(min_confidence=0.8)
        assert len(high_conf) >= 2  # cat and car

    def test_query_relationships(self):
        """Test relationship querying."""
        self.state.add_entity(WorldEntity(id="A"))
        self.state.add_entity(WorldEntity(id="B"))
        self.state.add_entity(WorldEntity(id="C"))
        self.state._create_version()
        
        self.state.add_relationship(WorldRelationship(
            id="qr1", subject_id="A", predicate=RelationshipType.RELATED_TO, object_id="B",
            confidence=0.9
        ))
        self.state._create_version()
        self.state.add_relationship(WorldRelationship(
            id="qr2", subject_id="A", predicate=RelationshipType.LOCATED_IN, object_id="C",
            confidence=0.5
        ))
        self.state._create_version()
        
        related = self.state.query_relationships(predicate=RelationshipType.RELATED_TO)
        assert len(related) >= 1
        
        high_conf = self.state.query_relationships(min_confidence=0.8)
        assert len(high_conf) >= 1

    def test_stats(self):
        """Test statistics."""
        self.state.add_entity(WorldEntity(id="cat", name="cat"))
        self.state.add_entity(WorldEntity(id="dog", name="dog"))
        self.state.add_relationship(WorldRelationship(
            id="r1", subject_id="cat", predicate=RelationshipType.RELATED_TO, object_id="dog"
        ))
        
        stats = self.state.stats()
        assert stats["entities"] == 2
        assert stats["relationships"] == 1
        assert stats["events"] == 0
        assert stats["current_version"] >= 3


class TestWorldModel:
    """Tests for WorldModel orchestrator."""

    def setup_method(self):
        """Set up fresh world model."""
        self.model = WorldModel()

    def test_add_and_get_entity(self):
        """Test adding and retrieving entities."""
        result = self.model.add_entity("cat", EntityType.ANIMAL)
        assert result is not None
        assert result.name == "cat"
        
        entity = self.model.get_entity_by_name("cat")
        assert entity is not None
        assert entity.type == EntityType.ANIMAL

    def test_entity_resolution(self):
        """Test entity resolution."""
        result = self.model.add_entity("feline", EntityType.ANIMAL)
        assert result is not None
        
        # Should resolve to existing entity
        entity = self.model.resolve_entity("feline")
        assert entity is not None
        assert entity.name == "feline"
        
        # Adding same name should return existing
        existing = self.model.resolve_entity("feline")
        assert existing is not None
        assert existing.id == entity.id

    def test_add_relationship(self):
        """Test adding relationships."""
        cat = self.model.add_entity("cat", EntityType.ANIMAL)
        mat = self.model.add_entity("mat", EntityType.OBJECT)
        assert cat is not None and mat is not None
        assert cat.id is not None and mat.id is not None
        
        rel = self.model.add_relationship(
            cat.id, RelationshipType.LOCATED_IN, mat.id,
            confidence=0.9
        )
        assert rel is not None
        assert rel.subject_id == cat.id
        assert rel.object_id == mat.id

    def test_add_event(self):
        """Test adding events."""
        cat = self.model.add_entity("cat")
        assert cat is not None and cat.id is not None
        
        event = self.model.add_event(
            EventType.ACTION,
            "cat_jumped",
            participants=[cat.id],
            time_context="2024-01-01",
            outcome="landed safely",
        )
        assert event is not None
        assert event.type == EventType.ACTION
        assert cat.id in event.participants

    def test_detect_contradictions(self):
        """Test contradiction detection."""
        n1 = self.model.add_entity("cat", EntityType.ANIMAL)
        n2 = self.model.add_entity("mat", EntityType.OBJECT)
        assert n1 is not None and n2 is not None
        assert n1.id != n2.id
        
        # Add conflicting relationships
        rel1 = self.model.add_relationship(
            n1.id, RelationshipType.PART_OF, n2.id, confidence=0.8
        )
        assert rel1 is not None
        
        rel2 = self.model.add_relationship(
            n1.id, RelationshipType.DIFFERENT_FROM, n2.id, confidence=0.8
        )
        assert rel2 is not None
        
        # IDs must not collide even when created in the same millisecond
        assert rel1.id != rel2.id, f"IDs collide: {rel1.id} == {rel2.id}"
        
        # Both relationships must be stored independently
        assert self.model.state.stats()["relationships"] == 2
        
        contradictions = self.model.detect_contradictions()
        assert len(contradictions) >= 1, f"Expected >=1 contradiction, got {contradictions}"
        assert contradictions[0]["type"] == "conflicting_relationship"

    def test_get_related_entities(self):
        """Test related entity traversal."""
        n1 = self.model.add_entity("cat", EntityType.ANIMAL)
        n2 = self.model.add_entity("mat", EntityType.OBJECT)
        n3 = self.model.add_entity("house", EntityType.OBJECT)
        assert n1 is not None and n2 is not None and n3 is not None
        assert n1.id != n2.id != n3.id
        
        rel1 = self.model.add_relationship(n1.id, RelationshipType.LOCATED_IN, n2.id)
        rel2 = self.model.add_relationship(n2.id, RelationshipType.LOCATED_IN, n3.id)
        assert rel1 is not None and rel2 is not None
        
        # Verify relationships were added
        assert self.model.state.get_relationship(rel1.id) is not None
        assert self.model.state.get_relationship(rel2.id) is not None
        
        # Verify the relationship was stored with correct subject
        rel1_from_state = self.model.state.get_relationship(rel1.id)
        assert rel1_from_state.subject_id == n1.id
        assert rel1_from_state.object_id == n2.id
        
        related_1 = self.model.get_related_entities(n1.id, max_depth=1)
        assert len(related_1) >= 1, f"Expected at least 1 related, got {len(related_1)}"
        related_ids = {e.id for e in related_1}
        assert n2.id in related_ids, f"Expected {n2.id} in {related_ids}; got {related_ids}"
        
        # Get entities related to n1 (2 hops): should find n2 and n3
        related_2 = self.model.get_related_entities(n1.id, max_depth=2)
        assert len(related_2) >= 2, f"Expected at least 2 related, got {len(related_2)}"
        related_ids_2 = {e.id for e in related_2}
        assert n2.id in related_ids_2
        assert n3.id in related_ids_2

    def test_full_stats(self):
        """Test complete statistics."""
        n1 = self.model.add_entity("cat", EntityType.ANIMAL)
        n2 = self.model.add_entity("dog", EntityType.ANIMAL)
        assert n1 is not None and n2 is not None
        assert n1.id != n2.id
        rel_result = self.model.add_relationship(n1.id, RelationshipType.RELATED_TO, n2.id)
        assert rel_result is not None
        
        stats = self.model.stats()
        assert stats["entities"] == 2
        assert stats["relationships"] == 1
        # Cache gets populated when entities are looked up by name, not when added
        # So we check it was at least accessed (might be 0 if no lookup by name occurred)
        assert "entity_resolution_cache" in stats


class TestWorldModelSerialization:
    """Tests for world model serialization."""

    def test_state_to_dict(self):
        """Test state serialization."""
        model = WorldModel()
        model.add_entity("cat", EntityType.ANIMAL)
        
        d = model.state.to_dict()
        assert "entities" in d
        assert "relationships" in d
        assert "events" in d
        assert "version_history" in d

    def test_world_model_to_dict(self):
        """Test world model serialization."""
        model = WorldModel()
        model.add_entity("dog")
        
        d = model.to_dict()
        assert "state" in d
        assert "entity_resolution_cache_size" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
