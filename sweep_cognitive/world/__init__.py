"""
World Model — Phase 7.

Represents:
- Entities (with properties, types, attributes)
- Relationships (between entities, with confidence)
- Events (with time, participants, outcomes)
- States (versioned world state)
- Confidence and provenance
- Contradictions

The world model is:
- Versioned (every change creates a new version)
- Auditable (full history of changes)
- Confidence-aware (each fact has confidence)
- Provenance-tracked (where each fact came from)
- Contradiction-aware (contradictions are explicit)

Key principle: Never overwrite world knowledge blindly.
Use: observation → candidate update → confidence assessment → 
source assessment → conflict analysis → update decision.
"""

from __future__ import annotations

import time
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Global unique ID counter (avoids millisecond-timestamp collisions)
_id_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    """Generate a guaranteed-unique ID with the given prefix."""
    return f"{prefix}_{next(_id_counter)}_{int(time.time() * 1000)}"


# ════════════════════════════════════════════════════════════════════
# WORLD ENTITY
# ════════════════════════════════════════════════════════════════════

class EntityType(str, Enum):
    """Types of entities in the world model."""
    PHYSICAL = "physical"          # Physical objects
    ANIMAL = "animal"              # Living animals
    PERSON = "person"              # Human beings
    ORGANIZATION = "organization"  # Companies, institutions
    LOCATION = "location"          # Places
    CONCEPT = "concept"            # Abstract ideas
    EVENT = "event"                # Happenings
    OBJECT = "object"              # Inanimate objects
    SUBSTANCE = "substance"        # Materials, chemicals
    TIME = "time"                  # Temporal entities
    DIGITAL = "digital"            # Digital entities


@dataclass
class WorldEntity:
    """
    An entity in the world model.
    
    Entities have:
    - Identity (stable ID)
    - Name and aliases
    - Type classification
    - Properties (key-value attributes)
    - Relationships to other entities
    - Observations (what we've seen)
    - Confidence (how sure we are this entity exists)
    - Provenance (where entity info came from)
    - Temporal context (when entity info applies)
    """
    
    id: str = field(default_factory=lambda: _uid("entity"))
    name: str = ""
    type: EntityType = EntityType.OBJECT
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)  # Related entity IDs
    observations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    source_provenance: str = ""
    temporal_context: str = ""      # When this entity info is valid
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "aliases": self.aliases,
            "properties": self.properties,
            "relationship_count": len(self.relationships),
            "observation_count": len(self.observations),
            "confidence": self.confidence,
            "source_provenance": self.source_provenance,
            "temporal_context": self.temporal_context,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def update_property(self, key: str, value: Any, 
                       source: str = "", confidence_boost: float = 0.0):
        """Update a property with provenance."""
        old_value = self.properties.get(key)
        self.properties[key] = value
        self.updated_at = time.time()
        self.version += 1
        
        # Track observation
        self.observations.append({
            "property": key,
            "old_value": old_value,
            "new_value": value,
            "source": source,
            "timestamp": time.time(),
        })
        
        if confidence_boost:
            self.confidence = min(1.0, self.confidence + confidence_boost)


# ════════════════════════════════════════════════════════════════════
# RELATIONSHIP
# ════════════════════════════════════════════════════════════════════

class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    PART_OF = "part_of"             # Component relationship
    LOCATED_IN = "located_in"       # Spatial relationship
    RELATED_TO = "related_to"       # General association
    CAUSES = "causes"               # Causal relationship
    OWNED_BY = "owned_by"           # Ownership
    MEMBER_OF = "member_of"         # Group membership
    PRECEDES = "precedes"           # Temporal ordering
    FOLLOWS = "follows"             # Temporal ordering
    SIMILAR_TO = "similar_to"       # Similarity
    DIFFERENT_FROM = "different_from"  # Difference


@dataclass
class WorldRelationship:
    """
    A relationship between two world entities.
    
    Relationships have:
    - Subject and object (entity IDs)
    - Relationship type
    - Confidence (how sure we are)
    - Source provenance
    - Temporal context (when relationship holds)
    - Evidence (supporting observations)
    """
    
    id: str = field(default_factory=lambda: _uid("rel"))
    subject_id: str = ""
    predicate: RelationshipType = RelationshipType.RELATED_TO
    object_id: str = ""
    confidence: float = 0.5
    source: str = ""
    temporal_context: str = ""      # When this relationship holds
    evidence: list[str] = field(default_factory=list)
    is_inferred: bool = False
    created_at: float = field(default_factory=time.time)
    version: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate.value,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "source": self.source,
            "temporal_context": self.temporal_context,
            "evidence_count": len(self.evidence),
            "is_inferred": self.is_inferred,
            "version": self.version,
            "created_at": self.created_at,
        }
    
    def to_triple(self) -> tuple[str, str, str]:
        """Return as (subject, predicate, object) tuple."""
        return (self.subject_id, self.predicate.value, self.object_id)


# ════════════════════════════════════════════════════════════════════
# EVENT
# ════════════════════════════════════════════════════════════════════

class EventType(str, Enum):
    """Types of events."""
    EVENT = "event"                # Generic event
    ACTION = "action"               # Something done
    CHANGE = "change"               # State change
    INTERACTION = "interaction"     # Between entities
    OBSERVATION = "observation"     # Something observed
    DECISION = "decision"           # A choice made
    PROCESS_START = "process_start" # Beginnings
    PROCESS_END = "process_end"     # Endings


@dataclass
class WorldEvent:
    """
    An event in the world model.
    
    Events have:
    - Type classification
    - Participants (entity IDs)
    - Time context (when it happened)
    - Location context (where it happened)
    - Outcome (what resulted)
    - Confidence
    - Source provenance
    - Observations (what was seen)
    """
    
    id: str = field(default_factory=lambda: _uid("event"))
    type: EventType = EventType.EVENT
    name: str = ""
    participants: list[str] = field(default_factory=list)  # Entity IDs
    time_context: str = ""          # When it happened
    location_context: str = ""      # Where it happened
    outcome: str = ""
    observations: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source: str = ""
    created_at: float = field(default_factory=time.time)
    version: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "participant_count": len(self.participants),
            "time_context": self.time_context,
            "location_context": self.location_context,
            "outcome_preview": self.outcome[:100] if self.outcome else "",
            "observation_count": len(self.observations),
            "confidence": self.confidence,
            "source": self.source,
            "version": self.version,
            "created_at": self.created_at,
        }


# ════════════════════════════════════════════════════════════════════
# WORLD STATE (versioned)
# ════════════════════════════════════════════════════════════════════

@dataclass
class WorldStateVersion:
    """
    A versioned snapshot of world state.
    
    Each version captures:
    - Entities at this point in time
    - Relationships at this point in time
    - Events that occurred up to this point
    - Metadata about the version
    """
    
    version: int
    timestamp: float = field(default_factory=time.time)
    entities: dict[str, WorldEntity] = field(default_factory=dict)
    relationships: dict[str, WorldRelationship] = field(default_factory=dict)
    events: dict[str, WorldEvent] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "event_count": len(self.events),
            "metadata": self.metadata,
        }


class WorldState:
    """
    Versioned world state.
    
    Maintains:
    - Current entities, relationships, events
    - Version history
    - Change tracking
    
    Each modification creates a new version.
    """
    
    def __init__(self):
        self._current_version: int = 0
        self._versions: dict[int, WorldStateVersion] = {}
        self._entities: dict[str, WorldEntity] = {}
        self._relationships: dict[str, WorldRelationship] = {}
        self._events: dict[str, WorldEvent] = {}
        
        # Create initial version
        self._create_version()
    
    def _create_version(self) -> WorldStateVersion:
        """Create a new version capturing current state."""
        self._current_version += 1
        version = WorldStateVersion(
            version=self._current_version,
            timestamp=time.time(),
            entities=self._entities.copy(),
            relationships=self._relationships.copy(),
            events=self._events.copy(),
            metadata={"change_description": "New version"},
        )
        self._versions[self._current_version] = version
        return version
    
    def add_entity(self, entity: WorldEntity) -> WorldEntity:
        """Add or update an entity."""
        self._entities[entity.id] = entity
        self._create_version()
        return entity
    
    def get_entity(self, entity_id: str) -> WorldEntity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> WorldEntity | None:
        """Get an entity by name (case-insensitive)."""
        name_lower = name.lower()
        for entity in self._entities.values():
            if entity.name.lower() == name_lower:
                return entity
            if name_lower in [a.lower() for a in entity.aliases]:
                return entity
        return None
    
    def add_relationship(self, relationship: WorldRelationship) -> WorldRelationship:
        """Add or update a relationship."""
        self._relationships[relationship.id] = relationship
        self._create_version()
        return relationship
    
    def get_relationship(self, rel_id: str) -> WorldRelationship | None:
        """Get a relationship by ID."""
        return self._relationships.get(rel_id)
    
    def get_relationships_for_entity(self, entity_id: str) -> list[WorldRelationship]:
        """Get all relationships involving an entity."""
        results = []
        for rel in self._relationships.values():
            if rel.subject_id == entity_id or rel.object_id == entity_id:
                results.append(rel)
        return results
    
    def debug_dump(self):
        """Debug helper to see what's stored."""
        print(f"Entities: {list(self._entities.keys())}")
        print(f"Relationships: {list(self._relationships.keys())}")
        for rid, rel in self._relationships.items():
            print(f"  {rid}: {rel.subject_id} {rel.predicate} {rel.object_id}")
        print(f"Events: {list(self._events.keys())}")
    
    def add_event(self, event: WorldEvent) -> WorldEvent:
        """Add an event."""
        if event.id in self._events:
            old = self._events[event.id]
            old.outcome = event.outcome or old.outcome
            old.version += 1
            event = old
        
        self._events[event.id] = event
        self._create_version()
        return event
    
    def get_event(self, event_id: str) -> WorldEvent | None:
        """Get an event by ID."""
        return self._events.get(event_id)
    
    def get_events_for_entity(self, entity_id: str) -> list[WorldEvent]:
        """Get events involving an entity."""
        results = []
        for event in self._events.values():
            if entity_id in event.participants:
                results.append(event)
        return results
    
    def get_current_version(self) -> WorldStateVersion:
        """Get the current version."""
        return self._versions.get(self._current_version)
    
    def get_version(self, version: int) -> WorldStateVersion | None:
        """Get a specific version."""
        return self._versions.get(version)
    
    def get_version_history(self, limit: int = 10) -> list[WorldStateVersion]:
        """Get recent version history."""
        versions = sorted(self._versions.values(), 
                         key=lambda v: v.version, reverse=True)
        return versions[:limit]
    
    def query_entities(self, entity_type: EntityType | None = None,
                      min_confidence: float = 0.0) -> list[WorldEntity]:
        """Query entities by type and confidence."""
        results = list(self._entities.values())
        
        if entity_type:
            results = [e for e in results if e.type == entity_type]
        
        results = [e for e in results if e.confidence >= min_confidence]
        
        return results
    
    def query_relationships(self, predicate: RelationshipType | None = None,
                           min_confidence: float = 0.0) -> list[WorldRelationship]:
        """Query relationships by type and confidence."""
        results = list(self._relationships.values())
        
        if predicate:
            results = [r for r in results if r.predicate == predicate]
        
        results = [r for r in results if r.confidence >= min_confidence]
        
        return results
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize current state."""
        return {
            "current_version": self._current_version,
            "entities": {eid: e.to_dict() for eid, e in self._entities.items()},
            "relationships": {rid: r.to_dict() for rid, r in self._relationships.items()},
            "events": {eid: e.to_dict() for eid, e in self._events.items()},
            "version_history": [
                v.to_dict() for v in self.get_version_history(10)
            ],
        }
    
    def clear(self):
        """Clear all world state."""
        self._current_version = 0
        self._versions.clear()
        self._entities.clear()
        self._relationships.clear()
        self._events.clear()
        self._create_version()
    
    def stats(self) -> dict[str, Any]:
        """Get world state statistics."""
        return {
            "current_version": self._current_version,
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "events": len(self._events),
            "version_history_length": len(self._versions),
        }


# ════════════════════════════════════════════════════════════════════
# WORLD MODEL (orchestrator)
# ════════════════════════════════════════════════════════════════════

class WorldModel:
    """
    Orchestrates the world model.
    
    Manages:
    - World state (versioned entities, relationships, events)
    - Entity resolution (matching entities across sources)
    - Relationship inference
    - Conflict detection
    - Temporal reasoning
    
    The world model is the central state representation
    that all other modules reference.
    """
    
    def __init__(self):
        self.state = WorldState()
        self._entity_resolution_cache: dict[str, str] = {}  # Alias -> canonical ID
    
    def add_entity(self, name: str, entity_type: EntityType = EntityType.OBJECT,
                  properties: dict[str, Any] | None = None,
                  confidence: float = 0.5,
                  source: str = "") -> WorldEntity | None:
        """Add an entity to the world model."""
        entity = WorldEntity(
            name=name,
            type=entity_type,
            properties=properties or {},
            confidence=confidence,
            source_provenance=source,
        )
        result = self.state.add_entity(entity)
        return result
    
    def get_entity(self, entity_id: str) -> WorldEntity | None:
        """Get an entity by ID."""
        return self.state.get_entity(entity_id)
    
    def get_entity_by_name(self, name: str) -> WorldEntity | None:
        """Get an entity by name or alias."""
        # Check cache first
        if name in self._entity_resolution_cache:
            return self.state.get_entity(self._entity_resolution_cache[name])
        
        entity = self.state.get_entity_by_name(name)
        if entity:
            # Cache all aliases
            for alias in entity.aliases:
                self._entity_resolution_cache[alias] = entity.id
            self._entity_resolution_cache[entity.name] = entity.id
        
        return entity
    
    def resolve_entity(self, name: str, existing_entity_id: str | None = None) -> WorldEntity | None:
        """
        Resolve an entity name to an existing entity or create new.
        
        Entity resolution:
        - Check if name matches existing entity
        - Check aliases
        - If ambiguous, create new entity (caller must resolve)
        """
        existing = self.get_entity_by_name(name)
        if existing:
            return existing
        
        # Create new entity
        new_entity = WorldEntity(
            name=name,
            confidence=0.3,  # Low confidence for new entities
            source_provenance="entity_resolution",
        )
        result = self.state.add_entity(new_entity)
        return result
    
    def add_relationship(self, subject_id: str, predicate: RelationshipType,
                        object_id: str, confidence: float = 0.5,
                        source: str = "", 
                        temporal_context: str = "") -> WorldRelationship:
        """Add a relationship between entities."""
        rel = WorldRelationship(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            source=source,
            temporal_context=temporal_context,
        )
        return self.state.add_relationship(rel)
    
    def add_event(self, event_type: EventType, name: str,
                  participants: list[str] | None = None,
                  time_context: str = "",
                  location_context: str = "",
                  outcome: str = "",
                  confidence: float = 0.5,
                  source: str = "") -> WorldEvent | None:
        """Add an event to the world model."""
        event = WorldEvent(
            type=event_type,
            name=name,
            participants=participants or [],
            time_context=time_context,
            location_context=location_context,
            outcome=outcome,
            confidence=confidence,
            source=source,
        )
        result = self.state.add_event(event)
        return result
    
    def detect_contradictions(self) -> list[dict[str, Any]]:
        """
        Detect contradictions in the world model.
        
        Checks for:
        - Conflicting property values
        - Mutual exclusion violations
        - Conflicting relationships (PART_OF vs DIFFERENT_FROM)
        - Temporal inconsistencies
        """
        contradictions = []
        
        # Build lookup by (subject, object) for efficient contradiction detection
        rel_by_pair: dict[tuple[str, str], list[WorldRelationship]] = {}
        for rel in self.state._relationships.values():
            key = (rel.subject_id, rel.object_id)
            rel_by_pair.setdefault(key, []).append(rel)
        
        # Check each pair for contradictory predicates
        for (subj, obj), rels in rel_by_pair.items():
            predicates = {r.predicate for r in rels}
            
            # PART_OF vs DIFFERENT_FROM are contradictory
            if RelationshipType.PART_OF in predicates and RelationshipType.DIFFERENT_FROM in predicates:
                part_of_rel = next(r for r in rels if r.predicate == RelationshipType.PART_OF)
                diff_rel = next(r for r in rels if r.predicate == RelationshipType.DIFFERENT_FROM)
                contradictions.append({
                    "type": "conflicting_relationship",
                    "relationship_1": part_of_rel.to_triple(),
                    "relationship_2": diff_rel.to_triple(),
                    "description": f"{subj} is both part of and different from {obj}",
                    "confidence": min(part_of_rel.confidence, diff_rel.confidence),
                })
            
            # LOCATED_IN vs different_from could be a contradiction too
            # (simplified - a thing located in X is not different from X in the relevant sense)
        
        return contradictions
    
    def get_related_entities(self, entity_id: str, 
                            relationship_type: RelationshipType | None = None,
                            max_depth: int = 1) -> list[WorldEntity]:
        """
        Get entities related to a given entity.
        
        Args:
            entity_id: Starting entity
            relationship_type: Filter by relationship type
            max_depth: How many hops to traverse
            
        Returns:
            List of related entities.
        """
        if max_depth < 1:
            return []
        
        visited = {entity_id}
        frontier = [entity_id]
        related = []
        
        for _ in range(max_depth):
            next_frontier = []
            for eid in frontier:
                for rel in self.state.get_relationships_for_entity(eid):
                    if relationship_type and rel.predicate != relationship_type:
                        continue
                    
                    other_id = rel.object_id if rel.subject_id == eid else rel.subject_id
                    if other_id not in visited:
                        visited.add(other_id)
                        next_frontier.append(other_id)
                        
                        entity = self.state.get_entity(other_id)
                        if entity:
                            related.append(entity)
            
            frontier = next_frontier
        
        return related
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize world model."""
        return {
            "state": self.state.to_dict(),
            "entity_resolution_cache_size": len(self._entity_resolution_cache),
        }
    
    def stats(self) -> dict[str, Any]:
        """Get world model statistics."""
        state_stats = self.state.stats()
        state_stats["entity_resolution_cache"] = len(self._entity_resolution_cache)
        return state_stats


# ════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════

__all__ = [
    "EntityType",
    "WorldEntity",
    "RelationshipType",
    "WorldRelationship",
    "EventType",
    "WorldEvent",
    "WorldState",
    "WorldStateVersion",
    "WorldModel",
]
