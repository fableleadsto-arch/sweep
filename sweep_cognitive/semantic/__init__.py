"""
Semantic Understanding Engine — Phase 3.

Responsible for extracting meaning from percepts:
- Concept representation
- Entity representation with attributes
- Relationship extraction
- Event representation
- Context representation
- Ambiguity detection

Key principle: Do NOT implement meaning primarily through rules.
This module provides the scaffolding for semantic understanding,
which will be enhanced with learned models in later phases.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..representation import (
    Representation,
    TextRepresentation,
    RepresentationQuality,
    RepresentationType,
    EmbeddingVector,
)


class SemanticType(str, Enum):
    """Types of semantic entities."""
    ENTITY = "entity"              # Physical or abstract thing
    CONCEPT = "concept"            # Abstract idea
    EVENT = "event"                # Something that happens
    ACTION = "action"              # Something done
    PROPERTY = "property"          # Attribute of entity/concept
    RELATIONSHIP = "relationship"  # Connection between entities
    ATTRIBUTE = "attribute"        # Specific property value
    CATEGORY = "category"          # Class/group membership
    QUANTITY = "quantity"          # Numerical value
    TEMPORAL = "temporal"          # Time-related
    LOCATION = "location"          # Place-related


@dataclass
class SemanticEntity:
    """
    A semantic entity with its properties.
    
    Entities are the basic units of semantic understanding.
    They can be physical (cat, building), abstract (justice, freedom),
    or events (meeting, collision).
    """
    id: str = field(default_factory=lambda: f"entity_{int(time.time() * 1000)}")
    name: str = ""
    entity_type: SemanticType = SemanticType.ENTITY
    kind: str = "unknown"           # Animal, person, organization, object, etc.
    attributes: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    mentioned_in: list[str] = field(default_factory=list)  # Source references
    confidence: float = 0.5
    is_mentioned: bool = False      # Appears in current context
    is_referenced: bool = False     # Referenced but not directly mentioned
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "kind": self.kind,
            "attributes": self.attributes,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "is_mentioned": self.is_mentioned,
            "is_referenced": self.is_referenced,
        }
    
    def with_attribute(self, key: str, value: Any) -> "SemanticEntity":
        """Return a copy with an additional attribute."""
        new = SemanticEntity(
            id=self.id,
            name=self.name,
            entity_type=self.entity_type,
            kind=self.kind,
            attributes=self.attributes.copy(),
            aliases=self.aliases.copy(),
            mentioned_in=self.mentioned_in.copy(),
            confidence=self.confidence,
        )
        new.attributes[key] = value
        return new


@dataclass
class SemanticRelationship:
    """
    A relationship between two semantic entities.
    
    Relationships connect entities and form the structure of
    semantic understanding.
    """
    id: str = field(default_factory=lambda: f"rel_{int(time.time() * 1000)}")
    subject: str = ""               # Entity ID or name
    predicate: str = ""             # Relationship type
    object_: str = ""               # Entity ID or name
    relationship_type: SemanticType = SemanticType.RELATIONSHIP
    confidence: float = 0.5
    source: str = ""
    temporal_context: str = ""      # When this relationship holds
    is_inferred: bool = False       # Derived, not directly stated
    evidence: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "type": self.relationship_type.value,
            "confidence": self.confidence,
            "source": self.source,
            "temporal_context": self.temporal_context,
            "is_inferred": self.is_inferred,
            "evidence_count": len(self.evidence),
        }


@dataclass
class SemanticEvent:
    """
    An event with its participants, time, and location.
    
    Events are things that happen, with temporal and often spatial
    extent.
    """
    id: str = field(default_factory=lambda: f"event_{int(time.time() * 1000)}")
    name: str = ""                  # Event type/name
    event_type: SemanticType = SemanticType.EVENT
    participants: list[str] = field(default_factory=list)  # Entity IDs
    time_context: str = ""          # When it happened
    location_context: str = ""      # Where it happened
    outcome: str = ""               # What resulted
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    source: str = ""
    is_observed: bool = False      # Directly observed
    is_inferred: bool = False      # Deduced from other info
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.event_type.value,
            "participants": self.participants,
            "time_context": self.time_context,
            "location_context": self.location_context,
            "outcome": self.outcome,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "source": self.source,
            "is_observed": self.is_observed,
            "is_inferred": self.is_inferred,
        }


@dataclass
class SemanticContext:
    """
    Context for semantic interpretation.
    
    Context includes temporal, spatial, social, and domain context
    that affects how entities and relationships should be understood.
    """
    id: str = field(default_factory=lambda: f"context_{int(time.time() * 1000)}")
    context_type: str = "general"   # temporal, spatial, social, domain, etc.
    description: str = ""
    entities: list[str] = field(default_factory=list)
    temporal_marker: str = ""
    location_marker: str = ""
    domain: str = ""                # Topic/domain context
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.context_type,
            "description": self.description,
            "entity_count": len(self.entities),
            "temporal_marker": self.temporal_marker,
            "location_marker": self.location_marker,
            "domain": self.domain,
            "assumption_count": len(self.assumptions),
            "confidence": self.confidence,
        }


class Ambiguity:
    """
    Represents an ambiguity in semantic interpretation.
    
    Ambiguities are tracked so the system can:
    - Recognize when multiple interpretations are possible
    - Ask for clarification when needed
    - Track which interpretation was chosen
    """
    
    def __init__(self, ambiguity_type: str, content: str,
                 alternatives: list[dict[str, Any]]):
        self.ambiguity_type = ambiguity_type
        self.content = content
        self.alternatives = alternatives
        self.resolution: Optional[str] = None  # Which alternative was chosen
        self.resolution_confidence: float = 0.0
        self.timestamp = time.time()
    
    def resolve(self, alternative_index: int, confidence: float):
        """Mark an alternative as the resolved interpretation."""
        if 0 <= alternative_index < len(self.alternatives):
            self.resolution = self.alternatives[alternative_index].get("interpretation", "")
            self.resolution_confidence = confidence
    
    def is_resolved(self) -> bool:
        return self.resolution is not None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguity_type": self.ambiguity_type,
            "content": self.content,
            "alternatives_count": len(self.alternatives),
            "is_resolved": self.is_resolved(),
            "resolution": self.resolution,
            "resolution_confidence": self.resolution_confidence,
        }


class SemanticUnderstandingEngine:
    """
    Engine for extracting semantic understanding from text representations.
    
    This is the scaffolding implementation. It extracts:
    - Entities (people, objects, organizations, etc.)
    - Concepts (abstract ideas)
    - Relationships between entities
    - Events with participants and time
    - Context (temporal, spatial, domain)
    - Ambiguities
    
    Current implementation is pattern-based. In later phases,
    this will be enhanced with learned models.
    """

    def __init__(self):
        self._entities: dict[str, SemanticEntity] = {}
        self._relationships: list[SemanticRelationship] = []
        self._events: list[SemanticEvent] = []
        self._contexts: list[SemanticContext] = []
        self._ambiguities: list[Ambiguity] = []
    
    def understand(self, text_representation: TextRepresentation) -> dict[str, Any]:
        """
        Extract semantic understanding from a text representation.
        
        Args:
            text_representation: The text to understand
            
        Returns:
            Dictionary with extracted semantic information.
        """
        t0 = time.perf_counter()
        
        text = text_representation.content
        if not text:
            return {
                "entities": [],
                "relationships": [],
                "events": [],
                "contexts": [],
                "ambiguities": [],
                "semantic_quality": RepresentationQuality.NONE,
                "processing_latency_ms": 0,
            }
        
        # Extract entities
        entities = self._extract_entities(text, text_representation)
        
        # Extract relationships
        relationships = self._extract_relationships(entities, text)
        
        # Extract events
        events = self._extract_events(text, entities)
        
        # Extract context
        contexts = self._extract_context(text, text_representation)
        
        # Detect ambiguities
        ambiguities = self._detect_ambiguities(text, entities)
        
        # Assess semantic quality
        quality = self._assess_semantic_quality(
            entities, relationships, events, contexts
        )
        
        latency = (time.perf_counter() - t0) * 1000
        
        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "events": [e.to_dict() for e in events],
            "contexts": [c.to_dict() for c in contexts],
            "ambiguities": [a.to_dict() for a in ambiguities],
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "event_count": len(events),
            "context_count": len(contexts),
            "ambiguity_count": len(ambiguities),
            "semantic_quality": quality,
            "processing_latency_ms": latency,
        }
    
    def _extract_entities(self, text: str, 
                          text_rep: TextRepresentation) -> list[SemanticEntity]:
        """Extract entities from text."""
        entities = []
        
        # Use entities from text representation if available
        if text_rep.entities:
            for ent in text_rep.entities:
                entity = SemanticEntity(
                    name=ent["text"],
                    entity_type=SemanticType.ENTITY,
                    kind=ent.get("type", "unknown"),
                    confidence=0.6,
                )
                entity.is_mentioned = True
                entities.append(entity)
        
        # Pattern-based entity extraction (temporary scaffolding)
        # People patterns: multi-word names ("Alice Smith") get higher
        # confidence; single capitalized mid-sentence words ("Alice met...")
        # are weaker evidence (could be sentence start), so we skip the
        # first word of the sentence for single-name extraction.
        person_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in person_pattern.finditer(text):
            name = match.group(0)
            # Check if already extracted
            if not any(e.name == name for e in entities):
                entity = SemanticEntity(
                    name=name,
                    entity_type=SemanticType.ENTITY,
                    kind="person",
                    confidence=0.5,
                )
                entity.is_mentioned = True
                entities.append(entity)
        
        # Single capitalized words NOT at sentence start → weaker entity signal
        if not text or not text[0].isupper() or True:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            offset = 0
            for sentence in sentences:
                # Skip the first word of each sentence (ambiguous)
                words = sentence.split()
                for i, word in enumerate(words):
                    clean = word.strip('.,!?;:"()')
                    if i == 0:
                        continue
                    if re.fullmatch(r'[A-Z][a-z]{1,}', clean) and \
                       not any(e.name == clean for e in entities):
                        entity = SemanticEntity(
                            name=clean,
                            entity_type=SemanticType.ENTITY,
                            kind="person",
                            confidence=0.4,
                        )
                        entity.is_mentioned = True
                        entities.append(entity)
                offset += len(words) + 1
        
        # Organization patterns (ALL CAPS or capitalized multi-word)
        org_pattern = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})\b')
        for match in org_pattern.finditer(text):
            name = match.group(0)
            if not any(e.name == name for e in entities):
                entity = SemanticEntity(
                    name=name,
                    entity_type=SemanticType.ENTITY,
                    kind="organization",
                    confidence=0.4,
                )
                entity.is_mentioned = True
                entities.append(entity)
        
        # Location patterns
        location_words = {"here", "there", "everywhere", "nowhere", "somewhere",
                         "city", "country", "state", "region", "area", "place"}
        for word in location_words:
            if word in text.lower():
                entity = SemanticEntity(
                    name=word,
                    entity_type=SemanticType.LOCATION,
                    kind="location",
                    confidence=0.3,
                )
                entity.is_referenced = True
                entities.append(entity)
        
        return entities
    
    def _extract_relationships(self, entities: list[SemanticEntity],
                               text: str) -> list[SemanticRelationship]:
        """Extract relationships between entities."""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Simple relationship patterns
        # "X of Y", "X's Y", "X and Y", "X with Y"
        relationship_patterns = [
            (r'\b(\w+)\s+of\s+(\w+)\b', 'has_property', 0.4),
            (r"\b(\w+)'s\s+(\w+)\b", 'possession', 0.4),
            (r'\b(\w+)\s+and\s+(\w+)\b', 'association', 0.3),
            (r'\b(\w+)\s+with\s+(\w+)\b', 'accompaniment', 0.3),
            (r'\b(\w+)\s+in\s+(\w+)\b', 'location', 0.3),
            (r'\b(\w+)\s+at\s+(\w+)\b', 'location', 0.3),
        ]
        
        for pattern, predicate, base_confidence in relationship_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj = match.group(1)
                obj = match.group(2)
                
                # Check if entities match
                subj_match = next((e for e in entities if e.name.lower() == subj.lower()), None)
                obj_match = next((e for e in entities if e.name.lower() == obj.lower()), None)
                
                if subj_match or obj_match:
                    rel = SemanticRelationship(
                        subject=subj_match.name if subj_match else subj,
                        predicate=predicate,
                        object_=obj_match.name if obj_match else obj,
                        confidence=base_confidence,
                        is_inferred=False,
                        source="pattern_extraction",
                    )
                    relationships.append(rel)
        
        return relationships
    
    def _extract_events(self, text: str,
                        entities: list[SemanticEntity]) -> list[SemanticEvent]:
        """Extract events from text."""
        events = []
        
        # Event patterns: verbs that indicate events
        event_indicators = [
            (r'\b(?:happened|occurred|took place|transpired)\b', 'occurrence'),
            (r'\b(?:met|met with|convened|gathered)\b', 'gathering'),
            (r'\b(?:started|began|commenced)\b', 'start'),
            (r'\b(?:ended|finished|completed|concluded)\b', 'end'),
            (r'\b(?:created|built|constructed|made)\b', 'creation'),
            (r'\b(?:destroyed|broke|demolished)\b', 'destruction'),
            (r'\b(?:moved|traveled|went|arrived|left)\b', 'movement'),
            (r'\b(?:said|stated|announced|declared)\b', 'communication'),
        ]
        
        for pattern, event_type in event_indicators:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                event = SemanticEvent(
                    name=event_type,
                    event_type=SemanticType.EVENT,
                    confidence=0.4,
                    source="pattern_extraction",
                )
                
                # Try to find participants
                words_before = text[max(0, match.start()-50):match.start()].split()
                words_after = text[match.end():match.end()+50].split()
                nearby_words = words_before[-3:] + words_after[:3]
                
                for entity in entities:
                    if entity.name.lower() in [w.lower() for w in nearby_words]:
                        event.participants.append(entity.name)
                
                if event.participants:
                    events.append(event)
        
        return events
    
    def _extract_context(self, text: str,
                         text_rep: TextRepresentation) -> list[SemanticContext]:
        """Extract context from text."""
        contexts = []
        
        # Temporal context
        temporal_markers = text_rep.temporal_markers if text_rep else []
        if temporal_markers:
            context = SemanticContext(
                context_type="temporal",
                description="Temporal context from markers",
                temporal_marker=", ".join(temporal_markers[:3]),
                confidence=0.6,
            )
            contexts.append(context)
        
        # Check for time words
        time_words = {"now", "then", "today", "yesterday", "tomorrow", "soon",
                     "later", "currently", "previously", "recently", "in the past",
                     "future", "currently", "at the same time"}
        found_time = [w for w in time_words if w in text.lower()]
        if found_time and not temporal_markers:
            context = SemanticContext(
                context_type="temporal",
                description="Temporal context from time words",
                temporal_marker=", ".join(found_time[:3]),
                confidence=0.4,
            )
            contexts.append(context)
        
        # Domain context from entities
        if text_rep.entities:
            domains = set()
            for ent in text_rep.entities:
                if ent.get("type") == "PERSON_ORGANIZATION":
                    domains.add("social")
                elif ent.get("type") == "DATE" or ent.get("type") == "TIME":
                    domains.add("temporal")
            
            if domains:
                context = SemanticContext(
                    context_type="domain",
                    description=f"Domain context from entities",
                    domain=", ".join(domains),
                    confidence=0.5,
                )
                contexts.append(context)
        
        return contexts
    
    def _detect_ambiguities(self, text: str,
                            entities: list[SemanticEntity]) -> list[Ambiguity]:
        """Detect ambiguities in the text."""
        ambiguities = []
        
        # Known ambiguous words
        ambiguous_words = {
            "bank": ["financial_institution", "river_edge", "airplane_maneuver"],
            "bat": ["flying_mammal", "sports_equipment"],
            "crane": ["bird", "construction_equipment"],
            "current": ["flow_of_electricity", "present_time", "water_flow"],
            "lead": ["metal", "to_guide", "heavy_weight"],
            "match": ["game_contest", "correspond", "fire_starter"],
            "park": ["green_space", "to_leave_vehicle", "to_rest"],
            "ring": ["jewelry", "sound", "circular_shape", "boxing_arena"],
            "scale": ["measurement", "fish_skin", "musical_notes", "climb"],
            "spring": ["season", "coil", "water_source", "jump"],
            "table": ["furniture", "data_chart", "postpone"],
            "wave": ["hand_gesture", "water_motion", "sound_propagation"],
            "light": ["illumination", "not_heavy", "color", "fire"],
            "rock": ["stone", "music_genre", "shake"],
        }
        
        text_lower = text.lower()
        for word, meanings in ambiguous_words.items():
            if word in text_lower:
                # Check if entity exists for this word
                entity_match = next((e for e in entities if e.name.lower() == word), None)
                
                ambiguity = Ambiguity(
                    ambiguity_type="lexical",
                    content=f"Word '{word}' has multiple meanings",
                    alternatives=[
                        {"interpretation": meaning, "meaning": meaning}
                        for meaning in meanings
                    ],
                )
                
                # If entity context helps disambiguate, note it
                if entity_match:
                    ambiguity.alternatives.append({
                        "interpretation": f"entity_context: {entity_match.kind}",
                        "meaning": f"entity_context: {entity_match.kind}",
                    })
                
                ambiguities.append(ambiguity)
        
        return ambiguities
    
    def _assess_semantic_quality(self, entities: list, relationships: list,
                                 events: list, contexts: list) -> RepresentationQuality:
        """Assess the quality of semantic extraction."""
        score = 0
        
        if entities:
            score += min(len(entities), 3)
        if relationships:
            score += min(len(relationships), 2)
        if events:
            score += 1
        if contexts:
            score += 1
        
        if score >= 5:
            return RepresentationQuality.HIGH
        elif score >= 3:
            return RepresentationQuality.MODERATE
        elif score >= 1:
            return RepresentationQuality.LOW
        return RepresentationQuality.UNKNOWN


class SemanticRepresentation(Representation):
    """
    A representation with semantic understanding attached.
    
    This wraps a base representation with extracted semantic information.
    """
    
    def __init__(self, base_representation: Representation,
                 semantic_understanding: dict[str, Any],
                 **kwargs):
        # Copy from base representation
        super().__init__(
            content=base_representation.content,
            representation_type=base_representation.representation_type,
            quality=base_representation.quality,
            confidence=base_representation.confidence,
            source=base_representation.source,
            modality=base_representation.modality,
            created_at=base_representation.created_at,
            embedding=base_representation.embedding,
            metadata=base_representation.metadata.copy(),
            id=base_representation.id,
        )
        
        self.semantic_understanding = semantic_understanding
        self.entity_count = semantic_understanding.get("entity_count", 0)
        self.relationship_count = semantic_understanding.get("relationship_count", 0)
        self.event_count = semantic_understanding.get("event_count", 0)
        self.ambiguity_count = semantic_understanding.get("ambiguity_count", 0)
        self.semantic_quality = semantic_understanding.get("semantic_quality", 
                                                           RepresentationQuality.UNKNOWN)
    
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "semantic_understanding": self.semantic_understanding,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "event_count": self.event_count,
            "ambiguity_count": self.ambiguity_count,
            "semantic_quality": self.semantic_quality.value if self.semantic_quality else None,
        })
        return d


__all__ = [
    "SemanticType",
    "SemanticEntity",
    "SemanticRelationship",
    "SemanticEvent",
    "SemanticContext",
    "Ambiguity",
    "SemanticUnderstandingEngine",
    "SemanticRepresentation",
]
