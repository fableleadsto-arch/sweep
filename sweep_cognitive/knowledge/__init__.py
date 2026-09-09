"""
Knowledge Base — Phase 1: Learned knowledge representation.

This converts the old world_knowledge.py (hard-coded entities) into an
updatable knowledge base that supports:
- Learned/updatable entries with confidence
- Provenance tracking
- Multiple knowledge types
- Conflict resolution
- Evidence-based updates

Key principle: Knowledge should not be hard-coded. It should be:
-Updatable based on evidence
-Tracked with confidence
-Traceable to source
-Able to represent uncertainty
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class KnowledgeType(str, Enum):
    """Types of knowledge entries."""
    ENTITY_PROPERTY = "entity_property"      # Properties of entities
    RELATION = "relation"                    # Relationships between entities
    FACT = "fact"                            # General facts
    CONCEPT = "concept"                      # Abstract concepts
    EVENT = "event"                          # Events
    CATEGORICAL = "categorical"              # Category membership
    QUANTITATIVE = "quantitative"            # Numerical knowledge
    TEMPORAL = "temporal"                    # Time-related knowledge


class KnowledgeConfidence(str, Enum):
    """Confidence levels for knowledge entries."""
    KNOWN = "known"              # High confidence, well-established
    LIKELY = "likely"            # Good confidence, some uncertainty
    POSSIBLE = "possible"        # Low confidence, plausible
    DISPUTED = "disputed"        # Conflicting evidence exists
    UNKNOWN = "unknown"          # Cannot determine
    FALSE = "false"              # Known to be false


@dataclass
class KnowledgeSource:
    """Source information for a knowledge entry."""
    name: str
    source_type: str = "unknown"    # "training", "observation", "retrieval", "inference"
    reliability: float = 0.5        # 0-1 source reliability
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "reliability": self.reliability,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeEntry:
    """
    A single entry in the knowledge base.

    Each entry has:
    - Subject: what the knowledge is about
    - Predicate: what aspect is being described
    - Object: the value/relationship
    - Confidence: how confident we are
    - Source: where this knowledge came from
    - Evidence: supporting evidence references
    - Contradictions: conflicting knowledge references
    - Temporal context: when this knowledge applies
    """

    subject: str
    predicate: str
    object: str
    knowledge_type: KnowledgeType = KnowledgeType.FACT
    confidence: KnowledgeConfidence = KnowledgeConfidence.UNKNOWN
    confidence_score: float = 0.5      # Numeric confidence 0-1
    sources: list[KnowledgeSource] = field(default_factory=list)
    evidence_count: int = 0
    contradiction_count: int = 0
    last_updated: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    temporal_context: str = ""          # When this applies
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"knowledge_{int(time.time() * 1000)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "type": self.knowledge_type.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "temporal_context": self.temporal_context,
            "sources": [s.to_dict() for s in self.sources],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
        }

    def with_updated_confidence(self, new_confidence: KnowledgeConfidence,
                                new_score: float, reason: str = "") -> "KnowledgeEntry":
        """Return a copy with updated confidence."""
        entry = KnowledgeEntry(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            knowledge_type=self.knowledge_type,
            confidence=new_confidence,
            confidence_score=new_score,
            sources=self.sources.copy(),
            evidence_count=self.evidence_count,
            contradiction_count=self.contradiction_count,
            last_updated=time.time(),
            created_at=self.created_at,
            temporal_context=self.temporal_context,
            metadata=self.metadata.copy(),
            id=self.id,
        )
        if reason:
            entry.metadata["last_update_reason"] = reason
        return entry


class KnowledgeBase:
    """
    A knowledge base that stores and retrieves knowledge entries.

    This replaces the hard-coded world_knowledge.py with an updatable
    knowledge system that can learn from evidence.

    The knowledge base supports:
    - Querying by subject, predicate, or object
    - Confidence-based retrieval
    - Conflict detection
    - Evidence-based updates
    - Temporal context
    """

    def __init__(self):
        self._entries: dict[str, KnowledgeEntry] = {}
        self._by_subject: dict[str, list[str]] = {}
        self._by_predicate: dict[str, list[str]] = {}
        self._by_object: dict[str, list[str]] = {}
        self._initialized = False

    def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Add a knowledge entry, merging with existing if same key."""
        key = self._make_key(entry.subject, entry.predicate, entry.object)

        if key in self._entries:
            # Merge: update confidence based on new evidence
            existing = self._entries[key]
            return self._merge(existing, entry)
        else:
            self._entries[key] = entry
            self._index(entry)
            return entry

    def get(self, subject: str, predicate: str = "",
            object_: str = "") -> list[KnowledgeEntry]:
        """Retrieve knowledge entries matching criteria."""
        results = []
        for entry in self._entries.values():
            if subject and entry.subject.lower() != subject.lower():
                continue
            if predicate and entry.predicate.lower() != predicate.lower():
                continue
            if object_ and entry.object.lower() != object_.lower():
                continue
            results.append(entry)
        return results

    def get_by_subject(self, subject: str) -> list[KnowledgeEntry]:
        """Get all knowledge about a subject."""
        results = []
        for entry in self._entries.values():
            if entry.subject.lower() == subject.lower():
                results.append(entry)
        return results

    def get_confident(self, min_confidence: float = 0.7) -> list[KnowledgeEntry]:
        """Get entries with confidence above threshold."""
        return [e for e in self._entries.values() 
                if e.confidence_score >= min_confidence]

    def get_disputed(self) -> list[KnowledgeEntry]:
        """Get entries with contradictions."""
        return [e for e in self._entries.values() 
                if e.contradiction_count > 0]

    def update_confidence(self, entry_id: str, new_confidence: KnowledgeConfidence,
                         new_score: float, reason: str = "") -> Optional[KnowledgeEntry]:
        """Update the confidence of a knowledge entry."""
        if not entry_id:
            return None
        # Find entry by ID
        for key, entry in self._entries.items():
            if entry.id == entry_id:
                updated = entry.with_updated_confidence(new_confidence, new_score, reason)
                self._entries[key] = updated
                return updated
        return None

    def add_evidence(self, entry_id: str, source: KnowledgeSource,
                    strength: float = 1.0) -> Optional[KnowledgeEntry]:
        """Add supporting evidence to a knowledge entry."""
        if not entry_id:
            return None
        # Find entry by ID
        for key, entry in self._entries.items():
            if entry.id == entry_id:
                entry.evidence_count += 1
                entry.sources.append(source)
                # Boost confidence based on evidence strength
                old_score = entry.confidence_score
                boost = 0.05 * strength
                new_score = min(1.0, old_score + boost)
                if new_score >= 0.8:
                    new_conf = KnowledgeConfidence.KNOWN
                elif new_score >= 0.6:
                    new_conf = KnowledgeConfidence.LIKELY
                elif new_score >= 0.4:
                    new_conf = KnowledgeConfidence.POSSIBLE
                else:
                    new_conf = KnowledgeConfidence.UNKNOWN
                updated = entry.with_updated_confidence(new_conf, new_score, 
                                                        f"added evidence (score: {old_score:.2f} -> {new_score:.2f})")
                self._entries[key] = updated
                return updated
        return None

    def add_contradiction(self, entry_id: str, contradictory_entry: KnowledgeEntry,
                         source: KnowledgeSource) -> Optional[KnowledgeEntry]:
        """Record a contradiction to a knowledge entry."""
        if not entry_id:
            return None
        # Find entry by ID
        for key, entry in self._entries.items():
            if entry.id == entry_id:
                entry.contradiction_count += 1
                # Reduce confidence
                old_score = entry.confidence_score
                reduction = 0.05
                new_score = max(0.0, old_score - reduction)
                if new_score <= 0.3:
                    new_conf = KnowledgeConfidence.DISPUTED
                elif new_score <= 0.5:
                    new_conf = KnowledgeConfidence.POSSIBLE
                else:
                    new_conf = entry.confidence
                updated = entry.with_updated_confidence(new_conf, new_score,
                                                        f"contradiction noted (score: {old_score:.2f} -> {new_score:.2f})")
                self._entries[key] = updated
                return updated
        return None

    def remove(self, entry_id: str) -> bool:
        """Remove a knowledge entry."""
        if not entry_id:
            return False
        # Find entry by ID
        key_to_remove = None
        for key, entry in self._entries.items():
            if entry.id == entry_id:
                key_to_remove = key
                break
        if key_to_remove is None:
            return False
        entry = self._entries[key_to_remove]
        # Remove from indexes
        for lst in self._by_subject.values():
            if entry.id in lst:
                lst.remove(entry.id)
        for lst in self._by_predicate.values():
            if entry.id in lst:
                lst.remove(entry.id)
        for lst in self._by_object.values():
            if entry.id in lst:
                lst.remove(entry.id)
        del self._entries[key_to_remove]
        return True

    def clear(self):
        """Clear all knowledge."""
        self._entries.clear()
        self._by_subject.clear()
        self._by_predicate.clear()
        self._by_object.clear()
        self._initialized = False

    def count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return statistics about the knowledge base."""
        by_confidence = {}
        by_type = {}
        for entry in self._entries.values():
            conf = entry.confidence.value
            by_confidence[conf] = by_confidence.get(conf, 0) + 1
            typ = entry.knowledge_type.value
            by_type[typ] = by_type.get(typ, 0) + 1
        return {
            "total_entries": len(self._entries),
            "by_confidence": by_confidence,
            "by_type": by_type,
            "disputed_count": sum(1 for e in self._entries.values() if e.contradiction_count > 0),
        }

    def _make_key(self, subject: str, predicate: str, object_: str) -> str:
        """Create a unique key for a knowledge entry."""
        return f"{subject.lower()}||{predicate.lower()}||{object_.lower()}"

    def _index(self, entry: KnowledgeEntry):
        """Index an entry for efficient retrieval."""
        subj = entry.subject.lower()
        pred = entry.predicate.lower()
        obj = entry.object.lower()

        self._by_subject.setdefault(subj, []).append(entry.id)
        self._by_predicate.setdefault(pred, []).append(entry.id)
        self._by_object.setdefault(obj, []).append(entry.id)

    def _merge(self, existing: KnowledgeEntry, new: KnowledgeEntry) -> KnowledgeEntry:
        """Merge new evidence into existing knowledge."""
        # Update evidence counts
        existing.evidence_count += new.evidence_count
        existing.contradiction_count += new.contradiction_count

        # Add sources
        for source in new.sources:
            # Avoid duplicate sources
            if not any(s.name == source.name and s.timestamp == source.timestamp 
                       for s in existing.sources):
                existing.sources.append(source)

        # Update confidence based on combined evidence
        total_evidence = existing.evidence_count
        total_contradictions = existing.contradiction_count

        if total_evidence == 0 and total_contradictions == 0:
            new_score = 0.5
        elif total_contradictions > total_evidence:
            new_score = max(0.0, 0.5 - 0.1 * (total_contradictions - total_evidence))
        else:
            new_score = min(1.0, 0.5 + 0.05 * (total_evidence - total_contradictions))

        if new_score >= 0.8:
            new_conf = KnowledgeConfidence.KNOWN
        elif new_score >= 0.6:
            new_conf = KnowledgeConfidence.LIKELY
        elif new_score >= 0.4:
            new_conf = KnowledgeConfidence.POSSIBLE
        elif new_score > 0.0:
            new_conf = KnowledgeConfidence.UNKNOWN
        else:
            new_conf = KnowledgeConfidence.FALSE

        existing.last_updated = time.time()
        existing.confidence = new_conf
        existing.confidence_score = new_score

        return existing

    def initialize_defaults(self):
        """
        Initialize the knowledge base with basic, well-established knowledge.
        
        This is TEMPORARY scaffolding. In the full architecture, this knowledge
        should come from training data, not be hard-coded. However, having some
        baseline knowledge is useful for the system to function initially.
        
        These entries have LOW confidence and are marked as needing verification.
        """
        if self._initialized:
            return

        # Basic physical facts (marked as needing evidence verification)
        basic_entries = [
            # Entity properties
            ("bird", "has_property", "has_wings"),
            ("bird", "has_property", "has_feathers"),
            ("bird", "reproduction", "lays_eggs"),
            ("cat", "has_property", "has_fur"),
            ("cat", "limbs", "4_legs"),
            ("dog", "has_property", "has_fur"),
            ("dog", "limbs", "4_legs"),
            ("fish", "habitat", "water"),
            ("fish", "respiration", "gills"),
            ("human", "limbs", "2_arms_and_2_legs"),
            ("human", "capability", "can_speak_human_language"),
            ("building", "is_a", "structure"),
            ("building", "habitat", "land"),
            ("tree", "is_a", "plant"),
            ("tree", "growth", "grows_from_seed"),
            
            # Physical properties
            ("water", "state_at_room_temperature", "liquid"),
            ("water", "freezes_at", "0_celsius"),
            ("water", "boils_at", "100_celsius"),
            ("earth", "shape", "oblate_sphere"),
            ("sun", "is_a", "star"),
            ("moon", "orbits", "earth"),
            ("earth", "orbits", "sun"),
            
            # Animal capabilities
            ("cat", "can", "climb"),
            ("cat", "can", "purr"),
            ("cat", "cannot", "fly"),
            ("dog", "can", "bark"),
            ("dog", "can", "run"),
            ("dog", "cannot", "fly"),
            ("bird", "can", "fly"),
            ("bird", "cannot", "swim_underwater"),
            ("fish", "can", "swim"),
            ("fish", "cannot", "walk_on_land"),
            ("human", "can", "speak"),
            ("human", "can", "walk"),
            ("human", "cannot", "fly_unassisted"),
            
            # Category memberships
            ("bird", "is_a", "animal"),
            ("cat", "is_a", "animal"),
            ("dog", "is_a", "animal"),
            ("fish", "is_a", "animal"),
            ("human", "is_a", "animal"),
            ("tree", "is_a", "plant"),
            ("water", "is_a", "substance"),
            ("earth", "is_a", "planet"),
            ("sun", "is_a", "star"),
            ("moon", "is_a", "satellite"),
            
            # Basic causality
            ("fire", "produces", "heat"),
            ("fire", "produces", "light"),
            ("fire", "requires", "fuel"),
            ("ice", "is", "frozen_water"),
            ("steam", "is", "gaseous_water"),
        ]

        for subject, predicate, obj in basic_entries:
            entry = KnowledgeEntry(
                subject=subject,
                predicate=predicate,
                object=obj,
                knowledge_type=self._classify_type(predicate),
                confidence=KnowledgeConfidence.POSSIBLE,  # Start with low confidence
                confidence_score=0.4,  # Low initial confidence — needs evidence
                sources=[KnowledgeSource(
                    name="baseline_initialization",
                    source_type="initialization",
                    reliability=0.3,  # Low reliability for initial defaults
                    metadata={"note": "Baseline knowledge needs evidence verification"}
                )],
                evidence_count=0,
                metadata={"status": "needs_verification"},
            )
            self.add(entry)

        self._initialized = True

    def _classify_type(self, predicate: str) -> KnowledgeType:
        """Classify a predicate into a knowledge type."""
        predicate_lower = predicate.lower()
        if "has_property" in predicate_lower or "has_" in predicate_lower:
            return KnowledgeType.ENTITY_PROPERTY
        if "can" in predicate_lower or "cannot" in predicate_lower:
            return KnowledgeType.ENTITY_PROPERTY
        if "is_a" in predicate_lower:
            return KnowledgeType.CATEGORICAL
        if "orbits" in predicate_lower or "part_of" in predicate_lower:
            return KnowledgeType.RELATION
        if "at" in predicate_lower or "temperature" in predicate_lower:
            return KnowledgeType.QUANTITATIVE
        return KnowledgeType.FACT


# Global knowledge base instance
_default_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base."""
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
        _default_kb.initialize_defaults()
    return _default_kb


def reset_knowledge_base():
    """Reset the global knowledge base (for testing)."""
    global _default_kb
    if _default_kb is not None:
        _default_kb.clear()
        _default_kb = None


__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeType",
    "KnowledgeConfidence",
    "KnowledgeSource",
    "get_knowledge_base",
    "reset_knowledge_base",
]
