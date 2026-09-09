"""
Memory Hierarchy — Phase 6.

Separates memory into distinct types:
- EPISODIC: Past experiences, events, episodes (what happened)
- SEMANTIC: Generalized knowledge, concepts (what we know)
- PROCEDURAL: How to do things (skills, procedures)
- SKILL: Reusable successful workflows
- FAILURE: Known failure patterns

With:
- Memory consolidation (episodes → semantic knowledge)
- Selective persistence (importance, relevance, novelty, reliability, recurrence)
- Memory provenance
- Expiration/downgrading for stale information
- Integration with working memory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..representation import Representation, RepresentationQuality
from ..knowledge import KnowledgeEntry, KnowledgeBase


# ════════════════════════════════════════════════════════════════════
# MEMORY TYPES
# ════════════════════════════════════════════════════════════════════

class MemoryType(str, Enum):
    """Types of long-term memory."""
    EPISODIC = "episodic"           # Personal experiences, events
    SEMANTIC = "semantic"           # Generalized knowledge, facts
    PROCEDURAL = "procedural"       # How to do things
    SKILL = "skill"                 # Reusable successful workflows
    FAILURE = "failure"             # Known failure patterns
    SOURCE = "source"               # Source reliability/history


class MemoryImportance(str, Enum):
    """Importance levels for memory persistence."""
    CRITICAL = "critical"           # Must persist
    HIGH = "high"                   # Should persist
    MEDIUM = "medium"               # May persist
    LOW = "low"                     # Candidate for expiration
    TRIVIAL = "trivial"             # Likely to expire


class MemoryStatus(str, Enum):
    """Status of a memory entry."""
    ACTIVE = "active"               # Currently relevant
    DORMANT = "dormant"             # Not currently relevant but retained
    STALE = "stale"                 # May be out of date
    ARCHIVED = "archived"           # Kept for history
    EXPIRED = "expired"             # Past expiration


# ════════════════════════════════════════════════════════════════════
# EPISODIC MEMORY
# ════════════════════════════════════════════════════════════════════

@dataclass
class EpisodicMemoryEntry:
    """
    A single episodic memory — a record of a past experience.
    
    Episodic memories are:
    - Time-stamped (when it happened)
    - Context-rich (what was the situation)
    - Experience-based (what was observed/done)
    - Potentially consolidable into semantic memory
    """
    
    id: str = field(default_factory=lambda: f"epi_{int(time.time() * 1000)}")
    episode_type: str = "general"   # task, investigation, error, success, interaction
    timestamp: float = field(default_factory=time.time)
    description: str = ""            # What happened
    task_id: str = ""                # What task this was part of
    goal: str = ""                   # What was being attempted
    actions_taken: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    outcome: str = ""                # What resulted
    success: bool = False           # Did it work?
    confidence: float = 0.5
    importance: MemoryImportance = MemoryImportance.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    source_provenance: str = ""      # Where this came from
    consolidation_candidates: list[str] = field(default_factory=list)  # Semantic entries to create
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "episode_type": self.episode_type,
            "timestamp": self.timestamp,
            "description_preview": self.description[:100] if self.description else "",
            "task_id": self.task_id,
            "goal_preview": self.goal[:50] if self.goal else "",
            "actions_count": len(self.actions_taken),
            "observations_count": len(self.observations),
            "outcome_preview": self.outcome[:50] if self.outcome else "",
            "success": self.success,
            "importance": self.importance.value,
            "status": self.status.value,
            "has_consolidation_candidates": len(self.consolidation_candidates) > 0,
        }


class EpisodicMemory:
    """
    Episodic memory — records of past experiences.
    
    Stores:
    - Task episodes (what we did, what happened)
    - Investigation episodes (what we found)
    - Error episodes (what went wrong)
    - Success episodes (what worked)
    
    Supports:
    - Time-based retrieval
    - Type-based retrieval
    - Task-based retrieval
    - Outcome-based filtering
    - Consolidation candidates
    """
    
    def __init__(self, max_entries: int = 10000):
        self._entries: dict[str, EpisodicMemoryEntry] = {}
        self._by_type: dict[str, list[str]] = {}
        self._by_task: dict[str, list[str]] = {}
        self._by_outcome: dict[str, list[str]] = {}
        self._max_entries = max_entries
    
    def add(self, entry: EpisodicMemoryEntry) -> EpisodicMemoryEntry:
        """Add an episodic memory entry."""
        if entry.id in self._entries:
            # Update existing
            old = self._entries[entry.id]
            old.description = entry.description or old.description
            old.outcome = entry.outcome or old.outcome
            old.success = entry.success if entry.success else old.success
            old.importance = entry.importance
            old.status = entry.status
            return old
        
        # Check capacity
        if len(self._entries) >= self._max_entries:
            self._evict_oldest()
        
        self._entries[entry.id] = entry
        
        # Index by type
        self._by_type.setdefault(entry.episode_type, []).append(entry.id)
        
        # Index by task
        if entry.task_id:
            self._by_task.setdefault(entry.task_id, []).append(entry.id)
        
        # Index by outcome
        outcome_key = "success" if entry.success else "failure"
        self._by_outcome.setdefault(outcome_key, []).append(entry.id)
        
        return entry
    
    def get(self, entry_id: str) -> EpisodicMemoryEntry | None:
        """Get an entry by ID."""
        return self._entries.get(entry_id)
    
    def get_by_type(self, episode_type: str) -> list[EpisodicMemoryEntry]:
        """Get entries by type."""
        ids = self._by_type.get(episode_type, [])
        return [self._entries[i] for i in ids if i in self._entries]
    
    def get_by_task(self, task_id: str) -> list[EpisodicMemoryEntry]:
        """Get entries for a task."""
        ids = self._by_task.get(task_id, [])
        return [self._entries[i] for i in ids if i in self._entries]
    
    def get_successes(self) -> list[EpisodicMemoryEntry]:
        """Get successful episodes."""
        return self._get_by_outcome("success")
    
    def get_failures(self) -> list[EpisodicMemoryEntry]:
        """Get failure episodes."""
        return self._get_by_outcome("failure")
    
    def _get_by_outcome(self, outcome_key: str) -> list[EpisodicMemoryEntry]:
        """Get entries by outcome."""
        ids = self._by_outcome.get(outcome_key, [])
        return [self._entries[i] for i in ids if i in self._entries]
    
    def get_recent(self, n: int = 10, 
                   episode_type: str | None = None,
                   min_importance: MemoryImportance = MemoryImportance.LOW) -> list[EpisodicMemoryEntry]:
        """Get recent episodes, optionally filtered."""
        entries = list(self._entries.values())
        
        # Filter by type if specified
        if episode_type:
            entries = [e for e in entries if e.episode_type == episode_type]
        
        # Filter by importance
        importance_order = {
            MemoryImportance.TRIVIAL: 0,
            MemoryImportance.LOW: 1,
            MemoryImportance.MEDIUM: 2,
            MemoryImportance.HIGH: 3,
            MemoryImportance.CRITICAL: 4,
        }
        min_order = importance_order.get(min_importance, 0)
        
        entries = [
            e for e in entries
            if importance_order.get(e.importance, 0) >= min_order
        ]
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        return entries[:n]
    
    def get_for_consolidation(self, max_age_days: float = 7) -> list[EpisodicMemoryEntry]:
        """
        Get episodes that are candidates for consolidation into semantic memory.
        
        Criteria:
        - Older than max_age_days
        - High importance
        - Successful outcome
        - Has consolidation candidates
        """
        cutoff = time.time() - (max_age_days * 24 * 3600)
        
        candidates = []
        for entry in self._entries.values():
            if entry.timestamp < cutoff:
                if entry.importance in (MemoryImportance.HIGH, MemoryImportance.CRITICAL):
                    if entry.success:
                        candidates.append(entry)
        
        return candidates
    
    def _evict_oldest(self):
        """Evict oldest low-importance entries."""
        # Find oldest entries with low importance
        eviction_candidates = [
            (eid, entry) for eid, entry in self._entries.items()
            if entry.importance in (MemoryImportance.TRIVIAL, MemoryImportance.LOW)
        ]
        
        if not eviction_candidates:
            # If no low-importance entries, evict oldest overall
            eviction_candidates = [
                (eid, entry) for eid, entry in self._entries.items()
            ]
        
        eviction_candidates.sort(key=lambda x: x[1].timestamp)
        
        # Evict one
        if eviction_candidates:
            eid, _ = eviction_candidates[0]
            del self._entries[eid]
            
            # Clean up indexes
            for lst in self._by_type.values():
                if eid in lst:
                    lst.remove(eid)
            for lst in self._by_task.values():
                if eid in lst:
                    lst.remove(eid)
            for lst in self._by_outcome.values():
                if eid in lst:
                    lst.remove(eid)
    
    def update_status(self, entry_id: str, status: MemoryStatus):
        """Update entry status."""
        if entry_id in self._entries:
            self._entries[entry_id].status = status
    
    def count(self) -> int:
        return len(self._entries)
    
    def stats(self) -> dict[str, Any]:
        """Get episodic memory statistics."""
        by_type = {}
        by_outcome = {"success": 0, "failure": 0}
        
        for entry in self._entries.values():
            by_type[entry.episode_type] = by_type.get(entry.episode_type, 0) + 1
            if entry.success:
                by_outcome["success"] += 1
            else:
                by_outcome["failure"] += 1
        
        return {
            "total_entries": len(self._entries),
            "by_type": by_type,
            "by_outcome": by_outcome,
            "max_entries": self._max_entries,
        }


# ════════════════════════════════════════════════════════════════════
# SEMANTIC MEMORY
# ════════════════════════════════════════════════════════════════════

class SemanticMemory:
    """
    Semantic memory — generalized knowledge.
    
    This is separate from the KnowledgeBase (which is about facts/entities).
    Semantic memory is about:
    - Concepts and their relationships
    - Generalized patterns
    - Categories and taxonomies
    - Inherited knowledge from consolidated episodes
    
    Semantic memory entries are derived from episodic consolidation
    and represent what SWEEP has "learned" about the world.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self._knowledge_base = knowledge_base or KnowledgeBase()
        self._concepts: dict[str, dict[str, Any]] = {}
        self._consolidation_sources: dict[str, list[str]] = {}  # concept -> episode IDs
    
    def add_concept(self, name: str, concept_type: str, properties: dict[str, Any],
                    source_episode_id: str | None = None) -> dict[str, Any]:
        """Add a concept to semantic memory."""
        concept = {
            "name": name,
            "type": concept_type,
            "properties": properties,
            "created_from_episode": source_episode_id,
            "created_at": time.time(),
            "access_count": 0,
            "last_accessed": time.time(),
        }
        
        self._concepts[name.lower()] = concept
        
        if source_episode_id:
            self._consolidation_sources.setdefault(name.lower(), []).append(source_episode_id)
        
        return concept
    
    def get_concept(self, name: str) -> dict[str, Any] | None:
        """Get a concept by name."""
        concept = self._concepts.get(name.lower())
        if concept:
            concept["access_count"] += 1
            concept["last_accessed"] = time.time()
        return concept
    
    def get_related_concepts(self, concept_name: str, n: int = 5) -> list[dict[str, Any]]:
        """Get concepts related to a given concept."""
        # Simple similarity based on type matching and property overlap
        target = self._concepts.get(concept_name.lower())
        if not target:
            return []
        
        related = []
        for name, concept in self._concepts.items():
            if name == concept_name.lower():
                continue
            
            # Same type = related
            if concept["type"] == target["type"]:
                score = 0.3
            else:
                score = 0.0
            
            # Property overlap
            target_props = set(target["properties"].keys())
            concept_props = set(concept["properties"].keys())
            if target_props and concept_props:
                overlap = len(target_props & concept_props)
                union = len(target_props | concept_props)
                if union > 0:
                    score += 0.7 * (overlap / union)
            
            if score > 0:
                related.append({
                    "name": concept["name"],
                    "type": concept["type"],
                    "relevance_score": score,
                    "properties": concept["properties"],
                })
        
        related.sort(key=lambda r: r["relevance_score"], reverse=True)
        return related[:n]
    
    def consolidate_from_episode(self, episode: EpisodicMemoryEntry) -> list[str]:
        """
        Consolidate an episodic memory into semantic knowledge.
        
        Extracts generalizable concepts from the episode.
        Called during memory consolidation.
        """
        created_concepts = []
        
        if episode.description:
            # Extract potential concept from description
            description_words = episode.description.lower().split()
            if len(description_words) > 3:
                # Simple concept extraction (scaffolding)
                # In real implementation, would use NLP
                concept_name = description_words[0]
                if len(concept_name) > 2:
                    concept = self.add_concept(
                        concept_name,
                        "pattern",
                        {
                            "episode_description": episode.description[:200],
                            "outcome": episode.outcome,
                            "success_pattern": episode.success,
                        },
                        source_episode_id=episode.id,
                    )
                    created_concepts.append(concept_name)
        
        # Update episode with consolidation candidates
        if created_concepts:
            episode.consolidation_candidates = created_concepts
        
        return created_concepts
    
    def count(self) -> int:
        return len(self._concepts)
    
    def stats(self) -> dict[str, Any]:
        """Get semantic memory statistics."""
        by_type = {}
        for concept in self._concepts.values():
            by_type[concept["type"]] = by_type.get(concept["type"], 0) + 1
        
        return {
            "total_concepts": len(self._concepts),
            "by_type": by_type,
            "knowledge_base_entries": self._knowledge_base.count(),
        }


# ════════════════════════════════════════════════════════════════════
# PROCEDURAL / SKILL MEMORY
# ════════════════════════════════════════════════════════════════════

@dataclass
class Skill:
    """
    A reusable skill — a learned procedure for accomplishing tasks.
    
    Skills are extracted from successful episodes.
    """
    
    id: str = field(default_factory=lambda: f"skill_{int(time.time() * 1000)}")
    name: str = ""
    description: str = ""
    purpose: str = ""               # What this skill accomplishes
    preconditions: list[str] = field(default_factory=list)  # When it can be used
    steps: list[str] = field(default_factory=list)          # The procedure
    tool_requirements: list[str] = field(default_factory=list)
    success_rate: float = 0.0       # How often it works
    usage_count: int = 0
    last_used: float = 0.0
    source_episode_id: str = ""     # Where it came from
    version: int = 1
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description_preview": self.description[:100] if self.description else "",
            "purpose_preview": self.purpose[:50] if self.purpose else "",
            "steps_count": len(self.steps),
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "version": self.version,
            "confidence": self.confidence,
        }


class SkillRegistry:
    """
    Registry of learned skills.
    
    Skills are:
    - Extracted from successful episodes
    - Tested before being added
    - Versioned
    - Tracked for success rate and usage
    """
    
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._by_name: dict[str, str] = {}  # name -> id
    
    def register(self, skill: Skill) -> Skill:
        """Register a new skill."""
        if skill.id in self._skills:
            # Update existing
            existing = self._skills[skill.id]
            existing.steps = skill.steps
            existing.success_rate = (
                (existing.success_rate * existing.usage_count + skill.success_rate)
                / (existing.usage_count + 1)
            )
            existing.usage_count += 1
            existing.last_used = time.time()
            existing.version += 1
            return existing
        
        self._skills[skill.id] = skill
        
        if skill.name:
            # Handle name conflicts by appending version
            base_name = skill.name
            if base_name in self._by_name:
                skill.name = f"{base_name}_v{skill.version}"
            self._by_name[skill.name] = skill.id
        
        return skill
    
    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        skill = self._skills.get(skill_id)
        if skill:
            skill.last_used = time.time()
            skill.usage_count += 1
        return skill
    
    def get_by_name(self, name: str) -> Skill | None:
        """Get a skill by name."""
        skill_id = self._by_name.get(name)
        if skill_id:
            return self.get(skill_id)
        return None
    
    def list_skills(self) -> list[Skill]:
        """List all skills."""
        return list(self._skills.values())
    
    def get_reliable_skills(self, min_success_rate: float = 0.7) -> list[Skill]:
        """Get skills with high success rates."""
        return [
            s for s in self._skills.values()
            if s.success_rate >= min_success_rate
        ]
    
    def update_success_rate(self, skill_id: str, success: bool):
        """Update a skill's success rate after use."""
        if skill_id in self._skills:
            skill = self._skills[skill_id]
            n = skill.usage_count
            if success:
                skill.success_rate = (skill.success_rate * n + 1) / (n + 1)
            else:
                skill.success_rate = (skill.success_rate * n) / (n + 1)
    
    def count(self) -> int:
        return len(self._skills)
    
    def stats(self) -> dict[str, Any]:
        """Get skill registry statistics."""
        by_purpose = {}
        for skill in self._skills.values():
            purpose = skill.purpose or "general"
            by_purpose[purpose] = by_purpose.get(purpose, 0) + 1
        
        return {
            "total_skills": len(self._skills),
            "by_purpose": by_purpose,
            "avg_success_rate": (
                sum(s.success_rate for s in self._skills.values()) / len(self._skills)
                if self._skills else 0.0
            ),
        }


# ════════════════════════════════════════════════════════════════════
# FAILURE MEMORY
# ════════════════════════════════════════════════════════════════════

@dataclass
class FailurePattern:
    """
    A known failure pattern — what goes wrong and how to avoid it.
    
    Extracted from failure episodes.
    """
    
    id: str = field(default_factory=lambda: f"fail_{int(time.time() * 1000)}")
    pattern_name: str = ""
    description: str = ""           # What the failure looks like
    failure_type: str = "general"  # perception, understanding, memory, reasoning, planning, tool, environment
    symptoms: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    avoidance_strategies: list[str] = field(default_factory=list)
    detection_cues: list[str] = field(default_factory=list)
    occurrence_count: int = 0
    last_occurred: float = 0.0
    severity: float = 0.5          # How bad this failure is
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pattern_name": self.pattern_name,
            "description_preview": self.description[:100] if self.description else "",
            "failure_type": self.failure_type,
            "symptoms_count": len(self.symptoms),
            "causes_count": len(self.causes),
            "avoidance_count": len(self.avoidance_strategies),
            "occurrence_count": self.occurrence_count,
            "severity": self.severity,
        }


class FailureMemory:
    """
    Memory of known failure patterns.
    
    Stores:
    - What failures look like (symptoms)
    - What causes them
    - How to avoid them
    
    Used for:
    - Failure detection during tasks
    - Prevention strategies
    - Learning from mistakes
    """
    
    def __init__(self):
        self._patterns: dict[str, FailurePattern] = {}
    
    def record(self, pattern: FailurePattern) -> FailurePattern:
        """Record a failure pattern."""
        if pattern.id in self._patterns:
            existing = self._patterns[pattern.id]
            existing.occurrence_count += 1
            existing.last_occurred = time.time()
            return existing
        
        self._patterns[pattern.id] = pattern
        return pattern
    
    def get(self, pattern_id: str) -> FailurePattern | None:
        """Get a failure pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def detect(self, symptoms: list[str]) -> list[FailurePattern]:
        """
        Detect potential failure patterns based on symptoms.
        
        Args:
            symptoms: Observed symptoms
            
        Returns:
            List of matching failure patterns with match scores.
        """
        matches = []
        for pattern in self._patterns.values():
            score = 0.0
            for symptom in symptoms:
                if symptom.lower() in " ".join(pattern.symptoms).lower():
                    score += 0.25
            
            if score > 0:
                matches.append({
                    "pattern": pattern,
                    "match_score": score,
                })
        
        matches.sort(key=lambda m: m["match_score"], reverse=True)
        return [m["pattern"] for m in matches]
    
    def count(self) -> int:
        return len(self._patterns)
    
    def stats(self) -> dict[str, Any]:
        """Get failure memory statistics."""
        by_type = {}
        for pattern in self._patterns.values():
            by_type[pattern.failure_type] = by_type.get(pattern.failure_type, 0) + 1
        
        return {
            "total_patterns": len(self._patterns),
            "by_type": by_type,
        }


# ════════════════════════════════════════════════════════════════════
# MEMORY HIERARCHY (orchestrator)
# ════════════════════════════════════════════════════════════════════

class MemoryHierarchy:
    """
    Orchestrates the memory hierarchy.
    
    Manages:
    - Episodic memory (experiences)
    - Semantic memory (concepts/knowledge)
    - Procedural/skill memory (how to do things)
    - Failure memory (known failure patterns)
    
    With:
    - Consolidation (episodes → semantic)
    - Skill extraction (successes → skills)
    - Failure learning (failures → patterns)
    - Selective persistence
    - Memory provenance
    """
    
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory(knowledge_base)
        self.skills = SkillRegistry()
        self.failures = FailureMemory()
        self._knowledge_base = knowledge_base
    
    def record_episode(self, episode: EpisodicMemoryEntry) -> EpisodicMemoryEntry:
        """Record an episodic memory."""
        return self.episodic.add(episode)
    
    def consolidate(self, max_age_days: float = 7) -> list[str]:
        """
        Run memory consolidation.
        
        Converts important old episodes into semantic knowledge
        and extracts skills from successes.
        
        Returns:
            List of created concept names.
        """
        created_concepts = []
        
        # Get consolidation candidates
        candidates = self.episodic.get_for_consolidation(max_age_days)
        
        for episode in candidates:
            # Consolidate into semantic memory
            concepts = self.semantic.consolidate_from_episode(episode)
            created_concepts.extend(concepts)
            
            # Extract skill from success
            if episode.success and episode.actions_taken:
                skill = Skill(
                    name=f"skill_from_{episode.episode_type}",
                    description=episode.description[:200],
                    purpose=f"Accomplish {episode.goal}",
                    steps=episode.actions_taken,
                    source_episode_id=episode.id,
                    success_rate=0.8,  # Initial estimate
                )
                self.skills.register(skill)
            
            # Update episode status to archived
            self.episodic.update_status(episode.id, MemoryStatus.ARCHIVED)
        
        return created_concepts
    
    def learn_from_failure(self, episode: EpisodicMemoryEntry) -> FailurePattern | None:
        """
        Learn a failure pattern from a failure episode.
        
        Args:
            episode: A failure episode
            
        Returns:
            Created failure pattern, or None if not extractable.
        """
        if not episode.success and episode.observations:
            pattern = FailurePattern(
                pattern_name=f"failure_{episode.episode_type}",
                description=episode.description[:200],
                failure_type=episode.episode_type,
                symptoms=episode.observations[:5],
                causes=["Unknown — needs investigation"],
                avoidance_strategies=["Investigate root cause"],
                occurrence_count=1,
                severity=0.5,
            )
            return self.failures.record(pattern)
        
        return None
    
    def retrieve_relevant_memory(self, query: str, 
                                  memory_types: list[MemoryType] | None = None,
                                  top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories across the hierarchy.
        
        Args:
            query: Search query
            memory_types: Which memory types to search (default: all)
            top_k: Maximum results
            
        Returns:
            List of relevant memories with source type.
        """
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, 
                           MemoryType.SKILL, MemoryType.FAILURE]
        
        results = []
        query_lower = query.lower()
        
        for mem_type in memory_types:
            if MemoryType.EPISODIC in mem_type:
                # Search episodic memory
                for episode in self.episodic.get_recent(20):
                    if query_lower in episode.description.lower():
                        results.append({
                            "memory_type": "episodic",
                            "id": episode.id,
                            "content_preview": episode.description[:100],
                            "relevance": 1.0,
                            "timestamp": episode.timestamp,
                        })
            
            elif MemoryType.SEMANTIC in mem_type:
                # Search semantic memory
                for concept in self.semantic._concepts.values():
                    if query_lower in concept["name"].lower() or \
                       query_lower in str(concept["properties"]).lower():
                        results.append({
                            "memory_type": "semantic",
                            "id": concept["name"],
                            "content_preview": concept["name"],
                            "relevance": 0.8,
                            "timestamp": concept["created_at"],
                        })
            
            elif MemoryType.SKILL in mem_type:
                # Search skills
                for skill in self.skills.list_skills():
                    if query_lower in skill.description.lower() or \
                       query_lower in skill.purpose.lower():
                        results.append({
                            "memory_type": "skill",
                            "id": skill.id,
                            "content_preview": skill.description[:100],
                            "relevance": 0.9,
                            "success_rate": skill.success_rate,
                            "timestamp": skill.last_used,
                        })
            
            elif MemoryType.FAILURE in mem_type:
                # Search failure patterns
                for pattern in self.failures._patterns.values():
                    if query_lower in pattern.description.lower():
                        results.append({
                            "memory_type": "failure",
                            "id": pattern.id,
                            "content_preview": pattern.description[:100],
                            "relevance": 0.7,
                            "severity": pattern.severity,
                            "timestamp": pattern.last_occurred,
                        })
        
        # Sort by relevance
        results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
        return results[:top_k]
    
    def stats(self) -> dict[str, Any]:
        """Get complete memory hierarchy statistics."""
        return {
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "skills": self.skills.stats(),
            "failures": self.failures.stats(),
        }


# ════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════

__all__ = [
    "MemoryType",
    "MemoryImportance",
    "MemoryStatus",
    "EpisodicMemoryEntry",
    "EpisodicMemory",
    "SemanticMemory",
    "Skill",
    "SkillRegistry",
    "FailurePattern",
    "FailureMemory",
    "MemoryHierarchy",
]
