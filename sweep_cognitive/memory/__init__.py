"""
Working Memory — Phase 5: Harmonized Working Memory.

This module harmonizes the three existing working memory implementations:
1. sweep_cognitive/memory — basic key-value working memory
2. sweep_neural_mesh/neurons/working_memory — Baddeley-style with decay
3. sweep_neural_mesh/engine/memory — layered memory (working/evidence/semantic/user)

into a unified WorkingMemory system with:

- Bounded capacity (Miller's Law: 4-7 items)
- Temporal decay (items fade unless rehearsed)
- Rehearsal mechanism (refresh items)
- Priority management (important items kept longer)
- Slot types (query, goal, finding, hypothesis, action, context)
- Interference management
- Integration with associative memory
- Active task state tracking

Key principle: Working memory is the active context for the current task.
It holds what's relevant NOW, not everything permanently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..representation import Representation, RepresentationQuality
from ..knowledge import KnowledgeBase, KnowledgeEntry, KnowledgeSource


# ════════════════════════════════════════════════════════════════════
# RETRIEVAL RESULT (from Phase 4 associative memory)
# ════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    Result of an associative retrieval.
    
    Contains:
    - The retrieved representation/entry
    - Retrieval score (combined relevance score)
    - Individual scoring components
    - Retrieval metadata
    """
    item: Any                    # The retrieved item (Representation, KnowledgeEntry, etc.)
    item_type: str = ""          # Type of item: "representation", "knowledge", etc.
    retrieval_id: str = field(default_factory=lambda: f"retrieval_{int(time.time() * 1000)}")
    score: float = 0.0           # Combined relevance score (0-1)
    semantic_score: float = 0.0  # Semantic similarity component
    contextual_score: float = 0.0  # Contextual relevance component
    recency_score: float = 0.0   # Recency component
    source_reliability: float = 0.0  # Source reliability component
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_id": self.retrieval_id,
            "item_type": self.item_type,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "contextual_score": self.contextual_score,
            "recency_score": self.recency_score,
            "source_reliability": self.source_reliability,
            "retrieved_at": self.retrieved_at,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalQuery:
    """
    A query for associative retrieval.
    
    Can be:
    - A text query (for semantic search)
    - A representation to find similar items
    - A knowledge entry to find related knowledge
    - Combined with contextual filters
    """
    
    def __init__(self, query: str | Representation | KnowledgeEntry | None = None,
                 context: dict[str, Any] | None = None,
                 filters: dict[str, Any] | None = None,
                 top_k: int = 10,
                 scoring_weights: dict[str, float] | None = None):
        self.query = query
        self.context = context or {}
        self.filters = filters or {}
        self.top_k = top_k
        self.scoring_weights = scoring_weights or {
            "semantic": 0.4,
            "contextual": 0.3,
            "recency": 0.15,
            "source_reliability": 0.15,
        }
        
        # Extract query text and embedding
        self.query_text = ""
        self.query_embedding = None
        self.query_type = "text"
        
        if isinstance(query, str):
            self.query_text = query
            self.query_type = "text"
        elif isinstance(query, Representation):
            self.query_text = query.content
            self.query_embedding = query.embedding
            self.query_type = "representation"
        elif isinstance(query, KnowledgeEntry):
            self.query_text = f"{query.subject} {query.predicate} {query.object}"
            self.query_type = "knowledge"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "query_text_preview": self.query_text[:100] if self.query_text else "",
            "has_embedding": self.query_embedding is not None,
            "context_keys": list(self.context.keys()),
            "filter_keys": list(self.filters.keys()),
            "top_k": self.top_k,
        }


class AssociativeMemory:
    """
    Associative memory with multiple retrieval mechanisms.
    
    This is the scaffolding implementation. It supports:
    - Semantic similarity retrieval (via word overlap)
    - Contextual retrieval (via context matching)
    - Recency-weighted retrieval
    - Source reliability weighting
    """
    
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self._knowledge_base = knowledge_base or KnowledgeBase()
        self._retrieved_items: list[dict[str, Any]] = []
        self._access_counts: dict[str, int] = {}
    
    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Perform associative retrieval based on the query."""
        results = []
        
        if query.query_type in ("text", "representation"):
            knowledge_results = self._retrieve_from_knowledge(query)
            results.extend(knowledge_results)
        
        if query.query_type == "knowledge":
            related_results = self._retrieve_related_knowledge(query)
            results.extend(related_results)
        
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:query.top_k]
        
        for result in results:
            self._track_retrieval(result)
        
        return results
    
    def _retrieve_from_knowledge(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Retrieve knowledge entries matching the query."""
        results = []
        search_text = query.query_text.lower()
        
        for entry in self._knowledge_base._entries.values():
            entry_text = f"{entry.subject} {entry.predicate} {entry.object}".lower()
            semantic_match = self._compute_semantic_match(search_text, entry_text)
            
            if semantic_match > 0:
                contextual_match = 0.1  # Simple default
                recency_score = 0.5  # Simple default
                source_rel = 0.3  # Default for entries without sources
                
                weights = query.scoring_weights
                combined_score = (
                    weights.get("semantic", 0.4) * semantic_match +
                    weights.get("contextual", 0.3) * contextual_match +
                    weights.get("recency", 0.15) * recency_score +
                    weights.get("source_reliability", 0.15) * source_rel
                )
                
                result = RetrievalResult(
                    item=entry,
                    item_type="knowledge",
                    score=combined_score,
                    semantic_score=semantic_match,
                    contextual_score=contextual_match,
                    recency_score=recency_score,
                    source_reliability=source_rel,
                    metadata={"entry_id": entry.id, "subject": entry.subject},
                )
                results.append(result)
        
        return results
    
    def _retrieve_related_knowledge(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Retrieve knowledge entries related to a knowledge entry."""
        results = []
        if not isinstance(query.query, KnowledgeEntry):
            return results
        
        query_entry = query.query
        for entry in self._knowledge_base._entries.values():
            if entry.id == query_entry.id:
                continue
            if entry.subject.lower() == query_entry.subject.lower():
                semantic_match = 0.5
                result = RetrievalResult(
                    item=entry,
                    item_type="knowledge",
                    score=0.5,
                    semantic_score=0.5,
                    metadata={"relation": "same_subject"},
                )
                results.append(result)
        return results
    
    def _compute_semantic_match(self, query_text: str, entry_text: str) -> float:
        """Compute semantic match using word overlap."""
        if not query_text or not entry_text:
            return 0.0
        
        query_words = set(query_text.split())
        entry_words = set(entry_text.split())
        
        if not query_words or not entry_words:
            return 0.0
        
        overlap = query_words & entry_words
        union = query_words | entry_words
        
        if not union:
            return 0.0
        
        jaccard = len(overlap) / len(union)
        return jaccard
    
    def _track_retrieval(self, result: RetrievalResult):
        """Track that an item was retrieved."""
        item_id = result.metadata.get("entry_id", "")
        if item_id:
            self._access_counts[item_id] = self._access_counts.get(item_id, 0) + 1
            self._retrieved_items.append({
                "retrieval_id": result.retrieval_id,
                "item_id": item_id,
                "score": result.score,
                "timestamp": result.retrieved_at,
            })
    
    def get_access_count(self, item_id: str) -> int:
        return self._access_counts.get(item_id, 0)
    
    def get_retrieval_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._retrieved_items[-limit:]


# ════════════════════════════════════════════════════════════════════
# WORKING MEMORY SLOTS (harmonized from neurons/working_memory.py)
# ════════════════════════════════════════════════════════════════════

class MemorySlot(Enum):
    """Types of working memory slots."""
    QUERY = "query"                # the current query being processed
    GOAL = "goal"                  # current reasoning goal
    FINDING = "finding"            # a finding from a processing center
    HYPOTHESIS = "hypothesis"      # a working hypothesis
    ACTION = "action"              # a pending action
    CONTEXT = "context"            # contextual information
    EVIDENCE = "evidence"          # evidence being considered
    ENTITY = "entity"              # active entity
    OBSERVATION = "observation"    # recent observation


# ════════════════════════════════════════════════════════════════════
# WORKING MEMORY ITEM (harmonized from neurons/working_memory.py)
# ════════════════════════════════════════════════════════════════════

@dataclass
class WorkingMemoryItem:
    """A single item in working memory with decay and rehearsal."""
    
    item_id: str
    slot_type: MemorySlot
    content: Any
    priority: float                # 0.0-1.0: how important to keep
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    rehearsal_count: int = 0       # how many times it's been refreshed
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def staleness(self) -> float:
        """How stale this item is (0.0 = fresh, 1.0 = very stale)."""
        time_since_access = time.time() - self.last_accessed
        return min(1.0, time_since_access / 300.0)  # decays over 5 minutes
    
    @property
    def retention_score(self) -> float:
        """Combined score for retention decisions."""
        # Higher priority + less staleness = better retention
        return self.priority * (1.0 - self.staleness * 0.5)
    
    def rehearse(self):
        """Rehearse this item to prevent decay."""
        self.last_accessed = time.time()
        self.rehearsal_count += 1
        # Rehearsal slightly boosts priority
        self.priority = min(1.0, self.priority + 0.05)
        self.access_count += 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "slot_type": self.slot_type.value,
            "content_preview": str(self.content)[:100] if self.content else "",
            "priority": self.priority,
            "age_seconds": self.age_seconds,
            "staleness": self.staleness,
            "retention_score": self.retention_score,
            "access_count": self.access_count,
            "rehearsal_count": self.rehearsal_count,
        }


# ════════════════════════════════════════════════════════════════════
# WORKING MEMORY (harmonized)
# ════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    Harmonized working memory combining the best of all implementations.
    
    Features:
    1. Bounded capacity (default 7, Miller's Law)
    2. Temporal decay: items fade unless rehearsed
    3. Priority management: important items kept longer
    4. Slot types: organized by purpose
    5. Rehearsal: refresh items to prevent decay
    6. Interference: similar items compete
    7. Active state: goal, entities, hypotheses, actions
    8. Context summary: introspection
    9. Eviction to episodic memory hint (when implemented)
    
    This replaces the need for separate WorkingMemory implementations.
    """
    
    def __init__(self, capacity: int = 7):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum number of items (default 7, Miller's Law)
        """
        self._items: dict[str, WorkingMemoryItem] = {}
        self._capacity = capacity
        self._total_inserts = 0
        self._total_evictions = 0
        self._total_rehearsals = 0
        
        # Active task state
        self._goal: str = ""
        self._active_entities: list[str] = []
        self._current_hypotheses: list[str] = []
        self._pending_actions: list[str] = []
        self._current_query: str = ""
        self._uncertainty_level: float = 0.0
        self._task_constraints: dict[str, Any] = {}
    
    # ────────────────────────────────────────────────────────────────
    # ITEM MANAGEMENT
    # ────────────────────────────────────────────────────────────────
    
    def insert(
        self,
        slot_type: MemorySlot,
        content: Any,
        priority: float = 0.5,
        item_id: str | None = None,
    ) -> WorkingMemoryItem:
        """
        Insert a new item into working memory.
        
        If at capacity, the lowest-retention item is evicted.
        
        Args:
            slot_type: Type of memory slot
            content: Item content
            priority: Importance (0.0-1.0)
            item_id: Optional explicit ID
            
        Returns:
            The created WorkingMemoryItem
        """
        self._total_inserts += 1
        
        if item_id is None:
            item_id = f"wm_{self._total_inserts}"
        
        item = WorkingMemoryItem(
            item_id=item_id,
            slot_type=slot_type,
            content=content,
            priority=priority,
        )
        
        # If at capacity, evict lowest retention item
        if len(self._items) >= self._capacity:
            evicted = self._evict_lowest_retention()
            if evicted:
                self._on_eviction(evicted)
        
        self._items[item_id] = item
        return item
    
    def retrieve(
        self,
        slot_type: MemorySlot | None = None,
        max_items: int = 10,
    ) -> list[WorkingMemoryItem]:
        """
        Retrieve items from working memory.
        
        Args:
            slot_type: Optional slot type filter
            max_items: Maximum items to return
            
        Returns:
            List of items sorted by retention score
        """
        items = list(self._items.values())
        
        if slot_type:
            items = [i for i in items if i.slot_type == slot_type]
        
        # Sort by retention score (most important first)
        items.sort(key=lambda i: i.retention_score, reverse=True)
        
        # Record access for returned items
        for item in items[:max_items]:
            item.last_accessed = time.time()
            item.access_count += 1
        
        return items[:max_items]
    
    def get(self, item_id: str) -> WorkingMemoryItem | None:
        """Get a specific item by ID."""
        if item_id in self._items:
            item = self._items[item_id]
            item.last_accessed = time.time()
            item.access_count += 1
            return item
        return None
    
    def has(self, item_id: str) -> bool:
        """Check if an item exists."""
        return item_id in self._items
    
    def update_item(
        self,
        item_id: str,
        content: Any | None = None,
        priority: float | None = None,
    ) -> bool:
        """Update an existing item's content or priority."""
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        if content is not None:
            item.content = content
        if priority is not None:
            item.priority = priority
        item.last_accessed = time.time()
        return True
    
    def rehearse_item(self, item_id: str) -> bool:
        """Rehearse an item to prevent decay."""
        if item_id in self._items:
            self._items[item_id].rehearse()
            self._total_rehearsals += 1
            return True
        return False
    
    def rehearse_slot_type(self, slot_type: MemorySlot):
        """Rehearse all items of a slot type."""
        for item in self._items.values():
            if item.slot_type == slot_type:
                item.rehearse()
                self._total_rehearsals += 1
    
    # ────────────────────────────────────────────────────────────────
    # EVICTION
    # ────────────────────────────────────────────────────────────────
    
    def _evict_lowest_retention(self) -> WorkingMemoryItem | None:
        """Evict the item with lowest retention score."""
        if not self._items:
            return None
        
        # Find item with lowest retention score
        evicted_id = min(
            self._items.keys(),
            key=lambda k: self._items[k].retention_score,
        )
        
        evicted = self._items.pop(evicted_id)
        self._total_evictions += 1
        return evicted
    
    def _on_eviction(self, item: WorkingMemoryItem):
        """
        Called when an item is evicted.
        
        High-priority evicted items leave a trace for episodic memory.
        (Implemented in Phase 6 when episodic memory exists.)
        """
        if item.priority > 0.6:
            # Would store to episodic memory here
            pass
    
    def decay_all(self) -> int:
        """
        Apply temporal decay to all items.
        
        Returns:
            Number of items that fell below minimum retention.
        """
        before = len(self._items)
        min_retention = 0.05
        
        to_remove = [
            k for k, item in self._items.items()
            if item.retention_score < min_retention
        ]
        
        for k in to_remove:
            del self._items[k]
        
        return before - len(self._items)
    
    def clear(self):
        """Clear all working memory."""
        self._items.clear()
        self._goal = ""
        self._active_entities.clear()
        self._current_hypotheses.clear()
        self._pending_actions.clear()
        self._current_query = ""
        self._uncertainty_level = 0.0
        self._task_constraints.clear()
    
    # ────────────────────────────────────────────────────────────────
    # ACTIVE TASK STATE
    # ────────────────────────────────────────────────────────────────
    
    def set_goal(self, goal: str):
        """Set the current task goal."""
        self._goal = goal
        self.insert(MemorySlot.GOAL, {"goal": goal, "timestamp": time.time()}, priority=1.0)
    
    def get_goal(self) -> str:
        """Get the current task goal."""
        return self._goal
    
    def set_query(self, query: str):
        """Set the current query."""
        self._current_query = query
        self.insert(MemorySlot.QUERY, {"query": query, "timestamp": time.time()}, priority=1.0)
    
    def get_query(self) -> str:
        """Get the current query."""
        return self._current_query
    
    def add_active_entity(self, entity_name: str, entity_type: str = "unknown"):
        """Add an active entity."""
        if entity_name not in self._active_entities:
            self._active_entities.append(entity_name)
            self.insert(
                MemorySlot.ENTITY,
                {"name": entity_name, "type": entity_type, "timestamp": time.time()},
                priority=0.7,
            )
    
    def get_active_entities(self) -> list[str]:
        """Get active entities."""
        return self._active_entities.copy()
    
    def add_hypothesis(self, hypothesis: str, confidence: float = 0.5):
        """Add a current hypothesis."""
        if hypothesis not in self._current_hypotheses:
            self._current_hypotheses.append(hypothesis)
            self.insert(
                MemorySlot.HYPOTHESIS,
                {"hypothesis": hypothesis, "confidence": confidence, "timestamp": time.time()},
                priority=confidence,
            )
    
    def get_hypotheses(self) -> list[str]:
        """Get current hypotheses."""
        return self._current_hypotheses.copy()
    
    def add_pending_action(self, action: str, priority: float = 0.5):
        """Add a pending action."""
        self._pending_actions.append({"action": action, "priority": priority, "timestamp": time.time()})
        self.insert(
            MemorySlot.ACTION,
            {"action": action, "priority": priority, "timestamp": time.time()},
            priority=priority,
        )
    
    def get_pending_actions(self) -> list[str]:
        """Get pending actions."""
        return [a["action"] for a in self._pending_actions]
    
    def set_uncertainty(self, level: float):
        """Set current uncertainty level (0.0-1.0)."""
        self._uncertainty_level = max(0.0, min(1.0, level))
    
    def get_uncertainty(self) -> float:
        """Get current uncertainty level."""
        return self._uncertainty_level
    
    def set_constraint(self, key: str, value: Any):
        """Set a task constraint."""
        self._task_constraints[key] = value
        self.insert(MemorySlot.CONTEXT, {"constraint": key, "value": value}, priority=0.5)
    
    def get_constraint(self, key: str) -> Any:
        """Get a task constraint."""
        return self._task_constraints.get(key)
    
    # ────────────────────────────────────────────────────────────────
    # CONTEXT & SUMMARY
    # ────────────────────────────────────────────────────────────────
    
    def get_context_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current working memory state.
        
        Used for introspection and providing context to other modules.
        """
        by_type: dict[str, int] = {}
        for item in self._items.values():
            slot_name = item.slot_type.value
            by_type[slot_name] = by_type.get(slot_name, 0) + 1
        
        return {
            "total_items": len(self._items),
            "capacity": self._capacity,
            "goal": self._goal,
            "query": self._current_query,
            "active_entities": self._active_entities,
            "hypotheses": self._current_hypotheses,
            "pending_actions": self.get_pending_actions(),
            "uncertainty": self._uncertainty_level,
            "by_type": by_type,
            "avg_priority": (
                sum(i.priority for i in self._items.values()) / len(self._items)
                if self._items else 0.0
            ),
            "avg_staleness": (
                sum(i.staleness for i in self._items.values()) / len(self._items)
                if self._items else 0.0
            ),
            "stats": {
                "total_inserts": self._total_inserts,
                "total_evictions": self._total_evictions,
                "total_rehearsals": self._total_rehearsals,
            },
        }
    
    def get_recent_findings(self, n: int = 3) -> list[dict[str, Any]]:
        """Get recent findings from working memory."""
        findings = [
            item for item in self._items.values()
            if item.slot_type == MemorySlot.FINDING
        ]
        findings.sort(key=lambda i: i.last_accessed, reverse=True)
        
        return [
            {
                "content": item.content,
                "priority": item.priority,
                "staleness": item.staleness,
                "access_count": item.access_count,
            }
            for item in findings[:n]
        ]
    
    # ────────────────────────────────────────────────────────────────
    # STATS
    # ────────────────────────────────────────────────────────────────
    
    @property
    def size(self) -> int:
        return len(self._items)
    
    @property
    def capacity(self) -> int:
        return self._capacity
    
    def stats(self) -> dict[str, Any]:
        """Get working memory statistics."""
        return {
            "size": len(self._items),
            "capacity": self._capacity,
            "utilization": len(self._items) / self._capacity if self._capacity > 0 else 0,
            "total_inserts": self._total_inserts,
            "total_evictions": self._total_evictions,
            "total_rehearsals": self._total_rehearsals,
            "avg_priority": (
                sum(i.priority for i in self._items.values()) / len(self._items)
                if self._items else 0.0
            ),
            "avg_staleness": (
                sum(i.staleness for i in self._items.values()) / len(self._items)
                if self._items else 0.0
            ),
        }


# ════════════════════════════════════════════════════════════════════
# WORKING MEMORY MANAGER (orchestrates working memory across tasks)
# ════════════════════════════════════════════════════════════════════

class WorkingMemoryManager:
    """
    Manages working memory across multiple tasks.
    
    Allows:
    - Multiple task contexts with separate working memory
    - Switching between tasks
    - Saving/restoring task state
    """
    
    def __init__(self, default_capacity: int = 7):
        self._default_capacity = default_capacity
        self._task_memories: dict[str, WorkingMemory] = {}
        self._current_task: str | None = None
    
    def create_task(self, task_id: str) -> WorkingMemory:
        """Create working memory for a new task."""
        if task_id not in self._task_memories:
            self._task_memories[task_id] = WorkingMemory(self._default_capacity)
        return self._task_memories[task_id]
    
    def get_task_memory(self, task_id: str) -> WorkingMemory | None:
        """Get working memory for a task."""
        return self._task_memories.get(task_id)
    
    def switch_to_task(self, task_id: str) -> WorkingMemory | None:
        """Switch to a task's working memory."""
        if task_id not in self._task_memories:
            self._task_memories[task_id] = WorkingMemory(self._default_capacity)
        self._current_task = task_id
        return self._task_memories[task_id]
    
    def current_memory(self) -> WorkingMemory | None:
        """Get the current task's working memory."""
        if self._current_task:
            return self._task_memories.get(self._current_task)
        return None
    
    def current_task_id(self) -> str | None:
        """Get the current task ID."""
        return self._current_task
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task's working memory."""
        if task_id in self._task_memories:
            del self._task_memories[task_id]
            if self._current_task == task_id:
                self._current_task = None
            return True
        return False
    
    def list_tasks(self) -> list[str]:
        """List all task IDs."""
        return list(self._task_memories.keys())


# ════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════

__all__ = [
    "MemorySlot",
    "WorkingMemoryItem",
    "WorkingMemory",
    "WorkingMemoryManager",
]
