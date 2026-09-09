"""
Context Module — Phase 1: Contextual representations.

Context is essential for understanding. The same content means different
things in different contexts. This module provides:

- Contextual representation of information
- Context stacking (multiple contexts apply simultaneously)
- Context-based interpretation
- Contextual retrieval cues

Key principle: Information should be interpreted as:
    concept + context + relationships + time + source + confidence
NOT as:
    word → fixed meaning
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ContextType(str, Enum):
    """Types of context."""
    TEMPORAL = "temporal"           # Time context
    SPATIAL = "spatial"             # Location context
    ENTITY = "entity"               # Entity context
    TASK = "task"                   # Task context
    SOURCE = "source"               # Source context
    DOMAIN = "domain"               # Domain context
    SOCIAL = "social"               # Social context
    MODAL = "modal"                 # Modal context (possibility, necessity)


@dataclass
class ContextFrame:
    """
    A single frame of context.

    Context frames can be stacked to represent complex situations.
    Each frame contributes to the interpretation of information.
    """
    context_type: ContextType
    content: str
    scope: str = "local"            # "local", "global", "task", "session"
    priority: float = 0.5          # Importance of this context
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # When this context expires
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"context_{int(time.time() * 1000)}")

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.context_type.value,
            "content": self.content,
            "scope": self.scope,
            "priority": self.priority,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired(),
            "metadata": self.metadata,
        }


@dataclass
class ContextualRepresentation:
    """
    A representation paired with its context.

    This is the primary way information enters the cognitive system.
    Every piece of information should be contextualized.
    """
    representation_content: str
    context_frames: list[ContextFrame] = field(default_factory=list)
    interpretation_notes: list[str] = field(default_factory=list)
    ambiguity_flags: list[str] = field(default_factory=list)
    alternative_interpretations: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"context_rep_{int(time.time() * 1000)}")

    def add_context(self, context_type: ContextType, content: str,
                   scope: str = "local", priority: float = 0.5,
                   expires_at: Optional[float] = None,
                   metadata: dict | None = None) -> ContextFrame:
        """Add a context frame to this representation."""
        frame = ContextFrame(
            context_type=context_type,
            content=content,
            scope=scope,
            priority=priority,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self.context_frames.append(frame)
        return frame

    def get_context(self, context_type: ContextType) -> list[ContextFrame]:
        """Get all context frames of a specific type."""
        return [f for f in self.context_frames if f.context_type == context_type]

    def get_active_context(self) -> list[ContextFrame]:
        """Get non-expired context frames, sorted by priority."""
        active = [f for f in self.context_frames if not f.is_expired()]
        active.sort(key=lambda f: f.priority, reverse=True)
        return active

    def mark_ambiguous(self, ambiguity: str, alternatives: list[dict] | None = None):
        """Mark this representation as ambiguous with alternative interpretations."""
        if ambiguity not in self.ambiguity_flags:
            self.ambiguity_flags.append(ambiguity)
        if alternatives:
            self.alternative_interpretations.extend(alternatives)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "representation_content": self.representation_content,
            "context_frames": [f.to_dict() for f in self.context_frames],
            "interpretation_notes": self.interpretation_notes,
            "ambiguity_flags": self.ambiguity_flags,
            "alternative_interpretations": self.alternative_interpretations,
            "created_at": self.created_at,
            "active_context_count": len(self.get_active_context()),
        }


class ContextStack:
    """
    A stack of context frames for the current task/session.

    This represents the ACTIVE context that should be applied to
    interpreting new information.
    """

    def __init__(self, capacity: int = 20):
        self._frames: list[ContextFrame] = []
        self._capacity = capacity
        self._by_type: dict[ContextType, list[ContextFrame]] = {}

    def push(self, frame: ContextFrame):
        """Push a context frame onto the stack."""
        self._frames.append(frame)
        self._by_type.setdefault(frame.context_type, []).append(frame)

        # Enforce capacity
        if len(self._frames) > self._capacity:
            # Remove lowest priority frame
            self._frames.sort(key=lambda f: f.priority)
            removed = self._frames.pop(0)
            # Remove from type index
            if removed.id in self._by_type.get(removed.context_type, []):
                self._by_type[removed.context_type] = [
                    f for f in self._by_type[removed.context_type] 
                    if f.id != removed.id
                ]

    def pop(self) -> Optional[ContextFrame]:
        """Pop the most recently added context frame."""
        if not self._frames:
            return None
        frame = self._frames.pop()
        if frame.id in self._by_type.get(frame.context_type, []):
            self._by_type[frame.context_type] = [
                f for f in self._by_type[frame.context_type] if f.id != frame.id
            ]
        return frame

    def get(self, context_type: ContextType) -> list[ContextFrame]:
        """Get all context frames of a type."""
        return self._by_type.get(context_type, []).copy()

    def get_all(self) -> list[ContextFrame]:
        """Get all context frames, sorted by priority."""
        return sorted(self._frames, key=lambda f: f.priority, reverse=True)

    def get_active(self) -> list[ContextFrame]:
        """Get all non-expired context frames."""
        return [f for f in self.get_all() if not f.is_expired()]

    def clear_expired(self):
        """Remove expired context frames."""
        expired_ids = {f.id for f in self._frames if f.is_expired()}
        self._frames = [f for f in self._frames if f.id not in expired_ids]
        for ctx_type in list(self._by_type.keys()):
            self._by_type[ctx_type] = [
                f for f in self._by_type[ctx_type] if f.id not in expired_ids
            ]

    def clear(self):
        """Clear all context."""
        self._frames.clear()
        self._by_type.clear()

    def size(self) -> int:
        return len(self._frames)

    def stats(self) -> dict[str, Any]:
        return {
            "total": len(self._frames),
            "active": len(self.get_active()),
            "by_type": {t.value: len(frames) 
                       for t, frames in self._by_type.items()},
        }


class ContextInterpreter:
    """
    Interprets information in context.

    Takes a representation and a context stack, and produces
    an interpreted contextual representation.
    """

    def interpret(self, content: str,
                  context_stack: ContextStack) -> ContextualRepresentation:
        """
        Interpret content given the current context.

        This is a scaffolding implementation. The full version would
        use learned models for context-sensitive interpretation.
        """
        ctx_rep = ContextualRepresentation(
            representation_content=content,
        )

        # Add context frames from the stack
        for frame in context_stack.get_active():
            ctx_rep.context_frames.append(frame)

        # Basic ambiguity detection (temporary — will be learned)
        self._detect_basic_ambiguities(ctx_rep)

        return ctx_rep

    def _detect_basic_ambiguities(self, ctx_rep: ContextualRepresentation):
        """Detect basic ambiguities (temporary scaffolding)."""
        content = ctx_rep.representation_content.lower()

        # Check for words with multiple meanings
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
        }

        for word, meanings in ambiguous_words.items():
            if word in content:
                ctx_rep.mark_ambiguous(
                    f"word '{word}' has multiple meanings",
                    [{"word": word, "meanings": meanings}]
                )


__all__ = [
    "ContextFrame",
    "ContextualRepresentation",
    "ContextStack",
    "ContextInterpreter",
    "ContextType",
]
