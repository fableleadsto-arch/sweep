"""Phase 3 — Persistent Cognitive Memory.

Working Memory and Long-Term Memory that are independent of the LLM context
window. Task events are stored to LTM; a new session retrieves them with
their provenance and uses them in reasoning.

Layout:
- `WorkingMemory`: bound working set for the current task/run.
- `LongTermMemory`: append-only persistent store of remembered events with
  provenance (source task, timestamp, retrieval count).
- memory handoff (`remember -> retrieve` across sessions).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ids import new_id
from .store import CognitiveStore


class MemoryKind(str, enum.Enum):
    FACT = "fact"
    EVENT = "event"
    CONCLUSION = "conclusion"
    STRATEGY = "strategy"
    OBSERVATION = "observation"


@dataclass
class MemoryEntry:
    """A unit of memory with provenance retained."""

    content: str
    kind: MemoryKind = MemoryKind.FACT
    source_task: str = ""
    provenance: str = ""
    created_at: float = field(default_factory=time.time)
    evidence_ids: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("mem"))
    retrievals: int = 0
    last_retrieved_at: Optional[float] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        vals = {k: v for k, v in d.items() if k in MemoryEntry.__dataclass_fields__}
        vals["kind"] = MemoryKind(d["kind"])
        return cls(**vals)


class WorkingMemory:
    """Bounded working set for the current task/run.

    Independent of any LLM context window, but transient.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._items: List[MemoryEntry] = []
        self._by_key: Dict[str, MemoryEntry] = {}

    def store(self, content: str, kind: MemoryKind = MemoryKind.FACT,
              source_task: str = "", provenance: str = "", key: str = "") -> MemoryEntry:
        entry = MemoryEntry(content=content, kind=kind, source_task=source_task, provenance=provenance,
                            metadata={"wm_key": key})
        if key:
            if key in self._by_key:
                return self._by_key[key]
            self._by_key[key] = entry
        self._items.append(entry)
        if len(self._items) > self.capacity:
            evicted = self._items.pop(0)
            keys_to_drop = [k for k, v in self._by_key.items() if v is evicted]
            for k in keys_to_drop:
                del self._by_key[k]
        return entry

    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        entry = self._by_key.get(key)
        if entry:
            self._touch(entry)
        return entry

    def _touch(self, entry: MemoryEntry) -> None:
        entry.retrievals += 1
        entry.last_retrieved_at = time.time()

    def recall(self, query: str = "") -> List[MemoryEntry]:
        if not query:
            return list(self._items)
        q = query.lower()
        return [e for e in self._items if q in e.content.lower()]

    def recent(self, n: int = 10) -> List[MemoryEntry]:
        return self._items[-n:]

    def size(self) -> int:
        return len(self._items)


class LongTermMemory:
    """Persistent memory independent of any context window.

    Entries survive process/session boundaries; provenance and history are
    retained so a later session can retrieve and reason with them.
    """

    def __init__(self, store: CognitiveStore | None = None, store_path=None, in_memory: bool = False):
        self._store = store or CognitiveStore(path=store_path, in_memory=in_memory)
        self._load()

    def _load(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}
        self._ordered: List[str] = []
        for blob in self._store._memory_records():
            entry = MemoryEntry.from_dict(blob["payload"])
            self._entries[entry.id] = entry
            if entry.id not in self._ordered:
                self._ordered.append(entry.id)

    def remember(self, content: str, kind: MemoryKind = MemoryKind.FACT,
                 source_task: str = "", provenance: str = "",
                 evidence_ids: List[str] | None = None, metadata: Dict[str, Any] | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            kind=kind,
            source_task=source_task,
            provenance=provenance,
            evidence_ids=evidence_ids or [],
            metadata=metadata or {},
        )
        self._entries[entry.id] = entry
        self._ordered.append(entry.id)
        self._store._memory_records({"payload": entry.to_dict()})
        return entry

    def retrieve(self, query: str = "", n: int = 10, kind: MemoryKind | None = None) -> List[MemoryEntry]:
        """Retrieve memories relevant to a query.

        Relevance here is substring/word-overlap matching against content,
        with retrieval stats (provenance/history) updated.
        """
        matches = []
        for oid in self._ordered:
            e = self._entries[oid]
            if kind is not None and e.kind != kind:
                continue
            if not query or self._score(e, query) > 0:
                matches.append(e)
        matches.sort(key=lambda e: (self._score(e, query) if query else 1.0, e.created_at), reverse=True)
        out = matches[:n]
        for e in out:
            e.retrievals += 1
            e.last_retrieved_at = time.time()
            self._store._memory_records({"payload": e.to_dict(), "_update": True})
        return out

    @staticmethod
    def _score(entry: MemoryEntry, query: str) -> float:
        q = query.lower()
        tokens = [t for t in q.replace(",", " ").replace(".", " ").split() if len(t) > 2]
        if not tokens:
            return 1.0
        content = entry.content.lower()
        hits = sum(1 for t in tokens if t in content)
        # words shared between query and content, and long-phrase containment
        extra = 2 if q in content else 0
        return (hits / len(tokens)) + extra

    def recall_all(self) -> List[MemoryEntry]:
        return [self._entries[oid] for oid in self._ordered]

    def by_task(self, source_task: str) -> List[MemoryEntry]:
        return [e for e in self.recall_all() if e.source_task == source_task]

    def count(self) -> int:
        return len(self._entries)


class CognitiveMemory:
    """Facade combining working + long-term memory with cross-session handoff."""

    def __init__(self, store: CognitiveStore | None = None, store_path=None, in_memory: bool = False):
        self.working = WorkingMemory()
        self.longterm = LongTermMemory(store=store, store_path=store_path, in_memory=in_memory)

    def observe(self, content: str, provenance: str = "") -> MemoryEntry:
        """Record an observation into working + LTM (handoff)."""
        self.working.store(content, source_task="current", provenance=provenance)
        return self.longterm.remember(content, kind=MemoryKind.OBSERVATION, source_task="current",
                                      provenance=provenance)

    def remember(self, content: str, **kw) -> MemoryEntry:
        return self.longterm.remember(content, **kw)

    def retrieve(self, query: str = "", n: int = 10, kind: MemoryKind | None = None) -> List[MemoryEntry]:
        return self.longterm.retrieve(query, n=n, kind=kind)

    def clear_working(self) -> None:
        self.working = WorkingMemory()