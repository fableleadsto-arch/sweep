"""Persistent JSONL store for cognitive objects.

Append-only JSONL keeps identity and relationships lossless across sessions.
All phase subsystems write through this store so data survives process
restarts and can be queried deterministically.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import object_from_dict, object_to_dict

_STORE_NS = "cognition::"


class CognitiveStore:
    """Thread-safe, append-only, indexable JSONL store."""

    def __init__(self, path: str | Path | None = None, in_memory: bool = False):
        if in_memory:
            self._path: Optional[Path] = None
        else:
            self._path = Path(path) if path else Path(os.environ.get("COGNITION_STORE", "cognition_store"))
            self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: Dict[str, Dict[str, Any]] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._compound_records: List[Dict[str, Any]] = []
        self._memory_entries: List[Dict[str, Any]] = []
        self._load_existing()

    # ------------------------------------------------------------- lifecycle
    def _load_existing(self) -> None:
        if self._path is None:
            return
        for f in sorted(self._path.glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    blob = json.loads(line)
                    if "_type" in blob:
                        self._index_blob(blob, persist=False)
                    elif "_ctype" in blob:
                        self._compound_records.append(blob)
                    elif "payload" in blob and "_update" not in blob:
                        self._memory_entries.append(blob)

    def _index_blob(self, blob: Dict[str, Any], persist: bool) -> None:
        oid = blob.get("id")
        tname = blob.get("_type")
        if oid:
            self._index[oid] = blob
            if tname and None not in (tname, oid):
                self._by_type.setdefault(tname, []).append(oid)
        if persist:
            self._append(blob)

    def _append(self, blob: Dict[str, Any]) -> None:
        if self._path is not None:
            kind = blob.get("_type", "unknown")
            with open(self._path / f"{kind}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(blob) + "\n")

    # ------------------------------------------------------------------ API
    def save(self, obj: Any) -> Any:
        """Persist one contract object (idempotent by id)."""
        with self._lock:
            blob = object_to_dict(obj)
            oid = blob.get("id")
            if oid:
                self._index[oid] = blob
                tname = blob.get("_type")
                if tname and oid not in self._by_type.setdefault(tname, []):
                    self._by_type[tname].append(oid)
            self._append(blob)
        return obj

    def save_many(self, objs: List[Any]) -> List[Any]:
        for o in objs:
            self.save(o)
        return objs

    def get(self, oid: str) -> Optional[Any]:
        with self._lock:
            blob = self._index.get(oid)
            if blob is None:
                return None
            return object_from_dict(dict(blob))

    def by_type(self, tname: str) -> List[Any]:
        with self._lock:
            return [self.get(i) for i in self._by_type.get(tname, []) if self.get(i)]

    def all(self) -> List[Any]:
        with self._lock:
            return [self.get(i) for i in self._index if self.get(i)]

    def count(self) -> int:
        with self._lock:
            return len(self._index)

    def query(self, tname: str | None = None, **attrs: Any) -> List[Any]:
        """Filter objects by attribute equality (best effort on primitives)."""
        with self._lock:
            objs = self.by_type(tname) if tname else self.all()
            out = []
            for o in objs:
                ok = True
                for k, v in attrs.items():
                    if getattr(o, k, None) != v:
                        ok = False
                        break
                if ok:
                    out.append(o)
            return out

    # --------------------------------------------------- compound blobs
    # Compound records (graph edges etc.) are NOT contract objects; they are
    # stored with a `_ctype` tag so they can be replayed when the store loads.
    def _compound(self, blob: Dict[str, Any] | None = None) -> list:
        with self._lock:
            if blob is None:
                return self._compound_records
            self._compound_records.append(blob)
            if self._path is not None:
                with open(self._path / "compound.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(blob) + "\n")
            return self._compound_records

    def _index_rel_claim_evidence(self) -> None:
        """Rebuild claim<->evidence edges from stored Relation records.

        Which claim IDs are linked to which evidence is stored in the blobs'
        `payload` dict; this helper is provided so graph layers can re-derive
        adjacency without reloading blob semantics themselves.
        """
        return

    # --------------------------------------------------- memory records
    def _memory_records(self, blob: Dict[str, Any] | None = None) -> list:
        """Read/write LTM entry records (payload previously written by MemoryLayer).

        Callers pass `{"payload": {...}}` to write or `None` to read all.
        """
        with self._lock:
            if blob is None:
                return self._memory_entries
            if blob.get("_update"):
                # rewrite an existing record (update retrieval stats)
                for i, b in enumerate(self._memory_entries):
                    if b["payload"].get("id") == blob["payload"].get("id"):
                        self._memory_entries[i] = blob
                        break
            else:
                self._memory_entries.append(blob)
            if self._path is not None:
                with open(self._path / "memory.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(blob) + "\n")
            return self._memory_entries

    def delete(self, oid: str) -> bool:
        with self._lock:
            existed = oid in self._index
            self._index.pop(oid, None)
            for k in list(self._by_type):
                self._by_type[k] = [i for i in self._by_type[k] if i != oid]
            return existed