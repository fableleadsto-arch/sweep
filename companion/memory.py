"""Persistent memory — mirrors `src/RelAI/memory/store.server.ts` + `qdrant.server.ts`.

Layered exactly like the TypeScript stack:
  1. Qdrant semantic search (vector) when configured
  2. Keyword relevance search over the JSON file store
  3. Graceful degradation at every layer

The file store keeps an in-memory cache with a short TTL so chat/voice turns
don't pay a full disk read each time.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient as QdrantHttp

from .config import BrainSettings
from .embeddings import embed_batch

MEMORY_VERSION = 1
MEMORY_KINDS = ("fact", "preference", "context", "artifact")

logger = logging.getLogger("companion.memory")

# Mirrors the TS latch in src/RelAI/memory/qdrant.server.ts: when Qdrant
# rejects our API key (its auth error is the distinctive `unauthorised: Invalid
# key`), stop hammering the server for a while and fall back to the file store
# silently after one clear warning. After the TTL the client re-probes, so a
# rotated/valid key is picked up without a restart.
AUTH_REJECT_TTL_S = 300.0  # 5 minutes
_AUTH_ERROR_TOKENS = ("unauthoris", "invalid api key", "invalid key", "forbidden")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i if in is it its of on
    or that the their there these this to was what when where which who will with
    you your""".split()
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryEntry:
    id: str
    user_id: str
    workspace_id: Optional[str] = None
    content: str = ""
    kind: str = "fact"
    tags: list[str] = field(default_factory=list)
    source: str = "relai"
    confidence: float = 0.7
    created_at: str = ""
    updated_at: str = ""
    expires_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9'\-]{2,}", text.lower()) if t not in STOPWORDS]


def _from_storage(data: dict[str, Any]) -> MemoryEntry:
    """Parse either file-store (camelCase) or Qdrant payload shapes."""
    return MemoryEntry(
        id=str(data.get("id", "")),
        user_id=str(data.get("userId", data.get("user_id", ""))),
        workspace_id=data.get("workspaceId", data.get("workspace_id")),
        content=str(data.get("content", "")),
        kind=str(data.get("kind", "fact")),
        tags=list(data.get("tags", [])),
        source=str(data.get("source", "relai")),
        confidence=float(data.get("confidence", 0.7)),
        created_at=str(data.get("createdAt", data.get("created_at", ""))),
        updated_at=str(data.get("updatedAt", data.get("updated_at", ""))),
        expires_at=data.get("expiresAt", data.get("expires_at")),
    )


def _to_storage(entry: MemoryEntry) -> dict[str, Any]:
    """File store shape (camelCase) — byte-compatible with the TS file."""
    return {
        "id": entry.id,
        "userId": entry.user_id,
        "workspaceId": entry.workspace_id,
        "content": entry.content,
        "kind": entry.kind,
        "tags": entry.tags,
        "source": entry.source,
        "confidence": entry.confidence,
        "createdAt": entry.created_at,
        "updatedAt": entry.updated_at,
        "expiresAt": entry.expires_at,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Relevance scoring (mirrors `scoreEntry` / `dedupeByContent`)
# ─────────────────────────────────────────────────────────────────────────


def score_entry(query: str, entry: MemoryEntry, half_life_days: float = 30.0) -> float:
    query_terms = set(_tokenize(query))
    content_terms = set(_tokenize(entry.content))
    overlap = len(query_terms & content_terms)
    phrase_boost = (
        3.0 if query and _normalize(query).lower() in entry.content.lower() else 0.0
    )

    age_ms = math.inf
    try:
        if entry.updated_at:
            updated = datetime.fromisoformat(entry.updated_at)
            age_ms = (datetime.now(timezone.utc) - updated).total_seconds() * 1000
    except ValueError:
        pass
    age_days = age_ms / 86_400_000
    recency = math.pow(0.5, age_days / half_life_days)
    confidence_factor = 0.5 + min(max(entry.confidence, 0), 1) * 0.5

    return (overlap * 2 + phrase_boost) * (0.5 + recency) * confidence_factor


def dedupe_by_content(scored: list[tuple[MemoryEntry, float]]) -> list[tuple[MemoryEntry, float]]:
    """Drop near-duplicate memories (>70% term overlap on the smaller set)."""
    kept: list[tuple[MemoryEntry, float]] = []
    for entry, score in scored:
        terms = set(_tokenize(entry.content))
        if len(terms) < 4:
            kept.append((entry, score))
            continue
        duplicate = False
        for kept_entry, _ in kept:
            kept_terms = set(_tokenize(kept_entry.content))
            if not kept_terms:
                continue
            if len(terms & kept_terms) / min(len(terms), len(kept_terms)) > 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append((entry, score))
    return kept


# ─────────────────────────────────────────────────────────────────────────
#  File store (sync; disk I/O is cheap and cached)
# ─────────────────────────────────────────────────────────────────────────


class FileMemoryStore:
    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings
        self.path: Path = Path(settings.memory_file)
        self._cache: Optional[tuple[float, list[MemoryEntry]]] = None

    def _load(self) -> list[MemoryEntry]:
        if self._cache and time.time() * 1000 - self._cache[0] < self.settings.memory_cache_ttl_ms:
            return self._cache[1]
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            entries = [_from_storage(e) for e in raw.get("memories", []) if isinstance(e, dict)]
        except (OSError, ValueError):
            entries = []
        return entries

    def _save(self, entries: list[MemoryEntry]) -> None:
        self._cache = (time.time() * 1000, entries)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": MEMORY_VERSION, "memories": [_to_storage(e) for e in entries]},
                indent=2,
            ),
            encoding="utf-8",
        )

    def remember(self, user_id: str, content: str, **kw: Any) -> MemoryEntry:
        content = _normalize(content)
        if not content:
            raise ValueError("Memory content cannot be empty.")

        entries = self._load()
        existing = next(
            (
                e
                for e in entries
                if e.user_id == user_id
                and e.workspace_id == kw.get("workspace_id")
                and _normalize(e.content) == content
            ),
            None,
        )
        now = utcnow()
        if existing:
            existing.content = content
            existing.kind = kw.get("kind", existing.kind)
            existing.tags = sorted(set(kw.get("tags", []) or []) | set(existing.tags))
            existing.source = kw.get("source", existing.source)
            existing.confidence = float(kw.get("confidence", existing.confidence))
            existing.updated_at = now
            existing.expires_at = kw.get("expires_at", existing.expires_at)
        else:
            existing = MemoryEntry(
                id=f"{user_id}:{int(time.time() * 1000)}",
                user_id=user_id,
                workspace_id=kw.get("workspace_id"),
                content=content,
                kind=kw.get("kind", "fact"),
                tags=sorted(set(kw.get("tags", []) or [])),
                source=kw.get("source", "relai"),
                confidence=float(kw.get("confidence", 0.7)),
                created_at=now,
                updated_at=now,
                expires_at=kw.get("expires_at"),
            )
            entries.append(existing)

        self._save(entries)
        return existing

    def search(
        self,
        user_id: str,
        query: str = "",
        workspace_id: Optional[str] = None,
        limit: int = 8,
    ) -> list[MemoryEntry]:
        entries = [
            e
            for e in self._load()
            if e.user_id == user_id
            and (not workspace_id or not e.workspace_id or e.workspace_id == workspace_id)
        ]
        if not query:
            entries.sort(key=lambda e: e.updated_at, reverse=True)
            return entries[:limit]

        scored = [
            (e, score_entry(query, e, self.settings.memory_recency_half_life_days))
            for e in entries
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [entry for entry, _ in dedupe_by_content(scored)[:limit]]


# ─────────────────────────────────────────────────────────────────────────
#  Qdrant vector adapter (async — embedding I/O must not block the loop)
# ─────────────────────────────────────────────────────────────────────────


def _stable_point_id(memory_id: str) -> str:
    """Qdrant wants UUID-ish point ids; derive one deterministically."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))


class QdrantMemoryStore:
    def __init__(self, settings: BrainSettings, file_store: FileMemoryStore) -> None:
        self.settings = settings
        self.file_store = file_store
        self._client: Optional[QdrantHttp] = None
        self._collection = settings.qdrant_collection
        self._auth_rejected_at: Optional[float] = None

    # ------------------------------------------------------------------
    #  Credential rejection latch (mirrors the TS implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_auth_rejection(err: Exception) -> bool:
        """True when a Qdrant error is a rejected/expired API key."""
        status = getattr(err, "status_code", None) or getattr(err, "status", None)
        if status in (401, 403):
            return True
        msg = str(err).lower()
        return any(token in msg for token in _AUTH_ERROR_TOKENS)

    def _auth_rejected(self) -> bool:
        return (
            self._auth_rejected_at is not None
            and time.time() - self._auth_rejected_at < AUTH_REJECT_TTL_S
        )

    def _note_auth_rejection(self, err: Exception) -> bool:
        """Record a rejected key; print one actionable warning per window."""
        if not self._is_auth_rejection(err):
            return False
        now = time.time()
        if self._auth_rejected_at is None or now - self._auth_rejected_at > AUTH_REJECT_TTL_S:
            logger.warning(
                "[Qdrant] API key rejected (unauthorised: Invalid key). Memory falls back to the "
                "file store until the key is fixed. Regenerate QDRANT_API_KEY at cloud.qdrant.io, "
                "or unset QDRANT_API_URL / QDRANT_API_KEY to keep the file store."
            )
        self._auth_rejected_at = now
        return True

    def _get_client(self) -> Optional[QdrantHttp]:
        if self._auth_rejected():
            return None
        if self._client:
            return self._client
        if not self.settings.qdrant_configured:
            return None
        try:
            from qdrant_client import QdrantClient as QdrantHttp

            self._client = QdrantHttp(
                url=self.settings.qdrant_api_url,
                api_key=self.settings.qdrant_api_key or None,
            )
            return self._client
        except Exception:  # noqa: BLE001 - fall back to file store
            return None

    def _ensure_collection(self, client: QdrantHttp) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams

            existing = client.collection_exists(self._collection)
            if not existing:
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self.settings.qdrant_vector_size, distance=Distance.COSINE
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - Qdrant may be transiently down
            self._note_auth_rejection(exc)

    async def remember(self, user_id: str, content: str, **kw: Any) -> tuple[MemoryEntry, bool]:
        client = self._get_client()
        if client is None:
            return self.file_store.remember(user_id, content, **kw), False

        self._ensure_collection(client)
        memory_id = f"{user_id}:{int(time.time() * 1000)}"
        now = utcnow()

        embeddings = await embed_batch([content], self.settings)
        vector = embeddings[0] if embeddings else None
        if vector is None:
            return self.file_store.remember(user_id, content, **kw), False

        try:
            from qdrant_client.models import PointStruct

            client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=_stable_point_id(memory_id),
                        vector=vector,
                        payload={
                            "userId": user_id,
                            "workspaceId": kw.get("workspace_id"),
                            "content": content,
                            "kind": kw.get("kind", "fact"),
                            "tags": sorted(set(kw.get("tags", []) or [])),
                            "source": kw.get("source", "relai"),
                            "confidence": float(kw.get("confidence", 0.7)),
                            "createdAt": now,
                            "updatedAt": now,
                            "expiresAt": kw.get("expires_at"),
                        },
                    )
                ],
            )
            entry = MemoryEntry(
                id=memory_id,
                user_id=user_id,
                workspace_id=kw.get("workspace_id"),
                content=content,
                kind=kw.get("kind", "fact"),
                tags=sorted(set(kw.get("tags", []) or [])),
                source=kw.get("source", "relai"),
                confidence=float(kw.get("confidence", 0.7)),
                created_at=now,
                updated_at=now,
                expires_at=kw.get("expires_at"),
            )
            return entry, True
        except Exception as exc:  # noqa: BLE001 - fall back to file store
            self._note_auth_rejection(exc)
            return self.file_store.remember(user_id, content, **kw), False

    async def search(
        self,
        user_id: str,
        query: str = "",
        workspace_id: Optional[str] = None,
        limit: int = 8,
    ) -> list[MemoryEntry]:
        client = self._get_client()
        if client is None:
            return self.file_store.search(user_id, query, workspace_id, limit)

        self._ensure_collection(client)
        embeddings = await embed_batch([query], self.settings)
        vector = embeddings[0] if embeddings else None
        if vector is None:
            return self.file_store.search(user_id, query, workspace_id, limit)

        try:
            must: list[dict[str, Any]] = [{"key": "userId", "match": {"value": user_id}}]
            if workspace_id:
                must.append({"key": "workspaceId", "match": {"value": workspace_id}})

            points = client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=limit,
                query_filter={"must": must},
                with_payload=True,
                score_threshold=0.2,
            )
            entries = [_from_storage({**p.payload, "id": p.id}) for p in points if p.payload]
            if entries:
                return entries
            return self.file_store.search(user_id, query, workspace_id, limit)
        except Exception as exc:  # noqa: BLE001 - fall back to file store
            self._note_auth_rejection(exc)
            return self.file_store.search(user_id, query, workspace_id, limit)

    def count(self, user_id: Optional[str] = None) -> int:
        client = self._get_client()
        if client is None:
            return 0
        try:
            result = client.count(
                collection_name=self._collection,
                count_filter=(
                    {"must": [{"key": "userId", "match": {"value": user_id}}]} if user_id else None
                ),
                exact=True,
            )
            return result.count
        except Exception as exc:  # noqa: BLE001
            self._note_auth_rejection(exc)
            return 0


# ─────────────────────────────────────────────────────────────────────────
#  Public facade
# ─────────────────────────────────────────────────────────────────────────


class MemoryService:
    """Single entry point for memory — Qdrant vector search, file fallback."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings
        self.file_store = FileMemoryStore(settings)
        self.qdrant = QdrantMemoryStore(settings, self.file_store)

    async def remember(
        self,
        user_id: str,
        content: str,
        *,
        workspace_id: Optional[str] = None,
        kind: str = "fact",
        tags: Optional[list[str]] = None,
        source: str = "relai",
        confidence: float = 0.7,
        expires_at: Optional[str] = None,
    ) -> MemoryEntry:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"Unknown memory kind: {kind!r}")
        entry, _ = await self.qdrant.remember(
            user_id,
            content,
            workspace_id=workspace_id,
            kind=kind,
            tags=tags or [],
            source=source,
            confidence=confidence,
            expires_at=expires_at,
        )
        return entry

    async def search(
        self,
        user_id: str,
        query: str = "",
        *,
        workspace_id: Optional[str] = None,
        limit: int = 8,
    ) -> list[MemoryEntry]:
        return await self.qdrant.search(user_id, query, workspace_id, limit)

    def profile(self, user_id: str, workspace_id: Optional[str] = None) -> dict[str, Any]:
        memories = self.file_store.search(user_id, "", workspace_id, limit=12)
        return {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "facts": [m.content for m in memories if m.kind != "context"],
            "recent_context": [m.content for m in memories if m.kind == "context"],
            "last_updated": memories[0].updated_at if memories else utcnow(),
        }

    def summary(self, user_id: str) -> dict[str, Any]:
        return {
            "totalMemories": self.qdrant.count(user_id),
            "byKind": {},
            "lastUpdated": None,
        }

    async def build_memory_context(
        self,
        user_id: str,
        query: str,
        *,
        workspace_id: Optional[str] = None,
        limit: int = 6,
    ) -> str:
        """Format a prompt-ready memory block, mirroring `buildMemoryContext`."""
        entries = await self.qdrant.search(user_id, query, workspace_id, limit)
        if not entries:
            return ""
        lines = ["Persistent memory context:"]
        lines += [f"- [{e.kind}] {e.content}" for e in entries]
        return "\n".join(lines)
