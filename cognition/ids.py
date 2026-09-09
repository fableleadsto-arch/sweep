"""Stable identity and timestamp helpers for the cognitive system.

Identifiers must survive serialization/deserialization unchanged so that
relationships between cognitive objects never silently break.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str = "") -> str:
    """Generate a stable, collision-resistant id, optionally namespaced."""
    raw = uuid.uuid4().hex
    return f"{prefix}-{raw}" if prefix else raw


def content_id(prefix: str, *parts: Any) -> str:
    """Deterministic id derived from content (for idempotent ingestion)."""
    blob = "|".join(str(p) for p in parts)
    return f"{prefix}-{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:32]}"


@dataclass
class Timestamped:
    """Base mixin for objects carrying creation time and access ordering."""

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


def utc_iso(ts: float | None = None) -> str:
    """ISO-8601 UTC string (deterministic-ish; used for records)."""
    import datetime

    value = datetime.datetime.fromtimestamp(ts or time.time(), tz=datetime.timezone.utc)
    return value.isoformat(timespec="microseconds")