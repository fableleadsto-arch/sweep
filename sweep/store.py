"""Persistent state for the terminal controller.

Everything the controller remembers about this machine lives under
``~/.sweep/controller/`` as small JSON files:

  config.json   — user knobs (confirm destructive actions, ...)
  aliases.json  — "mycode" -> "C:\\Projects\\code" style shortcuts
  notes.json    — the "remember / remind me" store
  stats.json    — intent usage counters (which skills get used)
  history.txt   — recent REPL commands

The ``SWEEP_CONTROLLER_DIR`` env var overrides the base directory so tests and
headless setups can use an isolated store.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_BASE = Path.home() / ".sweep" / "controller"


def _load_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class Note:
    """One persisted note / reminder."""

    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tags": self.tags,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Note":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            text=str(data.get("text") or ""),
            tags=list(data.get("tags") or []),
            created=float(data.get("created") or time.time()),
        )


class ControllerStore:
    """Read/write the controller's persistent state."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        env_dir = os.environ.get("SWEEP_CONTROLLER_DIR")
        self.base_dir = Path(base_dir) if base_dir else Path(env_dir) if env_dir else DEFAULT_BASE
        self.config_path = self.base_dir / "config.json"
        self.aliases_path = self.base_dir / "aliases.json"
        self.notes_path = self.base_dir / "notes.json"
        self.stats_path = self.base_dir / "stats.json"
        self.history_path = self.base_dir / "history.txt"

    # ── config ─────────────────────────────────────────────────────────

    @property
    def config(self) -> dict[str, Any]:
        return _load_json(self.config_path, {})

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        data = self.config
        data[key] = value
        _save_json(self.config_path, data)

    # ── aliases ────────────────────────────────────────────────────────

    @property
    def aliases(self) -> dict[str, str]:
        return _load_json(self.aliases_path, {})

    def get_alias(self, name: str) -> Optional[str]:
        if not name:
            return None
        lower = name.lower()
        for key, value in self.aliases.items():
            if key.lower() == lower:
                return str(value)
        return None

    def set_alias(self, name: str, target: str) -> None:
        data = self.aliases
        data[name] = target
        _save_json(self.aliases_path, data)

    def remove_alias(self, name: str) -> Optional[str]:
        data = self.aliases
        removed = None
        for key in list(data):
            if key.lower() == name.lower():
                removed = data.pop(key)
                break
        if removed is not None:
            _save_json(self.aliases_path, data)
        return removed

    # ── notes ──────────────────────────────────────────────────────────

    @property
    def notes(self) -> list[Note]:
        raw = _load_json(self.notes_path, [])
        return [Note.from_dict(item) for item in raw if isinstance(item, dict)]

    def add_note(self, text: str, tags: Optional[list[str]] = None) -> Note:
        note = Note(id=uuid.uuid4().hex[:12], text=text.strip(), tags=list(tags or []))
        data = [n.to_dict() for n in self.notes]
        data.append(note.to_dict())
        _save_json(self.notes_path, data)
        return note

    def search_notes(self, query: str, limit: int = 10) -> list[Note]:
        q = query.lower()
        if not q:
            return self.notes[-limit:]
        return [n for n in self.notes[-100:] if q in n.text.lower()][-limit:]

    def clear_notes(self) -> int:
        data = self.notes
        _save_json(self.notes_path, [])
        return len(data)

    # ── stats ──────────────────────────────────────────────────────────

    def record_intent(self, intent: str) -> None:
        data = self.stats
        data["intents"] = data.get("intents", {})
        data["intents"][intent] = int(data["intents"].get(intent, 0)) + 1
        data["total"] = int(data.get("total", 0)) + 1
        _save_json(self.stats_path, data)

    @property
    def stats(self) -> dict[str, Any]:
        return _load_json(self.stats_path, {})

    # ── history ────────────────────────────────────────────────────────

    def append_history(self, command: str) -> None:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(command.strip().replace("\n", " ") + "\n")
        except OSError:
            pass

    def read_history(self, limit: int = 100) -> list[str]:
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [line for line in lines if line.strip()][-limit:]

    # ── conversation ──────────────────────────────────────────────────

    @property
    def conversation_path(self) -> Path:
        return self.base_dir / "conversation.jsonl"

    def append_conversation(self, role: str, text: str) -> None:
        """Append one turn to the multi-turn discussion log."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.conversation_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"role": role, "text": text.strip(), "t": time.time()}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def read_conversation(self, limit: int = 40) -> list[dict[str, Any]]:
        try:
            lines = self.conversation_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        turns: list[dict[str, Any]] = []
        for line in lines:
            try:
                turns.append(json.loads(line))
            except ValueError:
                continue
        return turns[-limit:]

    def clear_conversation(self) -> int:
        try:
            return len(self.read_conversation())
        finally:
            try:
                self.conversation_path.unlink()
            except OSError:
                pass
