"""Persistent desktop task records, without GUI or model dependencies.

Only the desktop process writes these records. Workers consume an approved
running request and report events separately. A recovered or retried task never
inherits permission to execute an earlier request.
"""
from __future__ import annotations

import heapq
import json
import os
import re
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sweep.store import _save_json

MAX_TASK_BYTES = 1_048_576
MAX_LIST_TASKS = 1000
_TASK_ID = re.compile(r"[0-9a-f]{32}")
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
_TRANSITIONS = {
    "queued": frozenset({"running", "cancelled", "interrupted"}),
    "running": _TERMINAL,
    **{state: frozenset() for state in _TERMINAL},
}
_MUTABLE_FIELDS = frozenset({"state", "approved", "error", "started", "finished"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _check_id(task_id: str) -> str:
    if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
        raise ValueError("Invalid task ID.")
    return task_id


def _redirected(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("Invalid task timestamp.")
    parsed = datetime.fromisoformat(value)
    # Earlier desktop versions used local, naive timestamps. Keep those readable.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


class TaskStore:
    """Atomically store ``root/tasks/<id>/task.json`` and enforce state changes.

    ``get`` raises FileNotFoundError for an absent record and ValueError for an
    unsafe or malformed record. ``list`` and recovery skip those records so one
    damaged task cannot prevent opening the desktop.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.tasks_root = self.root / "tasks"

    def _check_root(self) -> None:
        if self.root.resolve() != self.root or _redirected(self.tasks_root):
            raise ValueError("Task storage must not redirect outside its directory.")
        if self.tasks_root.resolve() != self.tasks_root:
            raise ValueError("Task storage must not redirect outside its directory.")

    def directory(self, task_id: str) -> Path:
        """Return a validated task directory, whether or not it exists yet."""
        _check_id(task_id)
        self._check_root()
        path = self.tasks_root / task_id
        if _redirected(path) or path.resolve() != path:
            raise ValueError("Task directory must not redirect outside its directory.")
        return path

    @staticmethod
    def _validate(task: Any, task_id: str) -> dict:
        if not isinstance(task, dict) or task.get("id") != task_id:
            raise ValueError("Invalid task record.")
        for key in ("capability", "text"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise ValueError(f"Invalid task {key}.")
        if not isinstance(task.get("state"), str) or task["state"] not in _TRANSITIONS:
            raise ValueError("Invalid task state.")
        _timestamp(task.get("created"))
        for key in ("started", "finished"):
            if key in task:
                _timestamp(task[key])
        if "error" in task and not isinstance(task["error"], str):
            raise ValueError("Invalid task error.")
        if task.get("retry_of") is not None:
            _check_id(task["retry_of"])
        approved = task.setdefault("approved", False)
        if not isinstance(approved, bool):
            raise ValueError("Invalid task approval.")
        if task["state"] != "running":
            task["approved"] = False
        paths = task.setdefault("granted_paths", [])
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError("Invalid task file grants.")
        history = task.setdefault("history", [])
        if not isinstance(history, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("content"), str)
            for item in history
        ):
            raise ValueError("Invalid task conversation history.")
        return task

    def get(self, task_id: str) -> dict:
        """Read a fresh record with a bounded allocation; reject links and devices."""
        path = self.directory(task_id) / "task.json"
        if _redirected(path):
            raise ValueError("Task metadata must be a regular file.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TASK_BYTES:
                raise ValueError("Task metadata must be a regular file under 1 MB.")
            raw = stream.read(MAX_TASK_BYTES + 1)
        if len(raw) > MAX_TASK_BYTES:
            raise ValueError("Task metadata exceeds 1 MB.")
        try:
            task = json.loads(raw)
            return self._validate(task, task_id)
        except (UnicodeError, RecursionError, OverflowError) as exc:
            raise ValueError("Invalid task metadata.") from exc

    def _save(self, task: dict) -> dict:
        # Copy callers' nested values so no live reference can mutate stored data.
        try:
            encoded = json.dumps(task, ensure_ascii=False, allow_nan=False, indent=2)
            if len(encoded.encode("utf-8")) > MAX_TASK_BYTES:
                raise ValueError("Task metadata exceeds 1 MB.")
            task = self._validate(json.loads(encoded), task["id"])
        except (TypeError, UnicodeError, RecursionError, OverflowError) as exc:
            raise ValueError("Invalid task metadata.") from exc
        path = self.directory(task["id"]) / "task.json"
        if _redirected(path):
            raise ValueError("Task metadata must be a regular file.")
        _save_json(path, task)
        return task

    def create(
        self,
        capability: str,
        text: str,
        granted_paths: list[str] | None = None,
        history: list[dict] | None = None,
        retry_of: str | None = None,
    ) -> dict:
        """Queue a new request. Execution approval must be obtained at dispatch."""
        if retry_of is not None:
            _check_id(retry_of)
        task = {
            "id": uuid.uuid4().hex,
            "capability": capability,
            "text": text,
            "granted_paths": [] if granted_paths is None else granted_paths,
            "history": [] if history is None else history,
            "state": "queued",
            "approved": False,
            "created": _now(),
        }
        if retry_of is not None:
            task["retry_of"] = retry_of
        return self._save(task)

    def update(self, task_id: str, **changes: Any) -> dict:
        """Apply a legal transition without changing the original request."""
        if changes.keys() - _MUTABLE_FIELDS:
            raise ValueError("Task requests are immutable; create a new task to retry.")
        task = self.get(task_id)
        previous = task["state"]
        state = changes.get("state", previous)
        if not isinstance(state, str) or (
            state != previous and state not in _TRANSITIONS[previous]
        ):
            raise ValueError(f"Invalid task transition: {previous} -> {state}.")
        if changes.get("approved") is True and state != "running":
            raise ValueError("Only a running task can receive execution approval.")
        task.update(changes)
        if state == "running" and state != previous:
            task.setdefault("started", _now())
        if state in _TERMINAL:
            task.setdefault("finished", _now())
        if state != "running":
            task["approved"] = False
        return self._save(task)

    def _records(self):
        self._check_root()
        try:
            entries = os.scandir(self.tasks_root)
        except FileNotFoundError:
            return
        with entries:
            for entry in entries:
                if not _TASK_ID.fullmatch(entry.name):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    yield self.get(entry.name)
                except (OSError, ValueError):
                    continue

    def list(self, limit: int = MAX_LIST_TASKS) -> list[dict]:
        """Return at most 1000 valid tasks, newest first, with stable ID ties."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("Task history limit must be a non-negative integer.")
        return heapq.nlargest(
            min(limit, MAX_LIST_TASKS),
            self._records(),
            key=lambda task: (_timestamp(task["created"]), task["id"]),
        )

    def recover_interrupted(self) -> int:
        """Mark unfinished work interrupted at startup; never replay approvals."""
        count = 0
        for task in self._records():
            if task["state"] not in {"running", "queued"}:
                continue
            try:
                self.update(
                    task["id"], state="interrupted",
                    error="Sweep closed before this task finished. Review it and retry to run it again.",
                )
            except (OSError, ValueError):
                continue
            count += 1
        return count
