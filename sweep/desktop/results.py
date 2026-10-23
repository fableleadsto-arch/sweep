"""One bounded result contract for workers, history, exports and follow-ups.

Result files are observations, never execution authority. TaskStore validates
their parent directory and task state; this module validates the file and JSON.
"""
from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from pathlib import Path

MAX_RESULT_BYTES = 2_000_000
MAX_RESULT_DEPTH = 64
MAX_RESULT_NODES = 100_000


def _redirected(path: Path) -> bool:
    info = path.lstat()
    return (stat.S_ISLNK(info.st_mode) or path.is_junction()
            or bool(getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def validate_result(result: dict) -> None:
    """Bound structure and validate fields consumed by the native views.

    Capability-specific evidence remains extensible. Unknown fields must still
    contain finite JSON values; nothing is silently dropped or coerced.
    """
    if not isinstance(result, dict):
        raise ValueError("The saved result must contain a result object.")
    pending = [(result, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > MAX_RESULT_DEPTH or count > MAX_RESULT_NODES:
            raise ValueError("The result contains too much nested data.")
        if isinstance(value, dict):
            if len(value) + count + len(pending) > MAX_RESULT_NODES:
                raise ValueError("The result contains too many values.")
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Result field names must be text.")
                pending.append((child, depth + 1))
        elif isinstance(value, list):
            if len(value) + count + len(pending) > MAX_RESULT_NODES:
                raise ValueError("The result contains too many values.")
            pending.extend((child, depth + 1) for child in value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Result numbers must be finite.")
        elif value is not None and not isinstance(value, (str, int, bool)):
            raise ValueError("The result must contain JSON values only.")
    for key in ("title", "message", "text", "url", "path", "modified", "image"):
        if key in result and not isinstance(result[key], str):
            raise ValueError(f"Invalid result field: {key} must be text.")
    for key in ("exif", "location", "analysis"):
        if key in result and not isinstance(result[key], dict):
            raise ValueError(f"Invalid result field: {key} must be an object.")
    for key in ("location", "analysis"):
        if "status" in result.get(key, {}) and not isinstance(result[key]["status"], str):
            raise ValueError(f"Invalid result field: {key}.status must be text.")
    for key in ("hits", "sources", "evidence", "segments", "artifacts", "context_sources"):
        if key in result and (not isinstance(result[key], list)
                              or any(not isinstance(item, dict) for item in result[key])):
            raise ValueError(f"Invalid result field: {key} must be a list of objects.")
    for key in ("hits", "sources", "evidence"):
        for item in result.get(key, []):
            for field in ("title", "url", "snippet", "excerpt", "source_url", "source_title"):
                if field in item and not isinstance(item[field], str):
                    raise ValueError(f"Invalid result field: {key}.{field} must be text.")
    for item in result.get("artifacts", []):
        for field in ("name", "title", "format", "type", "path"):
            if field in item and not isinstance(item[field], str):
                raise ValueError(f"Invalid artifact field: {field} must be text.")
    columns, rows = result.get("columns", []), result.get("rows", [])
    if not isinstance(columns, list) or any(not isinstance(item, str) for item in columns):
        raise ValueError("Result columns must be a list of names.")
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        raise ValueError("Result rows must be a list of rows.")
    if "bytes" in result and (isinstance(result["bytes"], bool)
                              or not isinstance(result["bytes"], int) or result["bytes"] < 0):
        raise ValueError("Result file size must be a nonnegative integer.")
    if result.get("location", {}).get("status") == "embedded_gps":
        if not all(isinstance(result["location"].get(key), (int, float))
                   and not isinstance(result["location"][key], bool) for key in ("latitude", "longitude")):
            raise ValueError("Embedded coordinates must contain latitude and longitude.")
    for item in result.get("context_sources", []):
        if any(not isinstance(item.get(key), str) for key in ("label", "filename", "scope", "excerpt")):
            raise ValueError("A context source is missing its label or evidence.")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate result fields are ambiguous.")
        result[key] = value
    return result


def load_result(path: Path) -> dict:
    if _redirected(path):
        raise ValueError("Saved results must not use redirected files.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RESULT_BYTES:
            raise ValueError("Saved results must be regular files under 2 MB.")
        raw = stream.read(MAX_RESULT_BYTES + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
        if (_redirected(path) or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)
                or (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or (info.st_size, info.st_mtime_ns) != (current.st_size, current.st_mtime_ns)):
            raise ValueError("The saved result changed while it was being read.")
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError("The saved result exceeds 2 MB.")
    try:
        result = json.loads(raw, object_pairs_hook=_object)
        validate_result(result)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("This saved result is damaged. Retry its original task.") from exc
    return result


def save_result(path: Path, result: dict) -> None:
    """Validate and atomically persist exact UTF-8 bytes before completion.

    A retry has its own task directory; never replace an earlier result. Temporary
    files are removed even if serialization, flushing or replacement fails.
    """
    validate_result(result)
    if path.exists() or path.is_symlink():
        raise ValueError("This task already has a saved result. Retry as a new task.")
    descriptor, name = tempfile.mkstemp(prefix="result.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            size = 0
            encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for chunk in encoder.iterencode(result):
                raw = chunk.encode("utf-8")
                size += len(raw)
                if size > MAX_RESULT_BYTES:
                    raise ValueError("The result exceeds 2 MB. Request a smaller output.")
                stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
