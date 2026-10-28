"""Versioned local document snapshots and exact, verifiable text references.

This is a source-only shared foundation, not a TradeOps extractor or authority
service. Callers must check task/workspace access before exposing a snapshot.
File reading reuses the desktop's granted, bounded reader; no model is loaded.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

MAX_TEXT = 80_000
MAX_UNITS = 100
MAX_SOURCE_BYTES = 20_000_000
_FORMATS = frozenset({"pdf", "docx", "txt", "md", "log", "xml", "yaml", "yml", "py"})
_HASH = re.compile(r"[0-9a-f]{64}")


def _integer(value: Any, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}.")


def _text(value: Any, name: str, maximum: int, *, nonempty: bool = False) -> None:
    if not isinstance(value, str) or len(value) > maximum or (nonempty and not value.strip()):
        raise ValueError(f"Invalid {name}.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"Invalid Unicode in {name}.") from exc


def _hash(value: Any) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Invalid source or extraction hash.")


def _keys(value: Any, expected: set[str]) -> None:
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError("Missing or unknown document contract fields.")


@dataclass(frozen=True, slots=True)
class TextUnit:
    """Saved text; page is a real PDF page, or None when pagination is unknown."""

    page: int | None
    text: str

    def __post_init__(self) -> None:
        if self.page is not None:
            _integer(self.page, "page", 1, 1000)
        _text(self.text, "unit text", MAX_TEXT)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    file_sha256: str
    extraction_sha256: str
    unit: int
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        _hash(self.file_sha256)
        _hash(self.extraction_sha256)
        _integer(self.unit, "unit index", 0, MAX_UNITS - 1)
        _integer(self.start, "span start", 0, MAX_TEXT - 1)
        _integer(self.end, "span end", 1, MAX_TEXT)
        _text(self.quote, "quote", MAX_TEXT, nonempty=True)
        if self.start >= self.end or len(self.quote) != self.end - self.start:
            raise ValueError("Source span and quote lengths differ.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SourceSpan:
        _keys(value, {"file_sha256", "extraction_sha256", "unit", "start", "end", "quote"})
        return cls(**value)


@dataclass(frozen=True, slots=True)
class DocumentSnapshot:
    schema_version: int
    file_sha256: str
    source_bytes: int
    format: str
    parser: str
    parser_version: str
    units: tuple[TextUnit, ...]
    total_pages: int | None
    truncated: bool

    def __post_init__(self) -> None:
        _integer(self.schema_version, "document schema version", 1, 1)
        _hash(self.file_sha256)
        _integer(self.source_bytes, "source byte count", 0, MAX_SOURCE_BYTES)
        if not isinstance(self.format, str) or self.format not in _FORMATS:
            raise ValueError("Unsupported snapshot format.")
        _text(self.parser, "parser", 100, nonempty=True)
        _text(self.parser_version, "parser version", 100, nonempty=True)
        if type(self.truncated) is not bool:
            raise ValueError("Invalid truncation status.")
        if (type(self.units) is not tuple or len(self.units) > MAX_UNITS
                or any(type(unit) is not TextUnit for unit in self.units)):
            raise ValueError("Document units must be a bounded immutable tuple of TextUnit values.")
        if sum(len(unit.text) for unit in self.units) > MAX_TEXT:
            raise ValueError("Document text exceeds its snapshot limit.")
        if self.format == "pdf":
            _integer(self.total_pages, "total pages", 0, 1000)
            if (len(self.units) > self.total_pages
                    or any(unit.page != i + 1 for i, unit in enumerate(self.units))
                    or (len(self.units) < self.total_pages and not self.truncated)):
                raise ValueError("Inconsistent PDF page coverage.")
        elif (self.total_pages is not None or len(self.units) != 1
              or self.units[0].page is not None):
            raise ValueError("Text and DOCX snapshots cannot invent physical page numbers.")

    @property
    def text_status(self) -> str:
        return "available" if any(unit.text.strip() for unit in self.units) else "no_text"

    @property
    def empty_pages(self) -> tuple[int, ...]:
        """PDF pages without selectable text; possibly blank or requiring OCR."""
        return tuple(unit.page for unit in self.units
                     if unit.page is not None and not unit.text.strip())

    @property
    def review_required(self) -> bool:
        return self.truncated or self.text_status == "no_text" or bool(self.empty_pages)

    @property
    def extraction_sha256(self) -> str:
        # Hash the versioned extraction, not its path/title or a model summary.
        raw = json.dumps(asdict(self), sort_keys=True, ensure_ascii=True,
                         separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["units"] = [asdict(unit) for unit in self.units]
        result["extraction_sha256"] = self.extraction_sha256
        return result

    @classmethod
    def from_dict(cls, value: dict) -> DocumentSnapshot:
        _keys(value, {"schema_version", "file_sha256", "source_bytes", "format", "parser",
                      "parser_version", "units", "total_pages", "truncated", "extraction_sha256"})
        if not isinstance(value["units"], list) or len(value["units"]) > MAX_UNITS:
            raise ValueError("Invalid serialized document units.")
        for unit in value["units"]:
            _keys(unit, {"page", "text"})
        data = {key: item for key, item in value.items() if key not in {"units", "extraction_sha256"}}
        snapshot = cls(**data, units=tuple(TextUnit(**unit) for unit in value["units"]))
        if snapshot.extraction_sha256 != value["extraction_sha256"]:
            raise ValueError("The extraction does not match its saved hash.")
        return snapshot

    def reference(self, unit: int, start: int, end: int) -> SourceSpan:
        _integer(unit, "unit index", 0, len(self.units) - 1)
        _integer(start, "span start", 0, len(self.units[unit].text) - 1)
        _integer(end, "span end", start + 1, len(self.units[unit].text))
        return SourceSpan(self.file_sha256, self.extraction_sha256, unit, start, end,
                          self.units[unit].text[start:end])

    def resolve(self, reference: SourceSpan) -> TextUnit:
        """Verify exact saved evidence; never open a path or grant access."""
        if type(reference) is not SourceSpan:
            raise ValueError("Expected a validated source reference.")
        if (reference.file_sha256 != self.file_sha256
                or reference.extraction_sha256 != self.extraction_sha256
                or reference.unit >= len(self.units)):
            raise ValueError("Reference belongs to a different document extraction.")
        unit = self.units[reference.unit]
        if reference.end > len(unit.text) or unit.text[reference.start:reference.end] != reference.quote:
            raise ValueError("Reference quote does not match the saved source range.")
        return unit


def inspect_source(path: str | Path, granted_paths: list[str]) -> DocumentSnapshot:
    """Read one explicitly granted local document with the existing bounded reader.

    No model calls, network access, persistence, permission prompting or UI work.
    Physical boxes, table structure and extracted business fields are not supplied.
    """
    from sweep.desktop.data_tools import inspect_document

    result = inspect_document(str(path), granted_paths)
    kind = result["format"]
    if kind == "pdf":
        units = tuple(TextUnit(item["page"], item["text"]) for item in result["pages"])
        parser, parser_version = "sweep.pdf-text/pypdf", "1/" + version("pypdf")
        total_pages = result["page_count"]
    else:
        units = (TextUnit(None, result["text"]),)
        parser = "sweep.docx-paragraphs" if kind == "docx" else "sweep.plain-text"
        parser_version, total_pages = "1", None
    return DocumentSnapshot(1, result["sha256"], result["bytes"], kind, parser,
                            parser_version, units, total_pages, result["truncated"])
