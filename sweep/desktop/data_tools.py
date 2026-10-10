"""Bounded, entirely local document and data tools for desktop attachments.

The desktop grants individual files. Original files are never modified; data
operations produce separate task artifacts. This small stdlib adapter shares the
profile vocabulary of companion.tools.data without loading its optional numeric
stack into the desktop. User text is parsed into fixed operations, never code/SQL.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import stat
import time
import uuid
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

MAX_BYTES = 20_000_000
MAX_ROWS = 20_000
MAX_COLUMNS = 128
MAX_TEXT = 80_000
MAX_CELL = 40_000
MAX_SECONDS = 15
MAX_PREVIEW_BYTES = 120_000
MAX_PROFILE_BYTES = 550_000
MAX_DISPLAY_CHARS = 120
DATA_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".sqlite", ".sqlite3", ".db"})
DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx", ".txt", ".md", ".log", ".xml", ".yaml", ".yml", ".py"})
_DEDUPLICATE = r"deduplicat\w*|(?:remove|drop) duplicate(?:s| rows)"


def _read(path: str, granted_paths: list[str]) -> tuple[Path, bytes, dict]:
    supplied = Path(path).expanduser()
    if str(supplied).startswith(("\\\\", "//")):
        raise ValueError("Choose a file stored on this computer, not a network share.")
    selected = supplied.resolve(strict=True)
    if str(selected).startswith(("\\\\", "//")):
        raise ValueError("Choose a file stored on this computer, not a network share.")
    allowed = {Path(item).expanduser().resolve() for item in granted_paths}
    if selected not in allowed:
        raise PermissionError("Attach this file before asking Sweep to read it.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(selected, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Attach a regular local file.")
        if info.st_size > MAX_BYTES:
            raise ValueError("Local document and data tasks support files up to 20 MB.")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Local document and data tasks support files up to 20 MB.")
    return selected, raw, {"title": selected.name, "path": str(selected), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "modified": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(),
        "execution": "local", "source_unchanged": True}


def _deadline(start: float) -> None:
    if time.monotonic() - start > MAX_SECONDS:
        raise ValueError("This file exceeded the local processing time limit; use a smaller extract.")


def _decode(raw: bytes) -> str:
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        return raw.decode(encoding)
    except UnicodeError as exc:
        raise ValueError("Save this text file as UTF-8 or UTF-16 and attach it again.") from exc


class _DocumentTreeBuilder(ElementTree.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError("Document XML contains unsupported entities.")


def _xml(raw: bytes):
    if len(raw) > MAX_BYTES or re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.I):
        raise ValueError("Document XML contains unsupported entities or exceeds the size limit.")
    try:
        return ElementTree.fromstring(raw, parser=ElementTree.XMLParser(target=_DocumentTreeBuilder()))
    except ElementTree.ParseError as exc:
        raise ValueError("The document contains damaged XML.") from exc


def _pdf(raw: bytes, start: float) -> tuple[str, bool, dict]:
    try:
        from pypdf import PdfReader, apply_configuration
        from pypdf.errors import LimitReachedError
    except ImportError as exc:
        raise ValueError("PDF reading is unavailable in this installation. Install the current Sweep desktop build.") from exc
    pages = []
    truncated = False
    try:
        # Context-local bounds cover decompression before it allocates a huge
        # page, including compressed streams referenced by fonts or XObjects.
        with apply_configuration(
            maximum_declared_stream_length=MAX_BYTES,
            array_based_stream_maximum_output_length=MAX_BYTES,
            zlib_maximum_output_length=MAX_BYTES,
            lzw_maximum_output_length=MAX_BYTES,
            run_length_maximum_output_length=MAX_BYTES,
            jbig2_maximum_output_length=MAX_BYTES,
            image_maximum_buffer_size=MAX_BYTES,
            page_tree_maximum_entries=1000,
            page_tree_maximum_depth=30,
            xform_maximum_invocations_per_extraction=100,
            jbig2dec_binary=None,
        ):
            reader = PdfReader(io.BytesIO(raw), strict=False)
            if reader.is_encrypted:
                raise ValueError("Unlock this PDF locally, save a copy, and attach the unlocked copy.")
            count = len(reader.pages)
            used = 0
            content_bytes = 0
            for index, page in enumerate(reader.pages):
                _deadline(start)
                if index >= 100 or used >= MAX_TEXT:
                    truncated = True
                    break
                content = page.get_contents()
                if content is not None:
                    content_bytes += len(content.get_data())
                    if content_bytes > MAX_BYTES:
                        raise ValueError("The PDF exceeds the text extraction size limit.")
                extracted = page.extract_text() or ""
                remaining = MAX_TEXT - used
                value = extracted[:remaining]
                truncated = truncated or len(extracted) > remaining
                pages.append({"page": index + 1, "text": value})
                used += len(value)
            text = "\n\n".join(f"Page {item['page']}\n{item['text']}" for item in pages)
    except LimitReachedError as exc:
        raise ValueError("The PDF exceeds local processing limits; attach a smaller extract.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be read. It may be damaged or use unsupported protection.") from exc
    return text, truncated, {"pages": pages, "page_count": count}


def inspect_document(path: str, granted_paths: list[str]) -> dict:
    """Extract bounded plain text without running document macros or uploads."""
    selected, raw, result = _read(path, granted_paths)
    suffix = selected.suffix.casefold()
    start = time.monotonic()
    truncated = False
    pages = []
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if len(archive.infolist()) > 2000:
                    raise ValueError("The document contains too many archive entries.")
                part = archive.getinfo("word/document.xml")
                if part.file_size > MAX_BYTES or part.flag_bits & 1:
                    raise ValueError("The document is too large or encrypted.")
                with archive.open(part) as stream:
                    tree = _xml(stream.read(MAX_BYTES + 1))
                namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                paragraphs = []
                used = 0
                for paragraph in tree.iter(namespace + "p"):
                    _deadline(start)
                    value = "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
                    paragraphs.append(value)
                    used += len(value) + 1
                    if used > MAX_TEXT:
                        truncated = True
                        break
                text = "\n".join(paragraphs)
                result["paragraphs_extracted"] = len(paragraphs)
        except (zipfile.BadZipFile, KeyError, RuntimeError) as exc:
            raise ValueError("This is not a readable DOCX document.") from exc
    elif suffix == ".pdf":
        text, truncated, details = _pdf(raw, start)
        result.update(details)
        pages = details["pages"]
    elif suffix in DOCUMENT_SUFFIXES:
        text = _decode(raw)
    else:
        raise ValueError("Attach a PDF, DOCX, or plain-text document for text inspection.")
    result.update(text=text[:MAX_TEXT], truncated=truncated or len(text) > MAX_TEXT, format=suffix.lstrip("."))
    result["message"] = f"Read {selected.name} locally. {len(result['text']):,} characters extracted."
    if result["truncated"]:
        result["message"] += " This is a limited text preview, not the complete document."
    if suffix == ".pdf" and not any(item["text"].strip() for item in pages):
        result["message"] += " No selectable text was found; this PDF may require page OCR."
    return result


def _cell(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Data contains non-finite numbers; replace them with null before processing.")
        if isinstance(value, str) and len(value) > MAX_CELL:
            raise ValueError("A data cell exceeds the 40,000-character limit.")
        return value
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded) > MAX_CELL:
        raise ValueError("A nested data value exceeds the cell size limit.")
    return value


def _display_cell(value):
    """Render structured cells without changing their values in JSON exports."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return value


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains a duplicate object key {key!r}; resolve it before processing.")
        result[key] = value
    return result


def _columns(values) -> list[str]:
    columns = [str(value) for value in values]
    if len(columns) > MAX_COLUMNS:
        raise ValueError("This dataset exceeds 128 columns. Attach a smaller extract.")
    if any(not name.strip() or len(name) > 250 for name in columns) or len(set(columns)) != len(columns):
        raise ValueError("Each column needs a unique, non-empty name of at most 250 characters.")
    return columns


def _records(values: list) -> tuple[list[str], list[dict]]:
    if len(values) > MAX_ROWS:
        raise ValueError("Data transformations support up to 20,000 rows; attach a smaller extract.")
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Use a JSON array of objects or one object per JSONL line.")
    names = list(dict.fromkeys(str(key) for value in values for key in value))
    columns = _columns(names)
    return columns, [{name: _cell(value.get(name)) for name in columns} for value in values]


def _sqlite(raw: bytes, table: str | None, start: float) -> tuple[list[str], list[dict], dict]:
    if not raw.startswith(b"SQLite format 3\0"):
        raise ValueError("This file is not a SQLite database.")
    if raw[18:20] != b"\x01\x01":
        raise ValueError("Export a standalone SQLite backup first; live WAL databases may have uncheckpointed data.")
    # Deserialize the selected bytes into an isolated in-memory database. SQLite
    # cannot read a journal, attached file, extension, or a substituted source.
    connection = sqlite3.connect(":memory:", timeout=1)
    try:
        # A small database can contain virtual/generated expressions that allocate
        # huge values. Bound SQLite before parsing its schema or executing reads.
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_BYTES)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, MAX_COLUMNS)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        connection.deserialize(raw)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() - start > MAX_SECONDS), 1000)
        schema = connection.execute("SELECT name, sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 101").fetchall()
        if len(schema) > 100:
            raise ValueError("This database exceeds the 100-table inspection limit.")
        table_types = {row[1]: row[2] for row in connection.execute("PRAGMA table_list")}
        tables = [{"name": name, "available": table_types.get(name) == "table"} for name, _statement in schema
                  if table_types.get(name) != "shadow"]
        choices = [item["name"] for item in tables if item["available"]]
        if table is None and len(choices) != 1:
            return [], [], {"tables": tables, "needs_table": bool(choices), "no_tables": not choices}
        table = table or (choices[0] if choices else None)
        if table not in choices:
            raise ValueError("Choose an ordinary table shown in this database; views and virtual tables are not executed.")
        identifier = '"' + table.replace('"', '""') + '"'
        if any(column[6] for column in connection.execute(f"PRAGMA table_xinfo({identifier})")):
            raise ValueError("This table contains generated columns. Export ordinary stored values before inspecting it.")
        cursor = connection.execute(f"SELECT * FROM {identifier} LIMIT {MAX_ROWS + 1}")
        columns = _columns(item[0] for item in cursor.description)
        values = cursor.fetchall()
        if len(values) > MAX_ROWS:
            raise ValueError("This table exceeds 20,000 rows; export a smaller extract first.")
        rows = [{name: _cell(value) if not isinstance(value, bytes) else f"[binary: {len(value)} bytes]"
                 for name, value in zip(columns, row, strict=True)} for row in values]
        return columns, rows, {"tables": tables, "table": table,
                               "binary_cells": any(isinstance(value, bytes) for row in values for value in row)}
    except sqlite3.DatabaseError as exc:
        raise ValueError("This SQLite database could not be read within local safety limits.") from exc
    finally:
        connection.close()


def _load(path: str, granted_paths: list[str], table: str | None = None):
    selected, raw, result = _read(path, granted_paths)
    suffix = selected.suffix.casefold()
    start = time.monotonic()
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        columns, rows, details = _sqlite(raw, table, start)
        result.update(details)
    elif suffix in {".csv", ".tsv"}:
        text = _decode(raw)
        reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t" if suffix == ".tsv" else ",", strict=True)
        try:
            columns = _columns(next(reader, []))
            rows = []
            for values in reader:
                _deadline(start)
                if not values:
                    continue
                if len(rows) >= MAX_ROWS:
                    raise ValueError("Data transformations support up to 20,000 rows; attach a smaller extract.")
                if len(values) != len(columns):
                    raise ValueError(f"Row {reader.line_num} has {len(values)} cells; expected {len(columns)}. No data was changed.")
                rows.append({name: _cell(value) for name, value in zip(columns, values, strict=True)})
        except csv.Error as exc:
            raise ValueError("The delimited file has a malformed or oversized field.") from exc
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        try:
            text = _decode(raw)
            if suffix == ".json":
                value = json.loads(text, object_pairs_hook=_json_object)
                if isinstance(value, dict):
                    value = value.get("rows", [value])
                if not isinstance(value, list):
                    raise ValueError("Use a JSON array of records or an object containing a rows array.")
            else:
                value = []
                for line in text.splitlines():
                    if line.strip():
                        if len(value) >= MAX_ROWS:
                            raise ValueError("Data transformations support up to 20,000 rows; attach a smaller extract.")
                        value.append(json.loads(line, object_pairs_hook=_json_object))
            columns, rows = _records(value)
        except (json.JSONDecodeError, RecursionError, OverflowError) as exc:
            raise ValueError("The JSON data is malformed or nested too deeply.") from exc
    else:
        raise ValueError("Data tools support CSV, TSV, JSON, JSONL and standalone SQLite files.")
    result["format"] = suffix.lstrip(".")
    _deadline(start)
    return columns, rows, result


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _key(row: dict, columns: list[str]) -> str:
    return json.dumps([row.get(name) for name in columns], ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _profile(columns: list[str], rows: list[dict]) -> dict:
    start = time.monotonic()
    profile = {"rows": len(rows), "columns": len(columns), "column_names": columns,
        "missing": {}, "numeric_columns": [], "numeric_summary": {}, "categorical_summary": {}, "dtypes": {}}
    profile["duplicate_rows"] = len(rows) - len({_key(row, columns) for row in rows})
    for name in columns:
        _deadline(start)
        values = [row.get(name) for row in rows]
        nonempty = [value for value in values if value is not None and str(value).strip()]
        profile["missing"][name] = len(values) - len(nonempty)
        numbers = [_number(value) for value in nonempty]
        if numbers and all(value is not None for value in numbers):
            profile["dtypes"][name] = "number"
            profile["numeric_columns"].append(name)
            # Scaling avoids overflow for valid finite values near float's limit.
            scale = max(abs(value) for value in numbers)
            mean = math.fsum(value / scale for value in numbers) / len(numbers) * scale if scale else 0.0
            ordered = sorted(numbers)
            middle = len(ordered) // 2
            median = ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2
            profile["numeric_summary"][name] = {"count": len(numbers), "min": min(numbers), "max": max(numbers),
                "mean": mean, "median": median}
        else:
            profile["dtypes"][name] = "text" if nonempty else "empty"
            counts = Counter(str(value) for value in nonempty)
            profile["categorical_summary"][name] = [{"value": value[:250], "count": count} for value, count in counts.most_common(5)]
    return profile


def _describe(result: dict, columns: list[str], rows: list[dict]) -> dict:
    profile = _profile(columns, rows)
    missing_cells = sum(profile["missing"].values())

    def encoded_size(value):
        # ASCII escaping is the worst-case worker JSON representation, including
        # astral Unicode and the default separators used by event serialization.
        return len(json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8"))

    def display(value):
        value = _display_cell(value)
        if isinstance(value, str) and len(value) > MAX_DISPLAY_CHARS:
            return value[:MAX_DISPLAY_CHARS - 1] + "…", True
        return value, False

    # Preserve complete column names. Limit visible columns when a wide schema
    # or large Unicode labels would leave no space for even one preview row.
    visible_columns = []
    column_cost = 0
    for name in columns:
        cell_cost = max((encoded_size(display(row.get(name))[0]) for row in rows[:30]), default=4)
        cost = encoded_size(name) + cell_cost + 4
        if column_cost + cost > MAX_PREVIEW_BYTES // 2:
            break
        visible_columns.append(name)
        column_cost += cost
    preview_rows = []
    clipped_cells = 0
    preview_size = encoded_size({"columns": visible_columns, "rows": []})
    for row in rows[:30]:
        cells = [display(row.get(name)) for name in visible_columns]
        values = [value for value, _clipped in cells]
        cost = encoded_size(values) + (2 if preview_rows else 0)
        if preview_size + cost > MAX_PREVIEW_BYTES:
            break
        preview_rows.append(values)
        clipped_cells += sum(clipped for _value, clipped in cells)
        preview_size += cost

    # Full names recur across profile dictionaries, and category strings can
    # make those details much larger than the source. Retain complete details
    # only while they fit. Totals always describe the entire data, not the view.
    bounded = {"rows": profile["rows"], "columns": profile["columns"],
               "duplicate_rows": profile["duplicate_rows"], "missing_cells": missing_cells,
               "column_names": [], "missing": {}, "numeric_columns": [],
               "numeric_summary": {}, "categorical_summary": {}, "dtypes": {}}
    maps = ("missing", "numeric_summary", "categorical_summary", "dtypes")
    omitted = 0
    for name in columns:
        bounded["column_names"].append(name)
        numeric = name in profile["numeric_columns"]
        if numeric:
            bounded["numeric_columns"].append(name)
        for key in maps:
            if name in profile[key]:
                bounded[key][name] = profile[key][name]
        if encoded_size(bounded) > MAX_PROFILE_BYTES - 200:
            omitted += 1
            bounded["column_names"].pop()
            if numeric:
                bounded["numeric_columns"].pop()
            for key in maps:
                bounded[key].pop(name, None)
    bounded.update(truncated=bool(omitted), omitted_column_details=omitted)
    preview = {"truncated": bool(clipped_cells or len(preview_rows) < len(rows) or len(visible_columns) < len(columns)),
               "rows_shown": len(preview_rows), "columns_shown": len(visible_columns),
               "cells_clipped": clipped_cells}
    result.update(columns=visible_columns, rows=preview_rows, profile=bounded, preview=preview)
    result["message"] = (f"{result['title']}: {len(rows):,} rows, {len(columns)} columns. "
        f"{profile['duplicate_rows']:,} duplicate rows and {missing_cells:,} missing cells. "
        "Processed locally; the original file is unchanged.")
    if preview["truncated"] or omitted:
        result["display_notice"] = (f"Display limited to {len(preview_rows)} rows and {len(visible_columns)} columns; "
            f"{clipped_cells} long display cells shortened and {omitted} column profiles omitted. "
            "Full values are preserved in data exports.")
        result["message"] += " " + result["display_notice"]
    if result.get("needs_table"):
        result["message"] = "Choose a table to inspect: " + ", ".join(item["name"] for item in result["tables"] if item["available"]) + '. Say “inspect table \'name\'”.'
    elif result.get("no_tables"):
        result["message"] = "This database contains no ordinary tables to inspect. Views and virtual tables are not executed."
    return result


def inspect_data(path: str, granted_paths: list[str], *, table: str | None = None) -> dict:
    columns, rows, result = _load(path, granted_paths, table)
    return _describe(result, columns, rows)


def _unquote(value: str) -> str:
    value = value.strip()
    return value[1:-1] if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'`" else value


def parse_data_request(text: str) -> dict:
    """Parse common data requests into a finite, reviewable operation description.

    Filters deliberately accept one comparison rather than a SQL/Python fragment.
    Quoted names support spaces. Unrecognized transformation requests fail clearly.
    """
    if not isinstance(text, str) or len(text) > 8_000:
        raise ValueError("Use a short description of the data operation you want.")
    text = text.strip().rstrip(".!?")
    lowered = text.casefold()
    result = {"operation": "inspect", "output_format": None, "table": None}
    table = re.search(r"\btable\s+([\"'`])(.+?)\1", text, re.I)
    if table:
        result["table"] = table[2]
        text = text[:table.start()] + text[table.end():]
        lowered = text.casefold()
    conversion = re.search(r"\b(?:convert|export|save)\b.*?\b(?:to|as)\s+(?:a\s+)?(csv|tsv|jsonl|ndjson|json)(?:\s+file)?\b", text, re.I)
    if conversion:
        if text[conversion.end():].strip().casefold() not in {"", "please"}:
            raise ValueError("Put the export format last, for example: filter where age >= 18 and export to JSON.")
        conversion_words = text[conversion.start():conversion.end()]
        if re.search(r"\b(?:where|filter|deduplicat\w*|clean|duplicates|join|merge|delete|update|sql|sort|drop)\b", conversion_words, re.I):
            raise ValueError("Describe one transformation first, followed by the export format.")
        result["output_format"] = conversion[1].casefold().replace("ndjson", "jsonl")
        result["operation"] = "convert"
        text = re.sub(r"\b(?:and|then)\s*$", "", text[:conversion.start()].strip(), flags=re.I).strip()
        lowered = text.casefold()
    elif re.search(r"\b(?:convert|export|save)\b", lowered):
        raise ValueError("Choose CSV, TSV, JSON, or JSONL as the export format, for example: convert to JSON.")
    transformations = re.findall(rf"\b(?:{_DEDUPLICATE}|clean(?: up)?|filter|keep rows)\b", lowered)
    if len(transformations) > 1:
        raise ValueError("Apply one data transformation at a time. An export format can follow the transformation.")
    unsupported = r"\b(?:join|merge|delete|update|sql|normalize|normalise|aggregate|group|sort|replace|fill|impute|rename|drop(?! duplicate(?:s| rows)\b))\b"
    if re.search(rf"\b(?:{_DEDUPLICATE})\b", lowered):
        if re.search(unsupported, lowered):
            raise ValueError("Apply one supported transformation at a time: clean, remove duplicates, filter, or convert.")
        result["operation"] = "deduplicate"
        keys = re.search(r"\b(?:based on|by|using|on)\s+(.+)$", text, re.I)
        if keys:
            result["columns"] = [_unquote(value) for value in keys[1].split(",")]
    elif re.search(r"\bclean(?: up)?\b", lowered):
        if re.search(unsupported, lowered):
            raise ValueError("Cleaning trims text, removes empty rows and removes exact duplicates. Other changes must be requested separately.")
        result["operation"] = "clean"
    elif re.search(r"\b(?:filter|keep rows|where)\b", lowered):
        operators = (r">=|<=|!=|==|=|>|<|\b(?:is\s+)?(?:greater than or equal to|less than or equal to|"
                     r"at least|at most|greater than|less than)\b|\bdoes not equal\b|\bis not\b|\bequals\b|\bcontains\b|\bis\b")
        condition = re.search(rf"\b(?:where|filter(?:\s+rows)?(?!.*\bwhere\b))\s+(.+?)\s*({operators})\s*(.+)$", text, re.I)
        if not condition:
            raise ValueError('Use one comparison, such as “filter where age >= 18” or “filter where city = Kathmandu”.')
        value = condition[3].strip()
        quoted_value = len(value) > 1 and value[0] == value[-1] and value[0] in "\"'`"
        if not quoted_value and re.search(r"\s+(?:and|or)\s+|;|\n", value, re.I):
            raise ValueError("Use one filter comparison at a time; compound expressions are not executed.")
        operator = re.sub(r"\s+", " ", condition[2].casefold())
        numerical_operator = operator.removeprefix("is ")
        operator = {"greater than or equal to": ">=", "less than or equal to": "<=", "at least": ">=", "at most": "<=",
                    "greater than": ">", "less than": "<", "does not equal": "!=", "is not": "!=", "equals": "="}.get(numerical_operator, operator)
        if operator == "is not":
            operator = "!="
        result.update(operation="filter", column=_unquote(condition[1]), operator=operator, value=_unquote(value))
    elif re.search(unsupported, lowered):
        raise ValueError("That data transformation is not available yet. You can inspect, clean, remove duplicates, filter one column, or convert CSV/TSV/JSON/JSONL.")
    elif not conversion and not re.search(r"\b(?:inspect|analy[sz]e|summary|summari[sz]e|describe|preview|statistics|stats|read|what|show|explain|data|dataset|file)\b", lowered) and lowered:
        raise ValueError("Try “summarize this data”, “remove duplicates”, “filter where age >= 18”, or “convert to JSON”.")
    return result


def _column(name: str, columns: list[str]) -> str:
    if name in columns:
        return name
    matches = [item for item in columns if item.casefold() == name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Column {name!r} was not found. Available columns: {', '.join(columns)}")


def _filter(rows: list[dict], column: str, operator: str, value: str) -> list[dict]:
    target_number = _number(value)
    if operator in {">", ">=", "<", "<="} and target_number is None:
        raise ValueError("Greater/less-than filters need a numeric comparison value.")
    def matches(row):
        cell = row.get(column)
        if operator == "contains":
            return cell is not None and value.casefold() in str(cell).casefold()
        number = _number(cell)
        if operator in {">", ">=", "<", "<="}:
            if number is None:
                return False
            return {">": number > target_number, ">=": number >= target_number,
                    "<": number < target_number, "<=": number <= target_number}[operator]
        equal = number == target_number if number is not None and target_number is not None else str(cell).casefold() == value.casefold()
        return not equal if operator == "!=" else equal
    return [row for row in rows if matches(row)]


def _artifact(artifact_dir: str | Path, columns: list[str], rows: list[dict], output_format: str) -> dict:
    original = Path(os.path.abspath(artifact_dir))
    if str(original).startswith(("\\\\", "//")):
        raise ValueError("Store transformed data in local task storage.")
    if original.resolve() != original or original.is_symlink() or original.is_junction():
        raise ValueError("The artifact directory must not redirect outside task storage.")
    original.mkdir(parents=True, exist_ok=True)
    if original.resolve() != original:
        raise ValueError("The artifact directory changed while preparing the export.")
    if output_format in {"csv", "tsv"}:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter="\t" if output_format == "tsv" else ",", lineterminator="\n")
        writer.writerow(columns)
        # Data is exported faithfully. UI opening should use a text/data viewer,
        # not automatically execute a spreadsheet application or its formulas.
        writer.writerows([[_display_cell(row.get(name)) for name in columns] for row in rows])
        encoded = buffer.getvalue().encode("utf-8-sig")
    elif output_format == "jsonl":
        encoded = ("\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows) + ("\n" if rows else "")).encode("utf-8")
    else:
        encoded = json.dumps(rows, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("The transformed output exceeds 20 MB; use a smaller dataset.")
    target = original / f"data-{uuid.uuid4().hex[:12]}.{output_format}"
    with target.open("xb") as stream:
        stream.write(encoded)
    return {"name": target.name, "path": str(target), "bytes": len(encoded), "format": output_format,
            "sha256": hashlib.sha256(encoded).hexdigest()}


def process_data(path: str, granted_paths: list[str], text: str, artifact_dir: str | Path) -> dict:
    """Apply a deterministic data request and return a profile plus any new artifact."""
    plan = parse_data_request(text)
    columns, rows, result = _load(path, granted_paths, plan.get("table"))
    before = len(rows)
    if result.get("needs_table") or result.get("no_tables"):
        return _describe(result, columns, rows)
    operation = plan["operation"]
    if operation != "inspect" and result.get("binary_cells"):
        raise ValueError("This table contains binary cells. Export them explicitly before transforming its data; preview labels are not the original bytes.")
    if operation == "filter":
        rows = _filter(rows, _column(plan["column"], columns), plan["operator"], plan["value"])
    elif operation in {"clean", "deduplicate"}:
        if operation == "clean":
            rows = [{name: value.strip() if isinstance(value, str) else value for name, value in row.items()} for row in rows]
            rows = [row for row in rows if any(value is not None and str(value).strip() for value in row.values())]
        keys = [_column(name, columns) for name in plan.get("columns", columns)]
        seen = set()
        unique = []
        for row in rows:
            key = _key(row, keys)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        rows = unique
    result = _describe(result, columns, rows)
    result.update(operation=operation, input_rows=before, output_rows=len(rows), operation_details=plan)
    if operation != "inspect":
        output_format = plan["output_format"] or (result["format"] if result["format"] in {"csv", "tsv", "json", "jsonl"} else "csv")
        result["artifacts"] = [_artifact(artifact_dir, columns, rows, output_format)]
        action = {"filter": "Filtered", "clean": "Cleaned", "deduplicate": "Removed duplicates from", "convert": "Converted"}[operation]
        result["message"] = f"{action} {result['title']}. {before:,} input rows → {len(rows):,} output rows. Saved a new {output_format.upper()} file; the original is unchanged."
        if result.get("display_notice"):
            result["message"] += " " + result["display_notice"]
    return result
