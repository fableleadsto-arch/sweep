"""Dataset registry, cache layout and load dispatch.

Every dataset resolves to a uniform shape::

    {
      "name": str, "group": str, "source": str, "cached": bool,
      "columns": list[str], "rows": list[dict], "row_count": int,
      "path": str,                       # cache path (corpus/molecular)
      "notes": str,
    }

Reasoning rows are normalized to a common schema so the QA machinery can
prompt any of them: ``prompt`` (question text) and ``answer`` (expected
reference). Extra original fields are kept in the row for inspection.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_DIR = Path.home() / ".sweep" / "datasets"

# name -> (load callable, group, description, source label)
# Load callables return ("rows-or-path", meta dict); see individual modules.
_ENTRIES: dict[str, dict[str, Any]] = {}


def _register(name, group, description, source, loader: Callable) -> None:
    _ENTRIES[name] = {
        "name": name,
        "group": group,
        "description": description,
        "source": source,
        "loader": loader,
    }


def register_all() -> None:
    if _ENTRIES:
        return
    from .reasoning import load_reasoning
    from .corpus import load_corpus
    from .molecular import load_molecular

    _register("logiqa", "reasoning", "Deductive logical reasoning, 8,678 QA pairs", "lgw863/LogiQA", load_reasoning)
    _register("gsm8k", "reasoning", "Step-by-step grade-school math (8.5k problems)", "gsm8k", load_reasoning)
    _register("commonsense_qa", "reasoning", "Commonsense reasoning, 12k questions", "tau/commonsense_qa", load_reasoning)
    _register("babi", "reasoning", "20 synthetic reasoning tasks (FB bAbI)", "facebook/babi", load_reasoning)
    _register("mmlu", "reasoning", "General knowledge across 57 subjects", "cais/mmlu", load_reasoning)
    _register("cot_collection", "reasoning", "Chain-of-thought reasoning pairs", "kaist-ai/CoT-Collection", load_reasoning)
    _register("arc", "reasoning", "ARC Challenge science QA (cached)", "ai2_arc", load_reasoning)
    _register("super_glue", "reasoning", "BoolQ subset of SuperGLUE (cached)", "super_glue", load_reasoning)
    _register("folio", "reasoning", "FOLIO natural-language logic (cached)", "tasksource/folio", load_reasoning)
    _register("ruletaker", "reasoning", "RuleTaker contrapositive reasoning (cached)", "hitachi-nlp/ruletaker", load_reasoning)

    _register("pile", "corpus", "825 GB multi-domain web text (streamed)", "EleutherAI/pile", load_corpus)
    _register("fineweb", "corpus", "HF clean web corpus (streamed)", "HuggingFaceFW/fineweb", load_corpus)
    _register("wikipedia", "corpus", "Wikipedia reference KB (streamed)", "wikipedia", load_corpus)

    _register("pdb", "molecular", "Protein Data Bank structures (biotite)", "PDB via rcsb.org", load_molecular)
    _register("proteinnet", "molecular", "ProteinNet structure/sequence files", "proteinnetpy", load_molecular)
    _register("esm_atlas", "molecular", "ESM evolutionary-scale embeddings", "fair-esm", load_molecular)


def DATASETS() -> list[dict[str, Any]]:
    """Registry entries (lazy so importing never requires the data libs)."""
    register_all()
    return list(_ENTRIES.values())


def dataset_dir() -> Path:
    env = os.environ.get("SWEEP_DATASETS_DIR")
    return Path(env) if env else DEFAULT_DIR


def _cached_path(name: str) -> Path:
    return dataset_dir() / name


def list_datasets() -> list[dict[str, Any]]:
    out = []
    for entry in DATASETS():
        cached = _cached_path(entry["name"]).exists()
        out.append(
            {
                "name": entry["name"],
                "group": entry["group"],
                "description": entry["description"],
                "source": entry["source"],
                "cached": cached,
            }
        )
    return out


def dataset_status(name: str) -> dict[str, Any]:
    for entry in list_datasets():
        if entry["name"] == name:
            return entry
    return {"name": name, "group": "unknown", "description": "Unknown dataset.", "source": "", "cached": False}


def load_dataset(
    name: str,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Load (and if needed fetch) a dataset into the uniform shape."""
    register_all()
    key = name.lower().strip()
    entry = _ENTRIES.get(key)
    if not entry:
        raise ValueError(
            f"Unknown dataset {name!r}. Known: " + ", ".join(sorted(_ENTRIES)) + "."
        )
    started = time.time()
    data = entry["loader"](key, split=split, limit=limit)
    rows = data.get("rows")

    # Reuse a snapshot we already saved offline (don't re-download).
    if rows is None:
        cached = read_cached_rows(key)
        if cached:
            rows = cached[:limit] if limit else cached
            data = {**data, "rows": rows, "row_count": len(cached), "notes": (data.get("notes", "") + " Loaded from offline cache.").strip()}
    return {
        "name": key,
        "group": entry["group"],
        "description": entry["description"],
        "source": entry["source"],
        "cached": _cached_path(key).exists(),
        "columns": list(data.get("columns") or []),
        "rows": rows or [],
        "row_count": data.get("row_count", len(rows or [])),
        "path": str(data.get("path") or _cached_path(key)),
        "notes": data.get("notes", ""),
        "seconds": round(time.time() - started, 2),
    }


def cache_rows(name: str, rows: list[dict[str, Any]]) -> Path:
    """Snap a normalized reasoning/corpus dataset to disk for offline reuse."""
    import json

    path = _cached_path(name)
    path.mkdir(parents=True, exist_ok=True)
    file = path / "rows.jsonl"
    with open(file, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file


def read_cached_rows(name: str) -> list[dict[str, Any]]:
    import json

    file = _cached_path(name) / "rows.jsonl"
    if not file.exists():
        return []
    out = []
    with open(file, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def require_online(msg: str) -> None:
    """Raise a friendly error when a requested source needs a first download."""
    raise RuntimeError(msg + " This needs a one-time download; try 'data load <name>'.")