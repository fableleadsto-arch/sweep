"""Streaming access to the big web corpora (The Pile, FineWeb, Wikipedia).

Unlike QA datasets these are streamed by the ``datasets`` library — they are
too large to materialise locally. We pull a sample for inspection/knowledge
rows and note the offline cache location.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional


def _stream(repo: str, config: Optional[str] = None, split: str = "train"):
    from datasets import load_dataset

    kwargs = {"name": config} if config else {}
    ds = load_dataset(repo, split=split, streaming=True, **kwargs)
    stream = iter(ds)
    for _ in range(10 ** 9):
        try:
            yield next(stream)
        except StopIteration:
            return


def _norm_pile(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("text") or ""
    return {"title": f"Pile excerpt ({len(text)} chars)", "text": text[:1200], "prompt": "Summarize this web excerpt.", "answer": text[:800], **raw}


def _norm_fineweb(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("text") or ""
    return {"title": "FineWeb excerpt", "text": text[:1200], "prompt": "Summarize this web excerpt.", "answer": text[:800], **raw}


def _norm_wikipedia(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("text") or ""
    title = raw.get("title") or ""
    return {"title": title, "text": text[:1200], "prompt": f"Summarize: {title}", "answer": text[:800], **raw}


def load_corpus(name: str, split: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
    per_name = {
        "pile": ("EleutherAI/pile", None, _norm_pile, "Streaming — full corpus is 825 GB."),
        "fineweb": ("HuggingFaceFW/fineweb", None, _norm_fineweb, "Streaming — clean web corpus."),
        "wikipedia": ("wikipedia", "20220301.en", _norm_wikipedia, "Streaming — English Wikipedia dump."),
    }
    repo, config, norm, note = per_name.get(name, (*per_name.get("pile", (None, None, None, "")),))

    # try the local cache first (no network)
    cached = _rows_from_corpus_cache(name)
    if cached:
        normed = [norm(r) for r in cached]
        return {"rows": normed[:limit] if limit else normed, "row_count": len(cached),
                "columns": list(normed[0].keys()) if normed else [], "path": str(cached), "notes": f"{note} Served from cache."}

    n = limit or 25
    rows = []
    try:
        for i, raw in enumerate(_stream(repo, config, split or "train")):
            rows.append(norm(raw))
            if i + 1 >= n:
                break
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Couldn't stream {name}: {exc}. This corpus needs a one-time download.")
    return {
        "rows": rows,
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "path": "",
        "notes": f"{note} Sampled {len(rows)} rows streaming.",
    }


def _rows_from_corpus_cache(name: str) -> list[dict[str, Any]]:
    """Reuse a previous sample we snapshotted to disk."""
    from .registry import read_cached_rows

    return read_cached_rows(name)