"""Loaders + normalizers for the reasoning QA datasets.

Rows are normalized so downstream QA is uniform::

    {prompt, answer, split?, source?} + the dataset's own raw fields.

Prefer the Offline HF cache for known-cached sources; otherwise fetch once
through :func:`datasets.load_dataset` (a keyless data download, then cached).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Optional

HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


# ── generic HF helpers ────────────────────────────────────────────────


def _hf_rows(repo: str, split: str = "train", config: Optional[str] = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    configs = [c for c in ([config] if config else []) + [None, "all"] if c is not None]
    seen: Optional[object] = None
    last_err: Optional[Exception] = None
    for cfg in configs:
        try:
            kwargs = {"name": cfg} if cfg else {}
            ds = load_dataset(repo, split=split, **kwargs)
            seen = ds
            break
        except Exception as exc:  # noqa: BLE001 — try the next config on this repo
            last_err = exc
    if seen is None:
        raise RuntimeError(f"Couldn't load {repo}: {last_err}")
    return [dict(r) for r in seen]


def _peek_columns(repo: str, split: str = "train", config: Optional[str] = None) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(repo, split=split, **({"name": config} if config else {}))
        return list(ds.column_names)
    except Exception:  # noqa: BLE001
        return []


def _first_available_cached(repos: list[str]) -> Optional[Path]:
    """Return the snapshot dir of the first repo that is already on disk."""
    for repo in repos:
        key = "datasets--" + repo.replace("/", "--")
        snaps = HUB_CACHE / key / "snapshots"
        if snaps.is_dir():
            try:
                return next(snaps.iterdir())
            except StopIteration:
                continue
    return None


def _rows_from_local_json(data_dir: Path) -> list[dict[str, Any]]:
    """Build rows from raw json/jsonl files in an HF snapshot dir (offline)."""
    from datasets import load_dataset

    files = sorted(glob.glob(str(data_dir / "**" / "*.jsonl"), recursive=True))
    files += sorted(glob.glob(str(data_dir / "*.json"), recursive=False))
    if not files:
        return []
    ds = load_dataset("json", data_files=files[:5], split="train")
    return [dict(r) for r in ds]


# ── per-dataset normalizers ───────────────────────────────────────────


def _norm_logiqa(raw: dict[str, Any]) -> dict[str, Any]:
    question = str(raw.get("question") or "")
    options = raw.get("options") or []
    if options and all(not isinstance(o, str) for o in options):
        options = [str(o) if not isinstance(o, str) else o for o in options]
    opt_txt = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) if options else ""
    prompt = f"{question}\n{opt_txt}".strip()
    return {"prompt": prompt, "answer": str(raw.get("answer") or "").strip(), **raw}


def _norm_gsm8k(raw: dict[str, Any]) -> dict[str, Any]:
    answer = str(raw.get("answer") or "")
    final = re.search(r"####\s*(-?[\d.,]+)", answer)
    ref = final.group(1) if final else answer.strip()
    return {"prompt": str(raw.get("question") or ""), "answer": ref, **raw}


def _norm_commonsense(raw: dict[str, Any]) -> dict[str, Any]:
    question = str(raw.get("question") or "")
    choices = raw.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    opt_lines = []
    for lab, txt in zip(labels, texts):
        opt_lines.append(f"{lab}. {txt}")
    prompt = f"{question}\n" + "\n".join(opt_lines)
    return {"prompt": prompt.strip(), "answer": str(raw.get("answerKey") or "").strip(), **raw}


def _norm_babi(raw: dict[str, Any]) -> dict[str, Any]:
    story = raw.get("story") or []
    if isinstance(story, list):
        story = " ".join(str(s) for s in story)
    prompt = f"{story}\nQ: {raw.get('question')}"
    return {"prompt": prompt.strip(), "answer": str(raw.get("answer") or "").strip(), **raw}


def _norm_mmlu(raw: dict[str, Any]) -> dict[str, Any]:
    question = str(raw.get("question") or "")
    choices = raw.get("choices") or []
    if isinstance(choices, str):
        choices = [c for c in re.split(r"[A-D][).]", choices) if c.strip()]
    opt_txt = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)) if choices else ""
    prompt = f"{question}\n{opt_txt}".strip()
    answer = str(raw.get("answer") or "").strip()
    return {"prompt": prompt, "answer": answer, **raw}


def _norm_cot(raw: dict[str, Any]) -> dict[str, Any]:
    question = str(raw.get("question") or "")
    chain = raw.get("chain_of_thought") or raw.get("CoT") or raw.get("response") or ""
    answer = str(raw.get("answer") or "").strip() or str(chain).strip()
    return {"prompt": question, "answer": answer, **raw}


_NORMALIZERS = {
    "logiqa": _norm_logiqa,
    "gsm8k": _norm_gsm8k,
    "commonsense_qa": _norm_commonsense,
    "babi": _norm_babi,
    "mmlu": _norm_mmlu,
    "cot_collection": _norm_cot,
}

# known-cached repos → try the cache instead of the network first
_CACHED_REPOS = {
    "arc": ["ai2_arc"],
    "super_glue": ["super_glue"],
    "folio": ["tasksource/folio"],
    "ruletaker": ["hitachi-nlp/ruletaker"],
}


def load_reasoning(name: str, split: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
    """Load a reasoning dataset; prefer disk cache for the known-cached ones."""
    if name in _CACHED_REPOS:
        snapshot = _first_available_cached(_CACHED_REPOS[name])
        if snapshot:
            rows = _rows_from_local_json(snapshot)
            if rows:
                normalizer = _NORMALIZERS.get(name) or (lambda r: {"prompt": str(r.get("question") or ""), "answer": str(r.get("answer") or ""), **r})
                normalized = [normalizer(r) for r in rows]
                return {"rows": normalized[:limit] if limit else normalized,
                        "row_count": len(rows), "columns": list(rows[0].keys()), "path": str(snapshot),
                        "notes": "Served from the local Hugging Face cache (offline)."}
            raise RuntimeError(f"Found cached files for {name} but couldn't parse them.")
        raise RuntimeError(f"{name} isn't cached locally yet. Run 'data load {name}' once online to cache it.")

    if name == "logiqa":
        rows = _hf_rows("lgw863/LogiQA", split or "train")
        norm = _NORMALIZERS["logiqa"]
    elif name == "gsm8k":
        rows = _hf_rows("gsm8k", split or "test", "main")
        norm = _NORMALIZERS["gsm8k"]
    elif name == "commonsense_qa":
        rows = _hf_rows("tau/commonsense_qa", split or "train")
        norm = _NORMALIZERS["commonsense_qa"]
    elif name == "babi":
        try:
            rows = _hf_rows("facebook/babi", split or "train", "en-qa1")
        except Exception:  # noqa: BLE001
            rows = _hf_rows("facebook/babi", split or "train")
        norm = _NORMALIZERS["babi"]
    elif name == "mmlu":
        rows = _hf_rows("cais/mmlu", split or "test")
        norm = _NORMALIZERS["mmlu"]
    elif name == "cot_collection":
        rows = _hf_rows("kaist-ai/CoT-Collection", split or "train")
        norm = _NORMALIZERS["cot_collection"]
    else:
        raise ValueError(f"Unknown reasoning dataset: {name}")

    normalized = [norm(r) for r in rows]
    if limit and limit > 0:
        normalized = normalized[:limit]
    columns = sorted({k for r in normalized for k in r})
    return {
        "rows": normalized,
        "row_count": len(rows),
        "columns": columns,
        "path": str(HUB_CACHE),
        "notes": f"{len(rows)} rows loaded" + (", capped to first sample" if limit else ""),
    }