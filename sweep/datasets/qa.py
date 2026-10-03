"""Local reasoning over datasets — no external APIs.

Few-shot retrieval over a dataset's rows feeds a prompt to the local model
(sweep.models). Two entry points:

* ``ask_dataset(name, question)``  — answer a question grounded in the data.
* ``evaluate_sample(name, n)``     — score the local model on n sampled rows.

The local model is loaded lazily and cached per process, so a second question
is fast.
"""

from __future__ import annotations

from typing import Any

from .registry import load_dataset


def _scoring_tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _retrieve(rows: list[dict[str, Any]], query: str, k: int = 4) -> list[dict[str, Any]]:
    """Cheap lexical retrieval; a few rows is enough for few-shot grounding."""
    q = _scoring_tokens(query)
    if not q:
        return rows[:k]
    scored = []
    for row in rows:
        text = _scoring_tokens(str(row.get("prompt") or row.get("text") or ""))
        overlap = len(q & text)
        scored.append((overlap, row))
    scored.sort(key=lambda t: t[0], reverse=True)
    best = [row for score, row in scored if score > 0][:k]
    return best if best else rows[:k]


def _prompt_for(name: str, rows: list[dict[str, Any]], question: str) -> str:
    sections = [f"You are answering questions over the {name!r} dataset. Be concise and cite the given examples only."]
    for i, row in enumerate(rows[:4], 1):
        p = str(row.get("prompt") or row.get("text") or "").strip()[:600]
        a = str(row.get("answer") or "").strip()[:200]
        sections.append(f"Example {i}:\nQ: {p}\nA: {a}")
    sections.append(f"Now answer this question using the same style:\nQ: {question}\nA:")
    return "\n\n".join(sections)


def ask_dataset(name: str, question: str, max_tokens: int = 220) -> dict[str, Any]:
    from .models import generate

    data = load_dataset(name, limit=800)
    rows = data["rows"]
    if not rows:
        raise RuntimeError(f"No rows available for {name!r}.")
    examples = _retrieve(rows, question)
    prompt = _prompt_for(name, examples, question)
    answer = generate(prompt, max_tokens=max_tokens, temperature=0.2)
    return {
        "dataset": name,
        "question": question,
        "grounded_rows": len(examples),
        "answer": answer,
        "model": "local",
    }


def evaluate_sample(name: str, n: int = 5, timeout_per_row: int = 90) -> dict[str, Any]:
    from .models import generate

    data = load_dataset(name, limit=200)
    rows = data["rows"][:n]
    if not rows:
        raise RuntimeError(f"No rows available for {name!r}.")

    import time

    results: list[dict[str, Any]] = []
    correct = 0
    for i, row in enumerate(rows, 1):
        q = str(row.get("prompt") or "")
        expected = str(row.get("answer") or "").strip()
        started = time.time()
        try:
            reply = _timed_generate(_prompt_for(name, [row], q), timeout_per_row, max_tokens=160, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            reply = f"(failed: {exc})"
        ok = _looks_correct(reply, expected)
        if ok:
            correct += 1
        results.append({"i": i, "question": q[:120], "expected": expected[:120], "got": reply[:120], "ok": ok, "secs": round(time.time() - started, 1)})
    return {
        "dataset": name,
        "n": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 3) if results else 0.0,
        "results": results,
    }


def _looks_correct(reply: str, expected: str) -> bool:
    import re

    r = (reply or "").strip().lower()
    e = (expected or "").strip().lower()
    if not e or not r:
        return False
    if e == r:
        return True
    if len(e) <= 12 and (e + "." in r or e in r.split()):
        return True
    letters = re.findall(r"\b[A-D](?=[.)\s]|$)", reply, re.I)
    if re.fullmatch(r"[A-D]", e) and letters:
        return letters[0].upper() == e.upper()
    return False


def _timed_generate(prompt: str, timeout: int, **kwargs) -> str:
    """Run generation with a hard wall-clock timeout (Windows-safe)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_generate_pure, prompt, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise


def _generate_pure(prompt: str, **kwargs) -> str:
    from .models import generate

    return generate(prompt, **kwargs)