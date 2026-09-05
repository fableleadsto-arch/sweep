"""Deterministic extraction + scoring of Sweep output (no LLM judge).

Sweep returns (decision, confidence, reasoning-text, explanation_data).
For each case mode we map that raw output into a canonical answer without
ever looking at the expected answer, then compare with ground truth.
"""
from __future__ import annotations

import re
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Raw-output -> canonical answer (per mode)
# ──────────────────────────────────────────────────────────────────────
def extract_int(reasoning: str) -> int | None:
    """Find the router's emitted answer integer in reasoning text."""
    m = re.search(r"Task router \([^)]*\):\s*(-?\d+)", reasoning)
    if m:
        return int(m.group(1))
    m = re.search(r"\):\s*(-?\d+)", reasoning)
    if m:
        return int(m.group(1))
    # fallback: any isolated integer token
    nums = re.findall(r"-?\b\d+\b", reasoning)
    if nums:
        return int(nums[-1])
    return None


def extract_set(reasoning: str) -> set[int] | None:
    m = re.search(r"Task router \([^)]*\):\s*\{([^}]*)\}", reasoning)
    if m:
        body = m.group(1).strip()
        if not body:
            return set()
        return {int(x.strip()) for x in body.split(",") if x.strip().isdigit()}
    m = re.search(r"\{([^}]*)\}", reasoning)
    if m:
        body = m.group(1).strip()
        if not body:
            return set()
        return {int(x.strip()) for x in body.split(",") if x.strip().isdigit()}
    return None


def _router_word(reasoning: str) -> str | None:
    """Extract the trailing word token from 'Task router (...): WORD'."""
    m = re.search(r"Task router \([^)]*\):\s*([A-Za-z]+)", reasoning)
    if m:
        return m.group(1).lower()
    return None


def model_yesno(case: dict, decision: str, reasoning: str, confidence: float) -> str:
    """Map raw output to yes/no/unknown for yesno-mode cases."""
    w = _router_word(reasoning)
    if w in ("yes", "true"):
        return "yes"
    if w in ("no", "false"):
        return "no"
    # logic engines: supported/refuted/mixed/insufficient verdicts
    m = re.search(r"(Logical inference|Proof mesh)\s*\((supported|refuted|mixed)\)", reasoning)
    if m:
        v = m.group(2)
        return {"supported": "yes", "refuted": "no", "mixed": "unknown"}[v]
    if decision == "supported":
        return "yes"
    if decision == "refuted":
        return "no"
    return "unknown"


def model_claim(case: dict, decision: str, reasoning: str, confidence: float) -> str:
    """Map raw output to supported/refuted/unknown for claim-mode cases."""
    m = re.search(r"Neural evidence classifier \((supports|refutes|neutral)", reasoning)
    if m:
        return {"supports": "supported", "refutes": "refuted", "neutral": "unknown"}[m.group(1)]
    if decision == "supported":
        return "supported"
    if decision == "refuted":
        return "refuted"
    if decision == "insufficient":
        return "unknown"
    return "unknown"


def model_pair(case: dict, decision: str, reasoning: str, confidence: float) -> str:
    """Map raw output to consistent/contradiction/unknown for pair-mode cases."""
    m = re.search(r"Neural contradiction detector \((contradiction|consistent|partial|unknown)", reasoning)
    if m:
        return {"contradiction": "contradiction",
                "consistent": "consistent",
                "partial": "unknown",
                "unknown": "unknown"}[m.group(1)]
    if decision == "refuted":
        return "contradiction"
    if decision == "supported":
        return "consistent"
    return "unknown"


def model_answer(case: dict, decision: str, reasoning: str, confidence: float,
                 explanation: dict | None = None) -> tuple[str, str]:
    """Return (canonical_model_answer, detail) — detail notes the parse source."""
    mode = case["mode"]
    if mode == "int":
        v = extract_int(reasoning)
        if v is None:
            return "NA", "no-number"
        return str(v), "router-number"
    if mode == "set":
        s = extract_set(reasoning)
        if s is None:
            return "NA", "no-set"
        return str(sorted(s)), "router-set"
    if mode == "yesno":
        return model_yesno(case, decision, reasoning, confidence), "yesno"
    if mode == "claim":
        return model_claim(case, decision, reasoning, confidence), "claim"
    if mode == "pair":
        return model_pair(case, decision, reasoning, confidence), "pair"
    return "NA", "unknown-mode"


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────
def normalize_expected(expected: Any) -> str:
    if isinstance(expected, tuple):
        return str(sorted(expected))
    return str(expected)


def score_case(case: dict, decision: str, reasoning: str, confidence: float,
               explanation: dict | None = None, latency_ms: float = 0.0) -> dict:
    ans, detail = model_answer(case, decision, reasoning, confidence, explanation)
    expected = normalize_expected(case["expected"])
    correct = (ans == expected)
    # abstention: model returned unknown/NA
    abstained = ans in ("unknown", "NA")
    return {
        "id": case["id"],
        "group": case["group"],
        "family": case["family"],
        "difficulty": case["difficulty"],
        "query": case["query"],
        "expected": expected,
        "mode": case["mode"],
        "model_answer": ans,
        "decision": decision,
        "confidence": round(confidence, 4),
        "parse_detail": detail,
        "correct": correct,
        "abstained": abstained,
        "latency_ms": round(latency_ms, 3),
        "reasoning_snippet": str(reasoning)[:200],
    }
