"""HIDDEN evaluation set for Sweep (benchmark fairness protocol, directive §4/§31).

This dataset is the HELD-OUT evaluation set: it must never be used to tune,
select, or develop against. One protocol-frozen reference run is recorded;
after that, only final evaluations touch it.

Properties that distinguish it from suite_v1 (all 21 suite families reviewed;
none are reused):
  * Novel SKILLS: modulo arithmetic, min/max, set counting, duration
    arithmetic, event ordering over ISO dates, most/none quantifier grounding.
  * Novel VOCABULARY: entities/predicates disjoint from suite_v1 AND from
    sweep_neural_mesh training data (including train_evidence_pairs.py).
  * A content-generalization block: suite-STYLE claim verification (support /
    refute / irrelevant / hedged / conflicting) rendered in the new vocabulary,
    so saturated skills are probed on unseen content while new skills probe
    unseen structure.

Roles (directive §4): suite_v1 = development (SATURATED, 100% — no longer
informative), calibration pairs (train_evidence_pairs fresh seed) = validation
slice for model confidence, hidden_v1 (this file) = hidden evaluation.

Contamination: every case carries a contamination label computed by scanning
the repo's source/training files for the exact query text (same standard as
cli.contamination_scan, extended to the benchmark package itself).

Regenerating: python -m sweep_benchmark.hidden   (writes + re-freezes JSON;
compare printed SHA-256 against the recorded manifest before trusting runs).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

DATASET_PATH = "benchmark/datasets/hidden_v1.json"
MANIFEST_PATH = "benchmark/datasets/hidden_v1_manifest.json"
SEED = 20260905

# ──────────────────────────────────────────────────────────────────────
# Vocabulary — disjoint from suite_v1 (zorp/fleep/gloop/…, active/running/…)
# and from sweep_neural_mesh training vocabularies (krin/vost/…, stable/…)
# ──────────────────────────────────────────────────────────────────────
ENTITIES = ["gadjit", "plinx", "vorpan", "quindle", "sorbek", "thilse", "marnex", "veltrix"]
STATES = ["deployed", "sealed", "heated", "grounded", "flushed", "clamped", "mirrored", "zoned"]
HOLDERS = ["Delta Unit", "Kestrel Group", "Harbor Desk", "Talon Crew", "Meridian Office"]
HOURS = [3, 4, 5, 6, 7, 8]


def _rng(seed: int = SEED) -> random.Random:
    return random.Random(seed)


def _case(idx: int, group: str, family: str, difficulty: str, query: str,
          evidence: list, expected, mode: str, query_is_claim: bool = False) -> dict:
    return {
        "id": f"hb_{idx:04d}",
        "group": group,
        "family": family,
        "difficulty": difficulty,
        "query": query,
        "evidence": list(evidence),
        "expected": expected if isinstance(expected, tuple) else str(expected),
        "mode": mode,
        "query_is_claim": query_is_claim,
    }


# ──────────────────────────────────────────────────────────────────────
# Group 1: hidden_logic — skills absent from suite_v1
# ──────────────────────────────────────────────────────────────────────
def gen_modulo(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    for _ in range(n):
        a = rng.randint(30, 999)
        b = rng.randint(2, 12)
        q = f"What is the remainder of {a} divided by {b}?"
        out.append(_case(idx, "hidden_logic", "modulo", "easy", q, [], a % b, "int"))
        idx += 1
    return out, idx


def gen_minmax(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    for _ in range(n):
        nums = [rng.randint(10, 900) for _ in range(3)]
        pick_max = rng.random() < 0.5
        label = "largest" if pick_max else "smallest"
        ans = max(nums) if pick_max else min(nums)
        q = f"What is the {label} of {nums[0]}, {nums[1]} and {nums[2]}?"
        out.append(_case(idx, "hidden_logic", "minmax", "easy", q, [], ans, "int"))
        idx += 1
    return out, idx


def gen_set_count(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    for _ in range(n):
        a = rng.sample(range(1, 60), rng.randint(3, 5))
        b = rng.sample(range(1, 60), rng.randint(2, 4))
        if rng.random() < 0.5:
            res = len(set(a) | set(b))
            q = (f"How many elements are in the union of "
                 f"{{{', '.join(map(str, sorted(a)))}}} and "
                 f"{{{', '.join(map(str, sorted(b)))}}}?")
        else:
            res = len(set(a) & set(b))
            q = (f"How many elements are in the intersection of "
                 f"{{{', '.join(map(str, sorted(a)))}}} and "
                 f"{{{', '.join(map(str, sorted(b)))}}}?")
        out.append(_case(idx, "hidden_logic", "set_count", "medium", q, [], res, "int"))
        idx += 1
    return out, idx


# ──────────────────────────────────────────────────────────────────────
# Group 2: hidden_evidence — duration grounding (arithmetic over evidence)
# ──────────────────────────────────────────────────────────────────────
def gen_duration(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    for _ in range(n):
        ent = rng.choice(ENTITIES)
        holder = rng.choice(HOLDERS)
        hours = rng.choice(HOURS)
        start = rng.randint(6, 15)
        end = start + hours
        claim = f"The {ent} outage lasted {hours} hours."
        if rng.random() < 0.6:
            evidence = (f"{holder} logs show the {ent} outage began at "
                        f"{start:02d}:00 and ended at {end:02d}:00.")
            expected = "supported"
        else:
            wrong = hours + rng.choice([-2, -1, 1, 2])
            wend = start + wrong
            evidence = (f"{holder} logs show the {ent} outage began at "
                        f"{start:02d}:00 and ended at {wend:02d}:00.")
            expected = "refuted"
        out.append(_case(idx, "hidden_evidence", "duration_grounding", "hard",
                         claim, [evidence], expected, "claim", query_is_claim=True))
        idx += 1
    return out, idx


# ──────────────────────────────────────────────────────────────────────
# Group 3: hidden_temporal — event ordering over ISO dates
# ──────────────────────────────────────────────────────────────────────
def _rand_date(rng: random.Random, base: date) -> date:
    return base + timedelta(days=rng.randint(1, 400))


def gen_temporal_order(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    base = date(2024, 1, 1)
    for _ in range(n):
        ent = rng.choice(ENTITIES)
        incident = _rand_date(rng, base)
        patch = incident + timedelta(days=rng.randint(1, 30))
        if rng.random() < 0.5:
            claim = f"The {ent} incident happened before the patch."
            evidence = [f"The {ent} incident occurred on {incident.isoformat()}.",
                        f"The patch for the {ent} was deployed on {patch.isoformat()}."]
            expected = "supported"
        else:
            claim = f"The {ent} incident happened after the patch."
            evidence = [f"The {ent} incident occurred on {incident.isoformat()}.",
                        f"The patch for the {ent} was deployed on {patch.isoformat()}."]
            expected = "refuted"
        out.append(_case(idx, "hidden_temporal", "event_ordering", "hard",
                         claim, evidence, expected, "claim", query_is_claim=True))
        idx += 1
    return out, idx


# ──────────────────────────────────────────────────────────────────────
# Group 4: hidden_quantifier — most/none grounding over ratios
# ──────────────────────────────────────────────────────────────────────
def gen_quantifier(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    for _ in range(n):
        ent = rng.choice(ENTITIES)
        total = rng.choice([10, 12, 20])
        count = rng.randint(1, total - 1)
        if rng.random() < 0.5:
            claim = f"Most {ent}s are online."
            evidence = f"The dashboard shows {count} of {total} {ent}s are online."
            expected = "supported" if count > total / 2 else "refuted"
        else:
            claim = f"No {ent}s are online."
            evidence = f"The dashboard shows {count} of {total} {ent}s are online."
            expected = "refuted" if count > 0 else "supported"
        out.append(_case(idx, "hidden_quantifier", "quantifier_grounding", "hard",
                         claim, [evidence], expected, "claim", query_is_claim=True))
        idx += 1
    return out, idx


# ──────────────────────────────────────────────────────────────────────
# Group 5: hidden_content — suite-STYLE claim verification, NEW vocabulary
# (content-generalization probe of skills saturated by suite_v1)
# ──────────────────────────────────────────────────────────────────────
def gen_hidden_content(rng: random.Random, n: int, idx: int) -> list[dict]:
    out = []
    per = n // 5
    for _ in range(per):  # support
        ent, st = rng.choice(ENTITIES), rng.choice(STATES)
        claim = f"The {ent} is {st}."
        evidence = f"The {ent} is {st} and fully operational."
        out.append(_case(idx, "hidden_content", "claim_support", "easy", claim,
                         [evidence], "supported", "claim", query_is_claim=True))
        idx += 1
    for _ in range(per):  # refute
        ent, st = rng.choice(ENTITIES), rng.choice(STATES)
        claim = f"The {ent} is {st}."
        evidence = f"The {ent} is not {st}."
        out.append(_case(idx, "hidden_content", "claim_refute", "easy", claim,
                         [evidence], "refuted", "claim", query_is_claim=True))
        idx += 1
    for _ in range(per):  # irrelevant
        ent, st = rng.choice(ENTITIES), rng.choice(STATES)
        other = rng.choice([s for s in STATES if s != st])
        claim = f"The {ent} is {st}."
        evidence = f"The {ent} is {other} and idle."
        out.append(_case(idx, "hidden_content", "claim_irrelevant", "medium", claim,
                         [evidence], "unknown", "claim", query_is_claim=True))
        idx += 1
    for _ in range(per):  # hedged
        ent, st = rng.choice(ENTITIES), rng.choice(STATES)
        claim = f"The {ent} is {st}."
        hedge = rng.choice(["might be", "could be", "may be", "possibly is"])
        evidence = f"The {ent} {hedge} {st}."
        out.append(_case(idx, "hidden_content", "claim_hedged", "hard", claim,
                         [evidence], "unknown", "claim", query_is_claim=True))
        idx += 1
    for _ in range(n - 4 * per):  # conflicting sources
        ent, st = rng.choice(ENTITIES), rng.choice(STATES)
        claim = f"The {ent} is {st}."
        evidence = [f"Holder Alpha reports the {ent} is {st}.",
                    f"Holder Beta reports the {ent} is not {st}."]
        out.append(_case(idx, "hidden_content", "claim_conflicting", "hard", claim,
                         evidence, "unknown", "claim", query_is_claim=True))
        idx += 1
    return out, idx


# ──────────────────────────────────────────────────────────────────────
# Contamination labeling (directive §31)
# ──────────────────────────────────────────────────────────────────────
_CONTAM_SCAN_ROOTS = ("sweep_neural_mesh", "sweep_benchmark")


def _corpus_text() -> str:
    chunks = []
    for root in _CONTAM_SCAN_ROOTS:
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in (".py", ".json", ".jsonl", ".md", ".txt",
                                    ".yaml", ".yml", ".csv"):
                if any(part.startswith(".") for part in p.parts):
                    continue
                if "datasets" in p.parts or "results" in p.parts:
                    continue  # dataset/result files are not training sources
                try:
                    chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
    return "\n".join(chunks)


def label_contamination(cases: list[dict]) -> list[dict]:
    corpus = _corpus_text()
    for c in cases:
        q = c["query"].strip().lower()
        if len(q) >= 20 and q in corpus:
            c["contamination"] = "KNOWN_CONTAMINATION"
        else:
            # synthetic content is generated fresh; templates are new this
            # session, but "possibly" is the honest default for any task that
            # shares wording patterns with development material
            c["contamination"] = "POSSIBLY_CONTAMINATED"
    return cases


# ──────────────────────────────────────────────────────────────────────
# Assembly + freeze
# ──────────────────────────────────────────────────────────────────────
def generate(seed: int = SEED) -> tuple[list[dict], dict]:
    rng = _rng(seed)
    all_cases: list[dict] = []
    idx = 0
    for genfn, n in [
        (gen_modulo, 30),
        (gen_minmax, 30),
        (gen_set_count, 30),
        (gen_duration, 40),
        (gen_temporal_order, 40),
        (gen_quantifier, 40),
        (gen_hidden_content, 60),
    ]:
        cases, idx = genfn(rng, n, idx)
        all_cases.extend(cases)
    # deterministic shuffle, then re-id
    order = list(range(len(all_cases)))
    rng.shuffle(order)
    ordered = [all_cases[i] for i in order]
    for i, c in enumerate(ordered):
        c["id"] = f"hb_{i:04d}"
    ordered = label_contamination(ordered)
    summary = {}
    for c in ordered:
        summary.setdefault(c["group"], {"total": 0, "families": set()})
        summary[c["group"]]["total"] += 1
        summary[c["group"]]["families"].add(c["family"])
    out_summary = {g: {"total": v["total"], "families": sorted(v["families"])}
                   for g, v in sorted(summary.items())}
    return ordered, out_summary


def save(path: str = DATASET_PATH, seed: int = SEED) -> dict:
    cases, summary = generate(seed=seed)
    payload = {
        "version": "1.0.0",
        "role": "hidden",
        "seed": seed,
        "generator": "sweep_benchmark.hidden",
        "frozen_on": "2026-09-05",
        "total": len(cases),
        "per_group": summary,
        "protocol": {
            "access": "protocol-frozen; one reference run recorded; final evals only",
            "scoring": "sweep_benchmark.scoring (deterministic, no LLM judge)",
            "environment": "offline, CPU-only, fresh process per run",
        },
        "cases": cases,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    digest = hashlib.sha256(
        json.dumps([{k: c[k] for k in ("id", "group", "family", "query", "evidence",
                                       "expected", "mode")} for c in cases],
                   sort_keys=True).encode("utf-8")).hexdigest()
    counts = {"CLEAN": 0, "POSSIBLY_CONTAMINATED": 0, "KNOWN_CONTAMINATION": 0}
    for c in cases:
        counts[c["contamination"]] = counts.get(c["contamination"], 0) + 1
    manifest = {
        "dataset_file": path,
        "dataset_sha256": digest,
        "num_cases": len(cases),
        "per_group": summary,
        "contamination_counts": counts,
        "role": "hidden",
        "frozen_on": "2026-09-05",
        "note": ("Held-out evaluation set. Not used for tuning or selection. "
                 "All families and vocabulary are new relative to suite_v1 "
                 "and sweep_neural_mesh training data."),
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load(path: str = DATASET_PATH) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["cases"]


if __name__ == "__main__":
    m = save()
    print(json.dumps(m, indent=1))
