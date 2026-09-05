"""Phase 8-9 metrics: precision/recall/F1 per group, calibration, abstention,
hallucination rate, contradiction rate."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = "benchmark/results"


def _class_sets(mode: str) -> list[str]:
    if mode == "yesno":
        return ["yes", "no", "unknown"]
    if mode == "claim":
        return ["supported", "refuted", "unknown"]
    if mode == "pair":
        return ["consistent", "contradiction", "unknown"]
    return []


def group_prf(results: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for r in results:
        g = r["group"]
        d = groups.setdefault(g, {"total": 0, "tp": {}, "fp": {}, "fn": {},
                                  "correct_unknown": 0, "total_unknown": 0})
        d["total"] += 1
        labels = _class_sets(r.get("mode", ""))
        if not labels:
            continue
        expected = r.get("expected")
        predicted = r.get("model_answer")
        if predicted not in labels:
            predicted = "unknown"
        if expected not in labels:
            expected = "unknown"
        # abstention accounting
        if expected == "unknown":
            d["total_unknown"] += 1
            if predicted == "unknown":
                d["correct_unknown"] += 1
        for lab in labels:
            tp = fp = fn = 0
            if predicted == lab and expected == lab:
                tp = 1
            if predicted == lab and expected != lab:
                fp = 1
            if predicted != lab and expected == lab:
                fn = 1
            d["tp"][lab] = d["tp"].get(lab, 0) + tp
            d["fp"][lab] = d["fp"].get(lab, 0) + fp
            d["fn"][lab] = d["fn"].get(lab, 0) + fn
    out = {}
    for g, d in groups.items():
        labels = _class_sets(next((r["mode"] for r in results if r["group"] == g), ""))
        prf = {}
        for lab in labels:
            tp, fp, fn = d["tp"].get(lab, 0), d["fp"].get(lab, 0), d["fn"].get(lab, 0)
            prec = tp / (tp + fp) if (tp + fp) else None
            rec = tp / (tp + fn) if (tp + fn) else None
            f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
            prf[lab] = {"precision": round(prec, 4) if prec is not None else None,
                        "recall": round(rec, 4) if rec is not None else None,
                        "f1": round(f1, 4) if f1 is not None else None}
        out[g] = {
            "total": d["total"],
            "accuracy": round(sum(1 for r in results if r["group"] == g and r.get("correct")) / d["total"], 4),
            "per_class": prf,
            "abstention": {
                "expected_unknown": d["total_unknown"],
                "correct_abstentions": d["correct_unknown"],
                "abstention_accuracy": round(d["correct_unknown"] / d["total_unknown"], 4) if d["total_unknown"] else None,
            },
        }
    return out


def calibration_metrics(results: list[dict]) -> dict:
    """Reliability: mean confidence when correct vs wrong, overconfidence rate,
    hallucination rate (high-confidence wrong), ECE-ish bins."""
    scored = [r for r in results if r.get("confidence") and r.get("confidence") > 0]
    if not scored:
        return {}
    correct = [r for r in scored if r.get("correct")]
    wrong = [r for r in scored if not r.get("correct")]
    bins = {}
    for r in scored:
        b = int(r["confidence"] * 10)  # 0..9
        d = bins.setdefault(b, {"n": 0, "corr": 0})
        d["n"] += 1
        d["corr"] += 1 if r.get("correct") else 0
    ece = 0.0
    total = len(scored)
    for b, d in bins.items():
        conf = (b + 0.5) / 10
        acc = d["corr"] / d["n"]
        ece += (d["n"] / total) * abs(conf - acc)
    halluc = [r for r in wrong if r.get("confidence", 0) >= 0.75]
    overconf = [r for r in wrong if r.get("confidence", 0) > 0.5]
    return {
        "n_scored": len(scored),
        "mean_conf_correct": round(sum(r["confidence"] for r in correct) / len(correct), 4) if correct else None,
        "mean_conf_wrong": round(sum(r["confidence"] for r in wrong) / len(wrong), 4) if wrong else None,
        "ece": round(ece, 4),
        "hallucination_rate_high_conf": round(len(halluc) / max(len(wrong), 1), 4),
        "hallucination_count_high_conf": len(halluc),
        "overconfidence_rate": round(len(overconf) / max(len(wrong), 1), 4),
    }


def full_metrics(results: list[dict]) -> dict:
    prf = group_prf(results)
    cal = calibration_metrics(results)
    wrong = [r for r in results if not r.get("correct")]
    contrad = [r for r in results if r.get("model_answer") == "contradiction" or r.get("decision") == "refuted"]
    abstained = [r for r in results if r.get("abstained")]
    return {
        "per_group": prf,
        "calibration": cal,
        "contradiction_rate": round(len(contrad) / max(len(results), 1), 4),
        "abstention_rate": round(len(abstained) / max(len(results), 1), 4),
        "wrong_total": len(wrong),
    }


if __name__ == "__main__":
    run = json.loads(Path(f"{RESULTS}/raw/cpu_run.json").read_text(encoding="utf-8"))
    print(json.dumps(full_metrics(run["results"]), indent=1))
