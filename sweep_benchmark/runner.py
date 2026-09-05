"""Benchmark runner — executes Sweep on cases in a single process.

Design:
  - One ReasoningCortex per process (realistic deployment shape).
  - Offline mode: live network retrieval (Wikipedia/Wikidata/LLM) disabled so
    results are deterministic and reproducible on machines without internet.
  - Warm-up: the fine-tuned neural models load once (~60-90s); warm-up queries
    trigger + await loading before any measured iteration.
  - RAM is sampled on a background thread.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from . import scoring

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_cortex(offline: bool = True, enable_ml: bool = False, threads: int | None = None):
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:
            pass

    from sweep_neural_mesh.neurons.cortex import ReasoningCortex

    c = ReasoningCortex(enable_ml=enable_ml)
    if offline:
        c.retrieve_live_knowledge = lambda q: None
        c._try_live_knowledge = lambda q, e, t0: None
        c.web_research = lambda *a, **k: None
    return c


# Warm-up queries: cheap router hits first (model load starts on first call that
# reaches the neural path), then evidence queries that wait for models.
WARMUP = [
    ("What is 1 plus 1?", []),
    ("What comes next: 1, 2, 3, ?", []),
    ("true and false", []),
    ("The zorp is active.", ["The zorp is active and running."]),
    ("The zorp is active.", ["The zorp is not active."]),
    ("Compare these two statements.", ["The zorp is active.", "The zorp is not active."]),
    ("What is the union of {1, 2} and {2, 3}?", []),
    ("All zorbs are greeps. All greeps are bloops. Are all zorbs bloops?", []),
    ("What is 23 times 4?", []),
    ("What comes next: 3, 6, 9, 12, ?", []),
]


class RAMSampler:
    def __init__(self) -> None:
        self._stop = False
        self._peak = 0.0
        self._samples: list[float] = []
        self._proc = psutil.Process(os.getpid())
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
                self._samples.append(rss)
                self._peak = max(self._peak, rss)
            except Exception:
                pass
            time.sleep(0.02)

    def stop(self) -> dict:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._samples:
            import statistics
            return {
                "peak_rss_mb": round(self._peak, 1),
                "mean_rss_mb": round(statistics.mean(self._samples), 1),
                "n_samples": len(self._samples),
                "startup_rss_mb": round(self._samples[0], 1),
            }
        return {"peak_rss_mb": round(self._peak, 1), "mean_rss_mb": 0.0, "n_samples": 0,
                "startup_rss_mb": 0.0}


def run_case(c, case: dict) -> dict:
    t0 = time.perf_counter()
    r = c.reason(query=case["query"], evidence=case["evidence"])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return scoring.score_case(
        case, decision=r.decision, reasoning=r.reasoning,
        confidence=r.confidence,
        explanation=(r.explanation_data or {}),
        latency_ms=latency_ms,
    )


def warmup(c, n: int | None = None) -> dict:
    """Run warm-up queries, awaiting neural model readiness. Returns timings."""
    timings = []
    queries = WARMUP if n is None else WARMUP[:n]
    for q, ev in queries:
        t0 = time.perf_counter()
        try:
            c.reason(query=q, evidence=ev)
        except Exception:
            pass
        timings.append((time.perf_counter() - t0) * 1000)
    return {
        "warmup_queries": len(queries),
        "total_warmup_ms": round(sum(timings), 1),
        "warmup_timings_ms": [round(t, 1) for t in timings],
    }


def run_suite(cases: list[dict], cortex=None, offline: bool = True,
              label: str = "cpu", thread_config: str = "default",
              warmup_n: int | None = None, sampler: bool = True,
              enable_ml: bool = False, threads: int | None = None) -> dict:
    t_start = time.perf_counter()
    if cortex is None:
        c = make_cortex(offline=offline, enable_ml=enable_ml, threads=threads)
    else:
        c = cortex

    # cortex + startup timing
    startup_ms = (time.perf_counter() - t_start) * 1000

    ram = RAMSampler()
    if sampler:
        ram.start()

    # warm-up
    warm = warmup(c, warmup_n)
    warm_ms = (time.perf_counter() - t_start) * 1000 - startup_ms

    results = []
    t_run0 = time.perf_counter()
    for i, case in enumerate(cases):
        try:
            results.append(run_case(c, case))
        except Exception as e:
            results.append({
                "id": case["id"], "group": case["group"], "family": case["family"],
                "difficulty": case["difficulty"], "query": case["query"],
                "expected": scoring.normalize_expected(case["expected"]),
                "mode": case["mode"], "model_answer": "ERROR", "decision": "error",
                "confidence": 0.0, "correct": False, "abstained": False,
                "latency_ms": 0.0, "error": str(e)[:200],
            })
        if (i + 1) % 150 == 0:
            print(f"  [{label}] {i+1}/{len(cases)}", flush=True)
    run_ms = (time.perf_counter() - t_run0) * 1000

    ram_info = ram.stop() if sampler else {}
    elapsed = time.perf_counter() - t_start

    return {
        "label": label,
        "thread_config": thread_config,
        "num_cases": len(cases),
        "num_results": len(results),
        "results": results,
        "startup_ms": round(startup_ms, 1),
        "warmup": warm,
        "warmup_block_ms": round(warm_ms, 1),
        "measured_block_ms": round(run_ms, 1),
        "total_elapsed_s": round(elapsed, 2),
        "ram": ram_info,
    }


def accuracy_of(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    by_group: dict[str, dict] = {}
    for r in results:
        g = r["group"]
        d = by_group.setdefault(g, {"total": 0, "correct": 0, "latencies": []})
        d["total"] += 1
        d["correct"] += 1 if r.get("correct") else 0
        d["latencies"].append(r.get("latency_ms", 0.0))
    for d in by_group.values():
        d["accuracy"] = round(d["correct"] / d["total"], 4) if d["total"] else 0.0
        d["mean_latency_ms"] = round(sum(d["latencies"]) / len(d["latencies"]), 3) if d["latencies"] else 0.0
        del d["latencies"]
    overall = correct / total if total else 0.0
    return {"total": total, "correct": correct, "accuracy": round(overall, 4),
            "by_group": by_group}


def summarize_run(run: dict) -> dict:
    """Compute latency percentiles + accuracy summary for a run payload."""
    import statistics
    results = run["results"]
    lats = sorted(r.get("latency_ms", 0.0) for r in results)
    n = len(lats)
    def pct(p):
        if not lats:
            return 0.0
        return round(lats[min(int(p * n), n - 1)], 3)
    acc = accuracy_of(results)
    confs = [r.get("confidence", 0.0) for r in results]
    return {
        "label": run["label"],
        "thread_config": run["thread_config"],
        "num_cases": n,
        "accuracy": acc["accuracy"],
        "correct": acc["correct"],
        "by_group": acc["by_group"],
        "latency_ms": {
            "mean": round(statistics.mean(lats), 3) if lats else 0.0,
            "median_p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "min": round(min(lats), 3) if lats else 0.0,
            "max": round(max(lats), 3) if lats else 0.0,
            "stdev": round(statistics.stdev(lats), 3) if len(lats) > 1 else 0.0,
        },
        "confidence": {
            "mean": round(statistics.mean(confs), 4) if confs else 0.0,
            "min": round(min(confs), 4) if confs else 0.0,
            "max": round(max(confs), 4) if confs else 0.0,
        },
        "throughput_per_sec": round(n / (run["measured_block_ms"] / 1000.0), 2) if run["measured_block_ms"] > 0 else 0.0,
        "startup_ms": run["startup_ms"],
        "warmup_block_ms": run["warmup_block_ms"],
        "measured_block_ms": run["measured_block_ms"],
        "total_elapsed_s": run["total_elapsed_s"],
        "ram": run.get("ram", {}),
    }


def save_run(run: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(run, indent=1), encoding="utf-8")
