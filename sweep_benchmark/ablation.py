"""Ablation study (Phase 10) over Sweep's actual components.

Configurations toggle the real fast-path stack of ReasoningCortex:
  - neural fast path (fine-tuned BERT evidence/contradiction classifiers)
  - general intelligence (GI knowledge retrieval fast path)
  - logic engines (proof mesh + logical inference)
  - task router (deterministic rule handlers: math/logic/evidence/temporal/causal)

Configs (mapping to the spec's labels):
  BASELINE            rules-only backbone (router + full pipeline; neural/GI/logic OFF)
  BASELINE+GI         enable GI knowledge retrieval
  BASELINE+NEURAL     enable neural engine (evidence/contradiction BERT)  [=+memory? no, =neural mesh]
  + LOGIC ENGINES     full Sweep (all fast paths)
  FULL (shared cortex = with memory)   vs  FULL fresh-per-query (= no cross-query memory)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import runner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _disable(c, *names: str) -> None:
    """Disable cortex instance fast-path methods by shadowing with no-ops."""
    if "neural" in names:
        c._try_neural_fast_path = lambda q, e, t0: None
    if "contradiction" in names:
        c._try_contradiction_fast_path = lambda q, e, t0: None
    if "uncertainty" in names:
        c._try_uncertainty_fast_path = lambda q, e, t0: None
    if "gi" in names:
        c._try_gi_fast_path = lambda q, e, t0: None
    if "logic" in names:
        c._try_logic_engines = lambda q, e, t0: None
    if "router" in names:
        c._try_task_router = lambda q, e, t0: None
    if "live" in names:
        c._try_live_knowledge = lambda q, e, t0: None


CONFIGS = {
    # name -> disabled components
    "A_baseline_rules": ["neural", "contradiction", "uncertainty", "gi", "logic"],
    "B_gi":             ["neural", "contradiction", "uncertainty", "logic"],
    "C_neural_mesh":    ["gi", "logic"],          # neural + router + full pipeline
    "D_full_logic":     ["neural", "contradiction", "uncertainty"],  # GI + logic + router
    "E_full_sweep":     [],                       # everything
}


def run_ablation(cases: list[dict], subset: list[dict] | None = None,
                 out_path: str = "benchmark/results/raw/ablation.json") -> dict:
    """Run each config on the subset within a single process (shared cortex per config)."""
    sub = subset if subset is not None else cases
    print(f"Ablation: {len(CONFIGS)} configs x {len(sub)} cases", flush=True)

    # Warm the neural models once (process-level singleton), then build a FRESH
    # cortex per config so toggles cannot leak between configurations.
    first = runner.make_cortex(offline=True)
    runner.warmup(first)
    del first

    outputs = {}
    for name, disabled in CONFIGS.items():
        print(f"  config {name} (disable={disabled})", flush=True)
        cortex = runner.make_cortex(offline=True)
        _disable(cortex, *disabled)
        ram = runner.RAMSampler()
        ram.start()
        results = []
        for case in sub:
            t0 = time.perf_counter()
            try:
                r = cortex.reason(query=case["query"], evidence=case["evidence"])
                from .scoring import score_case
                results.append(score_case(case, r.decision, r.reasoning, r.confidence,
                                          r.explanation_data,
                                          latency_ms=(time.perf_counter() - t0) * 1000))
            except Exception as e:
                from .scoring import score_case, normalize_expected
                results.append({
                    "id": case["id"], "group": case["group"], "family": case["family"],
                    "difficulty": case["difficulty"], "query": case["query"],
                    "expected": normalize_expected(case["expected"]), "mode": case["mode"],
                    "model_answer": "ERROR", "decision": "error", "confidence": 0.0,
                    "correct": False, "abstained": False, "latency_ms": 0.0,
                    "error": str(e)[:200]})
        ram_info = ram.stop()
        summary = runner.summarize_run({
            "label": name, "thread_config": "default",
            "num_cases": len(results), "results": results,
            "startup_ms": 0.0, "warmup": {}, "warmup_block_ms": 0.0,
            "measured_block_ms": sum(r["latency_ms"] for r in results),
            "total_elapsed_s": sum(r["latency_ms"] for r in results) / 1000.0,
        })
        summary["ram"] = ram_info
        outputs[name] = summary

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(outputs, indent=1), encoding="utf-8")
    return outputs
