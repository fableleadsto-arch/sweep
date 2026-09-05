"""Write benchmark/REPORT.md (Phase 16-17)."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = "benchmark/results"
RAW = "benchmark/results/raw"


def _load(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(x, nd=2):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def write_report() -> str:
    summary = _load(f"{RESULTS}/summary.json")
    env = _load(f"{RESULTS}/environment.json")
    err = _load(f"{RESULTS}/error_analysis.json")
    met = _load(f"{RESULTS}/metrics.json")
    brk = _load(f"{RESULTS}/breakdown.json")
    scaling = _load(f"{RESULTS}/scaling.json")
    ablation = _load(f"{RESULTS}/ablation.json")
    comparison = _load(f"{RESULTS}/comparison.json")
    repeat = _load(f"{RESULTS}/repeatability.json")
    manifest = _load("benchmark/dataset_manifest.json")
    cpu = _load(f"{RAW}/cpu_run.json")

    L = []
    L.append("# Sweep Benchmark Results")
    L.append("")
    L.append("*Status: local, CPU-only, offline benchmark run on the machine below. "
             "Not an MLPerf submission; no MLPerf certification is claimed.*")
    L.append("")
    L.append("## 1. Executive Summary")
    L.append("")
    acc = summary.get("accuracy")
    by_group = summary.get("by_group", {})
    lat = summary.get("latency_ms", {})
    L.append(f"- **Overall accuracy: {_fmt(acc * 100, 1)}%** "
             f"({summary.get('correct')}/{summary.get('num_cases')} cases)")
    for g, v in by_group.items():
        L.append(f"  - {g}: {_fmt(v['accuracy'] * 100, 1)}%")
    L.append(f"- **Median latency (p50): {_fmt(lat.get('median_p50'), 1)} ms** | "
             f"p95: {_fmt(lat.get('p95'), 1)} ms | p99: {_fmt(lat.get('p99'), 1)} ms")
    L.append(f"- **Throughput: {_fmt(summary.get('throughput_per_sec'), 1)} cases/sec**")
    L.append(f"- **Peak RAM: {_fmt((cpu.get('ram') or {}).get('peak_rss_mb'), 1)} MB**")
    L.append("")
    L.append("## 2. Hardware")
    L.append("")
    L.append(f"- OS: {env.get('os')}")
    L.append(f"- CPU: {env.get('processor')} / {env.get('cpu_brand', '')}")
    L.append(f"- Physical cores: {env.get('cpu_physical_cores')} | "
             f"Logical cores: {env.get('cpu_logical_cores')}")
    L.append(f"- RAM: {env.get('ram_total_gb')} GB total, "
             f"{env.get('ram_available_gb')} GB available at capture")
    L.append(f"- GPU: {env.get('gpu')}")
    L.append("")
    L.append("## 3. Software")
    L.append("")
    L.append(f"- Python: {env.get('python_version')}")
    for k, v in (env.get("versions") or {}).items():
        L.append(f"- {k}: {v}")
    L.append(f"- Git SHA: {(env.get('git') or {}).get('sha')} "
             f"({(env.get('git') or {}).get('branch')}, "
             f"{(env.get('git') or {}).get('status')})")
    L.append("")
    L.append("## 4. Dataset")
    L.append("")
    L.append(f"- File: {manifest.get('dataset_file')}")
    L.append(f"- SHA-256: {manifest.get('dataset_sha256')}")
    L.append(f"- Cases: {manifest.get('num_cases')} | "
             f"per group: {json.dumps(manifest.get('per_group', {}))}")
    L.append("- Generated post-hoc with synthetic content (Phase 4); never routed into "
             "Sweep's training pipeline.")
    L.append("")
    L.append("## 5. Accuracy Results (by category)")
    L.append("")
    L.append("| Group | Accuracy | Correct/Total |")
    L.append("|---|---|---|")
    for g in ["basic_logic", "multi_step", "ambiguity", "evidence",
              "adversarial", "generalization"]:
        v = by_group.get(g, {})
        L.append(f"| {g} | {_fmt(v.get('accuracy', 0) * 100, 1)}% | "
                 f"{v.get('correct')}/{v.get('total')} |")
    L.append("")
    L.append("## 5b. Detailed Accuracy (precision/recall/F1, abstention)")
    L.append("")
    for g, d in (met.get("per_group") or {}).items():
        acc = d.get("accuracy")
        abst = d.get("abstention") or {}
        L.append(f"**{g}** (accuracy {_fmt((acc or 0) * 100, 1)}%):")
        for lab, m in (d.get("per_class") or {}).items():
            L.append(f"- {lab}: precision={_fmt(m['precision'] * 100, 1) if m.get('precision') is not None else '—'}% "
                     f"recall={_fmt(m['recall'] * 100, 1) if m.get('recall') is not None else '—'}% "
                     f"F1={_fmt(m['f1'] * 100, 1) if m.get('f1') is not None else '—'}%")
        if abst.get('expected_unknown'):
            L.append(f"- Abstention (correct refusal on {abst.get('expected_unknown')} ambiguous cases): "
                     f"{abst.get('correct_abstentions')} -> "
                     f"{_fmt((abst.get('abstention_accuracy') or 0) * 100, 1)}%")
    cal = met.get("calibration") or {}
    if cal:
        L.append("")
        L.append("**Confidence calibration:**")
        L.append(f"- mean confidence when correct: {_fmt(cal.get('mean_conf_correct'), 3)} | "
                 f"when wrong: {_fmt(cal.get('mean_conf_wrong'), 3)}")
        L.append(f"- ECE: {_fmt(cal.get('ece'), 4)}")
        L.append(f"- wrong answers at conf ≥0.75 (hallucination): "
                 f"{cal.get('hallucination_count_high_conf')} "
                 f"({_fmt((cal.get('hallucination_rate_high_conf') or 0) * 100, 1)}% of all errors)")
    L.append(f"- Contradiction rate: {_fmt((met.get('contradiction_rate') or 0) * 100, 1)}% | "
             f"abstention rate: {_fmt((met.get('abstention_rate') or 0) * 100, 1)}%")
    L.append("")
    L.append("## 6. Latency Results")
    L.append("")
    lat_ms = lat
    L.append(f"- p50: **{_fmt(lat_ms.get('median_p50'), 1)} ms**")
    L.append(f"- p90: {_fmt(lat_ms.get('p90'), 1)} ms")
    L.append(f"- p95: {_fmt(lat_ms.get('p95'), 1)} ms")
    L.append(f"- p99: {_fmt(lat_ms.get('p99'), 1)} ms")
    L.append(f"- mean: {_fmt(lat_ms.get('mean'), 1)} ms | "
             f"min: {_fmt(lat_ms.get('min'), 1)} ms | max: {_fmt(lat_ms.get('max'), 1)} ms | "
             f"stdev: {_fmt(lat_ms.get('stdev'), 1)} ms")
    L.append(f"- Startup (cortex+import): {_fmt(summary.get('startup_ms'), 1)} ms; "
             f"warm-up block: {_fmt(summary.get('warmup_block_ms'), 1)} ms")
    L.append("")
    L.append("## 7. Throughput")
    L.append("")
    L.append(f"- {_fmt(summary.get('throughput_per_sec'), 1)} cases/sec "
             f"(measured block {_fmt(summary.get('measured_block_ms') / 1000, 1)} s, "
             f"{summary.get('num_cases')} cases)")
    L.append("")
    L.append("## 8. Memory")
    L.append("")
    ram = cpu.get("ram") or {}
    L.append(f"- Peak RSS: {_fmt(ram.get('peak_rss_mb'), 1)} MB")
    L.append(f"- Mean RSS during run: {_fmt(ram.get('mean_rss_mb'), 1)} MB")
    L.append("")
    L.append("## 9. CPU Scaling")
    L.append("")
    L.append("| Config | threads | accuracy | median ms | p95 ms | p99 ms | cases/s |")
    L.append("|---|---|---|---|---|---|---|")
    for name, data in (scaling or {}).items():
        if isinstance(data, dict) and "error" not in data:
            L.append(f"| {name} | {data.get('threads')} | {_fmt((data.get('accuracy') or 0) * 100, 1)}% | "
                     f"{_fmt(data.get('median'), 1)} | {_fmt(data.get('p95'), 1)} | "
                     f"{_fmt(data.get('p99'), 1)} | {_fmt(data.get('throughput'), 1)} |")
        else:
            L.append(f"| {name} | — | error: {str(data)[:80]} | | | | |")
    L.append("")
    L.append("## 10. Ablation")
    L.append("")
    L.append("| Config | accuracy | vs baseline | mean latency ms | peak RSS MB |")
    L.append("|---|---|---|---|---|")
    impacts = ablation.get("impacts", {})
    configs = ablation.get("configs", {})
    for name, imp in impacts.items():
        L.append(f"| {name} | {_fmt((imp.get('accuracy') or 0) * 100, 1)}% | "
                 f"{imp.get('impact_vs_baseline', 0):+.3f} | "
                 f"{_fmt(imp.get('mean_latency_ms'), 1)} | {_fmt(imp.get('peak_rss_mb'), 1)} |")
    L.append("")
    L.append("## 11. Error Analysis")
    L.append("")
    L.append("| Error type | Count | % of failures |")
    L.append("|---|---|---|")
    for cat, cnt in sorted((err.get("counts") or {}).items(), key=lambda kv: -kv[1]):
        L.append(f"| {cat} | {cnt} | {_fmt(cnt / max(err.get('total_wrong', 1), 1) * 100, 1)}% |")
    L.append("")
    L.append("Top failures (by confidence):")
    L.append("")
    for f in (err.get("top_20") or [])[:10]:
        L.append(f"- `{f['id']}` [{f['error_type']}] expected `{f['expected']}` got "
                 f"`{f['model_answer']}` conf={f['confidence']}: {f['query'][:100]}")
    L.append("")
    L.append("## 12. Generalization")
    L.append("")
    gv = by_group.get("generalization", {})
    L.append(f"- Accuracy on unseen structures: **{_fmt(gv.get('accuracy', 0) * 100, 1)}%** "
             f"({gv.get('correct')}/{gv.get('total')})")
    L.append("- Note: cases use novel content (entities/values) and novel structures "
             "(deep syllogisms, 3-term arithmetic, nested booleans, multi-hop transitivity).")
    L.append("")
    L.append("## 13. Adversarial Robustness")
    L.append("")
    av = by_group.get("adversarial", {})
    L.append(f"- Accuracy: **{_fmt(av.get('accuracy', 0) * 100, 1)}%** "
             f"({av.get('correct')}/{av.get('total')})")
    L.append("")
    L.append("## 14. Public Comparison")
    L.append("")
    L.append("Sweep was evaluated on a **private synthetic benchmark**, so no published "
             "result shares its dataset. Per the fairness check, every comparison below "
             "is **NOT DIRECTLY COMPARABLE**. Reported for context only.")
    L.append("")
    L.append("| Model | Benchmark | Accuracy | Source | Comparable |")
    L.append("|---|---|---|---|---|")
    for row in (comparison.get("rows") or []):
        L.append(f"| {row['model']} | {row['benchmark']} | "
                 f"{row['accuracy_pct']}% | {row['source']} | {row['comparable']} |")
    L.append("")
    L.append("## 15. Limitations")
    L.append("")
    L.append("- **Offline mode**: live Wikipedia/Wikidata/LLM retrieval was disabled for "
             "determinism. Real-world questions that Sweep can only answer via live lookup "
             "are not exercised; in this mode they abstain.")
    L.append("- **Synthetic content**: entity names are invented so no KB memorisation is "
             "possible; that also means these are not MMLU-style knowledge questions.")
    L.append("- **CPU-only, 1 process, enable_ml=False**: sentiment/NER embedder engines in "
             "the full pipeline were not enabled (they add latency, not accuracy here).")
    L.append("- **Standard ML workloads (BERT/ResNet on public data): NOT RUN — RESOURCE "
             "LIMITATION** (no GPU, ~2.4 GB free RAM, CPU-only policy).")
    L.append("- **Throughput measured on a warm process**; first-ever cold start includes "
             f"~{_fmt(summary.get('warmup_block_ms', 0) / 1000, 0)} s of model loading.")
    L.append("")
    L.append("## 16. Conclusion")
    L.append("")
    L.append("This benchmark measures what Sweep's `ReasoningCortex` actually does on "
             "deterministic reasoning cases **without internet access**. The strongest "
             "measured areas and weakest measured areas are stated in the analysis and "
             "final report sections. Results here are raw measured numbers on the "
             "specified machine; they are not comparable to public LLM benchmarks "
             "because datasets differ.")
    L.append("")
    return "\n".join(L)


def main() -> None:
    text = write_report()
    Path("benchmark/REPORT.md").write_text(text, encoding="utf-8")
    print("wrote benchmark/REPORT.md")
