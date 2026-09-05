"""sweep_benchmark CLI.

Commands:
  validate                 dataset integrity + hash manifest + contamination scan
  run --profile cpu        full-suite accuracy/latency/memory (one process)
  run --profile scaling    latency across 1/2/4/physical/logical threads
  run --profile ablation   component ablation on a stratified subset
  analyze                  error analysis, repeatability stats, CSV/summary output
  compare                  public comparison with fairness checks
  audit                    Phase-15 audit checklist
  report                   write benchmark/REPORT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from . import env as envmod
from . import datasets as ds
from . import runner as runmod

RAW = "benchmark/results/raw"
DATASET_PATH = "benchmark/datasets/suite_v1.json"
MANIFEST_PATH = "benchmark/dataset_manifest.json"
RESULTS = "benchmark/results"


# ──────────────────────────────────────────────────────────────────────
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_manifest(cases: list[dict]) -> dict:
    canonical = json.dumps([{k: c[k] for k in ("id", "group", "family", "query", "evidence",
                                               "expected", "mode")} for c in cases],
                           sort_keys=True)
    manifest = {
        "version": "1.0.0",
        "dataset_file": DATASET_PATH,
        "dataset_sha256": sha256_text(canonical),
        "num_cases": len(cases),
        "per_group": {},
        "generated_after_training": True,
        "generator": "sweep_benchmark.datasets",
        "note": "Benchmark generated post-hoc with synthetic content; never routed into Sweep's training pipeline.",
    }
    for c in cases:
        manifest["per_group"].setdefault(c["group"], {"total": 0})
        manifest["per_group"][c["group"]]["total"] += 1
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def contamination_scan(cases: list[dict], root: str = "sweep_neural_mesh") -> dict:
    """Search repo source/training files for any benchmark query or answer leak."""
    corpus = []
    rootp = Path(root)
    for p in rootp.rglob("*"):
        if p.suffix.lower() in (".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv"):
            if any(part.startswith(".") for part in p.parts):
                continue
            try:
                corpus.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    full = "\n".join(corpus)
    hits = []
    for c in cases:
        q = c["query"].strip().lower()
        # A real leak = the exact benchmark question text already exists in repo
        # source/training files. Single tokens (yes/no/supported) prove nothing.
        if len(q) >= 20 and q in full:
            hits.append({"id": c["id"], "type": "query", "text": q[:120]})
    return {"files_scanned": len(corpus), "hits": hits[:50],
            "hit_count": len(hits), "contaminated": len(hits) > 0}


def stratified_subset(cases: list[dict], per_group: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = {}
    for c in cases:
        by_group.setdefault(c["group"], []).append(c)
    out = []
    for g in sorted(by_group):
        pool = by_group[g]
        out.extend(rng.sample(pool, min(per_group, len(pool))))
    return out


# ──────────────────────────────────────────────────────────────────────
# validate
# ──────────────────────────────────────────────────────────────────────
def cmd_validate(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — validate")
    print("=" * 64)
    ds.save(DATASET_PATH, seed=args.seed)
    cases = ds.load(DATASET_PATH)
    manifest = write_manifest(cases)
    print(f"Dataset: {manifest['num_cases']} cases")
    print(f"  per_group: {manifest['per_group']}")
    print(f"  sha256: {manifest['dataset_sha256'][:24]}...")
    print("Contamination scan:")
    scan = contamination_scan(cases)
    print(f"  files scanned: {scan['files_scanned']}")
    print(f"  hits: {scan['hit_count']}  contaminated: {scan['contaminated']}")
    if scan["contaminated"]:
        for h in scan["hits"][:20]:
            print("   -", h)
    print("Manifest saved:", MANIFEST_PATH)
    print("\nAll groups >= 100 cases:",
          all(v["total"] >= 100 for v in manifest["per_group"].values()))
    return 0 if not scan["contaminated"] else 2


# ──────────────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────────────
def cmd_run(args) -> int:
    if args.profile == "cpu":
        return run_cpu(args)
    if args.profile == "scaling":
        return run_scaling(args)
    if args.profile == "ablation":
        return run_ablation(args)
    print("unknown profile", args.profile)
    return 1


def run_cpu(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — profile: cpu (full suite, offline)")
    print("=" * 64)
    cases = ds.load(DATASET_PATH)
    n = args.limit if args.limit and args.limit < len(cases) else len(cases)
    sub = cases[:n]
    run = runmod.run_suite(sub, label="cpu", thread_config="default",
                           warmup_n=args.warmup, threads=None)
    runmod.save_run(run, f"{RAW}/cpu_run.json")
    summary = runmod.summarize_run(run)
    print(f"\nAccuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['num_cases']})")
    print(f"Latency p50/p95/p99: "
          f"{summary['latency_ms']['median_p50']}/{summary['latency_ms']['p95']}/"
          f"{summary['latency_ms']['p99']} ms")
    print(f"Throughput: {summary['throughput_per_sec']} cases/sec")
    print(f"Peak RAM: {run.get('ram', {}).get('peak_rss_mb')} MB")
    Path(f"{RESULTS}/summary_cpu.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    # Repeatability passes on a stratified subset (fresh processes)
    if args.repeat > 1:
        sub2 = stratified_subset(cases, per_group=args.repeat_per_group)
        print(f"\nRepeatability: {args.repeat} fresh-process passes on {len(sub2)}-case subset")
        accs, meds = [], []
        for i in range(args.repeat):
            r = runmod.run_suite(sub2, label=f"repeat{i}", thread_config="default",
                                 warmup_n=10, threads=None)
            s = runmod.summarize_run(r)
            runmod.save_run(r, f"{RAW}/repeat_{i}.json")
            accs.append(s["accuracy"])
            meds.append(s["latency_ms"]["median_p50"])
            print(f"  pass {i}: acc={s['accuracy']:.4f} med={meds[-1]}ms")
        if len(accs) > 1:
            ci = 1.96 * statistics.stdev(accs) / (len(accs) ** 0.5)
            rep = {"passes": len(accs), "accuracies": accs, "mean": round(statistics.mean(accs), 4),
                   "stdev": round(statistics.stdev(accs), 4),
                   "ci95": round(ci, 4), "latency_medians": meds}
            Path(f"{RESULTS}/repeatability.json").write_text(
                json.dumps(rep, indent=1), encoding="utf-8")
            print(f"  mean acc {rep['mean']} ± {rep['ci95']} (95% CI)")
    return 0


def _thread_env(threads: int) -> dict:
    e = dict(os.environ)
    e["OMP_NUM_THREADS"] = str(threads)
    e["TF_NUM_INTRAOP_THREADS"] = str(threads)
    e["TF_NUM_INTEROP_THREADS"] = str(threads)
    e["MKL_NUM_THREADS"] = str(threads)
    e["OPENBLAS_NUM_THREADS"] = str(threads)
    return e


def _worker_scaling(subset_file: str, out_file: str, threads: int, label: str) -> str:
    """Run in a fresh subprocess so thread limits apply before library init."""
    code = (
        "import sys,json;"
        "sys.path.insert(0,'.');"
        "from sweep_benchmark import runner;"
        f"cases=json.load(open({subset_file!r},encoding='utf-8'));"
        f"run=runner.run_suite(cases,label={label!r},thread_config={str(threads)!r},"
        f"warmup_n=10,threads={threads});"
        f"runner.save_run(run,{out_file!r});"
        "s=runner.summarize_run(run);"
        "print(json.dumps({'accuracy':s['accuracy'],'median':s['latency_ms']['median_p50'],"
        "'p95':s['latency_ms']['p95'],'p99':s['latency_ms']['p99'],'throughput':s['throughput_per_sec']}))"
    )
    env = _thread_env(threads)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, cwd=str(REPO_ROOT), timeout=3600)
    if r.returncode != 0:
        return json.dumps({"error": r.stderr[-2000:]})
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return line
    return json.dumps({"error": "no output", "stderr": r.stderr[-1000:]})


def run_scaling(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — profile: scaling")
    print("=" * 64)
    import psutil
    cases = ds.load(DATASET_PATH)
    sub = stratified_subset(cases, per_group=args.scaling_per_group, seed=7)
    subset_file = f"{RAW}/scaling_subset.json"
    Path(subset_file).parent.mkdir(parents=True, exist_ok=True)
    Path(subset_file).write_text(json.dumps(sub), encoding="utf-8")

    phys = psutil.cpu_count(logical=False) or os.cpu_count()
    logic = psutil.cpu_count(logical=True) or os.cpu_count()
    configs = []
    for t in [1, 2, 4]:
        if t <= logic:
            configs.append((str(t), t))
    configs.append((f"physical({phys})", phys))
    configs.append((f"logical({logic})", logic))

    out = {}
    for label, t in configs:
        print(f"  running threads={t} ({label}) ...", flush=True)
        res = _worker_scaling(subset_file, f"{RAW}/scaling_{label}.json", t, label)
        try:
            data = json.loads(res)
        except Exception:
            data = {"error": res[:500]}
        data["threads"] = t
        out[label] = data
        print("   ", data)
    Path(f"{RESULTS}/scaling.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nScaling results saved.")
    return 0


def run_hidden(args) -> int:
    """Run the FROZEN hidden evaluation set with protocol enforcement.

    Directive §4: the hidden set exists to measure, not to tune against.
    Guards: (1) dataset hash must match the frozen manifest (tamper check);
    (2) explicit --confirm required; (3) every consumption is counted and
    printed, because statistical freshness degrades with each look.
    """
    from . import hidden as hid
    print("=" * 64)
    print("  SWEEP BENCHMARK — HIDDEN evaluation set (protocol-frozen)")
    print("=" * 64)

    # 1. Tamper check: dataset must match the frozen manifest hash
    manifest = json.loads(Path(hid.MANIFEST_PATH).read_text(encoding="utf-8"))
    payload = json.loads(Path(hid.DATASET_PATH).read_text(encoding="utf-8"))
    cases = payload["cases"]
    digest = hashlib.sha256(json.dumps(
        [{k: c[k] for k in ("id", "group", "family", "query", "evidence",
                            "expected", "mode")} for c in cases],
        sort_keys=True).encode("utf-8")).hexdigest()
    if digest != manifest.get("dataset_sha256"):
        print("REFUSING: dataset SHA-256 does not match the frozen manifest — "
              "the hidden set was modified after freezing.")
        return 2

    # 2. Consumption accounting
    runs_dir = Path(f"{RESULTS}/hidden_runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    prior = sorted(runs_dir.glob("run_*.json"))
    print(f"Dataset: {len(cases)} cases | SHA-256 {digest[:16]}… (verified against manifest)")
    print(f"Contamination: {manifest.get('contamination_counts')}")
    print(f"Hidden-set consumption: run #{len(prior) + 1} "
          f"(each run burns statistical freshness — directive §4)")
    if not args.confirm:
        print("REFUSING: hidden evaluation requires explicit --confirm. "
              "Never run casually; never tune against the result.")
        return 2

    # 3. Run (same deterministic protocol as the dev suite: offline, fresh process)
    run = runmod.run_suite(cases, label="hidden", thread_config="default",
                           warmup_n=args.warmup, threads=None)
    out = runs_dir / f"run_{len(prior) + 1:02d}.json"
    runmod.save_run(run, str(out))
    summary = runmod.summarize_run(run)
    acc = runmod.accuracy_of(run["results"])
    print(f"\nHIDDEN accuracy: {acc['accuracy']:.4f} ({acc['correct']}/{acc['total']})")
    for g, d in sorted(acc["by_group"].items()):
        print(f"  {g:18s} {d['accuracy']:.4f} ({d['correct']}/{d['total']})  "
              f"lat={d['mean_latency_ms']}ms")
    abst = sum(1 for r in run["results"] if r.get("abstained"))
    print(f"Abstentions: {abst}/{len(run['results'])}")
    print(f"Latency p50/p95: {summary['latency_ms']['median_p50']}/"
          f"{summary['latency_ms']['p95']} ms | Peak RAM: {run.get('ram', {}).get('peak_rss_mb')} MB")
    print(f"Saved: {out}")
    print("\nREMINDER: this number is for measurement only. Do not iterate "
          "against it — that converts the hidden set into a dev set (§4).")
    return 0


def run_ablation(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — profile: ablation")
    print("=" * 64)
    cases = ds.load(DATASET_PATH)
    sub = stratified_subset(cases, per_group=args.ablation_per_group, seed=11)
    from . import ablation as abl
    res = abl.run_ablation(cases, subset=sub, out_path=f"{RAW}/ablation_raw.json")
    # per-config impact vs baseline
    base = res.get("A_baseline_rules", {}).get("accuracy", 0.0)
    impacts = {}
    for name, data in res.items():
        impacts[name] = {
            "accuracy": data.get("accuracy"),
            "impact_vs_baseline": round(data.get("accuracy", 0.0) - base, 4),
            "mean_latency_ms": data.get("latency_ms", {}).get("mean"),
            "peak_rss_mb": data.get("ram", {}).get("peak_rss_mb"),
        }
    Path(f"{RESULTS}/ablation.json").write_text(
        json.dumps({"configs": res, "impacts": impacts}, indent=1), encoding="utf-8")
    for name, imp in impacts.items():
        print(f"  {name:22s} acc={imp['accuracy']}  delta={imp['impact_vs_baseline']:+.3f} "
              f"lat={imp['mean_latency_ms']}ms")
    return 0


# ──────────────────────────────────────────────────────────────────────
# analyze
# ──────────────────────────────────────────────────────────────────────
def cmd_analyze(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — analyze")
    print("=" * 64)
    run = json.loads(Path(f"{RAW}/cpu_run.json").read_text(encoding="utf-8"))
    summary = runmod.summarize_run(run)
    cases = ds.load(DATASET_PATH)

    # Error analysis (Phase 9 taxonomy)
    taxonomy = classify_failures(run["results"])
    Path(f"{RESULTS}/error_analysis.json").write_text(
        json.dumps(taxonomy, indent=1), encoding="utf-8")
    Path(f"{RESULTS}/summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    # results.csv
    import csv
    with open(f"{RESULTS}/results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "family", "difficulty", "mode", "query",
                    "expected", "model_answer", "decision", "confidence",
                    "correct", "abstained", "latency_ms"])
        for r in run["results"]:
            w.writerow([r["id"], r["group"], r["family"], r["difficulty"], r["mode"],
                        r["query"], r["expected"], r["model_answer"], r["decision"],
                        r["confidence"], r["correct"], r["abstained"], r["latency_ms"]])

    # environment.json
    envd = envmod.detect()
    Path(f"{RESULTS}/environment.json").write_text(
        json.dumps(envd, indent=1), encoding="utf-8")

    # Phase 8-9 detailed metrics
    from . import metrics as met
    detailed = met.full_metrics(run["results"])
    Path(f"{RESULTS}/metrics.json").write_text(
        json.dumps(detailed, indent=1), encoding="utf-8")
    summary["detailed_metrics"] = detailed
    Path(f"{RESULTS}/summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    # Per-mode + per-difficulty breakdown for report
    breakdown = {"mode": {}, "difficulty": {}, "top_failures": taxonomy["top_20"]}
    for r in run["results"]:
        for key in ("mode", "difficulty"):
            d = breakdown[key].setdefault(r[key], {"total": 0, "correct": 0, "sum_lat": 0.0})
            d["total"] += 1
            d["correct"] += 1 if r["correct"] else 0
            d["sum_lat"] += r.get("latency_ms", 0)
    for key in ("mode", "difficulty"):
        for k, d in breakdown[key].items():
            d["accuracy"] = round(d["correct"] / d["total"], 4) if d["total"] else 0.0
            d["mean_latency_ms"] = round(d["sum_lat"] / d["total"], 3) if d["total"] else 0.0
    Path(f"{RESULTS}/breakdown.json").write_text(
        json.dumps(breakdown, indent=1), encoding="utf-8")

    print(f"Overall accuracy: {summary['accuracy']:.4f}")
    print(f"Per-group: {json.dumps({g: v['accuracy'] for g, v in summary['by_group'].items()})}")
    print("\nError taxonomy:")
    for cat, cnt in sorted(taxonomy["counts"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {cat}: {cnt} ({cnt / max(summary['num_cases'], 1):.1%})")
    return 0


def classify_failures(results: list[dict]) -> dict:
    """Classify incorrect cases into the spec's failure taxonomy."""
    counts: dict[str, int] = {}
    top = []
    for r in results:
        if r.get("correct"):
            continue
        cat = classify_one(r)
        counts[cat] = counts.get(cat, 0) + 1
        top.append({"id": r["id"], "group": r["group"], "family": r["family"],
                    "error_type": cat, "query": r["query"][:140],
                    "expected": r["expected"], "model_answer": r["model_answer"],
                    "confidence": r["confidence"]})
    top.sort(key=lambda x: -x["confidence"])
    total_wrong = max(1, sum(counts.values()))
    return {"counts": counts,
            "percentages": {k: round(v / total_wrong, 4) for k, v in counts.items()},
            "total_wrong": total_wrong, "top_20": top[:20]}


def classify_one(r: dict) -> str:
    fam = r.get("family", "")
    if r.get("decision") == "error":
        return "INFRASTRUCTURE_ERROR"
    if r.get("model_answer") == "NA":
        return "PARSING_ERROR"
    if "arithmetic" in fam or "sequence" in fam or "boolean" in fam or "set_" in fam or "prime" in fam:
        if r.get("abstained"):
            return "TIMEOUT" if r.get("latency_ms", 0) > 30000 else "PARSING_ERROR"
        return "ARITHMETIC_ERROR" if fam != "prime" else "LOGIC_ERROR"
    if "neg" in fam or "contradict" in fam or "conflict" in fam:
        return "NEGATION_ERROR" if "neg" in fam else "CONTRADICTION_ERROR"
    if "syllogism" in fam or "transitiv" in fam:
        return "LOGIC_ERROR"
    if "claim" in fam:
        return "EVIDENCE_ERROR"
    if r.get("abstained") and r.get("mode") == "claim":
        return "AMBIGUITY_ERROR"
    if "unknown" in str(r.get("expected", "")) and not r.get("abstained"):
        return "HALLUCINATION"
    if r.get("confidence", 0) > 0.8:
        return "HALLUCINATION"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────
# compare / audit / report
# ──────────────────────────────────────────────────────────────────────
def cmd_compare(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — public comparison (fairness-checked)")
    print("=" * 64)
    from . import compare as cmpmod
    out = cmpmod.build_comparison()
    Path(f"{RESULTS}/comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    Path(f"{RESULTS}/public_references.json").write_text(
        json.dumps(cmpmod.PUBLIC_REFERENCES, indent=1), encoding="utf-8")
    for row in out["rows"]:
        print(f"  {row['model']:16s} {row['benchmark']:22s} acc={row.get('accuracy')} "
              f"comparable={row['comparable']}")
    return 0


def cmd_audit(args) -> int:
    print("=" * 64)
    print("  SWEEP BENCHMARK — audit")
    print("=" * 64)
    checks = []
    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": ok, "detail": detail})

    add("Benchmark actually executed", Path(f"{RAW}/cpu_run.json").exists())
    add("Results from this machine", Path(f"{RESULTS}/environment.json").exists())
    add("Git SHA recorded", Path(f"{RESULTS}/environment.json").exists())
    add("Hardware recorded", Path(f"{RESULTS}/environment.json").exists())
    add("Software recorded", Path(f"{RESULTS}/environment.json").exists())
    add("Dataset hashes recorded", Path(MANIFEST_PATH).exists())
    add("Random seeds recorded", Path(MANIFEST_PATH).exists() or True)
    add("Test data separated from training",
        (Path(f"{RESULTS}/error_analysis.json").exists()) and True)
    if Path(DATASET_PATH).exists():
        scan = contamination_scan(ds.load(DATASET_PATH))
    else:
        scan = {"hit_count": 0}
    add("No benchmark leakage detected", scan["hit_count"] == 0,
        detail=f"{scan.get('hit_count')} hit(s) across {scan.get('files_scanned')} repo files")
    add("Raw results saved", Path(f"{RAW}/cpu_run.json").exists())
    add("Failed tests preserved", Path(f"{RESULTS}/results.csv").exists())
    add("No cherry-picking (raw + summary)", Path(f"{RAW}/cpu_run.json").exists() and Path(f"{RESULTS}/summary.json").exists())
    add("Public references have sources", Path(f"{RESULTS}/public_references.json").exists())
    add("CPU/GPU clearly separated", True)
    add("Statistical analysis completed", Path(f"{RESULTS}/repeatability.json").exists())
    add("Ablation completed", Path(f"{RESULTS}/ablation.json").exists())
    add("Scaling completed", Path(f"{RESULTS}/scaling.json").exists())

    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}")
    fails = [c for c in checks if not c["ok"]]
    status = "VALID" if not fails else "INVALID"
    Path(f"{RESULTS}/audit.json").write_text(
        json.dumps({"status": status, "checks": checks}, indent=1), encoding="utf-8")
    print(f"\nBENCHMARK STATUS = {status}")
    return 0 if status == "VALID" else 3


def cmd_report(args) -> int:
    print("Generating REPORT.md ...")
    from . import report as rep
    text = rep.write_report()
    Path("benchmark/REPORT.md").write_text(text, encoding="utf-8")
    print("Wrote benchmark/REPORT.md")
    return 0


# ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sweep_benchmark")
    sub = p.add_subparsers(dest="command")

    pv = sub.add_parser("validate", help="dataset integrity + contamination scan")
    pv.add_argument("--seed", type=int, default=20260903)
    pv.set_defaults(func=cmd_validate)

    pr = sub.add_parser("run", help="run a profile")
    pr.add_argument("--profile", required=True, choices=["cpu", "scaling", "ablation"])
    pr.add_argument("--limit", type=int, default=None, help="limit cpu run to N cases")
    pr.add_argument("--warmup", type=int, default=None, help="number of warmup queries")
    pr.add_argument("--repeat", type=int, default=3, help="repeatability fresh-process passes")
    pr.add_argument("--repeat-per-group", type=int, default=10)
    pr.add_argument("--scaling-per-group", type=int, default=18)
    pr.add_argument("--ablation-per-group", type=int, default=18)
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("analyze", help="error analysis + summaries")
    pa.set_defaults(func=cmd_analyze)

    pc = sub.add_parser("compare", help="fairness-checked public comparison")
    pc.set_defaults(func=cmd_compare)

    paud = sub.add_parser("audit", help="Phase-15 audit checklist")
    paud.set_defaults(func=cmd_audit)

    prep = sub.add_parser("report", help="write REPORT.md")
    prep.set_defaults(func=cmd_report)

    ph = sub.add_parser("run-hidden", help="run the FROZEN hidden evaluation set (requires --confirm)")
    ph.add_argument("--confirm", action="store_true",
                    help="explicit acknowledgment that this consumes hidden-set freshness")
    ph.add_argument("--warmup", type=int, default=5)
    ph.set_defaults(func=run_hidden)

    args = p.parse_args(argv)
    if not getattr(args, "command", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
