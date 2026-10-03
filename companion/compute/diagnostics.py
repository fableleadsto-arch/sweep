"""Compute diagnostics — structured report + CLI.

Produces the single source of truth for "what hardware/frameworks/backends
does this host have, which ones work, and what should I install next". Used by:

  * the CLI:   python -m companion.compute.diagnostics
               python -m companion.compute.diagnostics install <backend> [--dry-run]
  * the API:   GET /api/brain/compute/diagnostics
  * the UI:    Settings -> Compute -> Run diagnostics

Smoke tests execute a tiny native op on each enabled backend (NumPy matmul,
a CUDA tensor product, an ONNX provider listing, ...) so "available" is
verified rather than assumed. These are fast and safe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .backend_manager import BackendManager
from .capability_detector import ComputeEnvironment, detect
from .base import ComputeTask

# Backends that run a smoke test during diagnostics (enabled + available only).
_SMOKE_BACKENDS = ("cpu", "pytorch", "tensorflow", "onnx", "cuda", "rocm", "mps")

# Heavy frameworks whose import cost (e.g. TensorFlow ~60s) is only worth
# paying when the operator explicitly asks for a full check.
_HEAVY_SMOKE_BACKENDS = ("tensorflow",)


@dataclass
class DiagnosticsReport:
    """Full diagnostics payload (JSON-serializable)."""

    generated_at: str
    python: dict[str, Any]
    os: dict[str, str]
    cpu: dict[str, Any]
    memory: dict[str, Any]
    disk: dict[str, Any]
    gpus: list[dict[str, Any]]
    compute_devices: list[str]
    frameworks: list[dict[str, Any]]
    backends: list[dict[str, Any]]
    smoke_tests: dict[str, dict[str, Any]]
    recommendations: list[str]
    status: str  # "ready" | "degraded" | "minimal"
    native_models: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "python": self.python,
            "os": self.os,
            "cpu": self.cpu,
            "memory": self.memory,
            "disk": self.disk,
            "gpus": self.gpus,
            "compute_devices": self.compute_devices,
            "frameworks": self.frameworks,
            "backends": self.backends,
            "smoke_tests": self.smoke_tests,
            "recommendations": self.recommendations,
            "status": self.status,
            "native_models": self.native_models,
        }


def _env_to_dicts(env: ComputeEnvironment) -> dict[str, Any]:
    return {
        "python": asdict(env.python),
        "os": {"name": env.os_name, "release": env.os_release},
        "cpu": asdict(env.cpu),
        "memory": asdict(env.memory),
        "disk": asdict(env.disk),
        "gpus": [asdict(g) for g in env.gpus],
        "compute_devices": env.compute_devices,
        "frameworks": [
            {
                "name": f.name,
                "installed": f.installed,
                "version": f.version,
                "profile": f.profile,
                "profile_requirements": f.profile_requirements,
            }
            for f in env.frameworks.values()
        ],
    }


def _run_smokes(
    manager: BackendManager, backends: list, include_heavy: bool = False
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for backend in backends:
        if backend.id not in _SMOKE_BACKENDS:
            continue
        if not include_heavy and backend.id in _HEAVY_SMOKE_BACKENDS:
            results[backend.id] = {"ok": False, "skipped": True, "note": "heavy import skipped (use --full)"}
            continue
        if not manager.is_enabled(backend.id) or not backend.available:
            results[backend.id] = {"ok": False, "skipped": True}
            continue
        started = time.time()
        try:
            result = backend.run(ComputeTask.smoke())
            results[backend.id] = {
                "ok": result.ok,
                "device": result.device,
                "summary": result.summary,
                "error": result.error,
                "result": result.result,
                "duration_ms": round((time.time() - started) * 1000, 1),
            }
        except Exception as exc:  # noqa: BLE001 - a failing smoke must not kill the report
            results[backend.id] = {
                "ok": False,
                "error": str(exc)[:500],
                "duration_ms": round((time.time() - started) * 1000, 1),
            }
    return results


def _recommendations(env: ComputeEnvironment, manager: BackendManager, backends: list) -> list[str]:
    recs: list[str] = []
    for backend in backends:
        if backend.id in ("cpu",) or backend.available:
            continue
        if manager.is_enabled(backend.id):
            recs.append(f"{backend.label} is enabled but unavailable — {backend.reason() or backend.install_hint}")
    if not env.has_framework("torch") and not env.has_framework("tensorflow"):
        recs.append(
            "No deep-learning framework installed. "
            "pip install -r requirements.companion-pytorch.txt  (or the -tensorflow / -onnx profile)."
        )
    if env.gpus and not env.has_framework("torch"):
        recs.append(
            "GPUs detected but no PyTorch — install the CUDA profile for GPU compute "
            "(python -m companion.compute.diagnostics install cuda)."
        )
    if not env.gpus and env.has_framework("torch"):
        recs.append("No GPU detected — compute will run on CPU only.")
    return recs


def diagnose(settings: Any = None, include_heavy: bool = False) -> DiagnosticsReport:
    """Build the full diagnostics report (fast; smoke tests only on enabled backends).

    ``include_heavy`` runs smoke tests on slow-import frameworks (TensorFlow).
    """
    env = detect()
    manager = BackendManager(settings)
    backends = manager.backends.values()
    statuses = [b.status(enabled=manager.is_enabled(b.id)).__dict__ for b in backends]
    smokes = _run_smokes(manager, list(backends), include_heavy=include_heavy)
    recs = _recommendations(env, manager, list(backends))

    working = sum(1 for s in statuses if s.get("available") and s.get("enabled"))
    status = "ready"
    if working <= 1:
        status = "minimal"
    elif any(not s.get("available") for s in statuses):
        status = "degraded"

    base = _env_to_dicts(env)

    native_models: list[dict[str, Any]] = []
    try:
        from ..neural.registry import ModelRegistry

        registry_dir = Path(__file__).resolve().parent.parent.parent / "data" / "neural"
        if registry_dir.is_dir():
            native_models = [m.to_dict() for m in ModelRegistry(registry_dir).list_models()]
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        native_models = [{"error": str(exc)[:300]}]

    return DiagnosticsReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=base["python"],
        os=base["os"],
        cpu=base["cpu"],
        memory=base["memory"],
        disk=base["disk"],
        gpus=base["gpus"],
        compute_devices=base["compute_devices"],
        frameworks=base["frameworks"],
        backends=statuses,
        smoke_tests=smokes,
        recommendations=recs,
        status=status,
        native_models=native_models,
    )


# ── CLI ─────────────────────────────────────────────────────────────────

def _fmt_mb(mb: int) -> str:
    return f"{mb / 1024:.1f} GB" if mb > 1024 else f"{mb} MB"


def _render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("RelayAI Compute Diagnostics")
    lines.append("=" * 64)
    lines.append(
        f"Python {report['python']['version']} ({report['python']['implementation']}, "
        f"{report['python']['bitness']}-bit) | {report['os']['name']} {report['os']['release']}"
    )
    lines.append(
        f"CPU: {report['cpu']['physical_cores']} physical / {report['cpu']['logical_cores']} logical "
        f"({report['cpu']['arch']})"
    )
    lines.append(
        f"RAM: {_fmt_mb(report['memory']['total_mb'])}"
        + (f" ({_fmt_mb(report['memory']['available_mb'])} free)" if report['memory']['available_mb'] else "")
    )
    lines.append(f"Disk: {_fmt_mb(report['disk']['free_mb'])} free")
    if report["gpus"]:
        for gpu in report["gpus"]:
            lines.append(
                f"GPU: {gpu['name']} [{gpu['api']}:{gpu['index']}] "
                f"{_fmt_mb(gpu['memory_total_mb'])}"
                + (f" ({_fmt_mb(gpu['memory_free_mb'])} free)" if gpu["memory_free_mb"] else "")
                + (f" driver={gpu['driver']}" if gpu.get("driver") else "")
            )
    else:
        lines.append("GPU: none detected (CPU-only compute)")
    lines.append("")
    lines.append("Frameworks:")
    installed = [f for f in report["frameworks"] if f["installed"]]
    for f in report["frameworks"]:
        mark = "  " if f["installed"] else "✗ "
        lines.append(f"  {mark}{f['name']:<16} {f['version'] or '(missing)'}  [{f['profile']}]")
    lines.append(f"  ({len(installed)}/{len(report['frameworks'])} frameworks installed)")
    lines.append("")
    lines.append("Backends:")
    for b in report["backends"]:
        state = "ON " if b["enabled"] else "OFF"
        avail = "ready" if b["available"] else "MISSING"
        extra = f" — {b['reason']}" if not b["available"] and b.get("reason") else ""
        devices = ""
        if b.get("devices"):
            ids = [d["id"] for d in b["devices"]]
            devices = f" devices={','.join(ids)}"
        lines.append(f"  {b['id']:<12} {state} {avail:<8} v{b['version'] or '-'}{devices}{extra}")
    lines.append("")
    if report["smoke_tests"]:
        lines.append("Smoke tests:")
        for backend_id, test in report["smoke_tests"].items():
            if test.get("skipped"):
                lines.append(f"  {backend_id:<12} skipped (disabled/unavailable)")
                continue
            ok = "PASS" if test.get("ok") else "FAIL"
            detail = test.get("summary") or test.get("error") or ""
            lines.append(f"  {backend_id:<12} {ok:<5} {detail} ({test.get('duration_ms', 0)} ms)")
        lines.append("")
    lines.append(f"Status: {report['status'].upper()}")
    if report["recommendations"]:
        lines.append("")
        lines.append("Recommendations:")
        for rec in report["recommendations"]:
            lines.append(f"  • {rec}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RelayAI compute diagnostics")
    sub = parser.add_subparsers(dest="command")

    install = sub.add_parser("install", help="Install a backend through the controlled wheel path")
    install.add_argument("backend", help="backend id: pytorch, tensorflow, onnx, cuda")
    install.add_argument("--dry-run", action="store_true", default=True, help="validate without installing (default)")
    install.add_argument("--yes", action="store_true", help="allow the real install (mutates the environment)")

    sub.add_parser("smoke", help="Run backend smoke tests (full import check, may be slow)")

    args = parser.parse_args(argv)

    if args.command == "install":
        from .backend_manager import INSTALLABLE_WHEELS

        if args.backend not in INSTALLABLE_WHEELS:
            print(f"No controlled wheel install path for '{args.backend}'.")
            print("Install its framework profile manually: pip install -r <profile>")
            return 2
        manager = BackendManager()
        dry_run = not args.yes
        result = manager.install(args.backend, dry_run=dry_run)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    if args.command == "smoke":
        report = diagnose(include_heavy=True)
    else:
        report = diagnose()
    print(_render(report.to_dict()))
    if report.to_dict()["status"] == "minimal":
        return 3  # non-zero so scripts can detect a bare environment
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
