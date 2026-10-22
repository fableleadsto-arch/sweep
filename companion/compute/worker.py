"""Subprocess worker entrypoint — `python -m companion.compute.worker`.

Spawned in isolation by `LocalSubprocessWorker` (or an operator/trainer) so
that heavy frameworks (PyTorch, TensorFlow, ...) load only in this process.
Reads a job spec, executes it through the compute backends, and writes status
to a JSON file the caller polls.

Usage:
    python -m companion.compute.worker --job <id> --backend <id> --kind <kind> \\
        --spec <spec.json> --status <status.json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..tools.common import CapabilityUnavailable
from .backend_manager import BackendManager
from .base import ComputeTask, ComputeTaskKind
from .workers import WorkerStatus


def _write(path: Path, status: WorkerStatus) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status.to_dict(), default=str), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows readers/indexers can briefly hold the destination open.
            # Keep the old complete record instead of exposing a partial write.
            if attempt == 5:
                raise
            time.sleep(0.01 * (2 ** attempt))


def _run_worker(job_id: str, backend_id: str, kind: str, spec: Path, status_path: Path) -> int:
    status_path = Path(status_path)
    started_at = time.time()
    manager = BackendManager()
    backend = manager.get(backend_id or "")
    if backend is None:
        _write(status_path, WorkerStatus(job_id=job_id, state="failed", error=f"unknown backend '{backend_id}'"))
        return 2

    try:
        spec_data: dict[str, Any] = json.loads(Path(spec).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _write(status_path, WorkerStatus(job_id=job_id, state="failed", error=f"bad spec: {exc}"))
        return 2

    _write(
        status_path,
        WorkerStatus(job_id=job_id, state="running", progress=0.05, message=f"backend '{backend_id}' loaded", started_at=started_at),
    )

    task = ComputeTask(
        kind=ComputeTaskKind(kind),
        capability=str(spec_data.get("capability") or spec_data.get("task") or ""),
        payload=spec_data,
        framework_hint=backend_id,
        device_preference=str(spec_data.get("device_preference") or ""),
    )
    try:
        result = backend.run(task)
    except CapabilityUnavailable as exc:
        _write(status_path, WorkerStatus(job_id=job_id, state="failed", error=str(exc)))
        return 3
    except Exception as exc:  # noqa: BLE001 - surface any worker failure
        _write(status_path, WorkerStatus(job_id=job_id, state="failed", error=str(exc)))
        return 4

    if not result.ok:
        _write(status_path, WorkerStatus(job_id=job_id, state="failed", error=result.error or result.summary))
        return 5

    _write(
        status_path,
        WorkerStatus(
            job_id=job_id,
            state="done",
            progress=1.0,
            message=result.summary,
            result=result.result,
            started_at=started_at,
            finished_at=time.time(),
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RelayAI compute worker")
    parser.add_argument("--job", required=True, help="job id")
    parser.add_argument("--backend", required=True, help="backend id (pytorch, tensorflow, cuda, cpu, ...)")
    parser.add_argument("--kind", default="capability", help="task kind")
    parser.add_argument("--spec", required=True, help="path to the job spec JSON")
    parser.add_argument("--status", required=True, help="path to write status JSON")
    args = parser.parse_args(argv)

    try:
        return _run_worker(args.job, args.backend, args.kind, args.spec, args.status)
    except Exception as exc:  # noqa: BLE001 - never crash silently
        print(f"worker fatal: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
