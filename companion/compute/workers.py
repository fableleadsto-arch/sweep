"""Compute worker protocol — local + distributed training/inference workers.

The brain service keeps heavy frameworks out of the request path: long-running
training (and optionally inference) runs in a *separate environment*. This
module defines the worker contract and ships a local subprocess worker that
spawns ``python -m companion.compute.worker`` in isolation.

The interface is transport-agnostic so a distributed fleet can swap the local
worker for a remote one without touching callers — a remote implementation
only needs to translate submit/status/cancel onto its wire protocol.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class WorkerJob:
    """One unit of work for a worker environment."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str = "training"  # training | inference
    backend: str = ""  # pytorch | tensorflow | cuda | ...
    spec: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkerStatus:
    """Snapshot of a worker's state."""

    job_id: str
    state: str  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class ComputeWorker(ABC):
    """Contract any worker environment implements."""

    @abstractmethod
    def submit(self, job: WorkerJob) -> str:
        """Queue ``job`` and return its id."""

    @abstractmethod
    def status(self, job_id: str) -> WorkerStatus:
        """Current state of a job."""

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a job; True if accepted for cancellation."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources (stop the local pool, close connections)."""


class LocalSubprocessWorker(ComputeWorker):
    """Runs jobs in an isolated `python -m companion.compute.worker` process.

    Heavy frameworks import only in the worker process — the caller stays
    lightweight. Status is persisted to a per-job JSON file in ``workdir``.
    """

    def __init__(self, workdir: str | Path = ".relayhub/workers") -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}

    def _status_path(self, job_id: str) -> Path:
        self._validate_id(job_id)
        return self.workdir / f"{job_id}.status.json"

    def _spec_path(self, job_id: str) -> Path:
        self._validate_id(job_id)
        return self.workdir / f"{job_id}.spec.json"

    @staticmethod
    def _validate_id(job_id: str) -> None:
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", job_id):
            raise ValueError("invalid worker job id")
        if job_id.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
            *(f"LPT{i}" for i in range(10)),
        }:
            raise ValueError("reserved worker job id")

    def submit(self, job: WorkerJob) -> str:
        spec = self._spec_path(job.id)
        # Exclusive creation prevents resubmission from replacing an active job.
        with spec.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(job.spec, allow_nan=False))
        status_path = self._status_path(job.id)
        with status_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(WorkerStatus(job_id=job.id, state="queued").to_dict()))
        command = self._command(job, spec, status_path)
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self._write_status(job.id, "failed", error=f"failed to spawn worker: {exc}")
            raise
        self._procs[job.id] = proc
        return job.id

    def _command(self, job: WorkerJob, spec: Path, status_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "companion.compute.worker",
            "--job",
            job.id,
            "--backend",
            job.backend or "",
            "--kind",
            job.kind,
            "--spec",
            str(spec),
            "--status",
            str(status_path),
        ]

    def status(self, job_id: str) -> WorkerStatus:
        path = self._status_path(job_id)
        proc = self._procs.get(job_id)
        running = proc is not None and proc.poll() is None
        try:
            with path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("oversized status record")
            current = WorkerStatus(**json.loads(raw))
            if current.job_id != job_id or current.state not in {
                "queued", "running", "done", "failed", "cancelled",
            }:
                raise ValueError("invalid status record")
        except (OSError, ValueError, TypeError):
            current = WorkerStatus(job_id=job_id, state="unknown", error="no valid status record found")
        if running:
            # A child may write 'done' before exiting. Do not release its slot yet.
            current.state = "running"
            current.result = None
        elif proc is not None:
            if proc.returncode != 0 or current.state not in {"done", "failed", "cancelled"}:
                current.state = "failed"
                current.result = None
                current.error = current.error or f"worker exited without a valid result (code {proc.returncode})"
                current.finished_at = time.time()
                try:
                    path.write_text(json.dumps(current.to_dict()), encoding="utf-8")
                except OSError:
                    # The supervising pool must still learn that the process exited.
                    pass
            self._procs.pop(job_id, None)
        return current

    def cancel(self, job_id: str) -> bool:
        self._validate_id(job_id)
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
            status = WorkerStatus(
                job_id, "cancelled", message="cancelled by caller", finished_at=time.time(),
            )
            self._status_path(job_id).write_text(json.dumps(status.to_dict()), encoding="utf-8")
            self._procs.pop(job_id, None)
            return True
        return False

    def shutdown(self) -> None:
        for job_id in list(self._procs):
            self.cancel(job_id)
            self.status(job_id)

    # ── internal ──────────────────────────────────────────────────────

    def _write_status(
        self,
        job_id: str,
        state: str,
        *,
        progress: float = 0.0,
        message: str = "",
        error: str = "",
    ) -> None:
        current = self.status(job_id)
        current.state = state
        current.progress = progress
        current.message = message or current.message
        current.error = error or current.error
        if state in ("failed", "cancelled"):
            current.result = None
        if state in ("done", "failed", "cancelled"):
            current.finished_at = time.time()
        path = self._status_path(job_id)
        path.write_text(json.dumps(current.to_dict(), default=str), encoding="utf-8")


# ── remote worker (wire-compatible stub for distributed fleets) ─────────

class RemoteWorker(ComputeWorker):
    """Transport-agnostic remote worker interface.

    A distributed deployment implements the same submit/status/cancel contract
    over its wire protocol (HTTP/gRPC/nats). This class documents the shape and
    provides a thin HTTP client example.
    """

    def __init__(self, endpoint: str, api_token: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_token = api_token

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def submit(self, job: WorkerJob) -> str:
        import httpx

        resp = httpx.post(
            f"{self.endpoint}/jobs",
            json={"kind": job.kind, "backend": job.backend, "spec": job.spec},
            headers=self._headers(),
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("id", job.id)

    def status(self, job_id: str) -> WorkerStatus:
        import httpx

        resp = httpx.get(
            f"{self.endpoint}/jobs/{job_id}", headers=self._headers(), timeout=15.0
        )
        resp.raise_for_status()
        return WorkerStatus(**resp.json())

    def cancel(self, job_id: str) -> bool:
        import httpx

        resp = httpx.post(
            f"{self.endpoint}/jobs/{job_id}/cancel", headers=self._headers(), timeout=15.0
        )
        return resp.status_code == 200

    def shutdown(self) -> None:
        return None


__all__ = ["ComputeWorker", "LocalSubprocessWorker", "RemoteWorker", "WorkerJob", "WorkerStatus"]
