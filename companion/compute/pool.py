"""Bounded, owner-scoped dispatch for a single trusted compute host.

Owners must come from an authenticated caller, never an untrusted request field.
This module supplies scheduling, not authentication, an OS sandbox or a network
service. Give it exclusive ownership of its worker. Desktop actions do not belong
in this pool; they remain on the requesting device.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass

from .workers import ComputeWorker, WorkerJob, WorkerStatus

_TERMINAL = frozenset({"done", "failed", "cancelled"})
_KINDS = frozenset({"capability", "smoke", "training", "inference", "transform"})


class QueueFullError(RuntimeError):
    """The host or this owner has reached its outstanding-job limit."""


@dataclass
class _Entry:
    owner: str
    job: WorkerJob | None
    status: WorkerStatus
    backend_id: str = ""
    started: float = 0.0
    timeout_requested: bool = False


class FairComputePool:
    """FIFO within each owner, round-robin between owners, no preemption.

    Limits count both running and queued requests. By default only one heavy
    process can run, preventing additional clients from multiplying model RAM.
    Completed status retention is bounded; expired IDs return the same response
    as unknown or foreign IDs. Worker result files have a separate lifecycle.
    """

    def __init__(
        self,
        worker: ComputeWorker,
        *,
        max_running: int = 1,
        max_running_per_owner: int = 1,
        max_outstanding: int = 32,
        max_per_owner: int = 8,
        history_limit: int = 64,
        job_timeout: float = 300.0,
    ) -> None:
        for name, value, upper in (
            ("max_running", max_running, 32),
            ("max_running_per_owner", max_running_per_owner, 32),
            ("max_outstanding", max_outstanding, 1024),
            ("max_per_owner", max_per_owner, 1024),
            ("history_limit", history_limit, 1024),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")
        if isinstance(job_timeout, bool) or not math.isfinite(job_timeout) or job_timeout <= 0:
            raise ValueError("job_timeout must be finite and positive")
        self._worker = worker
        self._max_running = max_running
        self._max_running_per_owner = max_running_per_owner
        self._max_outstanding = max_outstanding
        self._max_per_owner = max_per_owner
        self._history_limit = history_limit
        self._job_timeout = job_timeout
        self._entries: dict[str, _Entry] = {}
        self._queues: dict[str, deque[str]] = {}
        self._ready: deque[str] = deque()
        self._completed: deque[str] = deque()
        self._last_owner = ""
        self._condition = threading.Condition(threading.RLock())
        self._thread: threading.Thread | None = None
        self._closed = False

    @staticmethod
    def _owner(owner: str) -> None:
        if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", owner):
            raise ValueError("invalid owner identity")

    def submit(self, owner: str, job: WorkerJob) -> str:
        """Admit a bounded JSON snapshot and assign a host-generated job ID."""
        self._owner(owner)
        if job.kind not in _KINDS:
            raise ValueError("unsupported job kind")
        if not isinstance(job.backend, str) or not re.fullmatch(r"[A-Za-z0-9_-]{0,64}", job.backend):
            raise ValueError("invalid backend")
        if type(job.spec) is not dict:
            raise ValueError("job spec must be a JSON object")
        encoded = json.dumps(job.spec, allow_nan=False).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise ValueError("job spec exceeds 64 KiB")
        snapshot = WorkerJob(id=uuid.uuid4().hex, kind=job.kind, backend=job.backend, spec=json.loads(encoded))
        with self._condition:
            if self._closed:
                raise RuntimeError("compute pool is closed")
            active = [e for e in self._entries.values() if e.status.state not in _TERMINAL]
            if len(active) >= self._max_outstanding:
                raise QueueFullError("host queue is full; retry after a job completes")
            if sum(e.owner == owner for e in active) >= self._max_per_owner:
                raise QueueFullError("your queue is full; retry after a job completes")
            self._entries[snapshot.id] = _Entry(
                owner, snapshot, WorkerStatus(snapshot.id, "queued"),
            )
            if owner not in self._queues:
                self._queues[owner] = deque()
                self._ready.append(owner)
            self._queues[owner].append(snapshot.id)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="sweep-compute", daemon=True)
                self._thread.start()
            self._condition.notify_all()
            return snapshot.id

    def status(self, owner: str, job_id: str) -> WorkerStatus:
        self._owner(owner)
        with self._condition:
            entry = self._entries.get(job_id)
            if entry is None or entry.owner != owner:
                return WorkerStatus(job_id, "unknown", error="job not found")
            return copy.deepcopy(entry.status)

    def cancel(self, owner: str, job_id: str) -> bool:
        self._owner(owner)
        with self._condition:
            entry = self._entries.get(job_id)
            if entry is None or entry.owner != owner or entry.status.state in _TERMINAL:
                return False
            if entry.status.state == "queued":
                queue = self._queues[owner]
                queue.remove(job_id)
                if not queue:
                    del self._queues[owner]
                    self._ready.remove(owner)
                self._finish(entry, WorkerStatus(job_id, "cancelled", message="cancelled before start"))
                self._condition.notify_all()
                return True
            try:
                accepted = self._worker.cancel(entry.backend_id)
            except Exception:  # the slot stays occupied until termination is observed
                entry.status.message = "Cancellation failed; worker is still being monitored"
                return False
            if accepted:
                entry.status.message = "Cancellation requested; waiting for worker exit"
                self._condition.notify_all()
            return accepted

    def shutdown(self) -> None:
        """Reject new work, cancel queued work, and stop owned worker processes."""
        with self._condition:
            self._closed = True
            for entry in list(self._entries.values()):
                if entry.status.state == "queued":
                    self.cancel(entry.owner, entry.status.job_id)
            self._worker.shutdown()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("compute workers have not confirmed shutdown")

    def _finish(self, entry: _Entry, status: WorkerStatus) -> None:
        status.job_id = entry.status.job_id
        status.finished_at = status.finished_at or time.time()
        entry.status = status
        entry.job = None
        self._completed.append(status.job_id)
        while len(self._completed) > self._history_limit:
            self._entries.pop(self._completed.popleft(), None)

    def _advance(self) -> None:
        for entry in list(self._entries.values()):
            if entry.status.state != "running":
                continue
            try:
                status = self._worker.status(entry.backend_id)
            except Exception:
                status = None
            if status is not None and status.state in _TERMINAL:
                if entry.timeout_requested and status.state == "cancelled":
                    status.state = "failed"
                    status.error = "job exceeded its time limit"
                self._finish(entry, status)
                continue
            if status is not None and status.state == "running":
                status.job_id = entry.status.job_id
                entry.status = status
            if time.monotonic() - entry.started >= self._job_timeout and not entry.timeout_requested:
                entry.timeout_requested = True
                self.cancel(entry.owner, entry.status.job_id)
                entry.status.message = "Job time limit reached; waiting for worker exit"

        running = Counter(e.owner for e in self._entries.values() if e.status.state == "running")
        while self._ready and sum(running.values()) < self._max_running and not self._closed:
            if len(self._ready) > 1 and self._ready[0] == self._last_owner:
                self._ready.rotate(-1)
            owner = next((o for o in self._ready if running[o] < self._max_running_per_owner), None)
            if owner is None:
                break
            self._ready.remove(owner)
            queue = self._queues[owner]
            entry = self._entries[queue.popleft()]
            if queue:
                self._ready.append(owner)
            else:
                del self._queues[owner]
            self._last_owner = owner
            try:
                assert entry.job is not None
                entry.backend_id = self._worker.submit(entry.job)
            except Exception:
                self._finish(entry, WorkerStatus(entry.status.job_id, "failed", error="worker could not start job"))
                continue
            entry.job = None
            entry.started = time.monotonic()
            entry.status = WorkerStatus(entry.status.job_id, "running", started_at=time.time())
            running[owner] += 1

    def _run(self) -> None:
        with self._condition:
            while True:
                self._advance()
                if not self._ready and not any(e.status.state == "running" for e in self._entries.values()):
                    self._thread = None
                    return
                self._condition.wait(timeout=0.1)


__all__ = ["FairComputePool", "QueueFullError"]
