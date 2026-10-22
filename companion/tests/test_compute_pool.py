"""Scheduling guarantees, caller isolation and actual subprocess execution."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from companion.compute.pool import FairComputePool, QueueFullError
from companion.compute.workers import ComputeWorker, LocalSubprocessWorker, WorkerJob, WorkerStatus


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("worker condition did not become true before deadline")


class ControlledWorker(ComputeWorker):
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = []
        self.states = {}
        self.peak = 0
        self.cancelled = []
        self.fail_start = False
        self.fail_poll = False
        self.fail_cancel = False
        self.delayed_cancel = False

    def submit(self, job):
        with self.lock:
            if self.fail_start:
                self.fail_start = False
                raise OSError("cannot spawn")
            self.jobs.append(copy.deepcopy(job))
            self.states[job.id] = WorkerStatus(job.id, "running")
            self.peak = max(self.peak, sum(s.state == "running" for s in self.states.values()))
            return job.id

    def status(self, job_id):
        with self.lock:
            if self.fail_poll:
                raise OSError("temporarily unreadable")
            return copy.deepcopy(self.states[job_id])

    def complete(self, job_id, state="done"):
        with self.lock:
            self.states[job_id] = WorkerStatus(job_id, state, result={"rows": [1, 2]})

    def cancel(self, job_id):
        with self.lock:
            if self.fail_cancel:
                raise OSError("could not stop process")
            self.cancelled.append(job_id)
            if not self.delayed_cancel:
                self.states[job_id] = WorkerStatus(job_id, "cancelled")
            return True

    def shutdown(self):
        with self.lock:
            self.fail_poll = False
            self.fail_cancel = False
            self.delayed_cancel = False
            for job_id, state in list(self.states.items()):
                if state.state == "running":
                    self.cancel(job_id)


@pytest.fixture
def controlled():
    worker = ControlledWorker()
    pools = []

    def make(**limits):
        pool = FairComputePool(worker, **limits)
        pools.append(pool)
        return pool, worker

    yield make
    for pool in pools:
        pool.shutdown()


def job(label="example"):
    return WorkerJob(kind="capability", backend="cpu", spec={"label": label})


def test_round_robin_prevents_busy_owner_starving_new_owner(controlled):
    pool, worker = controlled()
    a1 = pool.submit("alice", job("a1"))
    wait_for(lambda: len(worker.jobs) == 1)
    a2 = pool.submit("alice", job("a2"))
    a3 = pool.submit("alice", job("a3"))
    b1 = pool.submit("bob", job("b1"))
    b2 = pool.submit("bob", job("b2"))
    for index, expected in enumerate((a1, b1, a2, b2, a3), start=1):
        wait_for(lambda index=index: len(worker.jobs) == index)
        assert worker.jobs[-1].id == expected
        worker.complete(expected)
    wait_for(lambda: pool.status("alice", a3).state == "done")
    assert worker.peak == 1


def test_limits_count_running_and_queued_jobs_without_losing_accepted_work(controlled):
    pool, worker = controlled(max_outstanding=3, max_per_owner=2)
    a1 = pool.submit("a", job())
    wait_for(lambda: len(worker.jobs) == 1)
    pool.submit("a", job())
    with pytest.raises(QueueFullError, match="your queue"):
        pool.submit("a", job())
    pool.submit("b", job())
    with pytest.raises(QueueFullError, match="host queue"):
        pool.submit("c", job())
    worker.complete(a1)
    wait_for(lambda: pool.status("a", a1).state == "done")
    assert pool.submit("c", job())


def test_owner_running_cap_allows_other_owners_to_use_available_slots(controlled):
    pool, worker = controlled(max_running=2)
    a1 = pool.submit("a", job())
    a2 = pool.submit("a", job())
    b1 = pool.submit("b", job())
    wait_for(lambda: len(worker.jobs) == 2)
    assert {j.id for j in worker.jobs} == {a1, b1}
    assert pool.status("a", a2).state == "queued"
    assert worker.peak == 2


def test_owner_boundaries_and_immutable_job_and_result_snapshots(controlled):
    pool, worker = controlled()
    original = job()
    original.id = "../../client-chosen-path"
    job_id = pool.submit("owner", original)
    original.spec["label"] = "changed"
    wait_for(lambda: len(worker.jobs) == 1)
    assert worker.jobs[0].spec == {"label": "example"}
    assert job_id != original.id
    assert pool.status("other", job_id).state == "unknown"
    assert pool.status("other", job_id).error == pool.status("other", "missing").error
    assert not pool.cancel("other", job_id)
    assert worker.cancelled == []
    worker.complete(job_id)
    wait_for(lambda: pool.status("owner", job_id).state == "done")
    pool.status("owner", job_id).result["rows"].clear()
    assert pool.status("owner", job_id).result == {"rows": [1, 2]}
    assert not pool.cancel("owner", job_id)


def test_queued_cancel_never_reaches_worker_and_running_cancel_waits_for_exit(controlled):
    pool, worker = controlled()
    first = pool.submit("a", job())
    wait_for(lambda: len(worker.jobs) == 1)
    second = pool.submit("b", job())
    third = pool.submit("c", job())
    assert pool.cancel("b", second)
    assert pool.status("b", second).state == "cancelled"
    worker.delayed_cancel = True
    assert pool.cancel("a", first)
    assert pool.status("a", first).state == "running"
    assert pool.status("c", third).state == "queued"
    worker.complete(first, state="cancelled")
    wait_for(lambda: len(worker.jobs) == 2)
    assert [j.id for j in worker.jobs] == [first, third]


def test_spawn_failure_does_not_block_next_owner(controlled):
    pool, worker = controlled()
    worker.fail_start = True
    first = pool.submit("a", job())
    second = pool.submit("b", job())
    wait_for(lambda: pool.status("a", first).state == "failed")
    wait_for(lambda: len(worker.jobs) == 1)
    assert worker.jobs[0].id == second


def test_monitor_or_cancel_error_never_frees_a_live_slot(controlled):
    pool, worker = controlled()
    first = pool.submit("a", job())
    wait_for(lambda: len(worker.jobs) == 1)
    worker.fail_poll = worker.fail_cancel = True
    second = pool.submit("b", job())
    assert not pool.cancel("a", first)
    time.sleep(0.25)
    assert pool.status("a", first).state == "running"
    assert pool.status("b", second).state == "queued"
    assert len(worker.jobs) == 1
    worker.fail_poll = worker.fail_cancel = False
    worker.complete(first)
    wait_for(lambda: len(worker.jobs) == 2)


def test_timeout_stops_work_before_reusing_slot(controlled):
    pool, worker = controlled(job_timeout=0.1)
    first = pool.submit("a", job())
    wait_for(lambda: pool.status("a", first).state == "failed")
    assert "time limit" in pool.status("a", first).error
    assert worker.cancelled == [first]


def test_history_and_idle_thread_are_bounded_and_pool_restarts(controlled):
    pool, worker = controlled(history_limit=2)
    ids = []
    for index in range(4):
        job_id = pool.submit("a", job(str(index)))
        ids.append(job_id)
        wait_for(lambda index=index: len(worker.jobs) == index + 1)
        worker.complete(job_id)
        wait_for(lambda job_id=job_id: pool.status("a", job_id).state == "done")
        wait_for(lambda: pool._thread is None)
    assert len(pool._entries) == 2
    assert pool.status("a", ids[0]).state == "unknown"
    assert pool.status("a", ids[-1]).state == "done"
    assert not pool._queues and not pool._ready


def test_concurrent_admission_is_atomic(controlled):
    pool, worker = controlled(max_outstanding=4, max_per_owner=4)

    def submit(_):
        try:
            return pool.submit("owner", job())
        except QueueFullError:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(submit, range(30)))
    assert len({result for result in results if result}) == 4
    wait_for(lambda: len(worker.jobs) == 1)
    assert worker.peak == 1


def test_shutdown_cancels_all_and_prevents_future_submission(controlled):
    pool, worker = controlled()
    running = pool.submit("a", job())
    wait_for(lambda: len(worker.jobs) == 1)
    queued = pool.submit("b", job())
    pool.shutdown()
    assert pool.status("a", running).state == "cancelled"
    assert pool.status("b", queued).state == "cancelled"
    assert pool._thread is None
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit("a", job())


@pytest.mark.parametrize("limits", [
    {"max_running": 0}, {"max_running": True}, {"max_running_per_owner": 33},
    {"max_outstanding": -1}, {"max_per_owner": 1.5}, {"history_limit": 0},
    {"job_timeout": float("nan")}, {"job_timeout": float("inf")}, {"job_timeout": -1},
])
def test_invalid_limits_fail_closed(limits):
    with pytest.raises(ValueError):
        FairComputePool(ControlledWorker(), **limits)


@pytest.mark.parametrize("owner", ["", "a/b", "a\\b", "x" * 129, None])
def test_invalid_owner(controlled, owner):
    pool, worker = controlled()
    with pytest.raises(ValueError):
        pool.submit(owner, job())
    assert not worker.jobs


@pytest.mark.parametrize("spec", [{"data": "x" * 65536}, {"x": float("nan")}, []])
def test_oversize_or_non_json_spec(controlled, spec):
    pool, _ = controlled()
    with pytest.raises(ValueError):
        pool.submit("a", WorkerJob(spec=spec))


def test_real_cpu_subprocesses_for_two_owners(tmp_path):
    worker = LocalSubprocessWorker(tmp_path / "workers")
    pool = FairComputePool(worker)
    try:
        ids = {
            owner: pool.submit(owner, WorkerJob(
                kind="capability", backend="cpu", spec={"capability": "math", "data": "1 2 3 4"},
            )) for owner in ("alice", "bob")
        }
        wait_for(lambda: all(pool.status(o, i).state in {"done", "failed"} for o, i in ids.items()), timeout=60)
        for owner, job_id in ids.items():
            status = pool.status(owner, job_id)
            assert status.state == "done", status.error
            assert status.result is not None
            assert pool.status("stranger", job_id).state == "unknown"
        assert worker._procs == {}
    finally:
        pool.shutdown()


@pytest.mark.parametrize("job_id", ["../escape", "..\\escape", "CON", "NUL", "x:y", "", "a" * 65])
def test_local_worker_rejects_unsafe_ids(tmp_path, job_id):
    worker = LocalSubprocessWorker(tmp_path)
    for operation in (lambda: worker.submit(WorkerJob(id=job_id)), lambda: worker.status(job_id), lambda: worker.cancel(job_id)):
        with pytest.raises(ValueError):
            operation()
    assert not list(tmp_path.iterdir())


def test_local_cancel_does_not_invent_or_overwrite_finished_jobs(tmp_path):
    worker = LocalSubprocessWorker(tmp_path)
    assert not worker.cancel("missing")
    assert not list(tmp_path.iterdir())
    record = WorkerStatus("done", "done", result={"answer": 42}).to_dict()
    path = tmp_path / "done.status.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert not worker.cancel("done")
    assert json.loads(path.read_text()) == record


def test_local_worker_never_overwrites_existing_spec(tmp_path):
    worker = LocalSubprocessWorker(tmp_path)
    path = tmp_path / "existing.spec.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        worker.submit(WorkerJob(id="existing"))
    assert path.read_text() == "original"


def test_process_exit_is_required_before_done_and_crashes_are_reported(tmp_path):
    worker = LocalSubprocessWorker(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    worker._procs["test"] = proc
    path = tmp_path / "test.status.json"
    path.write_text(json.dumps(WorkerStatus("test", "done", result={"ok": True}).to_dict()), encoding="utf-8")
    try:
        assert worker.status("test").state == "running"
        assert worker.status("test").result is None
        proc.terminate()
        proc.wait(timeout=5)
        assert worker.status("test").state == "failed"
        assert worker.status("test").result is None
        assert worker._procs == {}
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_running_cancel_waits_for_real_process_exit(tmp_path):
    worker = LocalSubprocessWorker(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    worker._procs["test"] = proc
    try:
        assert worker.cancel("test")
        assert proc.poll() is not None
        assert worker.status("test").state == "cancelled"
        assert not worker.cancel("test")
        assert worker._procs == {}
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_local_worker_does_not_trust_malformed_or_oversize_status(tmp_path):
    worker = LocalSubprocessWorker(tmp_path)
    path = tmp_path / "test.status.json"
    for text in ('{', '[]', '{"job_id":"someone-else", "state":"done"}', 'x' * (1024 * 1024 + 1)):
        path.write_text(text, encoding="utf-8")
        assert worker.status("test").state == "unknown"


@pytest.mark.parametrize("kind", ["training", "inference", "smoke", "transform", "capability"])
def test_worker_preserves_requested_task_kind(tmp_path, monkeypatch, kind):
    from companion.compute import worker as entrypoint
    from companion.compute.base import ComputeJobResult, ComputeTaskKind

    captured = []

    class Backend:
        def run(self, task):
            captured.append(task)
            return ComputeJobResult(ok=True, backend="test", summary="finished", result={})

    class Manager:
        def get(self, _):
            return Backend()

    monkeypatch.setattr(entrypoint, "BackendManager", Manager)
    spec = tmp_path / "spec.json"
    spec.write_text('{"capability":"test"}', encoding="utf-8")
    status = tmp_path / "status.json"
    assert entrypoint._run_worker("test", "test", kind, spec, status) == 0
    assert captured[0].kind == ComputeTaskKind(kind)
    result = json.loads(status.read_text())
    assert result["started_at"] <= result["finished_at"]
    assert not status.with_suffix(".tmp").exists()


def test_importing_pool_does_not_load_heavy_frameworks_or_start_threads():
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import sys, threading; before = threading.active_count(); "
            "from companion.compute import FairComputePool; "
            "from companion.compute.backends.cpu import CPUBackend; CPUBackend().version(); "
            "assert not ({'torch', 'tensorflow', 'numpy', 'transformers', 'onnxruntime'} & sys.modules.keys()); "
            "assert threading.active_count() == before"
        )],
        capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert probe.returncode == 0, probe.stderr


def test_cpu_availability_and_supported_kinds_are_honest(monkeypatch):
    from companion.compute import base
    from companion.compute.backends.cpu import CPUBackend

    cpu = CPUBackend()
    for kind in (base.ComputeTaskKind.TRAINING, base.ComputeTaskKind.INFERENCE, base.ComputeTaskKind.TRANSFORM):
        assert not cpu.supports(base.ComputeTask(kind=kind))
    monkeypatch.setattr(base, "module_available", lambda _: False)
    assert not cpu.available
    assert cpu.missing_libraries == ["numpy"]


@pytest.mark.parametrize("failures", [2, 20])
def test_atomic_status_update_retries_only_bounded_transient_locks(tmp_path, monkeypatch, failures):
    from companion.compute import worker as entrypoint

    path = tmp_path / "status.json"
    path.write_text('{"old":true}', encoding="utf-8")
    replace = Path.replace
    calls = []
    pauses = []

    def locked(source, destination):
        calls.append(source)
        if len(calls) <= failures:
            raise PermissionError("temporary Windows sharing lock")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(entrypoint.time, "sleep", pauses.append)
    if failures > 5:
        with pytest.raises(PermissionError):
            entrypoint._write(path, WorkerStatus("test", "done"))
        assert json.loads(path.read_text()) == {"old": True}
        assert len(calls) == 6
    else:
        entrypoint._write(path, WorkerStatus("test", "done"))
        assert json.loads(path.read_text())["state"] == "done"
        assert len(calls) == 3
    assert sum(pauses) < 0.5
