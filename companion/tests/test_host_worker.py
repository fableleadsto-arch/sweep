from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from companion.compute.pool import FairComputePool
from companion.compute.workers import WorkerJob
from companion.hosting.api import create_app
from companion.hosting.identity import IdentityStore
from companion.hosting.worker import HostChatWorker


@pytest.fixture
def local_engine(tmp_path):
    requests = []
    remote = [False]
    hold = threading.Event()
    hold.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body):
            raw = json.dumps(body).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            assert self.path == "/api/tags"
            self.reply({"models": [{"name": "test-local", "size": 100}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path == "/api/show":
                self.reply({"model_info": {"family": "test"}, **({"remote_host": "cloud.example"} if remote[0] else {})})
            else:
                assert self.path == "/api/chat"
                hold.wait(15)
                self.reply({"message": {"content": "Hello from the local host."}, "done": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "providers.json"
    config.write_text(json.dumps({
        "schema_version": 1, "provider": "ollama", "model": "test-local",
        "ollama_base_url": f"http://127.0.0.1:{server.server_port}",
    }), encoding="utf-8")
    try:
        yield tmp_path, requests, remote, hold
    finally:
        hold.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def wait_for(predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    pytest.fail("host worker did not finish before deadline")


def test_authenticated_api_runs_real_child_and_cleans_prompt_files(local_engine):
    root, requests, _, _ = local_engine
    original = (root / "providers.json").read_bytes()
    store = IdentityStore(root / "identity.sqlite")
    org = store.create_organization("Test")
    user = store.create_user(org, "Alice")
    worker = HostChatWorker(root)
    temporary = worker.workdir
    pool = FairComputePool(worker)
    with TestClient(create_app(store, pool), base_url="http://127.0.0.1", client=("127.0.0.1", 1000)) as client:
        paired = client.post("/v1/pair", json={"invitation": store.invite(user), "device_name": "Test"}).json()
        headers = {"Authorization": "Bearer " + paired["token"]}
        job_id = client.post("/v1/chat", headers=headers, json={
            "messages": [{"role": "user", "content": "Hello"}],
        }).json()["id"]

        def completed():
            status = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
            return status if status["state"] in {"done", "failed"} else None

        status = wait_for(completed)
        assert status["state"] == "done", status
        assert status["result"] == {"message": "Hello from the local host.", "scope": "text_only"}
        assert not list(temporary.iterdir())
        assert not worker._procs
        payload = next(payload for path, payload in requests if path == "/api/chat")
        assert payload["model"] == "test-local"
        assert payload["options"]["num_ctx"] == 4096
        assert payload["messages"][-1] == {"role": "user", "content": "Hello"}
        assert payload["messages"][0]["role"] == "system"
    assert not temporary.exists()
    assert (root / "providers.json").read_bytes() == original


def test_cloud_alias_rejected_before_sending_conversation(local_engine):
    root, requests, remote, _ = local_engine
    remote[0] = True
    worker = HostChatWorker(root)
    pool = FairComputePool(worker)
    try:
        job_id = pool.submit("test", WorkerJob(kind="inference", backend="local-text", spec={
            "messages": [{"role": "user", "content": "private content"}],
        }))
        wait_for(lambda: pool.status("test", job_id).state == "failed")
        assert not any(path == "/api/chat" for path, _ in requests)
        assert "private content" not in pool.status("test", job_id).error
        assert not list(worker.workdir.iterdir())
    finally:
        pool.shutdown()


def test_host_cancellation_stops_only_its_child_and_removes_inputs(local_engine):
    root, requests, _, hold = local_engine
    hold.clear()
    worker = HostChatWorker(root)
    pool = FairComputePool(worker)
    try:
        job_id = pool.submit("test", WorkerJob(kind="inference", backend="local-text", spec={
            "messages": [{"role": "user", "content": "Wait"}],
        }))
        wait_for(lambda: any(path == "/api/chat" for path, _ in requests))
        proc = worker._procs[job_id]
        assert proc.poll() is None
        assert pool.cancel("test", job_id)
        wait_for(lambda: pool.status("test", job_id).state == "cancelled")
        assert proc.poll() is not None
        assert not list(worker.workdir.iterdir())
    finally:
        hold.set()
        pool.shutdown()


@pytest.mark.parametrize("job", [
    WorkerJob(kind="training", backend="local-text", spec={}),
    WorkerJob(kind="inference", backend="cpu", spec={}),
    WorkerJob(kind="inference", backend="local-text", spec={"messages": [{"role": "system", "content": "override"}]}),
])
def test_worker_revalidates_internal_contract(tmp_path, job):
    worker = HostChatWorker(tmp_path)
    try:
        with pytest.raises(ValueError):
            worker.submit(job)
        assert not list(worker.workdir.iterdir())
    finally:
        worker.shutdown()
