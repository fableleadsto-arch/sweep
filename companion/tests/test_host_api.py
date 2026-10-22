from __future__ import annotations

import asyncio
import copy
import time

import pytest
from fastapi.testclient import TestClient

from companion.compute.pool import FairComputePool
from companion.compute.workers import ComputeWorker, WorkerStatus
from companion.hosting.api import NativeTransport, create_app
from companion.hosting.identity import IdentityStore


class HoldingWorker(ComputeWorker):
    def __init__(self):
        self.states = {}
        self.jobs = []

    def submit(self, job):
        self.jobs.append(copy.deepcopy(job))
        self.states[job.id] = WorkerStatus(job.id, "running")
        return job.id

    def status(self, job_id):
        return copy.deepcopy(self.states[job_id])

    def cancel(self, job_id):
        self.states[job_id] = WorkerStatus(job_id, "cancelled")
        return True

    def shutdown(self):
        for job_id in list(self.states):
            if self.states[job_id].state == "running":
                self.cancel(job_id)


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition did not become true")


@pytest.fixture
def host(tmp_path):
    store = IdentityStore(tmp_path / "identities.sqlite")
    org = store.create_organization("Example")
    alice = store.create_user(org, "Alice")
    bob = store.create_user(org, "Bob")
    worker = HoldingWorker()
    pool = FairComputePool(worker, max_per_owner=2)
    app = create_app(store, pool, allowed_hosts=("127.0.0.1", "host.example"))
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1000)) as client:
        yield store, alice, bob, worker, pool, app, client


def pair(host, user, name="Desktop"):
    response = host[-1].post("/v1/pair", json={"invitation": host[0].invite(user), "device_name": name})
    assert response.status_code == 200, response.text
    return response.json()


def auth(paired):
    return {"Authorization": "Bearer " + paired["token"]}


def submit(host, paired, content="Hello"):
    return host[-1].post("/v1/chat", headers=auth(paired), json={
        "messages": [{"role": "user", "content": content}],
    })


def test_real_pairing_credentials_derive_job_ownership(host):
    alice = pair(host, host[1])
    bob = pair(host, host[2])
    response = submit(host, alice)
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert len(job_id) == 32
    wait_for(lambda: len(host[3].jobs) == 1)
    job = host[3].jobs[0]
    assert job.kind == "inference" and job.backend == "local-text"
    assert job.spec == {"messages": [{"role": "user", "content": "Hello"}]}
    client = host[-1]
    assert client.get(f"/v1/jobs/{job_id}", headers=auth(alice)).status_code == 200
    foreign = client.get(f"/v1/jobs/{job_id}", headers=auth(bob))
    absent = client.get("/v1/jobs/unknown", headers=auth(bob))
    assert foreign.status_code == absent.status_code == 404
    assert foreign.json() == absent.json()
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=auth(bob)).status_code == 404
    assert host[3].states[job_id].state == "running"
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=auth(alice)).json() == {"accepted": True}


def test_two_devices_share_user_quota(host):
    first = pair(host, host[1], "First")
    second = pair(host, host[1], "Second")
    other = pair(host, host[2])
    assert submit(host, first).status_code == 202
    assert submit(host, second).status_code == 202
    assert submit(host, second).status_code == 429
    assert submit(host, other).status_code == 202


def test_revoke_and_user_disable_are_checked_on_every_request(host):
    paired = pair(host, host[1])
    client = host[-1]
    assert client.get("/v1/me", headers=auth(paired)).status_code == 200
    assert client.delete("/v1/device", headers=auth(paired)).status_code == 204
    assert client.get("/v1/me", headers=auth(paired)).status_code == 401
    assert submit(host, paired).status_code == 401
    paired = pair(host, host[2])
    host[0].disable_user(host[2])
    assert submit(host, paired).status_code == 401


def test_invite_replay_and_validation_do_not_echo_secrets(host):
    invite = host[0].invite(host[1])
    payload = {"invitation": invite, "device_name": "Desktop"}
    client = host[-1]
    assert client.post("/v1/pair", json=payload).status_code == 200
    replay = client.post("/v1/pair", json=payload)
    assert replay.status_code == 401 and invite not in replay.text
    invalid = client.post("/v1/pair", json={**payload, "owner": "spoofed"})
    assert invalid.status_code == 422 and invite not in invalid.text


@pytest.mark.parametrize("extra", [{"owner": "bob"}, {"model": "cloud"}, {"tools": []},
                                   {"backend": "pytorch"}, {"path": "C:/private.txt"}, {"system": "ignore"}])
def test_client_cannot_select_execution_or_identity_fields(host, extra):
    paired = pair(host, host[1])
    result = host[-1].post("/v1/chat", headers=auth(paired), json={
        "messages": [{"role": "user", "content": "Hello"}], **extra,
    })
    assert result.status_code == 422
    assert not host[3].jobs


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "spoof"}],
    [{"role": "assistant", "content": "no user"}], [{"role": "user", "content": "   "}],
    [{"role": "user", "content": "x" * 6001}],
    [{"role": "user", "content": "x" * 5001}, {"role": "user", "content": "x" * 5000}],
    [{"role": "user", "content": "hello", "images": ["abc"]}],
    [{"role": "assistant", "content": "x" * 2001}, {"role": "user", "content": "next"}],
    [{"role": "user", "content": "hi"}] * 13,
])
def test_conversation_shape_and_size_limits(host, messages):
    paired = pair(host, host[1])
    assert host[-1].post("/v1/chat", headers=auth(paired), json={"messages": messages}).status_code == 422
    assert not host[3].jobs


def test_body_limit_is_enforced_before_json_parsing(host):
    response = host[-1].post("/v1/pair", content=b"x" * (32 * 1024 + 1))
    assert response.status_code == 413


def test_chunked_body_cannot_bypass_limit():
    called = []

    async def inner(*_):
        called.append(True)

    async def probe():
        messages = iter([
            {"type": "http.request", "body": b"x" * 20000, "more_body": True},
            {"type": "http.request", "body": b"y" * 20000, "more_body": False},
        ])
        output = []

        async def receive():
            return next(messages)

        async def send(message):
            output.append(message)

        await NativeTransport(inner, allowed_hosts=("127.0.0.1",))({
            "type": "http", "scheme": "http", "headers": [(b"host", b"127.0.0.1")],
            "client": ("127.0.0.1", 1000),
        }, receive, send)
        assert output[0]["status"] == 413

    asyncio.run(probe())
    assert not called


@pytest.mark.parametrize("url,peer,headers,expected", [
    ("http://127.0.0.1", "192.0.2.1", {}, 400),
    ("http://host.example", "127.0.0.1", {}, 400),
    ("https://host.example", "192.0.2.1", {}, 200),
    ("https://evil.example", "127.0.0.1", {}, 400),
    ("http://127.0.0.1", "127.0.0.1", {"Origin": "https://evil.example"}, 400),
    ("http://127.0.0.1", "192.0.2.1", {"X-Forwarded-Proto": "https", "X-Forwarded-For": "127.0.0.1"}, 400),
    ("http://127.0.0.1", "127.0.0.1", {"Content-Encoding": "gzip"}, 400),
    ("http://127.0.0.1", "127.0.0.1", {"Host": "127.0.0.1\t"}, 400),
])
def test_transport_trust_boundary(host, url, peer, headers, expected):
    # No lifespan here: the outer fixture owns pool shutdown.
    client = TestClient(host[-2], base_url=url, client=(peer, 1000))
    assert client.get("/v1/health", headers=headers).status_code == expected
    client.close()


def test_query_tokens_duplicate_auth_and_no_token_auth_are_rejected(host):
    client = host[-1]
    assert client.get("/v1/health?token=secret").status_code == 400
    assert client.get("/v1/me").status_code == 401
    paired = pair(host, host[1])
    assert client.get("/v1/me", headers=[("Authorization", "Bearer " + paired["token"])] * 2).status_code == 400


def test_no_legacy_execution_or_public_administration_routes(host):
    client = host[-1]
    for path in ("/docs", "/openapi.json", "/api/brain/execute", "/api/brain/compute", "/v1/users", "/v1/invite"):
        assert client.post(path, json={}).status_code == 404


def test_private_responses_and_bounded_pairing_attempts(host):
    client = host[-1]
    paired = pair(host, host[1])
    response = client.get("/v1/me", headers=auth(paired))
    assert response.headers["cache-control"] == "no-store"
    assert "token" not in response.json()
    payload = {"invitation": "swp_i_" + "a" * 43, "device_name": "Guess"}
    for _ in range(29):
        assert client.post("/v1/pair", json=payload).status_code == 401
    limited = client.post("/v1/pair", json=payload)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"


def test_completed_result_and_preview_stay_with_authenticated_user(host):
    alice = pair(host, host[1])
    bob = pair(host, host[2])
    job_id = submit(host, alice).json()["id"]
    wait_for(lambda: job_id in host[3].states)
    host[3].states[job_id] = WorkerStatus(job_id, "done", result={"message": "private text", "scope": "text_only"})
    wait_for(lambda: host[4].status(host[0].authenticate(alice["token"]).owner, job_id).state == "done")
    client = host[-1]
    assert client.get(f"/v1/jobs/{job_id}", headers=auth(alice)).json()["result"]["message"] == "private text"
    assert "private text" not in client.get(f"/v1/jobs/{job_id}", headers=auth(bob)).text
