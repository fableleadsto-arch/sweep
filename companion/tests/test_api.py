from __future__ import annotations

from fastapi.testclient import TestClient

from companion.config import BrainSettings, get_settings
from companion.main import app


def _client(settings: BrainSettings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, headers={"Authorization": "Bearer test-only"})


def test_health_endpoint(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "python" in body
    assert set(body["providers"]) == {"gemini", "openai", "ollama", "anthropic"}


def test_root_lists_docs(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["docs"] == "/docs"


def test_turn_validates_message_required(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post("/api/brain/turn", json={"user_id": "u1", "message": ""})
    assert resp.status_code == 422


def test_memory_remember_and_search_roundtrip(settings: BrainSettings) -> None:
    client = _client(settings)
    remembered = client.post(
        "/api/brain/memory/remember",
        json={
            "user_id": "u1",
            "content": "prefers concise daily briefings",
            "kind": "preference",
        },
    )
    assert remembered.status_code == 200
    entry = remembered.json()
    assert entry["content"] == "prefers concise daily briefings"

    searched = client.post(
        "/api/brain/memory/search",
        json={"user_id": "u1", "query": "briefings", "limit": 5},
    )
    assert searched.status_code == 200
    assert any(e["content"].startswith("prefers") for e in searched.json())


def test_embed_returns_null_vectors_without_keys(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post("/api/brain/embed", json={"texts": ["hello world"]})
    assert resp.status_code == 200
    assert resp.json()["embeddings"] == [None]


def test_rag_short_circuits_without_supabase(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post("/api/brain/rag", json={"workspace_id": "w1", "query": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["context"] == ""


def test_overview_short_circuits_without_supabase(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post(
        "/api/brain/overview",
        json={"user_id": "u1", "workspace_id": "w1", "briefing_type": "morning"},
    )
    assert resp.status_code == 200
    assert resp.json()["briefing"] == ""


def test_context_bundle_without_supabase(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post(
        "/api/brain/context",
        json={"user_id": "u1", "workspace_id": "w1", "message": "what's on my plate today"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overview_requested"] is True
    assert body["overview"] == ""
    assert body["profile"]["name"] is None
    assert body["workspace"]["workspace_id"] == "w1"


def test_execute_runs_sandboxed_script(settings: BrainSettings) -> None:
    settings.allow_code_execution = True
    client = _client(settings)
    resp = client.post(
        "/api/brain/execute",
        json={
            "code": "import json\nresult = {'n': sum(env.get('values', []))}\n",
            "env": {"values": [1, 2, 3]},
            "timeout_ms": 10_000,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"] == {"n": 6}
    assert body["sandboxed"] is True


def test_execute_refuses_system_import(settings: BrainSettings) -> None:
    settings.allow_code_execution = True
    client = _client(settings)
    resp = client.post(
        "/api/brain/execute",
        json={"code": "import os\nresult = {'x': 1}\n"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["violations"]
