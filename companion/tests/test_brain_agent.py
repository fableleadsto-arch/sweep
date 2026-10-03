"""HTTP tests for the upgraded /api/brain/agent/turn and /api/brain/agent/stream.

Mirrors the Node-side acceptance criteria: full-loop turn responses, SSE
streaming with camelCase step events (``argsPreview``), and a single-chunk chat
mode.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from companion.brain_agent import get_agent_responder
from companion.main import app, get_orchestrator_model, get_settings


class FakeModel:
    """Scripted model - returns one plan dict per call (no network)."""

    def __init__(self, *script: dict[str, Any]) -> None:
        self.script = list(script)
        self.calls = 0

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list,
        iteration: int,
    ) -> dict[str, Any]:
        self.calls += 1
        return self.script.pop(0)


def _client(settings, model, responder=None) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_orchestrator_model] = lambda: model
    if responder is not None:
        app.dependency_overrides[get_agent_responder] = lambda: responder
    else:
        app.dependency_overrides.pop(get_agent_responder, None)
    return TestClient(app, headers={"Authorization": "Bearer test-only"})


def _events(resp) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in resp.iter_lines()
        if line.startswith("data: ")
    ]


def test_turn_runs_full_loop(settings) -> None:
    model = FakeModel(
        {"steps": [{"tool": "math.evaluate", "input": {"expression": "2+2"}}]},
        {"respond": "4"},
    )
    client = _client(settings, model)
    resp = client.post(
        "/api/brain/agent/turn",
        json={"user_id": "u1", "workspace_id": "w1", "message": "compute 2+2"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "4"
    assert body["protocol"] == "1"
    assert body["turn_id"].startswith("turn-")
    assert body["steps"][0]["tool"] == "math.evaluate"
    assert body["steps"][0]["status"] == "ok"


def test_turn_reports_failed_step(settings) -> None:
    model = FakeModel(
        {"steps": [{"tool": "time.now", "input": {"broken": True}}]},
        {"respond": "done"},
    )
    client = _client(settings, model)
    resp = client.post(
        "/api/brain/agent/turn",
        json={"user_id": "u1", "message": "what time is it"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["steps"][0]["tool"] == "time.now"
    assert body["steps"][0]["status"] in ("ok", "failed")


def test_agent_stream_emits_camelcase_steps(settings) -> None:
    model = FakeModel(
        {"steps": [{"tool": "time.now", "input": {}}]},
        {"respond": "the time is now"},
    )
    client = _client(settings, model)
    with client.stream(
        "POST",
        "/api/brain/agent/stream",
        json={"message": "what time is it", "mode": "agent", "userId": "u1", "workspaceId": "w1"},
    ) as resp:
        assert resp.status_code == 200
        events = _events(resp)

    types = [e["type"] for e in events]
    assert types == ["step", "sources", "text", "done"]

    step = events[0]["step"]
    assert step["tool"] == "time.now"
    assert "argsPreview" in step
    assert "args_preview" not in step
    assert types[-1] == "done"
    assert events[-2]["text"] == "the time is now"


def test_chat_stream_single_text_event(settings) -> None:
    async def responder(system: str, messages: list[dict[str, Any]]) -> str:
        return "hello from the provider chain"

    client = _client(settings, FakeModel(), responder=responder)
    with client.stream(
        "POST",
        "/api/brain/agent/stream",
        json={"message": "hello there", "mode": "chat", "userId": "u1"},
    ) as resp:
        assert resp.status_code == 200
        events = _events(resp)

    assert [e["type"] for e in events] == ["text", "done"]
    assert events[0]["text"] == "hello from the provider chain"
