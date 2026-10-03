"""Tests for the Relay Brain orchestrator (agent loop + tool system)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from companion.contracts import AgentTurnRequest, BrainPlanRequest
from companion.main import app, get_orchestrator_model
from companion.orchestrator import (
    Orchestrator,
    ToolContext,
    ToolError,
    ToolRegistry,
    safe_math_eval,
)
from companion.config import BrainSettings, get_settings


class FakeModel:
    """Scripted model — returns one plan dict per call (no network)."""

    def __init__(self, *script: dict[str, Any]) -> None:
        self.script = list(script)
        self.calls = 0

    async def plan(self, *, system: str, messages: list[dict[str, Any]], tools: list, iteration: int) -> dict[str, Any]:
        self.calls += 1
        return self.script.pop(0)


def _turn(text: str = "compute 2+2", tools: list[str] | None = None) -> AgentTurnRequest:
    return AgentTurnRequest(user_id="u1", workspace_id="w1", message=text, tools=tools or [])


# ── safe math evaluator ──────────────────────────────────────


def test_math_eval_basic() -> None:
    assert safe_math_eval("2 + 3 * 4") == 14
    assert safe_math_eval("(2 + 3) ** 2") == 25
    assert safe_math_eval("sqrt(16) + pi") == pytest.approx(7.141592653589793)


def test_math_eval_rejects_code() -> None:
    for bad in ["__import__('os').system('ls')", "1; import os", "open('/etc/passwd')"]:
        with pytest.raises(ToolError):
            safe_math_eval(bad)


def test_math_eval_rejects_unknown_names() -> None:
    with pytest.raises(ToolError):
        safe_math_eval("evil * 2")


# ── tool registry ────────────────────────────────────────────


def test_registry_specs_and_execute() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        registry.register_fn(
            "math.evaluate",
            "eval",
            lambda i, ctx: safe_math_eval(i["expression"]),
            input_schema={
                "type": "object",
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
            },
        )
        specs = registry.specs()
        assert specs[0].name == "math.evaluate"
        assert specs[0].input_schema["required"] == ["expression"]

        res = await registry.execute("math.evaluate", {"expression": "1 + 1"}, ToolContext())
        assert res.ok is True
        assert res.output == 2

    asyncio.run(run())


def test_registry_unknown_tool() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        res = await registry.execute("nope", {}, ToolContext())
        assert res.ok is False
        assert "unknown tool" in res.error

    asyncio.run(run())


# ── plan (decomposition only) ────────────────────────────────


def test_plan_decomposes_into_steps() -> None:
    async def run() -> None:
        orch = Orchestrator()
        model = FakeModel({"summary": "compute", "steps": [{"tool": "math.evaluate", "input": {"expression": "2+2"}}]})
        plan = await orch.plan(BrainPlanRequest(message="compute 2+2"), model)
        assert plan.plan_id.startswith("plan_")
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "math.evaluate"

    asyncio.run(run())


def test_plan_respond_shortcut() -> None:
    async def run() -> None:
        orch = Orchestrator()
        model = FakeModel({"respond": "hello!"})
        plan = await orch.plan(BrainPlanRequest(message="hi"), model)
        assert plan.steps[0].kind == "respond"

    asyncio.run(run())


# ── full agent loop ──────────────────────────────────────────


def test_run_turn_executes_tool_then_responds() -> None:
    async def run() -> None:
        orch = Orchestrator()
        model = FakeModel(
            {"steps": [{"tool": "math.evaluate", "input": {"expression": "2+2"}, "description": "add"}]},
            {"respond": "the answer is 4"},
        )
        result = await orch.run_turn(_turn(), model)
        assert model.calls == 2
        assert result.text == "the answer is 4"
        assert len(result.steps) == 1
        assert result.steps[0].status == "ok"
        assert result.steps[0].output["value"] == 4
        assert result.turn_id.startswith("turn_")

    asyncio.run(run())


def test_run_turn_failing_tool_stops() -> None:
    async def run() -> None:
        orch = Orchestrator()
        model = FakeModel({"steps": [{"tool": "does.not.exist", "input": {}}]})
        result = await orch.run_turn(_turn(), model)
        assert result.steps[0].status == "failed"
        assert "does.not.exist" in result.text

    asyncio.run(run())


def test_run_turn_caps_iterations() -> None:
    async def run() -> None:
        orch = Orchestrator()
        model = FakeModel(
            *({"steps": [{"tool": "math.evaluate", "input": {"expression": "1+1"}}]} for _ in range(20))
        )
        result = await orch.run_turn(_turn(), model)
        assert model.calls == 8  # MAX_STEPS
        assert result.text  # fallback summary, never empty
        assert result.steps[0].status == "ok"

    asyncio.run(run())


def test_run_turn_tool_whitelist_hides_tools() -> None:
    async def run() -> None:
        orch = Orchestrator()
        seen: list[str] = []

        class RecordingModel:
            async def plan(self, **kw: Any) -> dict[str, Any]:
                seen.extend(t.name for t in kw["tools"])
                return {"respond": "ok"}

        await orch.run_turn(_turn(tools=["math.evaluate"]), RecordingModel())
        assert "math.evaluate" in seen
        assert "http.get" not in seen
        assert "time.now" not in seen

    asyncio.run(run())


# ── HTTP API ─────────────────────────────────────────────────


def _client(settings: BrainSettings, model: Any) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_orchestrator_model] = lambda: model
    return TestClient(app, headers={"Authorization": "Bearer test-only"})


def test_agent_turn_endpoint_with_fake_model(settings: BrainSettings) -> None:
    model = FakeModel(
        {"steps": [{"tool": "math.evaluate", "input": {"expression": "2+2"}}]},
        {"respond": "4"},
    )
    client = _client(settings, model)
    resp = client.post("/api/brain/agent/turn", json={"user_id": "u1", "message": "compute 2+2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "4"
    assert body["protocol"] == "1"
    assert body["steps"][0]["tool"] == "math.evaluate"


def test_plan_endpoint_with_fake_model(settings: BrainSettings) -> None:
    model = FakeModel({"summary": "s", "steps": [{"tool": "time.now", "input": {}}]})
    client = _client(settings, model)
    resp = client.post("/api/brain/plan", json={"user_id": "u1", "message": "what time is it"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["steps"][0]["tool"] == "time.now"
    assert body["protocol"] == "1"
