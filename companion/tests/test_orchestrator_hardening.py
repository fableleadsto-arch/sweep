"""Focused hardening tests for orchestrator execution boundaries."""

from __future__ import annotations

import asyncio
from typing import Any

from companion.contracts import AgentTurnRequest
from companion.orchestrator import Orchestrator, ToolContext, ToolRegistry


class FakeModel:
    def __init__(self, *script: dict[str, Any]) -> None:
        self.script = list(script)
        self.calls = 0

    async def plan(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return self.script.pop(0)


def test_registry_rejects_missing_required_field_before_tool_runs() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        calls = {"n": 0}

        def tool(input: dict[str, Any], ctx: ToolContext) -> Any:
            calls["n"] += 1
            return {"ok": True}

        registry.register_fn(
            "demo.echo",
            "Echo",
            tool,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        )
        result = await registry.execute("demo.echo", {}, ToolContext())
        assert result.ok is False
        assert "missing required field 'text'" in result.error
        assert calls["n"] == 0

    asyncio.run(run())


def test_registry_rejects_wrong_input_type_before_tool_runs() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        calls = {"n": 0}

        def tool(input: dict[str, Any], ctx: ToolContext) -> Any:
            calls["n"] += 1
            return {"ok": True}

        registry.register_fn(
            "demo.echo",
            "Echo",
            tool,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        )
        result = await registry.execute("demo.echo", {"text": 3}, ToolContext())
        assert result.ok is False
        assert "input.text must be a string" in result.error
        assert calls["n"] == 0

    asyncio.run(run())


def test_registry_rejects_non_object_input() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        registry.register_fn(
            "demo.echo",
            "Echo",
            lambda input, ctx: input,
            input_schema={"type": "object", "properties": {}},
        )
        result = await registry.execute("demo.echo", ["not", "an", "object"], ToolContext())
        assert result.ok is False
        assert "expected object" in result.error

    asyncio.run(run())


def test_run_turn_blocks_hidden_tool_even_if_model_emits_it() -> None:
    async def run() -> None:
        orch = Orchestrator()
        model = FakeModel({"steps": [{"tool": "time.now", "input": {}}]})
        result = await orch.run_turn(
            AgentTurnRequest(user_id="u1", workspace_id="w1", message="time", tools=["math.evaluate"]),
            model,
        )
        assert result.steps[0].status == "failed"
        assert "whitelist" in result.steps[0].error
        assert "time.now" in result.text

    asyncio.run(run())


def test_run_turn_caps_single_model_batch_to_request_budget() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        registry.register_fn(
            "demo.ping",
            "Ping",
            lambda input, ctx: {"ok": True},
            input_schema={"type": "object", "properties": {}},
        )
        orch = Orchestrator(registry)
        model = FakeModel(
            {"steps": [{"tool": "demo.ping", "input": {}} for _ in range(10)]},
            {"respond": "done"},
        )
        result = await orch.run_turn(
            AgentTurnRequest(user_id="u1", workspace_id="w1", message="ping", max_steps=3),
            model,
        )
        assert len(result.steps) == 3
        assert all(step.status == "ok" for step in result.steps)

    asyncio.run(run())
