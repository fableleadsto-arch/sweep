"""Relay Brain Orchestrator — the agentic loop for the Python service.

Implements the OBSERVE → THINK/PLAN → ACT → OBSERVE RESULT → EVALUATE →
CONTINUE / MODIFY / FINISH loop (spec §31) plus a universal, machine-readable
tool system (spec §6). The Node gateway (and later the Java Runtime) call the
Brain through ``plan`` / ``run_turn``; the Brain decides which tools to run and
feeds observations back to the model until the task is finished.

This module is provider-agnostic: callers inject an ``OrchestratorModel`` (an
async callable). ``ProviderChainModel`` adapts the existing companion provider
chain, and tests inject a fake model — so the loop is fully unit-testable
without any API keys or network.
"""

from __future__ import annotations

import ast
import math
import operator
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .contracts import (
    AgentTurnRequest,
    AgentTurnResponse,
    BrainPlanRequest,
    BrainPlanResponse,
    PlanStep,
    TaskStatus,
    ToolExecutionResponse,
    ToolSpec,
)
from .providers import ProviderChain

MAX_STEPS = 8


# ─────────────────────────────────────────────────────────────
#  Tool system
# ─────────────────────────────────────────────────────────────


class ToolError(Exception):
    """Raised by a tool when execution fails (message becomes the error)."""


@dataclass
class ToolContext:
    """Immutable per-call context a tool may read (never writes secrets)."""

    user_id: str = ""
    workspace_id: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)


ToolRun = Callable[[dict[str, Any], ToolContext], "dict[str, Any] | Any"]


class Tool:
    """A single executable capability with a machine-readable schema."""

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}

    async def run(self, input: dict[str, Any], ctx: ToolContext) -> Any:
        raise NotImplementedError


class _SimpleTool(Tool):
    """Adapter for a plain async/sync callable tool."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[[dict[str, Any], ToolContext], Any],
        input_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self.output_schema = {}

    async def run(self, input: dict[str, Any], ctx: ToolContext) -> Any:
        out = self.fn(input, ctx)
        if hasattr(out, "__await__"):
            out = await out
        return out


class ToolRegistry:
    """Registry of every capability the Brain can invoke (spec §6)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_fn(
        self,
        name: str,
        description: str,
        fn: Callable[[dict[str, Any], ToolContext], Any],
        input_schema: Optional[dict[str, Any]] = None,
    ) -> None:
        self.register(_SimpleTool(name, description, fn, input_schema))

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def specs(self, names: Optional[list[str]] = None) -> list[ToolSpec]:
        names = set(names) if names else set(self._tools)
        return [
            ToolSpec(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
                output_schema=t.output_schema,
            )
            for t in self._tools.values()
            if t.name in names
        ]

    async def execute(
        self,
        name: str,
        input: dict[str, Any],
        ctx: ToolContext,
        task_id: str = "",
    ) -> ToolExecutionResponse:
        started = time.monotonic()
        tool = self.get(name)
        if tool is None:
            return ToolExecutionResponse(
                task_id=task_id, tool=name, ok=False, error=f"unknown tool: {name}"
            )
        try:
            out = await tool.run(input, ctx)
            return ToolExecutionResponse(
                task_id=task_id,
                tool=name,
                ok=True,
                output=out,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:  # tool errors are results, never crashes
            return ToolExecutionResponse(
                task_id=task_id,
                tool=name,
                ok=False,
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )


# ─────────────────────────────────────────────────────────────
#  Safe math evaluator (deterministic, no eval)
# ─────────────────────────────────────────────────────────────

_ALLOWED_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
}

_CONSTANTS = {"pi": math.pi, "e": math.e}


def safe_math_eval(expression: str) -> float | int:
    """Evaluate a numeric expression with a strict whitelist (no eval)."""

    def checked(value):
        if type(value) not in (int, float):
            raise ValueError("Only real numbers are supported")
        if isinstance(value, int) and value.bit_length() > 4096:
            raise ValueError("Integer result is too large")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite results are not supported")
        return value

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return checked(node.value)
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"unknown name: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 4096:
                raise ValueError("Exponent is too large")
            if isinstance(node.op, ast.Pow) and isinstance(left, int) and right > 0:
                if left.bit_length() * right > 4096:
                    raise ValueError("Power result is too large")
            return checked(_ALLOWED_BIN[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            fn = _ALLOWED_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"unknown function: {node.func.id}")
            args = [_eval(a) for a in node.args]
            return checked(fn(*args))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    try:
        if len(expression) > 2000:
            raise ValueError("Expression is too long")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 256:
            raise ValueError("Expression is too complex")
        return _eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
        raise ToolError(f"invalid expression: {exc}") from exc


def build_default_tools() -> ToolRegistry:
    """The standard tool set available to the Brain out of the box.

    Phase-1 tools are deterministic and dependency-light. Search, research and
    the compute toolbox join the registry as the migration progresses.
    """
    registry = ToolRegistry()

    def _math_eval(input: dict[str, Any], ctx: ToolContext) -> Any:
        expr = str(input.get("expression", ""))
        if not expr:
            raise ToolError("expression is required")
        value = safe_math_eval(expr)
        return {"value": value, "kind": "number"}

    registry.register_fn(
        "math.evaluate",
        "Evaluate a safe arithmetic expression (numbers, + - * / ** %, "
        "parentheses, abs/round/min/max/sqrt/sin/cos/tan/floor/ceil/log, pi, e).",
        _math_eval,
        input_schema={
            "type": "object",
            "required": ["expression"],
            "properties": {"expression": {"type": "string", "description": "Numeric expression"}},
        },
    )

    def _now(input: dict[str, Any], ctx: ToolContext) -> Any:
        return {"iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "epoch": int(time.time())}

    registry.register_fn(
        "time.now",
        "Return the current UTC time (ISO 8601 string and epoch seconds).",
        _now,
        input_schema={"type": "object", "properties": {}},
    )

    async def _http_get(input: dict[str, Any], ctx: ToolContext) -> Any:
        import httpx

        url = str(input.get("url", ""))
        if not url:
            raise ToolError("url is required")
        if not url.startswith(("http://", "https://")):
            raise ToolError("only http(s) urls are allowed")
        max_chars = int(input.get("max_chars", 4000))
        try:
            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=True, headers={"User-Agent": "RelayAI-Brain/1"}
            ) as client:
                resp = await client.get(url)
            return {
                "status": resp.status_code,
                "url": str(resp.url),
                "text": resp.text[:max_chars],
            }
        except httpx.HTTPError as exc:
            raise ToolError(f"request failed: {exc}") from exc

    registry.register_fn(
        "http.get",
        "Fetch a public http(s) URL and return status + truncated text.",
        _http_get,
        input_schema={
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "description": "default 4000"},
            },
        },
    )

    return registry


# ─────────────────────────────────────────────────────────────
#  Model protocol
# ─────────────────────────────────────────────────────────────


class OrchestratorModel(Protocol):
    """A model that can plan steps and produce a final response.

    ``plan`` must return one of:
      * ``{"respond": "<final text>"}`` — the task is finished.
      * ``{"steps": [{"tool": ..., "input": {...}, "description": ...}, ...]}``
      * ``{"summary": "<text>", "steps": [...]}`` — summary used if no respond.
    """

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        iteration: int,
    ) -> dict[str, Any]: ...


class ProviderChainModel:
    """Adapters the existing companion ``ProviderChain`` to the orchestrator."""

    def __init__(self, chain: ProviderChain) -> None:
        self.chain = chain
        self.provider = ""
        self.model = ""

    def _system_prompt(self, tools: list[ToolSpec]) -> str:
        specs = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        return (
            "You are the Relay Brain. Decompose the user's request into a short "
            "plan of tool steps (at most 6). Reply with STRICT JSON only:\n"
            '{"summary": "...", "steps": [{"tool": "<name>", "input": {...}, '
            '"description": "..."}]}\n'
            "If the request needs no tools, reply with "
            '{"respond": "<direct answer>"}.\n\n'
            "Available tools:\n"
            + __import__("json").dumps(specs, ensure_ascii=False)
        )

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        iteration: int,
    ) -> dict[str, Any]:
        from .providers import ProviderResult

        result: ProviderResult = await self.chain.generate(
            system=system or self._system_prompt(tools),
            messages=messages,
            temperature=0.2,
            max_tokens=900,
            json_mode=True,
        )
        self.provider = result.provider
        self.model = result.model
        return result.parsed


# ─────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────


class Orchestrator:
    """The agentic loop: plan → execute tools → observe → evaluate → finish."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or build_default_tools()

    # ── plan only (no execution) ──────────────────────────────
    async def plan(
        self,
        request: BrainPlanRequest,
        model: OrchestratorModel,
    ) -> BrainPlanResponse:
        system = _SYSTEM_PLAN_PROMPT
        plan = await model.plan(
            system=system,
            messages=[{"role": "user", "content": request.message}],
            tools=self.registry.specs(),
            iteration=0,
        )
        steps = plan.get("steps") or []
        respond = plan.get("respond")
        if respond:
            coerced_steps = [
                PlanStep(step_id=f"step_{uuid.uuid4().hex[:8]}", kind="respond", description=respond)
            ]
        else:
            coerced_steps = [_coerce_step(s, i) for i, s in enumerate(steps[: request.max_steps])]
        provider, model_name = _provider_meta(model)
        return BrainPlanResponse(
            plan_id=f"plan_{uuid.uuid4().hex[:12]}",
            message=request.message,
            summary=str(plan.get("summary") or ""),
            steps=coerced_steps,
            model=model_name,
            provider=provider,
        )

    # ── full agent turn (plan + execute + observe + finish) ──
    async def run_turn(
        self,
        request: AgentTurnRequest,
        model: OrchestratorModel,
    ) -> AgentTurnResponse:
        started = time.monotonic()
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        ctx = ToolContext(user_id=request.user_id, workspace_id=request.workspace_id or "")
        tools = self.registry.specs(request.tools or None)

        messages: list[dict[str, Any]] = [
            *[dict(h) for h in request.history],
            {"role": "user", "content": request.message},
        ]
        trace: list[PlanStep] = []
        final = ""

        for iteration in range(1, min(request.max_steps, MAX_STEPS) + 1):
            plan = await model.plan(
                system=_SYSTEM_AGENT_PROMPT,
                messages=messages,
                tools=tools,
                iteration=iteration,
            )
            respond = plan.get("respond")
            if respond:
                final = str(respond)
                break

            steps = plan.get("steps") or []
            if not steps:
                final = str(plan.get("summary") or "Done.")
                break

            observations: list[str] = []
            for raw in steps:
                if raw.get("kind") == "respond":
                    final = str(raw.get("text") or raw.get("description") or "Done.")
                    break
                tool = str(raw.get("tool") or "")
                step = PlanStep(
                    step_id=f"step_{iteration}_{uuid.uuid4().hex[:8]}",
                    kind="tool",
                    tool=tool,
                    description=str(raw.get("description") or raw.get("tool") or ""),
                    input=raw.get("input") or {},
                    status="running",
                )
                if not tool:
                    step.status = "skipped"
                    step.error = "no tool name in plan"
                    trace.append(step)
                    continue
                result = await self.registry.execute(tool, step.input, ctx, task_id=turn_id)
                step.status = "ok" if result.ok else "failed"
                step.output = result.output
                step.error = result.error
                trace.append(step)
                if result.ok:
                    observations.append(f"{tool} → {_compact(result.output)}")
                else:
                    final = f"Tool '{tool}' failed: {result.error}"
                    break

            if final:
                break
            if observations:
                messages.append(
                    {"role": "user", "content": "Observations:\n" + "\n".join(observations)}
                )
        else:
            # Ran out of iterations without a final response — summarize honestly.
            final = _fallback_summary(trace) or "Done."

        if not final:
            final = _fallback_summary(trace) or "Done."

        provider, model_name = _provider_meta(model)
        return AgentTurnResponse(
            turn_id=turn_id,
            text=final,
            steps=trace,
            model=model_name,
            provider=provider,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

_SYSTEM_PLAN_PROMPT = (
    "You are the Relay Brain planner. Decompose the user's request into a short "
    "ordered plan of tool steps (at most 6). Reply with STRICT JSON only:\n"
    '{"summary": "...", "steps": [{"tool": "<name>", "input": {...}, "description": "..."}]}\n'
    "If the request needs no tools, reply with {\"respond\": \"<direct answer>\"}."
)

_SYSTEM_AGENT_PROMPT = (
    "You are the Relay Brain. Execute the user's request by planning tool steps. "
    "Reply with STRICT JSON only: either "
    '{"respond": "<final answer>"} or {"steps": [{"tool": ..., "input": {...}, "description": ...}]}. '
    "Prefer finishing as soon as the question is answered."
)


def _provider_meta(model: OrchestratorModel) -> tuple[str, str]:
    if isinstance(model, ProviderChainModel):
        return model.provider, model.model
    return "", ""


def _coerce_step(raw: dict[str, Any], index: int) -> PlanStep:
    kind = raw.get("kind")
    if kind == "respond" or "text" in raw and "tool" not in raw:
        return PlanStep(
            step_id=f"step_{uuid.uuid4().hex[:8]}",
            kind="respond",
            description=str(raw.get("description") or raw.get("text") or ""),
        )
    return PlanStep(
        step_id=f"step_{uuid.uuid4().hex[:8]}",
        kind="tool",
        tool=str(raw.get("tool") or ""),
        description=str(raw.get("description") or raw.get("tool") or ""),
        input=raw.get("input") or {},
    )


def _compact(value: Any) -> str:
    try:
        import json

        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:400]


def _fallback_summary(trace: list[PlanStep]) -> str:
    if not trace:
        return ""
    ok = [s for s in trace if s.status == "ok"]
    if not ok:
        return "The request could not be completed — no tool step produced a result."
    return "Finished. Executed " + ", ".join(s.tool for s in ok) + "."


__all__ = [
    "AgentTurnResponse",
    "AgentTurnRequest",
    "BrainPlanRequest",
    "BrainPlanResponse",
    "MAX_STEPS",
    "Orchestrator",
    "OrchestratorModel",
    "PlanStep",
    "ProviderChainModel",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "TaskStatus",
    "build_default_tools",
    "safe_math_eval",
]
