"""The multipurpose terminal controller.

Turns plain-English commands ("open youtube", "please search for X", "remind me
to call mom") into day-to-day actions. Intents parsed deterministically first for
speed and predictability, with an optional LLM fallback for ambiguous wording.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .parser import ParsedCommand, looks_like_question, llm_parse, parse
from .skills import (
    ActionResult,
    SkillContext,
    skill_alias,
    skill_calc,
    skill_chat,
    skill_date,
    skill_files,
    skill_help,
    skill_notes,
    skill_open,
    skill_power,
    skill_run,
    skill_search,
    skill_system,
    skill_time,
    skill_weather,
)
from .store import ControllerStore


@dataclass
class Skill:
    """A registered controller capability."""

    name: str
    description: str
    handler: Callable[[dict[str, Any], SkillContext], Any]
    examples: list[str] = field(default_factory=list)
    needs_confirm: bool = False

    async def run(self, params: dict[str, Any], ctx: SkillContext) -> ActionResult:
        out = self.handler(params, ctx)
        if hasattr(out, "__await__"):
            out = await out
        return out if isinstance(out, ActionResult) else ActionResult("ok", str(out or ""))


class Controller:
    """Registry + dispatch: parse → confirm → run → report."""

    def __init__(
        self,
        store: Optional[ControllerStore] = None,
        settings: Optional[Any] = None,
    ) -> None:
        self.store = store or ControllerStore()
        self.settings = settings  # BrainSettings (used only for LLM fallback)
        self.skills: dict[str, Skill] = {}
        self._register_default_skills()

    # ── registry ───────────────────────────────────────────────────────

    def _register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def _register_default_skills(self) -> None:
        self._register(Skill("open", "Open a website, app or folder.", skill_open,
                             ["open youtube", "open calculator", "open my downloads"]))
        self._register(Skill("search", "Search the web.", skill_search,
                             ["search for python docs", "who is Isaac Newton"]))
        self._register(Skill("run", "Run a shell command.", skill_run,
                             ["run pip version", "run dir"], needs_confirm=True))
        self._register(Skill("calc", "Do math.", skill_calc,
                             ["what is 15% of 200", "2 + 2"]))
        self._register(Skill("time", "Current time.", skill_time, ["what time is it"]))
        self._register(Skill("date", "Today's date.", skill_date, ["what's today's date"]))
        self._register(Skill("weather", "Weather for a city.", skill_weather,
                             ["weather in london"]))
        self._register(Skill("files", "List, open, create or find files/folders.", skill_files,
                             ["list files", "create folder projects", "open folder downloads"]))
        self._register(Skill("notes", "Remember things and look them up later.", skill_notes,
                             ["remember my wifi password is X", "what did I note about wifi"]))
        self._register(Skill("system", "Report system info.", skill_system, ["system info"]))
        self._register(Skill("power", "Shutdown, restart, lock or sleep.", skill_power,
                             ["shutdown the computer"], needs_confirm=True))
        self._register(Skill("alias", "Manage custom shortcuts.", skill_alias,
                             ["alias set mycode C:\\code", "alias list"]))
        self._register(Skill("help", "Show this help.", skill_help, ["help"]))
        self._register(Skill("chat", "Small talk.", skill_chat, ["hi", "thanks"]))

    # ── confirmations ──────────────────────────────────────────────────

    def _confirm_enabled(self, ctx: SkillContext) -> bool:
        if ctx.yes:
            return False
        return bool(self.store.get_config("confirm_actions", True))

    async def _confirm(self, ctx: SkillContext, prompt: str) -> bool:
        ask = ctx.ask or input
        text = ""
        while True:
            try:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, ask, prompt)
            except (EOFError, KeyboardInterrupt):
                return False
            choice = str(text or "n").strip().lower()
            if choice in {"y", "yes", "1", "ok", "sure"}:
                return True
            if choice in {"n", "no", "0"}:
                return False

    def _confirm_prompt(self, cmd: ParsedCommand) -> str:
        if cmd.intent == "run":
            command = cmd.params.get("command", "")
            return f"Run command: {command!r}? [y/N] "
        action = cmd.params.get("action", cmd.intent)
        return f"Are you sure you want to {action} the computer? [y/N] "

    # ── execution ──────────────────────────────────────────────────────

    async def execute(
        self,
        text: str,
        ctx: Optional[SkillContext] = None,
        allow_llm: bool = True,
    ) -> ActionResult:
        """Full pipeline for one request: parse → confirm → run → report."""
        ctx = ctx or SkillContext()
        ctx.store = self.store  # the controller's store is authoritative
        raw = (text or "").strip()
        if not raw:
            return ActionResult("info", "Say something like \"open youtube\" or \"help\".")

        cmd = parse(raw)
        if cmd is None:
            if looks_like_question(raw):
                cmd = ParsedCommand("search", {"query": raw}, raw, "rule")
            elif allow_llm:
                cmd = await llm_parse(raw, self.settings)
            if cmd is None:
                cmd = ParsedCommand("search", {"query": raw}, raw, "rule")

        if cmd.intent == "exit":
            self.store.record_intent("exit")
            return ActionResult("exit", "Goodbye.")

        skill = self.skills.get(cmd.intent)
        if skill is None:
            # Unknown intent (e.g. from the LLM) → search as the safe default.
            cmd = ParsedCommand("search", {"query": raw}, raw, cmd.source)
            skill = self.skills["search"]

        self.store.record_intent(cmd.intent)

        if skill.needs_confirm and self._confirm_enabled(ctx):
            if not await self._confirm(ctx, self._confirm_prompt(cmd)):
                return ActionResult("info", "Alright, cancelled.")

        try:
            return await skill.run(cmd.params, ctx)
        except Exception as exc:  # noqa: BLE001 — a failing skill is a reportable result
            return ActionResult("error", f"Something went wrong: {exc}")

    async def help_text(self) -> str:
        skill = self.skills["help"]
        return (await skill.run({}, SkillContext(store=self.store))).message

    def intent_stats(self) -> dict[str, Any]:
        return self.store.stats.get("intents", {})


__all__ = ["ActionResult", "Controller", "ControllerStore", "Skill", "SkillContext"]
