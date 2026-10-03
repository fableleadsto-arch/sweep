"""Tests for the terminal controller: parser, store and dispatch loop."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import sweep.skills as skills_mod
from sweep import Controller
from sweep.parser import parse
from sweep.skills import SkillContext, skill_calc
from sweep.store import ControllerStore


@pytest.fixture()
def store(tmp_path: Path) -> ControllerStore:
    return ControllerStore(base_dir=tmp_path / "controller")


def _run(coro):
    return asyncio.run(coro)


# ── store ─────────────────────────────────────────────────────────────


def test_store_notes_roundtrip(store: ControllerStore) -> None:
    note = store.add_note("my wifi password is hunter2")
    assert store.notes[0].id == note.id
    assert store.search_notes("wifi")[0].text == "my wifi password is hunter2"
    assert store.search_notes("no-such-term") == []


def test_store_aliases_are_case_insensitive(store: ControllerStore) -> None:
    store.set_alias("MyCode", "C:\\code")
    assert store.get_alias("mycode") == "C:\\code"
    assert store.get_alias("MYCODE") == "C:\\code"
    assert store.remove_alias("mycode") == "C:\\code"
    assert store.get_alias("mycode") is None


def test_store_stats(store: ControllerStore) -> None:
    store.record_intent("open")
    store.record_intent("open")
    assert store.stats["intents"]["open"] == 2
    assert store.stats["total"] == 2


# ── parser: intent mapping ────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,intent",
    [
        ("open youtube", "open"),
        ("please open youtube", "open"),
        ("can you open gmail", "open"),
        ("could you please open Youtube", "open"),
        ("could you open youtube", "open"),
        ("will you open youtube", "open"),
        ("hey sweep open twitter", "open"),
        ("sweep open github", "open"),
        ("open youtube please", "open"),
        ("open youtube for me", "open"),
        ("go to netflix", "open"),
        ("launch calculator", "open"),
        ("youtube", "open"),
        ("search for python docs", "search"),
        ("look up tensorflow", "search"),
        ("who is Isaac Newton", "search"),
        ("what is the capital of france", "search"),
        ("search python", "search"),
        ("what time is it", "time"),
        ("what's the time", "time"),
        ("current time", "time"),
        ("what's today's date", "date"),
        ("what is 15% of 200", "calc"),
        ("what is 2 plus 2", "calc"),
        ("2 + 2", "calc"),
        ("calculate 45 * 3", "calc"),
        ("weather in london", "weather"),
        ("whats the weather like in paris", "weather"),
        ("run pip version", "run"),
        ("install requests", "run"),
        ("list files", "files"),
        ("show files", "files"),
        ("create folder projects", "files"),
        ("open folder downloads", "files"),
        ("find files called readme", "files"),
        ("remember that my wifi password is hunter2", "notes"),
        ("remind me to call mom", "notes"),
        ("what did I note about wifi", "notes"),
        ("show notes", "notes"),
        ("system info", "system"),
        ("shutdown the computer", "power"),
        ("lock the pc", "power"),
        ("restart", "power"),
        ("help", "help"),
        ("exit", "exit"),
        ("hello", "chat"),
        ("thanks", "chat"),
        ("alias set mycode C:\\code", "alias"),
    ],
)
def test_parser_intents(text: str, intent: str) -> None:
    cmd = parse(text)
    assert cmd is not None, f"no intent parsed for {text!r}"
    assert cmd.intent == intent, f"{text!r} → {cmd.intent!r}, expected {intent!r}"


def test_parser_open_target() -> None:
    cmd = parse("please open youtube")
    assert cmd.intent == "open"
    assert cmd.params["target"] == "youtube"


def test_parser_notes_query_not_remember() -> None:
    cmd = parse("what did I note about wifi")
    assert cmd.intent == "notes"
    assert cmd.params["action"] == "query"


def test_parser_remind_strips_me() -> None:
    cmd = parse("remind me to call mom")
    assert cmd.params["text"] == "to call mom"


def test_parser_sweep_prefix_variants() -> None:
    assert parse("sweep open youtube").intent == "open"
    assert parse("sweep, open youtube").intent == "open"


def test_parser_bare_known_site() -> None:
    assert parse("youtube").intent == "open"
    assert parse("calculator").params["target"] == "calculator"


# ── calc skill ────────────────────────────────────────────────────────


def test_calc_percent() -> None:
    result = skill_calc({"expression": "200*(15/100)"}, SkillContext())
    assert result.status == "ok"
    assert "30" in result.message


def test_calc_plain() -> None:
    result = skill_calc({"expression": "2 + 2"}, SkillContext())
    assert "4" in result.message


def test_calc_invalid() -> None:
    result = skill_calc({"expression": "foo + bar"}, SkillContext())
    assert result.status == "error"


# ── dispatch loop with mocked OS actions ──────────────────────────────


def test_execute_open_youtube(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("please open youtube", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert "youtube" in result.message
    assert opened == ["https://www.youtube.com"]
    assert controller.intent_stats().get("open", 0) == 1


def test_execute_open_unknown_target_searches(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("open skywarp the dragon", ctx=SkillContext(yes=True)))
    assert result.status == "info"
    assert opened and "duckduckgo" in opened[0]


def test_open_installed_app_launches_natively(tmp_path: Path, monkeypatch) -> None:
    launched: list[str] = []
    monkeypatch.setattr(skills_mod, "start_app", lambda cmd: (launched.append(cmd) or True))
    monkeypatch.setattr(skills_mod, "_lookup_menu_entry", lambda _name: None)
    monkeypatch.setattr(skills_mod.shutil, "which", lambda cmd: "C:/fake/" + cmd)
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("open calculator", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert "Launched" in result.message
    assert launched == ["C:/fake/calc.exe"]


def test_open_not_installed_app_falls_back_to_website(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    monkeypatch.setattr(skills_mod, "start_app", lambda _cmd: True)
    monkeypatch.setattr(skills_mod, "_lookup_menu_entry", lambda _name: None)
    monkeypatch.setattr(skills_mod.shutil, "which", lambda _cmd: None)
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("open spotify", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert opened == ["https://open.spotify.com"]


def test_open_in_browser_forces_website(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("open spotify in browser", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert opened == ["https://open.spotify.com"]


def test_open_in_new_tab_forces_website(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("open gmail in a new tab", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert opened == ["https://mail.google.com"]


def test_execute_run_requires_confirm(tmp_path: Path) -> None:
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    ctx = SkillContext(ask=lambda _prompt: "n")
    result = _run(controller.execute("run echo hello", ctx=ctx))
    assert result.status == "info"
    assert "cancelled" in result.message.lower()


@pytest.mark.parametrize("reply", ["y", "yes", "Y", "sure", "ok"])
def test_execute_run_confirmed(tmp_path: Path, monkeypatch, reply: str) -> None:
    ran: list[list[str]] = []

    def fake_run(command: str, timeout: int = 60):
        ran.append([command])
        return True, "hello", ""

    monkeypatch.setattr(skills_mod, "_run_command", fake_run)
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    ctx = SkillContext(ask=lambda _prompt: reply)
    result = _run(controller.execute("run echo hello", ctx=ctx))
    assert result.status == "ok"
    assert ran == [["echo hello"]]


def test_execute_run_yes_flag_skips_prompt(tmp_path: Path, monkeypatch) -> None:
    ran: list[list[str]] = []

    def fake_run(command: str, timeout: int = 60):
        ran.append([command])
        return True, "hello", ""

    monkeypatch.setattr(skills_mod, "_run_command", fake_run)
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("run echo hi", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert ran == [["echo hi"]]


def test_execute_notes_roundtrip(tmp_path: Path) -> None:
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    first = _run(controller.execute("remember that my wifi password is hunter2", ctx=SkillContext(yes=True)))
    assert first.status == "ok"
    second = _run(controller.execute("what did I note about wifi", ctx=SkillContext(yes=True)))
    assert second.status == "ok"
    assert "hunter2" in second.message


def test_execute_search_with_mocked_engine(tmp_path: Path, monkeypatch) -> None:
    async def fake_search(query, limit=10, **kwargs):
        return {
            "hits": [
                {"url": "https://example.com/a", "title": "Alpha", "snippet": "first result"},
                {"url": "https://example.com/b", "title": "Beta", "snippet": "second result"},
            ],
            "errors": [],
        }

    monkeypatch.setattr("app.search.engine.relai_search", fake_search)
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("search for python docs", ctx=SkillContext(yes=True)))
    assert result.status == "ok"
    assert "https://example.com/a" in result.message


def test_execute_search_fallback_opens_browser(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    opened: list[str] = []

    async def fake_search(query, limit=10, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("app.search.engine.relai_search", fake_search)
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("search for python docs", ctx=SkillContext(yes=True)))
    assert result.status == "info"
    assert opened and "duckduckgo" in opened[0]


def test_execute_unknown_defaults_to_search(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []

    async def fake_search(query, limit=10, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("app.search.engine.relai_search", fake_search)
    monkeypatch.setattr(skills_mod, "open_url", lambda url: opened.append(url))
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("zzzz gibberish phrase", ctx=SkillContext(yes=True), allow_llm=False))
    assert result.status == "info"
    assert "duckduckgo" in opened[0]


def test_execute_time_and_help(tmp_path: Path) -> None:
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    t = _run(controller.execute("what time is it", ctx=SkillContext(yes=True)))
    assert t.status == "ok" and "It's" in t.message
    h = _run(controller.execute("help", ctx=SkillContext(yes=True)))
    assert h.status == "ok" and "open youtube" in h.message


def test_execute_power_requires_confirm(tmp_path: Path) -> None:
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    ctx = SkillContext(ask=lambda _prompt: "n")
    result = _run(controller.execute("shutdown the computer", ctx=ctx))
    assert result.status == "info"
    assert "cancelled" in result.message.lower()


def test_execute_exit_signal(tmp_path: Path) -> None:
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    result = _run(controller.execute("exit", ctx=SkillContext(yes=True)))
    assert result.status == "exit"


def test_alias_roundtrip(tmp_path: Path) -> None:
    controller = Controller(store=ControllerStore(base_dir=tmp_path / "c"))
    set_res = _run(controller.execute("alias set mycode C:\\projects", ctx=SkillContext(yes=True)))
    assert set_res.status == "ok"
    list_res = _run(controller.execute("alias list", ctx=SkillContext(yes=True)))
    assert "mycode" in list_res.message
    assert controller.store.get_alias("mycode") == "C:\\projects"