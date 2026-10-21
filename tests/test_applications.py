"""Installed-app discovery and launch receipts without running user programs."""
import asyncio
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sweep import Controller, applications, skills
from sweep.applications import AmbiguousApplication, LaunchReceipt, matching_entry
from sweep.parser import parse
from sweep.skills import SkillContext
from sweep.store import ControllerStore


@pytest.mark.parametrize("phrase,target,desktop", [
    ("could you bring up the calculator", "calculator", False),
    ("launch the installed app called Telegram", "Telegram", True),
    ("open VS Code on my computer", "VS Code", True),
    ("open the Spotify desktop app", "Spotify", True),
    ("run the calculator", "calculator", False),
    ("run notepad.exe", "notepad.exe", False),
    ("run the software called Acme Editor", "Acme Editor", True),
    ("run Acme Editor app", "Acme Editor", True),
])
def test_conversational_app_commands(phrase, target, desktop):
    parsed = parse(phrase)
    assert parsed.intent == "open"
    assert parsed.params == {"target": target, **({"desktop": True} if desktop else {})}


def test_shell_arguments_remain_an_explicit_shell_task():
    parsed = parse("run notepad.exe example.txt")
    assert parsed.intent == "run"
    assert parsed.params["command"] == "notepad.exe example.txt"


def test_match_prefers_exact_name_and_rejects_partial_word_guesses():
    entries = (("Paint 3D", "3d"), ("Paint", "paint"), ("Painter", "painter"))
    assert matching_entry("paint", entries) == "paint"
    assert matching_entry("pain", entries) is None
    assert matching_entry("Visual Studio", (("Microsoft Visual Studio 2022", "vs"),)) == "vs"


@pytest.mark.parametrize("entries", [
    (("Acme Editor", "editor"), ("Acme Viewer", "viewer")),
    (("Acme", "different1"), ("Acme", "different2")),
])
def test_ambiguous_names_do_not_choose_first_app(entries):
    with pytest.raises(AmbiguousApplication, match="Several installed apps"):
        matching_entry("Acme", entries)


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "_menu_shortcuts", lambda: ())
    monkeypatch.setattr(skills, "windows_start_apps", lambda: ())
    monkeypatch.setattr(skills, "_registered_browser", lambda _name: None)
    monkeypatch.setenv("PATH", str(tmp_path))
    return tmp_path


def test_executable_alias_precedes_ambiguous_menu_items(discovery, monkeypatch):
    executable = discovery / "calc.exe"
    executable.write_bytes(b"fixture, never run")
    monkeypatch.setattr(skills, "_menu_shortcuts", lambda: (
        ("Calculator Professional", "pro.lnk"), ("Calculator Basic", "basic.lnk")))
    monkeypatch.setattr(skills.shutil, "which", lambda cmd, **_kwargs: str(executable) if cmd == "calc.exe" else None)
    assert skills._app_ready("calc.exe", ["calculator"]) == str(executable)


@pytest.mark.parametrize("alias", ["code", "vs code", "vscode", "visual studio code"])
def test_vscode_alias_resolves_native_shortcut(discovery, monkeypatch, alias):
    monkeypatch.setattr(skills, "_menu_shortcuts", lambda: (("Visual Studio Code", "Code.lnk"),))
    assert skills._app_ready(skills._app_target(alias), [alias]) == "Code.lnk"


def test_start_menu_scan_ignores_web_links_and_keeps_same_name_ambiguity(tmp_path, monkeypatch):
    for variable in ("APPDATA", "PROGRAMDATA"):
        root = tmp_path / variable
        menu = root / "Microsoft/Windows/Start Menu/Programs"
        menu.mkdir(parents=True)
        (menu / "Acme.lnk").write_text("fixture")
        (menu / "Acme online.url").write_text("fixture")
        monkeypatch.setenv(variable, str(root))
    skills._menu_shortcuts.cache_clear()
    try:
        entries = skills._menu_shortcuts()
        assert len(entries) == 2
        assert all(target.endswith(".lnk") for _, target in entries)
        with pytest.raises(AmbiguousApplication):
            skills._lookup_menu_entry("Acme")
    finally:
        skills._menu_shortcuts.cache_clear()


def test_unregistered_store_app_resolves_registered_app_id(discovery, monkeypatch):
    monkeypatch.setattr(skills, "windows_start_apps", lambda: (("Acme Editor", "shell:AppsFolder\\Acme_123!App"),))
    assert skills._app_ready("", ["Acme Editor"]) == "shell:AppsFolder\\Acme_123!App"


@pytest.mark.skipif(os.name != "nt", reason="Windows App Paths")
@pytest.mark.parametrize("requested,command", [("Acme", ""), ("acme.exe", "acme.exe")])
def test_generic_app_paths_lookup(discovery, monkeypatch, requested, command):
    executable = discovery / "acme.exe"
    executable.write_bytes(b"fixture, never run")
    queries = []
    def registered(name):
        queries.append(name)
        return str(executable.resolve()) if name.casefold() == "acme.exe" else None
    monkeypatch.setattr(skills, "_registered_browser", registered)
    assert skills._app_ready(command, [requested]) == str(executable.resolve())
    assert queries == [requested if command else requested + ".exe"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Start Apps")
def test_store_discovery_uses_fixed_script_and_validates_rows(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='[{"Name":"Acme","AppID":"Acme_123!App"},'
            '{"Name":"Bad","AppID":"abc; calc.exe"},{"Name":"Missing"}]')

    monkeypatch.setattr(applications.subprocess, "run", run)
    applications.windows_start_apps.cache_clear()
    try:
        assert applications.windows_start_apps() == (("Acme", "shell:AppsFolder\\Acme_123!App"),)
        assert applications.windows_start_apps() == (("Acme", "shell:AppsFolder\\Acme_123!App"),)
        assert len(calls) == 1
        assert calls[0][1]["shell"] is False and calls[0][1]["timeout"] == 6
        assert "Get-StartApps" in calls[0][0][-1]
    finally:
        applications.windows_start_apps.cache_clear()


@pytest.mark.skipif(os.name != "nt", reason="Windows Start Apps")
def test_store_discovery_timeout_is_a_missing_source(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("powershell", 6)
    monkeypatch.setattr(applications.subprocess, "run", timeout)
    applications.windows_start_apps.cache_clear()
    try:
        assert applications.windows_start_apps() == ()
    finally:
        applications.windows_start_apps.cache_clear()


def test_relative_working_directory_executable_is_never_discovered(discovery, monkeypatch):
    executable = discovery / "acme.exe"
    executable.write_bytes(b"fixture, never run")
    monkeypatch.chdir(discovery)
    monkeypatch.setenv("PATH", "." + os.pathsep)
    assert skills._app_ready("acme.exe", ["Acme"]) is None


@pytest.mark.parametrize("receipt,status,fragment", [
    (LaunchReceipt(False, error="Access denied"), "error", "Access denied"),
    (LaunchReceipt(True), "info", "couldn't verify"),
    (LaunchReceipt(True, True, 123), "ok", "window is visible"),
])
def test_controller_reports_launch_receipt_truthfully(tmp_path, monkeypatch, receipt, status, fragment):
    monkeypatch.setattr(skills, "_app_ready", lambda *_args: "C:/Installed/Acme.exe")
    monkeypatch.setattr(skills, "start_app", lambda _target: receipt)
    monkeypatch.setattr(skills, "open_url", lambda *_args: pytest.fail("Unexpected web fallback"))
    controller = Controller(store=ControllerStore(tmp_path))
    result = asyncio.run(controller.execute("open the Acme desktop app", allow_llm=False))
    assert result.status == status and fragment in result.message


@pytest.mark.parametrize("target,desktop", [("does not exist", False), ("spotify", True), ("example.com", True)])
def test_missing_app_never_turns_into_web_search(tmp_path, monkeypatch, target, desktop):
    monkeypatch.setattr(skills, "_app_ready", lambda *_args: None)
    monkeypatch.setattr(skills, "open_url", lambda *_args: pytest.fail("Unexpected web fallback"))
    result = skills.skill_open({"target": target, "desktop": desktop}, SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "error" and "couldn't find" in result.message


def test_ambiguous_app_returns_actionable_error_without_launch(tmp_path, monkeypatch):
    def ambiguous(*_args):
        raise AmbiguousApplication("Several installed apps match Acme: Acme Editor, Acme Viewer.")
    monkeypatch.setattr(skills, "_app_ready", ambiguous)
    monkeypatch.setattr(skills, "start_app", lambda *_args: pytest.fail("Unexpected launch"))
    result = skills.skill_open({"target": "Acme"}, SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "error" and "Acme Editor" in result.message


@pytest.mark.parametrize("suffix", [".cmd", ".bat", ".ps1", ".vbs", ".js"])
def test_scripts_require_explicit_shell_intent(tmp_path, monkeypatch, suffix):
    script = tmp_path / ("script" + suffix)
    script.write_text("fixture, never run")
    monkeypatch.setattr(skills, "start_app", lambda *_args: pytest.fail("Unexpected script launch"))
    result = skills.skill_open({"target": str(script)}, SkillContext(store=ControllerStore(tmp_path / "store")))
    assert result.status == "error" and "explicit shell task" in result.message


@pytest.mark.skipif(os.name != "nt", reason="Windows ShellExecuteEx")
@pytest.mark.parametrize("accepted,handle,visible,exit_code,expected", [
    (False, 0, False, 0, LaunchReceipt(False, error="Access denied")),
    (True, 0, False, 0, LaunchReceipt(True)),
    (True, 55, True, 259, LaunchReceipt(True, True, 123)),
    (True, 55, False, 259, LaunchReceipt(True, process_id=123)),
    (True, 55, False, 0, LaunchReceipt(True, process_id=123)),
    (True, 55, False, 7, LaunchReceipt(False, process_id=123, error="The application exited with code 7.")),
])
def test_native_launch_acceptance_visibility_failure_and_handle_cleanup(
    monkeypatch, accepted, handle, visible, exit_code, expected,
):
    shell, kernel = Mock(), Mock()
    seen = []

    def launch(pointer):
        info = pointer._obj
        seen.append((info.lpFile, info.lpParameters, info.lpVerb))
        info.hProcess = handle
        return accepted

    def exited(_handle, pointer):
        pointer._obj.value = exit_code
        return True

    shell.ShellExecuteExW.side_effect = launch
    kernel.GetProcessId.return_value = 123
    kernel.GetExitCodeProcess.side_effect = exited
    monkeypatch.setattr(applications.ctypes, "WinDLL", lambda name, **_kwargs: shell if name == "shell32" else kernel)
    monkeypatch.setattr(applications.ctypes, "FormatError", lambda _code: "Access denied")
    monkeypatch.setattr(applications, "_visible_window", lambda pid: visible and pid == 123)
    assert applications.launch_windows("C:/Installed/Acme.exe", verify_seconds=0) == expected
    assert seen == [("C:/Installed/Acme.exe", None, "open")]
    if accepted and handle:
        kernel.CloseHandle.assert_called_once_with(handle)
    else:
        kernel.CloseHandle.assert_not_called()
