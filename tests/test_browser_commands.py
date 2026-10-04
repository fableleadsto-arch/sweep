"""Browser-qualified commands launch a resolved URL, never type arbitrary text."""
import asyncio
from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace

import pytest

from sweep import Controller
from sweep.parser import parse
from sweep import skills
from sweep.skills import SkillContext
from sweep.store import ControllerStore


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch):
    launched = []
    monkeypatch.setattr(skills.subprocess, "Popen", lambda args, **kwargs: launched.append((args, kwargs)))
    monkeypatch.setattr(skills, "open_url", lambda *args: pytest.fail("Unexpected default browser fallback"))
    monkeypatch.setattr(skills, "start_app", lambda *args: pytest.fail("Unexpected application fallback"))
    return launched


@pytest.mark.parametrize("phrase,target,browser", [
    ("open youtube in brave", "youtube", "brave"),
    ("could you open YouTube in Brave", "YouTube", "brave"),
    ("Please OPEN YouTube IN BRAVE!", "YouTube", "brave"),
    ("open brave and go to youtube", "youtube", "brave"),
    ("go to youtube using brave", "youtube", "brave"),
    ("take me to youtube with Brave browser", "youtube", "brave"),
    ("navigate to GitHub in Google Chrome", "GitHub", "chrome"),
    ("launch Microsoft Edge and visit github", "github", "edge"),
    ("visit Wikipedia using Mozilla Firefox please", "Wikipedia", "firefox"),
    ("show me youtube in the Brave browser", "youtube", "brave"),
])
def test_natural_browser_commands_keep_destination_and_browser_separate(phrase, target, browser):
    command = parse(phrase)
    assert command.intent == "open"
    assert command.params == {"target": target, "browser": browser}


@pytest.mark.parametrize("url", [
    "https://example.com/Case?Key=Value&Token=A%2FB#Section",
    "https://example.com/Path/12-34?x=2+3",
    "https://example.com/Path?", "https://example.com/Path!",
])
def test_urls_survive_parser_case_punctuation_and_math_matching(url):
    assert parse("open " + url).params == {"target": url}
    assert parse("open " + url + " in Brave").params == {"target": url, "browser": "brave"}


@pytest.mark.parametrize("browser", ["brave", "chrome", "edge", "firefox"])
def test_controller_opens_resolved_site_in_requested_browser(tmp_path, monkeypatch, no_real_browser, browser):
    executable = str(tmp_path / f"{browser}.exe")
    requested = []
    monkeypatch.setattr(skills, "find_browser", lambda value: requested.append(value) or executable)
    controller = Controller(store=ControllerStore(tmp_path / "controller"))
    result = asyncio.run(controller.execute(f"could you open YouTube in {browser}",
                                           ctx=SkillContext(yes=True), allow_llm=False))
    assert result.status == "ok", result.message
    assert requested == [browser]
    assert len(no_real_browser) == 1
    arguments, options = no_real_browser[0]
    assert arguments == [executable, "https://www.youtube.com"]
    assert options["shell"] is False


def test_complete_url_is_one_unchanged_process_argument(tmp_path, monkeypatch, no_real_browser):
    monkeypatch.setattr(skills, "find_browser", lambda browser: str(tmp_path / "brave.exe"))
    url = "https://Example.com/Case?Key=A%2FB&Other=2#Section"
    result = skills.skill_open({"target": url, "browser": "BRAVE"}, SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "ok"
    assert no_real_browser[0][0] == [str(tmp_path / "brave.exe"), url]


def test_browser_qualified_custom_alias_preserves_url(tmp_path, monkeypatch, no_real_browser):
    store = ControllerStore(tmp_path)
    store.set_alias("MyDocs", "docs.example.com/Case?Key=AbC")
    monkeypatch.setattr(skills, "find_browser", lambda browser: str(tmp_path / "brave.exe"))
    result = skills.skill_open({"target": "mydocs", "browser": "brave"}, SkillContext(store=store))
    assert result.status == "ok"
    assert no_real_browser[0][0][-1] == "https://docs.example.com/Case?Key=AbC"


def test_missing_requested_browser_is_an_error_without_fallback(tmp_path, monkeypatch, no_real_browser):
    monkeypatch.setattr(skills, "find_browser", lambda browser: None)
    result = skills.skill_open({"target": "youtube", "browser": "Brave"},
                               SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "error" and "Brave" in result.message and "couldn't find" in result.message
    assert no_real_browser == []


def test_launch_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "find_browser", lambda browser: str(tmp_path / "brave.exe"))

    def fail(*args, **kwargs):
        raise OSError("Access denied")

    monkeypatch.setattr(skills.subprocess, "Popen", fail)
    result = skills.skill_open({"target": "youtube", "browser": "brave"},
                               SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "error" and "Couldn't launch Brave" in result.message


@pytest.mark.parametrize("target", ["javascript:alert(1)", "file:///C:/private.txt",
    "https://", "https://example.com:bad/", "https://example.com:99999/",
    "https://example.com\\evil", "https://user:password@example.com/",
    "https://example.com/\n--incognito", "--remote-debugging-port=9222", "something unknown"])
def test_browser_requests_reject_invalid_or_non_web_targets(tmp_path, target, no_real_browser):
    result = skills.skill_open({"target": target, "browser": "brave"},
                               SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "error"
    assert no_real_browser == []


def test_unsafe_alias_cannot_be_passed_to_browser(tmp_path, no_real_browser):
    store = ControllerStore(tmp_path)
    store.set_alias("bad", "javascript:alert(1)")
    assert skills.skill_open({"target": "bad", "browser": "brave"}, SkillContext(store=store)).status == "error"
    assert no_real_browser == []


def test_browser_name_is_not_an_executable_command(tmp_path, no_real_browser):
    result = skills.skill_open({"target": "youtube", "browser": "brave.exe & calc.exe"},
                               SkillContext(store=ControllerStore(tmp_path)))
    assert result.status == "error"
    assert no_real_browser == []


@pytest.mark.parametrize("phrase", ["go to youtube", "take me to youtube", "visit youtube", "navigate to youtube"])
def test_unqualified_navigation_resolves_site_without_navigation_words(tmp_path, monkeypatch, phrase):
    opened = []
    monkeypatch.setattr(skills, "open_url", opened.append)
    result = asyncio.run(Controller(store=ControllerStore(tmp_path)).execute(phrase, allow_llm=False))
    assert result.status == "ok"
    assert opened == ["https://www.youtube.com"]


def test_bare_browser_launch_uses_the_same_discovery(tmp_path, monkeypatch, no_real_browser):
    monkeypatch.setattr(skills, "find_browser", lambda browser: str(tmp_path / "brave.exe"))
    result = asyncio.run(Controller(store=ControllerStore(tmp_path)).execute("open Brave", allow_llm=False))
    assert result.status == "ok"
    assert no_real_browser[0][0] == [str(tmp_path / "brave.exe")]


@pytest.fixture
def isolated_windows_discovery(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Windows browser installation paths")
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA", "ProgramW6432", "PATH"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(skills, "_registered_browser", lambda name: None)
    return tmp_path


@pytest.mark.parametrize("browser,relative", [
    ("brave", "BraveSoftware/Brave-Browser/Application/brave.exe"),
    ("chrome", "Google/Chrome/Application/chrome.exe"),
    ("edge", "Microsoft/Edge/Application/msedge.exe"),
    ("firefox", "Mozilla Firefox/firefox.exe"),
])
def test_windows_known_install_locations(isolated_windows_discovery, monkeypatch, browser, relative):
    root = isolated_windows_discovery
    executable = root / relative
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture, never executed")
    monkeypatch.setenv("LOCALAPPDATA", str(root))
    assert skills.find_browser(browser) == str(executable.resolve())


def test_absolute_path_entry_discovers_requested_executable(isolated_windows_discovery, monkeypatch):
    root = isolated_windows_discovery
    executable = root / "brave.exe"
    executable.write_bytes(b"fixture, never executed")
    monkeypatch.setenv("PATH", str(root))
    assert skills.find_browser("brave") == str(executable.resolve())
    monkeypatch.chdir(root)
    monkeypatch.setenv("PATH", "." + os.pathsep)
    assert skills.find_browser("brave") is None


@pytest.mark.parametrize("kind,quoted", [(1, False), (1, True), (2, False)])
def test_windows_app_paths_reads_only_a_valid_executable(tmp_path, monkeypatch, kind, quoted):
    if os.name != "nt":
        pytest.skip("Windows App Paths")
    executable = tmp_path / "Browser Files" / "brave.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"fixture, never executed")
    value = f'"{executable}"' if quoted else str(executable)
    fake_registry = SimpleNamespace(HKEY_CURRENT_USER=1, HKEY_LOCAL_MACHINE=2,
        KEY_WOW64_64KEY=256, KEY_WOW64_32KEY=512, KEY_READ=1, REG_SZ=1, REG_EXPAND_SZ=2,
        OpenKey=lambda *args: nullcontext("key"), QueryValueEx=lambda *args: (value, kind),
        ExpandEnvironmentStrings=lambda text: str(executable))
    monkeypatch.setitem(sys.modules, "winreg", fake_registry)
    assert skills._registered_browser("brave.exe") == str(executable.resolve())


@pytest.mark.parametrize("suffix", ['" --incognito', '" & calc.exe'])
def test_app_paths_command_strings_are_never_executed(tmp_path, monkeypatch, suffix):
    executable = tmp_path / "brave.exe"
    executable.write_bytes(b"fixture, never executed")
    executable.chmod(0o700)
    assert skills._browser_executable('"' + str(executable) + suffix, ("brave.exe",)) is None
    assert skills._browser_executable("brave.exe", ("brave.exe",)) is None
    assert skills._browser_executable(str(executable), ("firefox.exe",)) is None
