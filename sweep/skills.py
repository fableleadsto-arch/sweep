"""Day-to-day skills the terminal controller can perform.

Each skill is a plain callable ``(params: dict, ctx: SkillContext) -> ActionResult``
so it is trivial to test and easy to extend. Handlers that touch the OS go
through tiny module-level helpers (``open_url``, ``start_app``, ...) so tests
can monkeypatch them without ever opening a real browser or process.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .store import ControllerStore

HOME = Path.home()

# ── wide catalogue of well-known sites ────────────────────────────────
SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "mail": "https://mail.google.com",
    "drive": "https://drive.google.com",
    "docs": "https://docs.google.com",
    "sheets": "https://sheets.google.com",
    "calendar": "https://calendar.google.com",
    "maps": "https://maps.google.com",
    "translate": "https://translate.google.com",
    "news": "https://news.google.com",
    "github": "https://github.com",
    "gitlab": "https://gitlab.com",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "netflix": "https://www.netflix.com",
    "hulu": "https://www.hulu.com",
    "prime video": "https://www.primevideo.com",
    "spotify": "https://open.spotify.com",
    "soundcloud": "https://soundcloud.com",
    "twitch": "https://www.twitch.tv",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "linkedin": "https://www.linkedin.com",
    "whatsapp": "https://web.whatsapp.com",
    "telegram": "https://web.telegram.org",
    "discord": "https://discord.com",
    "pinterest": "https://www.pinterest.com",
    "tiktok": "https://www.tiktok.com",
    "amazon": "https://www.amazon.com",
    "flipkart": "https://www.flipkart.com",
    "wikipedia": "https://www.wikipedia.org",
    "wiki": "https://www.wikipedia.org",
    "quora": "https://www.quora.com",
    "medium": "https://medium.com",
    "dev.to": "https://dev.to",
    "hacker news": "https://news.ycombinator.com",
    "hn": "https://news.ycombinator.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "perplexity": "https://www.perplexity.ai",
    "notion": "https://www.notion.so",
    "trello": "https://trello.com",
    "slack": "https://app.slack.com",
    "feedly": "https://feedly.com",
    "weather.com": "https://weather.com",
    "accuweather": "https://www.accuweather.com",
    "pypi": "https://pypi.org",
    "npm": "https://www.npmjs.com",
    "mdn": "https://developer.mozilla.org",
    "w3schools": "https://www.w3schools.com",
    "kaggle": "https://www.kaggle.com",
    "hugging face": "https://huggingface.co",
    "huggingface": "https://huggingface.co",
    "arxiv": "https://arxiv.org",
    "scholar": "https://scholar.google.com",
    "replit": "https://replit.com",
    "codesandbox": "https://codesandbox.io",
    "canva": "https://www.canva.com",
    "figma": "https://www.figma.com",
    "dribbble": "https://dribbble.com",
    "behance": "https://www.behance.net",
    "coeurl": "https://coeurl.dev",
    "duckduckgo": "https://duckduckgo.com",
    "bing": "https://www.bing.com",
    "yahoo": "https://www.yahoo.com",
    "ebay": "https://www.ebay.com",
    "paypal": "https://www.paypal.com",
    "bank": "https://www.chase.com",
    "microsoft": "https://www.microsoft.com",
    "apple": "https://www.apple.com",
    "adobe": "https://www.adobe.com",
    "coursera": "https://www.coursera.org",
    "udemy": "https://www.udemy.com",
    "khan academy": "https://www.khanacademy.org",
    "summit": "https://summit.cern.ch",
    "ollama": "https://ollama.com",
    "openai": "https://openai.com",
    "anthropic": "https://www.anthropic.com",
    "google colab": "https://colab.research.google.com",
    "colab": "https://colab.research.google.com",
    "jupyter": "https://jupyter.org",
}

# ── well-known desktop apps (win32 + common fallbacks) ────────────────
APPS: dict[str, str] = {
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "notepad": "notepad.exe",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "control panel": "control.exe",
    "controlpanel": "control.exe",
    "explorer": "explorer.exe",
    "file manager": "explorer.exe",
    "files": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "powershell 7": "pwsh.exe",
    "terminal": "wt.exe",
    "settings": "ms-settings:",
    "camera": "microsoft.windows.camera:",
    "photos": "ms-photos:",
    "store": "ms-windows-store:",
    "snipping tool": "SnippingTool.exe",
    "clipboard": "ms-settings:clipboard",
    "sound": "ms-settings:sound",
    "notifications": "ms-settings:notifications",
    "wifi": "ms-settings:network-wifi",
    "bluetooth": "ms-settings:bluetooth",
    "display": "ms-settings:display",
    "mouse": "ms-settings:mousetouchpad",
    "clock": "ms-settings:dateandtime",
    "region": "ms-settings:regionformatting",
    "disk cleanup": "cleanmgr.exe",
    "cleanmgr": "cleanmgr.exe",
    "character map": "charmap.exe",
    "magnifier": "magnify.exe",
    "on screen keyboard": "osk.exe",
    "osk": "osk.exe",
    "registry editor": "regedit.exe",
    "regedit": "regedit.exe",
    "run": "ms-settings:appsfeatures",
    "appwiz": "appwiz.cpl",
    "winword": "winword.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "power point": "powerpnt.exe",
    "outlook": "outlook.exe",
    "onenote": "onenote.exe",
    "teams": "ms-teams:",
    "meet": "https://meet.google.com",
    "zoom": "zoommtg:",
    "code": "code",
    "vs code": "code",
    "vscode": "code",
    "visual studio": "devenv.exe",
    "pycharm": "pycharm64.exe",
    "sublime": "subl",
    "sublime text": "subl",
    "atom": "atom",
    "intellij": "idea64.exe",
    "chrome": "chrome.exe",
    "browser": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "opera": "opera.exe",
    "brave": "brave.exe",
    "spotify app": "spotify.exe",
    "spotify": "spotify.exe",
    "discord app": "discord.exe",
    "discord": "discord.exe",
    "slack app": "slack.exe",
    "slack": "slack.exe",
    "telegram app": "Telegram.exe",
    "telegram": "Telegram.exe",
    "whatsapp app": "WhatsApp.exe",
    "whatsapp": "WhatsApp.exe",
    "steam": "steam.exe",
    "epic": "EpicGamesLauncher.exe",
    "obs": "obs64.exe",
    "vlc": "vlc.exe",
    "media player": "wmplayer.exe",
    "blender": "blender.exe",
    "unity": "Unity.exe",
    "unity hub": "Unity Hub.exe",
    "gimp": "gimp.exe",
    "photoshop": "Photoshop.exe",
    "office": "winword.exe",
}

# ── Windows well-known folders ────────────────────────────────────────
FOLDERS: dict[str, str] = {
    "downloads": str(HOME / "Downloads"),
    "download": str(HOME / "Downloads"),
    "desktop": str(HOME / "Desktop"),
    "documents": str(HOME / "Documents"),
    "music": str(HOME / "Music"),
    "pictures": str(HOME / "Pictures"),
    "videos": str(HOME / "Videos"),
    "home": str(HOME),
    "temp": (os.environ.get("TEMP") or str(HOME / "AppData" / "Local" / "Temp")),
}

_ARTICLES = {"the", "a", "an", "my"}
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass
class SkillContext:
    """Everything a skill may need, injected by the controller."""

    store: ControllerStore = field(default_factory=ControllerStore)
    yes: bool = False
    ask: Optional[callable] = None  # for interactive confirmations


@dataclass
class ActionResult:
    """Structured result of a skill run."""

    status: str  # "ok" | "info" | "error"
    message: str

    def __bool__(self) -> bool:
        return self.status != "error"


# ── low-level OS helpers (the only monkeypatch points) ────────────────


def open_url(url: str) -> None:
    """Open an http(s) URL in the default browser."""
    webbrowser.open(url, new=2)


def start_app(command: str) -> bool:
    """Launch a desktop app; returns True on success."""
    if os.name != "nt":
        try:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, command], start_new_session=True)
            return True
        except OSError:
            return False
    try:
        return os.startfile(command) is None  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        try:
            subprocess.Popen(
                [command],
                creationflags=CREATE_NO_WINDOW,
                shell=False,
            )
            return True
        except OSError:
            return False


def open_folder(path: str) -> None:
    if os.name == "nt":
        subprocess.Popen(["explorer", path])
    else:
        start_app(path)


def search_url(query: str, engine: str = "duckduckgo") -> str:
    from urllib.parse import quote_plus

    if engine == "google":
        return f"https://www.google.com/search?q={quote_plus(query)}"
    return f"https://duckduckgo.com/?q={quote_plus(query)}"


# ── open ──────────────────────────────────────────────────────────────


def _browser_target(target: str, ctx: SkillContext) -> Optional[str]:
    """Resolve a site-name / alias / url-ish string to a URL."""
    lowered = target.lower().strip()
    alias = ctx.store.get_alias(lowered)
    if alias:
        return alias if "://" in alias else f"https://{alias}"
    if lowered in SITES:
        return SITES[lowered]
    if lowered.startswith(("http://", "https://")):
        return target.strip()
    if len(target.split()) == 1 and "." in target and target.count(".") == 1 and not target.startswith("."):
        return f"https://{target}"
    return None


def _app_target(target: str) -> Optional[str]:
    return APPS.get(target.lower().strip())


_BROWSER_HINTS = (
    "in a new tab", "in the browser", "in my browser", "in a browser", "in browser",
    "in the web", "in web", "on the web", "on web", "web version", "website", "online",
)


def _strip_browser_hint(target: str) -> tuple[bool, str]:
    """Return (True, cleaned) when the user asked for the browser/web version."""
    lower = target.lower()
    for phrase in _BROWSER_HINTS:
        idx = lower.find(phrase)
        if idx != -1:
            cleaned = target[:idx] + " " + target[idx + len(phrase):]
            return True, " ".join(cleaned.split()).strip()
    return False, target


@lru_cache(maxsize=1)
def _menu_shortcuts() -> tuple[tuple[str, str], ...]:
    """Index of Start Menu .lnk/.url shortcuts → (stem, full path)."""
    roots: list[str] = []
    appdata = os.environ.get("APPDATA")
    progdata = os.environ.get("PROGRAMDATA")
    if appdata:
        roots.append(os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs"))
    if progdata:
        roots.append(os.path.join(progdata, "Microsoft", "Windows", "Start Menu", "Programs"))
    seen: set[str] = set()
    entries: list[tuple[str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            walks = os.walk(root)
            for dirpath, _dirs, files in walks:
                for fn in files:
                    if fn.lower().endswith((".lnk", ".url")):
                        stem = os.path.splitext(fn)[0]
                        key = stem.lower()
                        if key not in seen:
                            seen.add(key)
                            entries.append((stem, os.path.join(dirpath, fn)))
        except OSError:
            continue
    return tuple(entries)


def _lookup_menu_entry(name: str) -> Optional[str]:
    """Find a Start Menu shortcut that plausibly matches the app name."""
    name = name.strip().lower()
    if not name:
        return None
    ntoks = set(name.split())
    for stem, path in _menu_shortcuts():
        low = stem.lower()
        if low == name or low.startswith(name):
            return path
        stoks = set(low.split())
        if ntoks and ntoks <= stoks:
            return path
    return None


def _app_ready(cmd: str, names: list[str]) -> Optional[str]:
    """Resolve a runnable app (menu shortcut / PATH / on-disk command), or None.

    Settings-style URIs like ``ms-settings:`` are always treated as available.
    """
    if not cmd:
        return None
    if cmd.endswith(":") or cmd.endswith("://"):
        return cmd
    for name in names:
        entry = _lookup_menu_entry(name)
        if entry:
            return entry
    if shutil.which(cmd):
        return shutil.which(cmd)
    if Path(cmd).exists():
        return cmd
    return None


def _path_target(target: str) -> Optional[str]:
    """Resolve a 'my downloads' / well-known folder / real path."""
    lowered = target.lower().strip()
    words = lowered.split()
    if words and words[0] in {"my", "the", "a", "an"}:
        rest = " ".join(words[1:])
        if rest in FOLDERS:
            return FOLDERS[rest]
        if rest.lower() in {"pc", "computer", "this pc"}:
            return str(HOME)
        if rest:
            candidate = HOME / rest
            if candidate.exists():
                return str(candidate)
    if lowered in FOLDERS:
        return FOLDERS[lowered]
    expanded = str(Path(target).expanduser())
    if Path(expanded).exists():
        return expanded
    return None


def skill_open(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    target = str(params.get("target") or "").strip()
    if not target:
        return ActionResult("error", "What should I open? Say something like \"open youtube\".")

    forced_browser, target = _strip_browser_hint(target)
    if not target:
        return ActionResult("error", "What should I open? Say something like \"open youtube\".")

    browser_url = _browser_target(target, ctx)
    app_cmd = _app_target(target)
    app_names = [target]
    if app_cmd:
        app_names.append(os.path.splitext(os.path.basename(app_cmd))[0])
    ready_app = _app_ready(app_cmd, app_names) if app_cmd else None

    if forced_browser:
        if browser_url:
            open_url(browser_url)
            return ActionResult("ok", f"Opened {target} in your browser.")
        if ready_app:
            start_app(ready_app)
            return ActionResult("info", f"No website for {target!r}, so I launched the app.")
        open_url(search_url(target))
        return ActionResult("info", f"Opened a search for {target!r} in your browser.")

    # App first: if it's installed, launch it instead of the website.
    if ready_app:
        start_app(ready_app)
        return ActionResult("ok", f"Launched {target}.")
    if browser_url:
        open_url(browser_url)
        return ActionResult("ok", f"Opened {target} in your browser.")
    if app_cmd:
        if start_app(app_cmd):
            return ActionResult("ok", f"Launched {target}.")
        return ActionResult("error", f"Couldn't launch {target} — is it installed? (tried {app_cmd!r}).")

    path = _path_target(target)
    if path:
        if Path(path).is_dir():
            open_folder(path)
        else:
            open_url(f"file:///{Path(path).as_posix()}")
        return ActionResult("ok", f"Opened {target}.")
    # Unknown → search instead of failing silently.
    open_url(search_url(target))
    return ActionResult(
        "info",
        f"I didn't recognize {target!r} as a site, app or folder — searched for it in your browser.",
    )


# ── search ────────────────────────────────────────────────────────────


def _format_hits(hits: list[dict[str, Any]], limit: int) -> str:
    lines: list[str] = []
    for hit in hits[:limit]:
        url = hit.get("url", "")
        title = hit.get("title") or url
        snippet = (hit.get("snippet") or "").strip()
        lines.append(f"• {title}\n  {url}")
        if snippet:
            lines.append(f"  {snippet[:180]}")
    return "\n".join(lines)


async def skill_search(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    query = str(params.get("query") or "").strip()
    if not query:
        return ActionResult("error", "Search for what? Try \"search for python docs\".")
    try:
        from app.search.engine import relai_search

        result = await relai_search(query, limit=5)
        hits = result.get("hits", []) or []
        errors = result.get("errors", [])
        if hits:
            return ActionResult(
                "ok",
                f"Top results for {query!r}:\n{_format_hits(hits, 5)}"
                + (f"\n(Notes: {'; '.join(str(e) for e in errors[:2])})" if errors else ""),
            )
        if errors:
            raise RuntimeError(errors[0])
        open_url(search_url(query))
        return ActionResult("info", f"No results found — opened a search for {query!r} in your browser.")
    except Exception as exc:  # noqa: BLE001 — degrade to browser search
        open_url(search_url(query))
        return ActionResult(
            "info", f"Search engine unavailable ({exc}). Opened {query!r} in your browser instead."
        )


# ── run / shell ───────────────────────────────────────────────────────


def _run_command(command: str, timeout: int = 60) -> tuple[bool, str, str]:
    try:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", command] if os.name == "nt" else command,
            shell=(os.name != "nt"),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
        out = (completed.stdout or "").strip()
        err = (completed.stderr or "").strip()
        ok = completed.returncode == 0
        return ok, out[:6000], err[:2000]
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s."
    except OSError as exc:
        return False, "", str(exc)


def skill_run(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    command = str(params.get("command") or "").strip()
    if not command:
        return ActionResult("error", "What command should I run? Try \"run pip version\".")
    ok, out, err = _run_command(command)
    if ok:
        return ActionResult("ok", f"Ran: {command}\n{out}" if out else f"Ran: {command} (no output)")
    return ActionResult("error", f"Command failed ({command}):\n{err or out}")


# ── calc ──────────────────────────────────────────────────────────────


def skill_calc(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    expression = str(params.get("expression") or "").strip()
    if not expression:
        return ActionResult("error", "Give me a calculation, like \"what is 15% of 200\".")
    try:
        from companion.orchestrator import safe_math_eval

        value = safe_math_eval(expression)
    except Exception as exc:  # noqa: BLE001 — surface the math error
        return ActionResult("error", f"I couldn't evaluate that: {exc}")
    return ActionResult("ok", f"{expression} = {_pretty_number(value)}")


def _pretty_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


# ── time / date ───────────────────────────────────────────────────────


def skill_time(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    now = datetime.datetime.now()
    return ActionResult(
        "ok",
        f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B %d, %Y')}.",
    )


def skill_date(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    now = datetime.datetime.now()
    return ActionResult(
        "ok",
        f"Today is {now.strftime('%A, %B %d, %Y')}. It's {now.strftime('%I:%M %p').lstrip('0')}.",
    )


# ── weather ───────────────────────────────────────────────────────────


def skill_weather(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    city = str(params.get("city") or "").strip()
    if city:
        city_part = f", {city}" if city else ""
        open_url(f"https://weather.com/weather/today{city_part.replace(' ', '+')}")
        return ActionResult("ok", f"Opened the weather for {city} in your browser.")
    open_url("https://weather.com")
    return ActionResult("ok", "Opened the weather in your browser.")


# ── files ─────────────────────────────────────────────────────────────


def skill_files(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    action = str(params.get("action") or "list").lower()
    target = str(params.get("target") or ".").strip() or "."

    if action == "create":
        return _files_create(target, ctx)
    if action == "open":
        path = _resolve_path(target)
        if path is None:
            known = _path_target(target)
            path = known or str(Path(target).expanduser())
        if Path(path).exists():
            open_folder(str(Path(path).resolve()))
            return ActionResult("ok", f"Opened folder {path}.")
        return ActionResult("error", f"Folder not found: {path}")
    if action == "find":
        name = target.rsplit("/", 1)[-1]
        matches = list(Path.cwd().rglob(name))[:10]
        if not matches:
            matches = [p for p in Path.home().rglob(name) if p.is_file()][:5]
        if matches:
            return ActionResult("ok", "Found:\n" + "\n".join(f"• {p}" for p in matches))
        return ActionResult("info", f"No files named {name!r} found.")
    return _files_list(target, ctx)


def _resolve_path(target: str) -> Optional[str]:
    expanded = str(Path(target).expanduser())
    return expanded if Path(expanded).exists() else None


def _files_list(target: str, ctx: SkillContext) -> ActionResult:
    path = _resolve_path(target)
    if target in {"here", "."} or path is None:
        if target not in {"here", "."}:
            known = _path_target(target)
            if not known:
                return ActionResult("error", f"Folder not found: {target}")
            path = known
        else:
            path = str(Path.cwd())
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return ActionResult("error", f"Couldn't read {path}: {exc}")
    dirs = [f"📁 {p.name}" for p in entries if p.is_dir()][:20]
    files = [p.name for p in entries if p.is_file()][:50]
    body: list[str] = [f"Contents of {path}:"]
    if dirs:
        body.append("Folders:\n" + "\n".join(dirs))
    if files:
        body.append("Files:\n" + "\n".join(files))
    if len(entries) > len(dirs) + len(files):
        body.append("…")
    if not entries:
        body.append("(empty)")
    return ActionResult("ok", "\n\n".join(body))


def _files_create(target: str, ctx: SkillContext) -> ActionResult:
    is_dir = False
    name = target
    if name.startswith(("folder ", "dir ", "directory ")):
        is_dir = True
        name = name.split(maxsplit=1)[1]
    elif name.startswith(("file ")):
        name = name.split(maxsplit=1)[1]
    if not name:
        return ActionResult("error", "Create what? Try \"create folder myproject\".")
    path = Path(name).expanduser()
    if is_dir or "." not in path.name:
        path.mkdir(parents=True, exist_ok=True)
        verb = "folder" if is_dir or path.is_dir() else "file"
        return ActionResult("ok", f"Created {verb} {path}.")
    path.touch()
    return ActionResult("ok", f"Created file {path}.")


# ── notes ─────────────────────────────────────────────────────────────


def skill_notes(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    action = str(params.get("action") or "remember").lower()
    text = str(params.get("text") or "").strip()

    if action == "remember":
        if not text:
            return ActionResult("error", "What should I remember? Try \"remember my sister's birthday is June 4th\".")
        note = ctx.store.add_note(text)
        return ActionResult("ok", f"Noted: {note.text}")
    if action == "query":
        q = str(params.get("query") or text or "")
        matches = ctx.store.search_notes(q, limit=5)
        if not matches:
            return ActionResult("info", f"No notes match {q!r}.")
        now = datetime.datetime.now()
        body = [f"{m.text} (saved {datetime.datetime.fromtimestamp(m.created).strftime('%b %d')})" for m in matches]
        return ActionResult("ok", "\n".join(body))
    # list
    notes = ctx.store.notes
    if not notes:
        return ActionResult("info", "You have no saved notes yet.")
    body = [f"• {n.text}" for n in notes[-20:]]
    return ActionResult("ok", "Your notes:\n" + "\n".join(body))


# ── system info ───────────────────────────────────────────────────────


def _memory_gb() -> float:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32 = ctypes.windll.kernel32
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024**3)
        except Exception:  # noqa: BLE001
            pass
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024**3)
    except ImportError:
        return 0.0


def skill_system(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    gb = _memory_gb()
    mem = f"{gb:.1f} GB" if gb else "unknown"
    return ActionResult(
        "ok",
        "\n".join(
            [
                f"OS: {platform.platform()}",
                f"Python: {sys.version.split()[0]}",
                f"CPU cores: {os.cpu_count()}",
                f"RAM: {mem}",
                f"Host: {platform.node()}",
                f"Architecture: {platform.machine()}",
            ]
        ),
    )


# ── power ─────────────────────────────────────────────────────────────


def power_commands() -> dict[str, str]:
    if os.name == "nt":
        return {
            "shutdown": "shutdown /s /t 0",
            "restart": "shutdown /r /t 0",
            "lock": "rundll32.exe user32.dll,LockWorkStation",
            "sleep": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
        }
    return {
        "shutdown": "systemctl poweroff",
        "restart": "systemctl reboot",
        "lock": "loginctl lock-session",
        "sleep": "systemctl suspend",
    }


def skill_power(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    action = str(params.get("action") or "").lower()
    commands = power_commands()
    if action not in commands:
        return ActionResult("error", "Try \"shutdown\", \"restart\", \"lock\" or \"sleep\".")
    ok, out, err = _run_command(commands[action], timeout=10)
    if ok:
        return ActionResult("ok", f"{action.title()} command sent.")
    return ActionResult("error", f"Couldn't {action}: {err or out}")


# ── help ──────────────────────────────────────────────────────────────


def skill_help(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    text = (
        "I'm your multipurpose controller.\n\n"
        "Just say what you want in plain English, like:\n"
        "  • \"open youtube\" / \"open my downloads\" / \"open calculator\"\n"
        "     (installed apps open natively; add \"in browser\" for the website)\n"
        "  • \"search for python docs\" / \"who is Newton\"\n"
        "  • \"what is 15% of 200\" / \"what time is it\"\n"
        "  • \"remember that my wifi password is ...\"\n"
        "  • \"list files\" / \"create folder projects\"\n"
        "  • \"run pip version\" / \"system info\"\n"
        "  • \"weather in london\" / \"shutdown the computer\"\n"
        "\nPolite phrasing works too — \"please open youtube\", \"can you open gmail?\"\n"
        "Additional commands: help, alias <name> <target>, notes, exit\n"
        "Type \"alias set mycode C:\\code\" to teach me your own shortcuts.\n"
        "\nDiscussion & data:\n"
        "  • just type a question or topic → we discuss it (runs 100% locally)\n"
        "  • \"data list\" / \"data load <name>\" / \"data inspect <name>\"\n"
        "  • \"dataset ask <name> <question>\" · \"dataset solve <name>\"\n"
        "  • \"dataset protein 7tim\" · \"dataset esm <sequence>\" — \"clear\" resets our chat"
    )
    return ActionResult("ok", text)


# ── alias management ──────────────────────────────────────────────────


def skill_alias(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    action = str(params.get("action") or "list").lower()
    name = str(params.get("name") or "").strip()
    target = str(params.get("target") or "").strip()

    if action == "set":
        if not name or not target:
            return ActionResult("error", "Usage: alias set <name> <target>")
        ctx.store.set_alias(name, target)
        return ActionResult("ok", f"Saved alias {name!r} → {target}")
    if action == "remove":
        removed = ctx.store.remove_alias(name)
        return ActionResult("ok", f"Removed alias {name!r}." if removed else f"No alias {name!r}.")
    aliases = ctx.store.aliases
    if not aliases:
        return ActionResult("info", "No custom aliases. Add one with \"alias set mycode C:\\code\".")
    body = [f"• {k} → {v}" for k, v in aliases.items()]
    return ActionResult("ok", "Your aliases:\n" + "\n".join(body))


# ── discussion (local, no external APIs) ─────────────────────────────


_DISCUSS_SYSTEM = (
    "You are Sweep, a local-first terminal assistant. Answer conversationally and "
    "concisely (2-6 sentences). You run fully locally. You can suggest things like "
    "\"search for <topic>\", \"open <site>\", \"data list\", \"dataset protein <PDB id>\", "
    "or \"dataset ask <name> <question>\". Never claim to browse or use a search "
    "engine yourself unless asked."
)


def _canned_social(text: str) -> Optional[str]:
    t = text.lower()
    words = t.split()
    if len(words) > 5:
        return None
    if any(w in t for w in ("how are you", "what's up", "how's it going", "how are things")):
        return "Running smoothly! I can hold a discussion, work the datasets, or run anything from \"open youtube\" to \"dataset protein 7tim\"."
    if any(w in t for w in ("hello", "hi", "hey", "yo", "good morning", "good evening", "good afternoon", "sup")):
        return "Hey! I'm ready — chat with me about anything, or try \"help\" / \"data list\" for what I can do."
    if any(w in t for w in ("thank", "thanks", "thx")):
        return "Anytime! What's next?"
    return None


def skill_discuss(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    """Multi-turn discussion backed by the LOCAL model (never an external API)."""
    text = str(params.get("text") or "").strip()
    action = str(params.get("action") or "").lower()
    if action == "clear" or text.lower() in {"clear", "clear conversation", "reset chat", "new chat"}:
        removed = ctx.store.clear_conversation()
        return ActionResult("ok", f"Conversation cleared ({removed} turns removed).")

    canned = _canned_social(text)
    if canned:
        return ActionResult("ok", canned)

    if not text:
        return ActionResult("info", "Say something, or \"help\" for commands.")

    from . import models  # local model layer

    history = ctx.store.read_conversation(limit=16)
    turns = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        turns.append(f"{role}: {turn.get('text', '')[:400]}")
    turns.append(f"User: {text}")
    convo = "\n".join(turns)
    prompt = f"{convo}\nAssistant:"
    reply = models.generate(prompt, max_tokens=300, temperature=0.7, system=_DISCUSS_SYSTEM)
    if not reply:
        return ActionResult("error", "Local model produced no reply — try again, or \"help\".")
    ctx.store.append_conversation("user", text)
    ctx.store.append_conversation("assistant", reply)
    return ActionResult("ok", reply)


skill_chat = skill_discuss  # backwards-compatible name for the canned reply skill


# ── dataset exploration (reasoning / molecular / corpus) ──────────────


def _fmt_datasets(entries: list[dict[str, Any]]) -> str:
    lines = ["Available datasets (group ▸ name ▸ status):"]
    for e in entries:
        mark = "✓ cached" if e.get("cached") else "— not cached"
        lines.append(f"  [{e['group']:<10}] {e['name']:<16} {mark}  {e['description']}")
    lines.append("\nTry: 'data load <name>', 'data inspect <name>', 'data ask <name> <question>', 'dataset protein 7tim', 'dataset solve gsm8k'.")
    return "\n".join(lines)


def _fmt_rows(rows: list[dict[str, Any]], limit: int = 5) -> str:
    lines = []
    for row in rows[:limit]:
        prompt = str(row.get("prompt") or row.get("question") or row.get("text") or "").strip().replace("\n", " | ")[:160]
        answer = str(row.get("answer") or "").strip().replace("\n", " ")[:100]
        lines.append(f"  · Q: {prompt}\n    A: {answer}")
    return "\n".join(lines)


def skill_dataset(params: dict[str, Any], ctx: SkillContext) -> ActionResult:
    from . import datasets

    action = str(params.get("action") or "list").lower()
    name = str(params.get("name") or "").strip()
    query = str(params.get("query") or "").strip()
    limit = int(params.get("limit") or 0) or None
    target = str(params.get("target") or "").strip()

    if action in {"list", "help", ""}:
        return ActionResult("ok", _fmt_datasets(datasets.list_datasets()))

    if action == "load" or action == "download":
        if not name:
            return ActionResult("error", "Which dataset? e.g. 'data load gsm8k'.")
        try:
            data = datasets.load_dataset(name, split=params.get("split") or None, limit=limit)
        except Exception as exc:  # noqa: BLE001
            return ActionResult("error", str(exc))
        if data["rows"]:
            from .datasets.registry import cache_rows

            cache_rows(name, data["rows"])
        return ActionResult(
            "ok",
            f"Loaded {name} ({data['row_count']} rows, {data['columns'] and len(data['columns'])} cols) in {data['seconds']}s.\n"
            f"columns: {', '.join(data['columns'][:12])}\n{_fmt_rows(data['rows'], 3)}",
        )

    if action in {"info", "status", "inspect"}:
        if not name:
            return ActionResult("error", "Which dataset? e.g. 'data info logiqa'.")
        try:
            data = datasets.load_dataset(name, limit=limit or 5)
        except Exception as exc:  # noqa: BLE001
            return ActionResult("error", str(exc))
        head = f"{name} — {data['description']}\nsource: {data['source']} · rows: {data['row_count']} · cached: {data['cached']}"
        if data["notes"]:
            head += f"\nnotes: {data['notes']}"
        return ActionResult("ok", f"{head}\ncolumns: {', '.join(data['columns'][:12])}\n{_fmt_rows(data['rows'], limit or 5)}")

    if action in {"ask", "query"}:
        if not name:
            return ActionResult("error", "e.g. 'dataset ask gsm8k what is 15 percent of 80'")
        return _run_ask(name, query)

    if action in {"solve", "eval", "evaluate"}:
        if not name:
            return ActionResult("error", "e.g. 'dataset solve gsm8k' (local model evaluated on sampled rows.)")
        return _run_solve(name, limit)

    if action == "protein" or action == "structure":
        if not target:
            return ActionResult("error", "Give a PDB id or file path: 'dataset protein 7tim'.")
        return _run_protein(target)

    if action == "esm":
        if not target:
            return ActionResult("error", "Give a protein sequence: 'dataset esm MVLSPADKTNVKAAW...'")
        return _run_esm(target)

    if action == "remove":
        if not name:
            return ActionResult("error", "Which dataset to remove from cache? 'data remove gsm8k'.")
        return _run_remove(name)

    return ActionResult("error", f"Unknown dataset action {action!r}. Try 'data list' for the menu.")


def _run_ask(name: str, query: str) -> ActionResult:
    if not query:
        return ActionResult("error", f"Ask what? e.g. 'dataset ask {name} <your question>'.")
    from .datasets import qa

    try:
        result = qa.ask_dataset(name, query)
    except Exception as exc:  # noqa: BLE001
        return ActionResult("error", str(exc))
    return ActionResult("ok", f"[{result['dataset']}] Q: {result['question']}\n→ {result['answer']}")


def _run_solve(name: str, limit: Optional[int]) -> ActionResult:
    from . import models
    from .datasets import qa

    backends = ", ".join(models.available_backends())
    try:
        result = qa.evaluate_sample(name, n=min(limit or 4, 8))
    except Exception as exc:  # noqa: BLE001
        return ActionResult("error", str(exc))
    head = f"{result['dataset']} — {result['correct']}/{result['n']} correct ({round(result['accuracy'] * 100, 1)}%). Backends: {backends}"
    body = "\n".join(
        f"  {r['i']}. {'✓' if r['ok'] else '✗'} Q: {r['question'][:60]} | expected: {r['expected'][:40]!r} | got: {r['got'][:60]!r} ({r['secs']}s)"
        for r in result["results"]
    )
    return ActionResult("ok", f"{head}\n{body}")


def _run_protein(target: str) -> ActionResult:
    from .datasets import molecular

    try:
        stats = molecular.protein_stats(target)
    except Exception as exc:  # noqa: BLE001
        return ActionResult("error", str(exc))
    comp = " ".join(f"{k}: {v}" for k, v in (stats.get("composition") or {}).items())
    lines = [
        f"{target}: {stats.get('description', '')}",
        f"  chains: {', '.join(stats.get('chains') or [])} · residues: {stats.get('residue_count', stats.get('records', 0))}",
        f"  sequence length: {stats.get('sequence_length')} · Mw ≈ {stats.get('molecular_weight_approx')} Da",
    ]
    if stats.get("sequence"):
        lines.append(f"  sequence: {stats['sequence']}")
    if comp:
        lines.append(f"  composition: {comp}")
    return ActionResult("ok", "\n".join(lines))


def _run_esm(sequence: str) -> ActionResult:
    from .datasets import molecular

    try:
        result = molecular.esm_embed(sequence)
    except Exception as exc:  # noqa: BLE001
        return ActionResult("error", str(exc))
    return ActionResult(
        "ok",
        f"ESM embedding: model {result['model']} · dim {result['embedding_dim']} · tokens {result['tokens']} · norm {result['pooled_norm']:.3f}",
    )


def _run_remove(name: str) -> ActionResult:
    from .datasets import registry

    target = registry._cached_path(name)
    if not target.exists():
        return ActionResult("info", f"{name} has no cache to remove.")
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    return ActionResult("ok", f"Removed cached {name}.")


__all__ = [
    "APPS",
    "ActionResult",
    "FOLDERS",
    "HOME",
    "SITES",
    "SkillContext",
    "open_folder",
    "open_url",
    "search_url",
    "skill_alias",
    "skill_calc",
    "skill_chat",
    "skill_dataset",
    "skill_date",
    "skill_discuss",
    "skill_files",
    "skill_help",
    "skill_notes",
    "skill_open",
    "skill_power",
    "skill_run",
    "skill_search",
    "skill_system",
    "skill_time",
    "skill_weather",
    "start_app",
]
