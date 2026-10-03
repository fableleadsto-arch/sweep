"""Natural-language parser for the terminal controller.

The priority is *predictability*: politeness phrases ("please open youtube",
"can you open gmail?", "hey sweep, wait — open youtube") are stripped, then the
text is matched against deterministic intent rules. Anything the rules can't
classify goes to a compact LLM dispatch call (when a provider is configured)
and only falls back to a web search otherwise.

The parser never executes anything — it only produces a ``ParsedCommand``
(intent + params). The controller maps intents to skills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .skills import APPS, SITES

# ── politeness handling ───────────────────────────────────────────────


_LEADING_DROP = {
    "please", "pls", "plz", "hey", "hi", "hello", "yo", "heyya", "hiya",
    "just", "ok", "okay", "k", "could", "would", "can", "will", "do", "did",
    "may", "might", "shall", "so", "umm", "um", "ah", "oh", "aye", "sweep",
}

_TRAILING_DROP = {"please", "pls", "plz", "thanks", "thank", "thx", "ty", "bro", "dude", "mate", "man"}


def clean_text(text: str) -> str:
    """Collapse whitespace only; keeps case and punctuation (payload-preserving)."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    """Lowercased, punctuation-collapsed copy used ONLY for intent matching.

    Apostrophes are dropped ("what's" → "whats"); sentence punctuation becomes
    spaces. Colons/slashes/backslashes/dots/operators inside payloads are
    preserved because matching never needs them.
    """
    t = re.sub(r"'", "", text)
    t = re.sub(r"[.,!?;:\"`()\[\]{}<>|#@&^=]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def _tok(word: str) -> str:
    """Trimmed, lowercased token for politeness comparisons only."""
    return word.strip(".,!?;:'\"`()[]{}<>| ").lower()


def strip_politeness(text: str) -> str:
    """Remove polite lead-ins/trailers, returning the CORE phrase unchanged.

    Case and punctuation survive (so ``C:\\projects`` reaches skills intact);
    the caller compares with ``normalize_for_match`` when it needs a lowercased
    form.
    """
    words = clean_text(text).split()
    if not words:
        return ""

    # "can you please …" / "could you …" / "i want you to …"
    changed = True
    while changed and words:
        changed = False
        first, second = _tok(words[0]), (_tok(words[1]) if len(words) > 1 else "")
        third = _tok(words[2]) if len(words) > 2 else ""
        # pronoun pair must drop together ("can you" → "") before a lone "can"/"could"
        if first in {"can", "could", "would", "will", "do", "did", "may", "might", "shall"} and second in {"you", "u", "ya"}:
            del words[:2]
            changed = True
            continue
        if first in _LEADING_DROP:
            words.pop(0)
            changed = True
            continue
        if first == "i" and second == "want" and third in {"you", "u"} and len(words) > 3 and _tok(words[3]) == "to":
            del words[:4]
            changed = True
            continue
        if first == "i" and second == "want" and third == "to":
            del words[:3]
            changed = True
            continue
        if first == "i" and second in {"need", "would", "like"} and third == "to":
            del words[:3]
            changed = True
            continue
        if first == "i" and second == "need":
            del words[:2]
            changed = True
            continue

    while words and _tok(words[-1]) in _TRAILING_DROP:
        words.pop()
    if len(words) >= 2 and _tok(words[-2]) == "for" and _tok(words[-1]) == "me":
        words = words[:-2]
    if len(words) >= 2 and _tok(words[-1]) == "me" and _tok(words[-2]) in {"help", "tell", "show"}:
        words = words[:-2]

    joined = " ".join(words).rstrip(".,!?;:")
    return joined


# ── intent definitions ────────────────────────────────────────────────


@dataclass
class ParsedCommand:
    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    display: str = ""
    source: str = "rule"  # "rule" | "llm"


_SEARCH_LEADERS = {"what", "how", "why", "who", "where", "when", "which", "define", "whats", "whos", "hows", "whens", "wheres"}

_EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "stop", "leave", "logout"}
_HELP_WORDS = {"help", "commands", "help me"}
_GREETING_WORDS = {"hi", "hello", "hey", "yo", "goodmorning", "goodafternoon", "goodevening", "howareyou", "hiya", "sup"}
_THANKS_WORDS = {"thanks", "thank", "thx"}

_GREETING_PHRASES = {
    "hi", "hello", "hey", "yo", "hiya", "sup", "hey there", "hi there", "hello there",
    "how are you", "how are you doing", "how do you do", "how is it going", "hows it going",
    "good morning", "good afternoon", "good evening", "whats up", "what is up", "wassup",
    "hiya", "heyya",
}


def scanned_greeting(text: str) -> bool:
    """True when the input is purely a greeting/exchange, not a command."""
    return normalize_for_match(text) in _GREETING_PHRASES

_MATH_WORDS = {
    "plus": "+", "add": "+", "minus": "-", "subtract": "-", "times": "*",
    "multiplied": "*", "multiply": "*", "divided": "/", "over": "/", "by": "/",
    "to the power": "**", "power": "**", "modulo": "%", "mod": "%",
}


def _word_math(text: str) -> str:
    """Convert a words/mixed expression to a safe symbolic form."""
    lower = text.lower()
    if "percent of" in lower or "% of" in lower:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)", lower)
        if m:
            return f"{m.group(2)}*({m.group(1)}/100)"
    if re.fullmatch(r"half", lower):
        return "0.5"
    if "percent" in lower or "%" in lower:
        lower = re.sub(r"\bpercent of\b", "*", lower)
        lower = re.sub(r"%", "/100", lower)
    for word, symbol in _MATH_WORDS.items():
        lower = re.sub(rf"\b{re.escape(word)}\b", symbol, lower)
    return lower


def _looks_like_math(text: str) -> Optional[str]:
    """Return a safe expression if the text contains a recognizable calculation."""
    expr = _word_math(text)
    if re.search(r"\d", expr) and re.search(r"[\d].*[+\-*/%^()].*[\d]|[+\-*/%^=].*[\d]|[\d].*[+\-*/%^=]", expr):
        if not _has_injection(expr):
            return expr
    return None


def _has_injection(text: str) -> bool:
    return any(token in text.lower() for token in ("import", "open(", "system(", "exec", "eval", "__", ";", "&&", "|", ">", "<"))


def _find_calc(text: str) -> Optional[str]:
    """Extract a calculation from phrases like 'what is 15% of 200'."""
    m = re.match(r"^(?:what'?s|what is|whats|calculate|compute|solve|work out|how much is|how much)\s+(.+)", text)
    candidate = m.group(1) if m else text
    return _looks_like_math(candidate)


def parse_math_expression(text: str) -> Optional[str]:
    return _find_calc(strip_politeness(text))


# ── rule matcher ──────────────────────────────────────────────────────


def _bare_open_target(text: str) -> Optional[str]:
    """Single-token/known-name opens: 'youtube', 'calculator', 'home'."""
    lowered = text.lower().strip()
    if lowered in SITES or lowered in APPS or lowered in {"downloads", "download", "desktop", "documents", "home"}:
        return lowered
    return None


def _dataset_command(words: list[str], first: str, whole: str, t: str) -> Optional[ParsedCommand]:
    """Parse 'data/dataset <action> ...' into a dataset ParsedCommand."""
    is_data_prefix = first in {"data", "dataset"}
    whole_trigger = (
        whole.startswith(("list datasets", "load dataset ", "download dataset "))
        or whole in {"datasets", "which datasets do you have", "what datasets do you have", "list dataset"}
    )
    if not (is_data_prefix or whole_trigger):
        return None

    action, rest = "list", ["list"]
    if is_data_prefix and len(words) > 1:
        action = words[1].lower()
        rest = words[2:]
    elif not is_data_prefix:
        if whole.startswith(("load dataset ", "download dataset ")):
            action, rest = "load", words[2:]
        else:
            action, rest = "list", ["list"]

    if action in {"list", "help", "menu", "show", "datasets"}:
        return ParsedCommand("dataset", {"action": "list"}, t, "rule")
    if action in {"load", "download", "fetch"}:
        return ParsedCommand("dataset", {"action": "load", "name": " ".join(rest).strip()}, t, "rule")
    if action in {"info", "inspect", "status", "what", "about"}:
        return ParsedCommand("dataset", {"action": "info", "name": " ".join(rest).strip()}, t, "rule")
    if action in {"ask", "query", "question"}:
        name = rest[0] if rest else ""
        query = " ".join(rest[1:]).strip()
        return ParsedCommand("dataset", {"action": "ask", "name": name, "query": query}, t, "rule")
    if action in {"solve", "eval", "evaluate", "benchmark"}:
        return ParsedCommand("dataset", {"action": "solve", "name": " ".join(rest).strip()}, t, "rule")
    if action in {"protein", "structure", "pdb"}:
        return ParsedCommand("dataset", {"action": "protein", "target": " ".join(rest).strip()}, t, "rule")
    if action in {"esm", "embed", "embedding"}:
        return ParsedCommand("dataset", {"action": "esm", "target": " ".join(rest).strip()}, t, "rule")
    if action in {"remove", "delete"}:
        return ParsedCommand("dataset", {"action": "remove", "name": " ".join(rest).strip()}, t, "rule")
    return ParsedCommand("dataset", {"action": "list"}, t, "rule")


def parse_rules(text: str) -> Optional[ParsedCommand]:
    original = text.strip()
    cleaned = normalize_for_match(original)
    first_word = cleaned.split()[0] if cleaned else ""

    # Pure sayings must be caught BEFORE politeness stripping eats them.
    if first_word and first_word in _EXIT_WORDS and len(cleaned.split()) <= 2:
        return ParsedCommand("exit", {}, original.strip(), "rule")
    if cleaned in {"help", "help me", "what can you do", "what do you do", "commands", "list commands"} or (
        first_word == "help" and len(cleaned.split()) <= 2
    ):
        return ParsedCommand("help", {}, original.strip(), "rule")
    if scanned_greeting(original):
        return ParsedCommand("chat", {"text": original}, original.strip(), "rule")
    if first_word == "thanks" or cleaned in {"thank you", "thx", "ty", "thank"}:
        return ParsedCommand("chat", {"text": original}, original.strip(), "rule")

    t = strip_politeness(original)
    if not t:
        return None
    words = t.split()
    if not words:
        return None
    whole = normalize_for_match(t)
    first = words[0].lower()

    # time / date
    if first == "time" or whole.startswith("what time") or whole in {
        "current time", "the time", "whats the time", "what is the time", "time now", "whats the current time",
    } or (whole.startswith("time ") and "what" in whole):
        return ParsedCommand("time", {}, t, "rule")
    if first == "date" or first == "today" or whole.startswith("what date") or "todays date" in whole or whole in {
        "current date", "whats the date", "what is the date", "date today",
    }:
        return ParsedCommand("date", {}, t, "rule")

    # datasets: "data ..." / "dataset ..."
    dcmd = _dataset_command(words, first, whole, t)
    if dcmd:
        return dcmd
    if first == "protein" and len(words) > 1:
        return ParsedCommand("dataset", {"action": "protein", "target": " ".join(words[1:])}, t, "rule")
    if first == "esm" and len(words) > 1:
        return ParsedCommand("dataset", {"action": "esm", "target": " ".join(words[1:])}, t, "rule")

    # explicit discussion topics
    if first in {"discuss", "talk", "chat"} and len(words) > 1:
        return ParsedCommand("discuss", {"text": " ".join(words[1:])}, t, "rule")
    if first == "clear" or whole in {"clear conversation", "reset chat", "new chat", "clear chat"}:
        return ParsedCommand("discuss", {"action": "clear"}, t, "rule")

    # notes: remember / remind / note
    if first in {"remember", "remind", "note", "redit"} or (
        first == "remind" and len(words) >= 2 and words[1] in {"me", "myself"}
    ):
        if first == "remind" and len(words) >= 2 and words[1].lower() in {"me", "myself"}:
            body = " ".join(words[2:])
        else:
            body = " ".join(words[1:])
        body = re.sub(r"^that\s+", "", body).strip()
        return ParsedCommand("notes", {"action": "remember", "text": body or original}, t, "rule")
    if whole in {"show notes", "my notes", "list notes", "notes", "note list", "all my notes"}:
        return ParsedCommand("notes", {"action": "list"}, t, "rule")
    if first == "shownotes" or (first == "show" and len(words) > 1 and words[1] in {"notes", "note"}):
        return ParsedCommand("notes", {"action": "list"}, t, "rule")
    if re.search(r"^what did i (note|write)\s+about\s+", whole):
        query = re.sub(r"^what did i (note|write)\s+about\s+", "", whole).strip()
        return ParsedCommand("notes", {"action": "query", "query": query}, t, "rule")
    if re.search(r"^what did i (note|write)", whole):
        query = " ".join(words[4:]).strip()
        return ParsedCommand("notes", {"action": "query", "query": query}, t, "rule")
    if re.search(r"^find my note\s*$", whole):
        return ParsedCommand("notes", {"action": "list"}, t, "rule")
    if whole.startswith("note about") or whole.startswith("notes about") or whole.startswith("what do i know about"):
        query = re.sub(r"^(note|notes|what do i know)\s+about\s+", "", whole)
        return ParsedCommand("notes", {"action": "query", "query": query.strip()}, t, "rule")
    if first in {"noted", "notedown"} or (first == "note" and whole.startswith("note down")):
        body = " ".join(words[1:])
        return ParsedCommand("notes", {"action": "remember", "text": body or original}, t, "rule")

    # files: list / open folder / create / find file
    if first in {"ls", "dir", "list"} or whole.startswith("list files") or whole.startswith("list folder") or whole.startswith("list directory") or whole.startswith("show files"):
        target = " ".join(words[1:]) if len(words) > 1 else "here"
        return ParsedCommand("files", {"action": "list", "target": target}, t, "rule")
    if first in {"create", "make", "new", "mkdir"} and len(words) > 1 and words[1] in {"folder", "dir", "directory", "file", "directory"}:
        action = "create"
        target = " ".join(words[2:])
        if first in {"make", "mkdir"} and words[1] not in {"file"}:
            target = " ".join(words[1:])
        return ParsedCommand("files", {"action": action, "target": target or "newfolder"}, t, "rule")
    if first in {"create", "make", "new", "mkdir"} and len(words) > 1 and words[1] in {"a", "an", "the"}:
        if len(words) > 2 and words[2] in {"folder", "directory", "dir", "file"}:
            return ParsedCommand("files", {"action": "create", "target": " ".join(words[3:])}, t, "rule")
    if whole.startswith("open folder") or whole.startswith("open directory") or whole.startswith("open the folder") or whole.startswith("open the directory"):
        target = re.sub(r"^open (the |a |an )?(folder|directory|dir)\s+", "", whole)
        return ParsedCommand("files", {"action": "open", "target": target.strip()}, t, "rule")
    if whole.startswith("find file") or whole.startswith("find files") or whole.startswith("search file") or whole.startswith("search for file"):
        target = re.sub(r"^(find|search)\s+(for\s+)?(files?|dirs?)\s+", "", whole)
        return ParsedCommand("files", {"action": "find", "target": target.strip()}, t, "rule")

    # power
    if first in {"shutdown", "restart", "reboot", "lock", "sleep"} or (first == "power" and len(words) > 1 and words[1] in {"off", "down"}):
        action = {"poweroff": "shutdown", "powerdown": "shutdown", "power": "shutdown"}.get(first, first)
        return ParsedCommand("power", {"action": action}, t, "rule")

    # weather
    if first == "weather" or whole.startswith("forecast") or whole.startswith("weather forecast"):
        city = " ".join(words[1:]) if first == "weather" else " ".join(words[1:])
        return ParsedCommand("weather", {"city": city}, t, "rule")
    if whole.startswith("whats the weather") or whole.startswith("what is the weather"):
        city = re.sub(r"^(whats|what is) the weather (like )?(in|for)?\s*", "", whole).strip()
        return ParsedCommand("weather", {"city": city}, t, "rule")

    # system
    if any(k in whole for k in ("system info", "system details", "system specs", "computer specs", "machine info", "my specs", "my system", "how much ram", "how many cores", "hardware info", "specs of my")):
        return ParsedCommand("system", {}, t, "rule")
    if first in {"whoami"}:
        return ParsedCommand("system", {}, t, "rule")

    # run / shell
    if first in {"run", "execute", "shell"} or whole.startswith("run command") or whole.startswith("run the command"):
        command = " ".join(words[1:])
        if command.startswith("command "):
            command = command[len("command "):].strip()
        command = command.strip("`")
        return ParsedCommand("run", {"command": command}, t, "rule")
    if whole.startswith("install "):
        package = whole[len("install "):].strip()
        command = f"pip install {package}" if package else "pip install"
        return ParsedCommand("run", {"command": command}, t, "rule")

    # calculator (after run so "run 2+2" wins as a command, before search)
    expr = _find_calc(t)
    if expr and _looks_like_math(expr):
        # "what is 2+2" → expr is "2+2"; whole math strings pass directly
        return ParsedCommand("calc", {"expression": expr}, t, "rule")
    if _looks_like_math(whole):
        return ParsedCommand("calc", {"expression": _looks_like_math(whole)}, t, "rule")

    # alias management
    if first == "alias":
        if len(words) >= 2 and words[1] == "set" and len(words) >= 4:
            return ParsedCommand("alias", {"action": "set", "name": words[2], "target": " ".join(words[3:])}, t, "rule")
        if len(words) >= 3 and words[1] == "remove":
            return ParsedCommand("alias", {"action": "remove", "name": words[2]}, t, "rule")
        if len(words) >= 2 and words[1] in {"list", "show", "all"}:
            return ParsedCommand("alias", {"action": "list"}, t, "rule")
        if len(words) == 1:
            return ParsedCommand("alias", {"action": "list"}, t, "rule")
        if len(words) >= 2:
            return ParsedCommand("alias", {"action": "set", "name": words[1], "target": " ".join(words[2:])}, t, "rule")

    # open / launch / go to
    if first in {"open", "launch", "start", "show"} or (len(words) > 1 and words[0] in {"go", "take", "navigate", "visit", "goto"} and words[1] in {"to", "me", "at"}):
        if first == "show" and len(words) > 1 and words[1] in {"me", "files"}:
            if words[1] == "files":
                return ParsedCommand("files", {"action": "list", "target": "here"}, t, "rule")
            body = " ".join(words[2:])
            if body:
                return ParsedCommand("open", {"target": body}, t, "rule")
        body = " ".join(words[1:])
        body = re.sub(r"^(the|a|an|my|that)\s+", "", body)
        if body:
            return ParsedCommand("open", {"target": body}, t, "rule")

    # search (question-style leads)
    if first in _SEARCH_LEADERS or first in {"search", "google", "look", "find", "tell", "describe", "explain", "give", "define"}:
        return ParsedCommand("search", {"query": t}, t, "rule")
    if whole.startswith("tell me about") or whole.startswith("what is the best") or whole.startswith("what are the"):
        return ParsedCommand("search", {"query": t}, t, "rule")

    # bare known target: "youtube" → open
    if first in {"sweep", "sweep,"}:
        remaining = " ".join(words[1:])
        if remaining:
            return parse_rules(remaining)
    bare = _bare_open_target(t)
    if bare:
        return ParsedCommand("open", {"target": bare}, t, "rule")

    return None


# ── LLM fallback ──────────────────────────────────────────────────────


_LLM_SYSTEM = """You are the Sweep terminal controller dispatcher. Map the user's English request to exactly one action. Reply with STRICT JSON only:
{"intent": "<intent>", "params": {...}}

Allowed intents and their params:
- open: {"target": "site alias, app name, url, or path"}
- search: {"query": "text to search for"}
- run: {"command": "shell command"}
- calc: {"expression": "safe math expression"}
- time: {} or {"kind": "date"}
- files: {"action": "list|open|create|find", "target": "..."}
- notes: {"action": "remember|list|query", "text": "...", "query": "..."}
- weather: {"city": "city or empty"}
- system: {}
- power: {"action": "shutdown|restart|lock|sleep"}
- chat: {"text": "..."} for greetings, thanks, or small talk
- alias: {"action": "set|list|remove", "name": "...", "target": "..."}
- help: {}

Rules:
- If the request is about opening/launching/going to something, choose "open".
- If it has numbers and arithmetic/percent words, choose "calc" with the expression.
- Never invent values the user didn't provide. Use only the intents above."""


async def llm_parse(text: str, settings: Optional[Any] = None) -> Optional[ParsedCommand]:
    """Ask the LLM to classify an ambiguous request. Never raises."""
    try:
        from companion.config import BrainSettings, get_settings
        from companion.providers import ProviderChain

        effective = settings or get_settings()
        chain = ProviderChain(effective)
        if not any(p.available for p in chain.providers.values()):
            return None
        result = await chain.generate(
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=300,
            json_mode=True,
            preferred="gemini",
        )
        parsed = result.parsed or {}
        intent = str(parsed.get("intent") or "").strip()
        params = parsed.get("params") or {}
        if not intent or intent == "unknown":
            return None
        if not isinstance(params, dict):
            params = {}
        return ParsedCommand(intent=intent, params=params, display=text, source="llm")
    except Exception:  # noqa: BLE001 — LLM fallback is best-effort
        return None


# ── top-level parse ───────────────────────────────────────────────────


def clean_command(text: str) -> str:
    """Strip a leading \"sweep\" so BOTH \"youtube\" and \"sweep youtube\" work."""
    stripped = text.strip()
    if stripped.lower().startswith("sweep "):
        return stripped[len("sweep "):].strip()
    if stripped.lower() == "sweep":
        return ""
    return stripped


def parse(text: str) -> Optional[ParsedCommand]:
    """Deterministic parse only (never calls the network)."""
    return parse_rules(clean_command(text))


def looks_like_question(text: str) -> bool:
    """True when the request reads like an informational query."""
    t = normalize_for_match(text)
    if not t:
        return False
    first = t.split()[0].lower()
    if first in _SEARCH_LEADERS:
        return True
    return any(
        phrase in t
        for phrase in ("tell me about", "show me results for", "find out about", "about the best", "what are the")
    )