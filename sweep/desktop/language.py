"""Conservative desktop intent evidence; chat is the default, not a command line."""
from __future__ import annotations

import re


def request_core(text: str) -> str:
    """Remove conversational wrappers without rewriting URLs, paths or note bodies."""
    value = text.strip()
    prefix = re.compile(
        r"^(?:(?:hey|hi|hello|yo|okay|ok|sweep|please|pls|plz)[,!]?\s+"
        r"|(?:can|could|would|will)\s+(?:you|u|ya)\s+"
        r"|(?:couldya|wouldya)\s+|(?:i\s+(?:want|need)\s+(?:you|u)\s+to)\s+"
        r"|i\s+(?:want\s+to|would\s+like\s+to|wanna)\s+)", re.I,
    )
    for _ in range(12):
        updated = prefix.sub("", value, count=1)
        if updated == value:
            break
        value = updated
    # A URL's trailing punctuation may be part of its query or path.
    value = re.sub(r"\s+(?:please|pls|plz|thanks|thx|ty|bro|dude|mate)[.!?]*$", "", value, flags=re.I)
    if not re.search(r"https?://\S+$", value, re.I):
        value = value.rstrip(".!?,")
    return value.strip()


def command_core(text: str) -> str:
    value = request_core(text)
    # Expand only an explicit destination, never the user's general prose.
    return re.sub(r"^((?:open|launch|start|visit|go to|bring up)\s+)yt(?=\s+(?:in|on|using|with)\b|$)",
                  r"\1youtube", value, flags=re.I)


def explicit_command(core: str, command) -> bool:
    """A parser match alone is not evidence that someone asked for an action."""
    if command is None:
        return False
    value = core.casefold()
    intent = command.intent
    if intent == "open":
        if re.match(r"open[ -]source\b|open (?:your|my|our|their) mind\b|go to (?:bed|sleep|hell)\b", value):
            return False
        # Explanation requests and idioms are common false positives in the CLI.
        if re.match(r"(?:show(?: me)?|open|start|launch|bring up|take me to)\s+"
                    r"(?:(?:an?|the)\s+)?(?:how|why|what|examples?|ideas?|explanations?|"
                    r"conversation|discussion|story|joke|debate|mind|up\b|thinking|over\b)", value):
            return False
        if re.match(r"show(?: me)?\s+", value):
            from sweep.skills import APPS, SITES
            target = str(command.params.get("target", "")).casefold()
            return target in APPS or target in SITES or target in {"downloads", "documents", "desktop"}
        if re.match(r"start\s+", value):
            from sweep.skills import APPS, SITES
            target = str(command.params.get("target", "")).casefold()
            return bool(target in APPS or target in SITES or command.params.get("browser")
                        or command.params.get("desktop") or re.match(r"https?://", target))
        return bool(re.match(r"(?:open|launch|start|bring up|go to|take me to|navigate to|visit|goto|run)\s+", value))
    if intent == "calc":
        # The terminal parser also recognizes numbers embedded in prose. Require
        # a whole arithmetic request here, so "I worked 9-5" remains a remark.
        candidate = re.sub(r"^(?:what'?s|what is|calculate|compute|solve|work out|how much(?: is)?)\s+",
                           "", value)
        remainder = re.sub(r"\b(?:plus|add|minus|subtract|times|multiplied|multiply|divided|"
                           r"over|by|to|the|power|modulo|mod|percent|of|half)\b", "", candidate)
        return bool(re.fullmatch(r"[\d\s.+*/%^()=\-]+", remainder))
    if intent in {"time", "date"}:
        return bool(re.fullmatch(
            r"(?:time|date|today|current (?:time|date)|the time|time now|date today|"
            r"what(?:'s| is)? (?:the )?(?:current )?(?:time|date)(?: is it| today| now)?|"
            r"what day is (?:it|today)|today(?:'s|s) date)", value))
    if intent == "files":
        return bool(re.match(r"(?:(?:find|search(?: for)?)\s+files?\b|"
                             r"(?:list|show)\s+(?:files|folders?|directories)\b|"
                             r"(?:open|create|make|new)\s+(?:(?:a|an|the)\s+)?(?:file|folder|directory)\b|"
                             r"(?:ls|dir|mkdir)(?:\s|$))", value))
    if intent == "notes":
        if re.match(r"(?:remember|remind me)\s+(?:when|how|why|what|who|where|if)\b", value):
            return False
        return bool(re.match(r"(?:remember\s+|note\s+|note down\s+|notes(?: about|$)|"
                             r"(?:show|list)(?: my)? notes$|my notes$|all my notes$|"
                             r"find my note(?:\s|$)|what did i (?:note|write)\b)", value))
    if intent == "weather":
        return bool(re.match(r"(?:weather(?:\s+(?:in|for)\s+|$)|forecast(?:\s+(?:in|for)\s+|$)|"
                             r"what(?:'s| is) the weather\b)", value))
    if intent == "system":
        return bool(re.fullmatch(r"(?:(?:show|check|tell me)(?: me)? (?:my |the )?)?"
                                 r"(?:system (?:info|details|specs)|computer specs|hardware info|my specs|my system|whoami)|"
                                 r"how (?:much ram|many cores)(?: do i have| does (?:my|this) (?:computer|pc) have)?", value))
    return False


def conversational_search(core: str) -> bool:
    """These requests ask for reasoning/help, not transmission to a search engine."""
    return bool(re.match(r"look up at\b|find(?: me)?\s+(?:it\b|(?:a |the )?(?:way|solution|reason|meaning|answer)\b|out (?:if|why|how)\b)|"
                         r"(?:research|search(?: engines?| results?)?)\s+(?:is|are|was|were|feels?|seems?|takes?|can|has|makes?)\b",
                         core, re.I))


def local_lookup(core: str) -> tuple[str, str] | None:
    """Local lookup wording must never fall through to an external web query."""
    match = re.fullmatch(r"(?:find|search(?: for)?)\s+(?:(?:my|the|a|an)\s+)?"
                         r"(files?|notes?|documents?|folders?|directories)\b"
                         r"(?:\s+(?:for|about))?\s*(.*)", core, re.I)
    if match:
        kind, target = match[1].casefold(), match[2].strip()
        if kind.startswith("note"):
            return "computer.command", "notes about " + target if target else "list notes"
        if target:
            return "computer.command", "find file " + target
        return "chat.clarify", "Which local file or folder should I look for? Give me its name, or attach it here."
    if re.match(r"(?:find|search(?: for)?)\s+my\b", core, re.I):
        return "chat.clarify", "Are you looking for a local file or note? Tell me which one, or say 'search the web for' if you want a public search."
    return None
