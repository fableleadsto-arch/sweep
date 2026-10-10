"""Local chat threads and deterministic routing for the single chat surface."""
from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sweep.store import _save_json


def understand(text: str, attachments: list[str] | None = None) -> tuple[str, str]:
    """Resolve explicit tasks without turning every question into a web search."""
    from sweep.parser import parse, strip_politeness
    original = text.strip()
    core = strip_politeness(original)
    if not original:
        raise ValueError("Write a message or attach a file first.")
    if attachments:
        from .data_tools import DATA_SUFFIXES, DOCUMENT_SUFFIXES, parse_data_request
        suffix = Path(attachments[0]).suffix.lower()
        if suffix in DATA_SUFFIXES:
            try:
                operation = parse_data_request(original)["operation"]
            except ValueError as exc:
                return "chat.clarify", str(exc)
            return ("data.inspect" if operation == "inspect" else "data.transform"), original
        if suffix in DOCUMENT_SUFFIXES:
            return "documents.inspect", original
        return ("images.inspect" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
                else "files.inspect"), original
    if re.search(r"\b(?:this|that|attached) (?:document|pdf|csv|dataset|spreadsheet|data file)\b", core, re.I):
        return "chat.clarify", "Attach the document or data file here. I can read PDF, DOCX and text, or inspect, clean, filter and convert supported data files locally."
    if re.search(r"\b(?:this|that) (?:image|photo|picture|screenshot)\b", core, re.I):
        return "chat.clarify", "Attach the image here so I can inspect it. You can then ask about its text, metadata, objects or location clues."
    if re.search(r"\b(?:this|that) (?:person(?:'s|s)?|man|woman|face)\b", core, re.I) and re.search(r"social|profile|identity|who|find", core, re.I):
        return "chat.clarify", "Give me a name or public handle to search for public profiles. I can inspect an attached image's visible content, but I can't identify a private person or find their accounts from a photo."
    if core.casefold() in {"clear chat", "clear conversation", "reset chat", "reset conversation", "new chat", "new conversation"}:
        return "chat.new", original
    command = parse(original)
    # These explicit local requests must precede generic "find/search" web routing.
    if command and command.intent in {"files", "notes"}:
        return "computer.command", original
    # The terminal's discussion skill has its own model and history. All desktop
    # discussion stays with the selected provider and the current chat thread.
    if command and command.intent == "discuss":
        return "conversation", original
    match = re.match(r"(?:research|investigate|gather (?:information|sources) (?:on|about))\s+(.+)", core, re.I)
    if match:
        return "web.research", match[1]
    match = re.match(r"(?:scrape|web\s*scrape|extract (?:the )?(?:text|data) from|read (?:the )?(?:web ?page|website))\s+(https?://\S+)", core, re.I)
    if match:
        return "web.scrape", match[1].rstrip(".,")
    if re.fullmatch(r"https?://\S+", core):
        return "web.scrape", core
    match = re.match(r"(?:search(?: the web)?(?: for)?|look up|find(?: me)?(?: sources (?:for|on))?)\s+(.+)", core, re.I)
    if match:
        return "web.search", match[1]
    if command and command.intent not in {"chat", "search", "exit"}:
        return "computer.command", original
    return "conversation", original


def validate_proposal(value: object) -> dict | None:
    """A model can propose a registered task, never executable code or a shell."""
    if not isinstance(value, dict):
        return None
    capability, text = value.get("capability"), value.get("text")
    if capability not in {"computer.command", "web.search", "web.scrape", "web.research"}:
        return None
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        return None
    if capability == "computer.command":
        from sweep.parser import parse
        command = parse(text)
        if not command or command.intent not in {"open", "calc", "time", "date", "weather", "system", "notes", "help"}:
            return None
    if capability == "web.scrape" and not text.startswith(("http://", "https://")):
        return None
    return {"capability": capability, "text": text.strip()}


def result_text(result: dict) -> str:
    text = str(result.get("message") or result.get("text") or result.get("title") or "Task completed.")
    if result.get("text") and result.get("message") and result["text"] != result["message"]:
        text += "\n\n" + str(result["text"])
    if result.get("columns"):
        text += "\n\n" + " · ".join(map(str, result["columns"]))
        text += "\n" + "\n".join(" · ".join(map(str, row)) for row in result.get("rows", [])[:30])
    evidence = result.get("evidence", [])
    if isinstance(evidence, list):
        excerpts = []
        for item in evidence[:8]:
            if not isinstance(item, dict) or not isinstance(item.get("excerpt"), str) or not item["excerpt"].strip():
                continue
            excerpts.append(f"{item['excerpt'][:1200]}\n{str(item.get('source_title', 'Source'))[:200]}: {str(item.get('source_url', ''))[:2048]}")
        if excerpts:
            text += "\n\nEvidence excerpts (source content):\n" + "\n\n".join(excerpts)
    hits = result.get("hits", result.get("sources", []))
    if hits:
        text += "\n\n" + "\n".join(f"{hit.get('title', '')}: {hit.get('snippet', '')}\n{hit.get('url', '')}" for hit in hits[:8])
    return text[:20000]


class ChatStore:
    def __init__(self, root: Path):
        self.root = root / "chats"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, chat_id: str) -> Path:
        if not isinstance(chat_id, str) or not re.fullmatch(r"[0-9a-f]{32}", chat_id):
            raise ValueError("Invalid conversation ID")
        path = self.root / f"{chat_id}.json"
        if path.is_symlink() or self.root.is_symlink() or self.root.is_junction():
            raise ValueError("Conversation storage cannot use redirected files")
        return path

    def create(self) -> dict:
        thread = {"id": uuid.uuid4().hex, "title": "New chat", "created": datetime.now(UTC).isoformat(), "turns": []}
        self.save(thread)
        return thread

    def save(self, thread: dict):
        _save_json(self.path(thread["id"]), thread)

    def get(self, chat_id: str) -> dict:
        with self.path(chat_id).open(encoding="utf-8") as stream:
            raw = stream.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Conversation history is too large")
        try:
            value = json.loads(raw)
        except RecursionError as exc:
            raise ValueError("Invalid conversation history") from exc
        if (not isinstance(value, dict) or value.get("id") != chat_id
            or not isinstance(value.get("title"), str) or len(value["title"]) > 200
            or not isinstance(value.get("created"), str) or len(value["created"]) > 64
            or not isinstance(value.get("turns"), list) or len(value["turns"]) > 200):
            raise ValueError("Invalid conversation history")
        try:
            datetime.fromisoformat(value["created"])
        except ValueError as exc:
            raise ValueError("Invalid conversation timestamp") from exc
        for turn in value["turns"]:
            if (not isinstance(turn, dict) or not isinstance(turn.get("role"), str)
                or turn["role"] not in {"user", "assistant"}
                or not isinstance(turn.get("text", ""), str) or len(turn.get("text", "")) > 60000
                or ("task_id" in turn and (not isinstance(turn["task_id"], str)
                    or not re.fullmatch(r"[0-9a-f]{32}", turn["task_id"])))
                or not isinstance(turn.get("attachments", []), list) or len(turn.get("attachments", [])) > 10
                or any(not isinstance(name, str) or len(name) > 1024 for name in turn.get("attachments", []))):
                raise ValueError("Invalid conversation turn")
        return value

    def list(self) -> list[dict]:
        threads = []
        for path in self.root.glob("*.json"):
            try:
                thread = self.get(path.stem)
                threads.append(thread)
            except (OSError, ValueError):
                continue
        return sorted(threads, key=lambda thread: thread.get("created", ""), reverse=True)[:100]

    def append(self, chat_id: str, turn: dict):
        thread = self.get(chat_id)
        if not thread["turns"] and turn.get("role") == "user":
            thread["title"] = str(turn.get("text", "Chat"))[:60]
        thread["turns"].append(turn)
        # Tasks keep their own complete artifacts; the chat stays bounded.
        thread["turns"] = thread["turns"][-200:]
        # Leave room for JSON formatting and Windows newline translation, so
        # even long Unicode turns cannot produce a record get() will reject.
        while len(thread["turns"]) > 1 and len(json.dumps(thread, indent=2, ensure_ascii=False).encode("utf-8")) > 1_900_000:
            thread["turns"].pop(0)
        self.save(thread)
