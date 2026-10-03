"""Desktop capability adapters, explicit permissions and task event contracts.

No GUI or model imports at startup. Existing Sweep modules do the actual work.
"""
from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Capability:
    id: str
    title: str
    permission: str
    execution: str
    description: str


CAPABILITIES = {item.id: item for item in (
    Capability("computer.command", "Desktop command", "COMPUTER_CONTROL", "local",
               "Use Sweep's existing desktop skills; approve actions before execution."),
    Capability("web.search", "Web search", "EXTERNAL", "remote", "Find public sources."),
    Capability("web.scrape", "Read a web page", "EXTERNAL", "remote", "Extract public-page text and links."),
    Capability("web.research", "Research", "EXTERNAL", "remote", "Collect sources and evidence within a time budget."),
    Capability("files.inspect", "Inspect a file", "READ", "local", "Preview a selected text, CSV, JSON or image file."),
    Capability("conversation", "Conversation", "EXTERNAL", "configured provider", "Talk using an existing configured AI provider."),
)}


def route(text: str, selected: str = "auto") -> tuple[str, str]:
    text = text.strip()
    if not text:
        raise ValueError("Enter a request first.")
    if selected != "auto":
        if selected not in CAPABILITIES:
            raise ValueError("Unknown capability")
        return selected, text
    for prefix, capability in (("research ", "web.research"), ("scrape ", "web.scrape"),
                               ("search ", "web.search"), ("find sources ", "web.search")):
        if text.lower().startswith(prefix):
            return capability, text[len(prefix):].strip()
    if text.startswith(("https://", "http://")):
        return "web.scrape", text
    from sweep.parser import parse
    command = parse(text)
    if command and command.intent not in {"chat", "search", "exit"}:
        return "computer.command", text
    if command and command.intent == "search":
        return "web.search", command.params.get("query", text)
    return "conversation", text


def needs_approval(capability: str, text: str) -> bool:
    if capability != "computer.command":
        return CAPABILITIES[capability].permission in {"READ", "EXTERNAL"}
    from sweep.parser import parse
    command = parse(text)
    return not command or command.intent not in {"calc", "time", "date", "system", "help"}


def inspect_file(path: str, granted_paths: list[str]) -> dict:
    selected = Path(path).resolve(strict=True)
    allowed = {Path(value).resolve(strict=True) for value in granted_paths}
    if selected not in allowed or not selected.is_file():
        raise PermissionError("Choose this file using the file picker before reading it.")
    info = selected.stat()
    result = {"title": selected.name, "path": str(selected), "bytes": info.st_size,
              "modified": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
              "message": "File inspected locally. Nothing was uploaded."}
    if info.st_size > 10_000_000:
        result["message"] += " Preview limited to files up to 10 MB."
        return result
    suffix = selected.suffix.lower()
    if suffix == ".csv":
        with selected.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows = []
            for index, row in enumerate(csv.reader(handle)):
                if index >= 101:
                    break
                rows.append([cell[:1000] for cell in row[:50]])
        result.update(columns=rows[0] if rows else [], rows=rows[1:],
                      message="Local CSV preview: up to 100 rows and 50 columns; source unchanged.")
    elif suffix in {".txt", ".md", ".log", ".json", ".jsonl", ".py", ".yaml", ".yml"}:
        with selected.open(encoding="utf-8-sig", errors="replace") as handle:
            result["text"] = handle.read(60000)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        result["image"] = str(selected)
    else:
        result["message"] += " Content preview is not available for this format yet."
    return result


async def execute(request: dict, emit: Callable[[dict], None]) -> dict:
    capability, text = request["capability"], request["text"]
    if capability not in CAPABILITIES:
        raise ValueError("Unknown capability")
    if needs_approval(capability, text) and request.get("approved") is not True:
        raise PermissionError("This task needs your approval.")

    def progress(message: str, **extra):
        emit({"kind": "progress", "message": message, **extra})

    progress(CAPABILITIES[capability].title, execution=CAPABILITIES[capability].execution)
    if capability == "files.inspect":
        progress("Reading the selected file locally")
        return inspect_file(text, request.get("granted_paths", []))
    if capability == "computer.command":
        from sweep import Controller, SkillContext
        from sweep.parser import parse
        command = parse(text)
        if not command or command.intent in {"search", "chat", "exit"}:
            raise ValueError("Choose Search or Conversation for this request.")
        progress(f"Running desktop skill: {command.intent}")
        result = await Controller().execute(text, SkillContext(yes=True), allow_llm=False)
        if result.status == "error":
            raise RuntimeError(result.message)
        return asdict(result)
    if capability == "web.search":
        from app.search.engine import relai_search
        progress(f"Searching public sources for: {text}")
        result = await relai_search(text, limit=8)
        result["message"] = f"Found {len(result.get('hits', []))} sources."
        return result
    if capability == "web.scrape":
        from app.core.http import relai_fetch
        from app.extraction.page_data import extract_page_data
        progress("Checking the public address and downloading the page")
        fetched = await relai_fetch(text)
        if not fetched.ok:
            raise RuntimeError(fetched.error or f"The page returned HTTP {fetched.status}.")
        progress("Extracting text, metadata and links")
        return extract_page_data(html=fetched.text, url=fetched.url, status=fetched.status,
                                 content_type=fetched.content_type).model_dump(mode="json")
    if capability == "web.research":
        from app.research.engine import start_research, get_research
        session = await start_research(text, max_searches=4, max_pages=8, max_runtime_ms=60000)
        seen = {}
        while session.status.value == "running":
            for action in session.actions:
                value = (action.status, action.detail)
                if seen.get(action.id) != value:
                    seen[action.id] = value
                    progress(action.description, status=action.status, detail=action.detail)
            await asyncio.sleep(0.15)
            session = get_research(session.id)
        result = session.model_dump(mode="json")
        result["message"] = f"Collected {len(session.evidence)} evidence items from {len(session.sources)} sources."
        if session.error:
            result["message"] += f" Partial result: {session.error}"
        return result
    from companion.config import get_settings
    from companion.providers import ProviderChain
    progress("Contacting your configured conversation provider")
    history = request.get("history", [])[-12:]
    messages = [{"role": item["role"], "content": str(item["content"])[:12000]}
                for item in history if item.get("role") in {"user", "assistant"}]
    try:
        result = await ProviderChain(get_settings()).generate(
            system="You are Sweep, a desktop companion. Answer clearly. You cannot execute tools in this conversation. Never claim you changed files or performed actions.",
            messages=[*messages, {"role": "user", "content": text}], json_mode=False,
        )
    except Exception:
        raise RuntimeError("No conversation provider responded. Configure your provider in Sweep's .env file, or run Ollama locally. Desktop, file and web tasks remain available.") from None
    return {"message": result.text, "provider": result.provider, "model": result.model}


def run_worker(task_path: str, events_path: str) -> int:
    """File-based IPC also works in a frozen Windows windowed executable."""
    request = json.loads(Path(task_path).read_text(encoding="utf-8"))
    with Path(events_path).open("a", encoding="utf-8", buffering=1) as stream:
        def emit(event):
            event["at"] = datetime.now(timezone.utc).isoformat()
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        try:
            async def bounded():
                async with asyncio.timeout(120):
                    return await execute(request, emit)
            result = asyncio.run(bounded())
            emit({"kind": "result", "data": result})
            return 0
        except Exception as exc:
            emit({"kind": "error", "message": str(exc)[:1000]})
            return 1
