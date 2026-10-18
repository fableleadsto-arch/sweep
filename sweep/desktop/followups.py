"""Resolve explicit follow-ups using only resources already in this conversation."""
from __future__ import annotations

import re
from pathlib import Path

from .chat import understand

SOURCE_CAPABILITIES = frozenset({
    "documents.inspect", "data.inspect", "data.transform", "files.inspect",
    "images.inspect", "images.analyze", "media.inspect", "media.analyze",
    "audio.transcribe", "web.search", "web.scrape", "web.research",
})


def data_output(root: Path, source: dict) -> str:
    """Resolve one previous transformation's output inside its owned workspace."""
    from .data_tools import DATA_SUFFIXES
    from .tasks import TaskStore

    task = source["task"]
    if task["capability"] != "data.transform":
        raise ValueError("This source task did not create a transformed data file.")
    directory = TaskStore(root).directory(task["id"]) / "artifacts"
    if directory.is_symlink() or directory.is_junction() or directory.resolve() != directory:
        raise ValueError("The earlier output directory is redirected.")
    files = source["result"].get("artifacts", [])
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("Choose the specific output file to continue this data task.")
    path = Path(files[0]["path"])
    if (not path.is_absolute() or path.parent != directory or path.suffix.lower() not in DATA_SUFFIXES
            or path.is_symlink() or path.is_junction() or path.resolve() != path or not path.is_file()):
        raise ValueError("The earlier data output is missing or outside its task workspace.")
    return str(path)


def recent_sources(thread: dict, store) -> list[dict]:
    """Find the latest resource group, including unfinished work queued before us."""
    members = {turn["task_id"] for turn in thread["turns"] if turn.get("task_id")}
    selected = []
    for turn in reversed(thread["turns"]):
        if turn["role"] == "user" and selected:
            break
        identifier = turn.get("task_id")
        if not identifier:
            continue
        try:
            task = store.get(identifier)
            if task.get("thread_id", thread["id"]) != thread["id"]:
                continue
            if task["capability"] == "context.answer":
                if selected:
                    # A later extraction/retry is the current resource group.
                    # An older answer must not replace it with stale sources.
                    break
                sources = task.get("source_tasks", [])
                if not sources or any(source not in members for source in sources):
                    continue
                return [store.get(source) for source in sources[:4]]
            if task["capability"] in SOURCE_CAPABILITIES:
                selected.append(task)
                if len(selected) == 4:
                    break
        except (OSError, ValueError, KeyError):
            continue
    return list(reversed(selected))


def followup_request(text: str, sources: list[dict]) -> dict | None:
    """Continue an explicit resource request; ordinary chat and new actions stay separate."""
    if not sources:
        return None
    lowered = text.casefold()
    named = [task for task in sources if any(Path(path).name.casefold() in lowered
             for path in task.get("granted_paths", []) if Path(path).name)]
    if named:
        sources = named
    referenced = bool(named or re.search(
        r"\b(?:this|that|these|those|it|its|them|their|they|earlier|previous|above)\b", lowered))
    requested = bool(re.search(r"\b(?:summari[sz]e|summary|compare|explain|transcrib\w*|subtitles?|clean|deduplicat\w*|filter|convert|export)\b", lowered))
    if not referenced and not requested:
        return None
    if not referenced and not re.search(r"\b(?:compare|clean|deduplicat\w*|filter|convert|export)\b", lowered):
        if lowered.strip(" .!?") not in {"summarize", "summarise", "summary", "explain more", "tell me more"}:
            return None
    # Explicit new web/computer commands must not become file operations merely
    # because a previous result exists. Derived web queries need a separate review.
    plain, _ = understand(text)
    if plain in {"computer.command", "web.search", "web.scrape", "web.research", "chat.new"}:
        return None
    if not requested and not re.search(r"\?|\b(?:what|when|where|which|why|how|tell|read|inspect|describe|mention|remove|trim|drop)\b", lowered):
        return None
    file_sources = [task for task in sources if task.get("granted_paths")]
    compare = bool(re.search(r"\b(?:compare|both|each|their|these|those|differences?|similarities)\b", lowered))
    if len(sources) > 1 and not compare:
        names = [Path(task["granted_paths"][0]).name if task.get("granted_paths") else task["text"][:80] for task in sources]
        return {"capability": "chat.clarify", "text": "Which source should I use? Name one of: " + ", ".join(names) + ". You can also ask to compare them."}
    source_ids = [task["id"] for task in sources]
    if len(file_sources) == 1 and len(sources) == 1:
        task = file_sources[0]
        paths = task["granted_paths"][:1]
        suffix = Path(paths[0]).suffix.lower()
        from .data_tools import DATA_SUFFIXES
        from .media import AUDIO_SUFFIXES, VIDEO_SUFFIXES

        if suffix in DATA_SUFFIXES and re.search(r"\b(?:clean|deduplicat\w*|filter|convert|export|remove|trim|drop)\b", lowered):
            command = re.sub(r"^(?:now|then)[,\s]+", "", text, flags=re.I)
            capability, request = understand(command, paths)
            return {"capability": capability, "text": request, "granted_paths": paths, "source_tasks": source_ids,
                    "source_output": task["capability"] == "data.transform"}
        if suffix in AUDIO_SUFFIXES | VIDEO_SUFFIXES and re.search(r"\b(?:transcrib\w*|subtitles?)\b", lowered):
            return {"capability": "audio.transcribe", "text": text, "granted_paths": paths, "source_tasks": source_ids}
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"} and re.search(r"\b(?:describe|inspect|look|analy[sz]e)\b", lowered):
            return {"capability": "images.inspect", "text": text, "granted_paths": paths, "source_tasks": source_ids}
    return {"capability": "context.answer", "text": text, "source_tasks": source_ids}
