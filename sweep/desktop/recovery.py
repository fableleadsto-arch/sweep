"""Explicit, conversation-scoped replacement of failed task dependencies."""
from __future__ import annotations

from .chat import ChatStore
from .tasks import TaskStore

MAX_RECOVERY_OPTIONS = 8


def recovery_sources(store: TaskStore, thread_id: str, task_id: str) -> list[dict]:
    """Offer completed retries; never infer a replacement from recency alone.

    Inspection does not create tasks, reread original files or replay approvals.
    The chat's bounded turn list limits both traversal and candidate enumeration.
    """
    thread = ChatStore(store.root).get(thread_id)
    members = {turn.get("task_id") for turn in thread["turns"] if turn.get("task_id")}

    def owned(identifier):
        if identifier not in members:
            raise ValueError("A recovery task no longer belongs to this conversation.")
        record = store.get(identifier)
        if record.get("thread_id") not in (None, thread_id):
            raise ValueError("A recovery task belongs to another conversation.")
        return record

    task = owned(task_id)
    if task["state"] in {"queued", "running"}:
        raise ValueError("Stop or finish this task before reviewing a retry.")
    records = {}
    for identifier in members:
        try:
            records[identifier] = owned(identifier)
        except (OSError, ValueError):
            continue
    sources = []
    for identifier in task.get("source_tasks", []):
        source = owned(identifier)
        options = []
        candidates = sorted(records.values(), key=lambda item: (item["created"], item["id"]), reverse=True)
        candidates = [source] + [item for item in candidates if item["id"] != identifier]
        for candidate in candidates:
            current = candidate
            seen = set()
            while current["id"] != identifier:
                if current["id"] in seen:
                    break
                seen.add(current["id"])
                # A retry keeps the request and file scope, even when its own
                # source references have been deliberately repaired.
                if any(current.get(key) != source.get(key) for key in
                       ("capability", "text", "granted_paths", "source_output", "history")):
                    break
                current = records.get(current.get("retry_of"))
                if current is None:
                    break
            else:
                try:
                    result = store.result(candidate["id"])
                except (OSError, ValueError):
                    continue
                options.append({"id": candidate["id"], "created": candidate["created"],
                                "original": candidate["id"] == identifier,
                                "excerpt": str(result.get("message") or result.get("text") or "Saved result")[:600]})
                if len(options) >= MAX_RECOVERY_OPTIONS:
                    break
        options.sort(key=lambda item: (not item["original"], item["created"], item["id"]))
        sources.append({"task": source, "options": options})
    return sources


def validate_selection(store: TaskStore, thread_id: str, task_id: str, selected: list[str]) -> list[str]:
    """Recheck the explicit selection after a modal review, before enqueueing."""
    sources = recovery_sources(store, thread_id, task_id)
    if len(selected) != len(sources) or len(set(selected)) != len(selected):
        raise ValueError("Choose one distinct completed result for each source.")
    for source, chosen in zip(sources, selected, strict=True):
        if chosen not in {item["id"] for item in source["options"]}:
            raise ValueError("A selected result is unavailable. Review the sources again.")
    return selected.copy()
