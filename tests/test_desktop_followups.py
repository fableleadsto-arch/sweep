"""Exercise resource continuation through saved tasks and real data adapters."""
import asyncio
import json

import pytest

from sweep.desktop.chat import ChatStore
from sweep.desktop.followups import data_output, followup_request, recent_sources
from sweep.desktop.runtime import execute
from sweep.desktop.tasks import TaskStore


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_DESKTOP_DIR", str(tmp_path))
    chats, tasks = ChatStore(tmp_path), TaskStore(tmp_path)
    thread = chats.create()
    return tmp_path, chats, tasks, thread


def save_source(workspace, path, capability="documents.inspect", state="completed"):
    root, chats, tasks, thread = workspace
    task = tasks.create(capability, "Inspect", granted_paths=[str(path)], thread_id=thread["id"])
    chats.append(thread["id"], {"role": "assistant", "task_id": task["id"]})
    if state != "queued":
        tasks.update(task["id"], state="running", approved=True)
        task = tasks.update(task["id"], state=state)
    return task


@pytest.mark.parametrize("text", ["Open YouTube in Brave", "Search for Python docs", "Explain photosynthesis"])
def test_new_requests_do_not_reuse_an_unrelated_document(workspace, text):
    task = save_source(workspace, workspace[0] / "contract.txt")
    assert followup_request(text, [task]) is None


def test_queued_source_is_retained_and_reloaded_after_restart(workspace):
    root, chats, tasks, thread = workspace
    task = save_source(workspace, root / "contract.txt", state="queued")
    sources = recent_sources(ChatStore(root).get(thread["id"]), TaskStore(root))
    request = followup_request("What does this document say about cancellation?", sources)
    assert request["capability"] == "context.answer"
    assert request["source_tasks"] == [task["id"]]


def test_multiple_sources_need_a_name_or_plural_reference(workspace):
    first = save_source(workspace, workspace[0] / "invoice.txt")
    second = save_source(workspace, workspace[0] / "packing.txt")
    assert followup_request("Explain it", [first, second])["capability"] == "chat.clarify"
    assert followup_request("Compare these documents", [first, second])["source_tasks"] == [first["id"], second["id"]]
    assert followup_request("Summarize invoice.txt", [first, second])["source_tasks"] == [first["id"]]


def test_context_answer_keeps_original_sources_for_later_followups(workspace):
    root, chats, tasks, thread = workspace
    source = save_source(workspace, root / "contract.txt")
    chats.append(thread["id"], {"role": "user", "text": "Summarize it"})
    answer = tasks.create("context.answer", "Summarize it", thread_id=thread["id"], source_tasks=[source["id"]])
    chats.append(thread["id"], {"role": "assistant", "task_id": answer["id"]})
    assert [task["id"] for task in recent_sources(chats.get(thread["id"]), tasks)] == [source["id"]]


def test_later_extraction_retry_supersedes_an_older_answers_sources(workspace):
    root, chats, tasks, thread = workspace
    source = save_source(workspace, root / "contract.txt", state="failed")
    answer = tasks.create("context.answer", "Summarize it", thread_id=thread["id"], source_tasks=[source["id"]])
    chats.append(thread["id"], {"role": "assistant", "task_id": answer["id"]})
    retry = save_source(workspace, root / "contract.txt")
    sources = recent_sources(chats.get(thread["id"]), tasks)
    assert [task["id"] for task in sources] == [retry["id"]]
    assert followup_request("Summarize it", sources)["source_tasks"] == [retry["id"]]


def test_new_chat_has_no_previous_resource(workspace):
    root, chats, tasks, thread = workspace
    save_source(workspace, root / "contract.txt")
    assert recent_sources(chats.create(), tasks) == []


def test_real_clean_then_convert_uses_cleaned_output(workspace):
    root, chats, tasks, thread = workspace
    original = root / "orders.csv"
    original.write_text("name,quantity\nbolt,2\nbolt,2\nnut,3\n", encoding="utf-8")
    before = original.read_bytes()
    clean = tasks.create("data.transform", "Remove duplicate rows", granted_paths=[str(original)], thread_id=thread["id"])
    tasks.update(clean["id"], state="running", approved=True)
    clean_dir = tasks.directory(clean["id"])
    result = asyncio.run(execute({**clean, "approved": True, "_artifact_dir": str(clean_dir / "artifacts")}, lambda event: None))
    (clean_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    tasks.update(clean["id"], state="completed")
    chats.append(thread["id"], {"role": "assistant", "task_id": clean["id"]})
    request = followup_request("Now convert it to JSON", recent_sources(chats.get(thread["id"]), tasks))
    assert request["source_output"] is True
    conversion = tasks.create(**request, thread_id=thread["id"])
    converted = asyncio.run(execute({**conversion, "approved": True,
        "_artifact_dir": str(tasks.directory(conversion["id"]) / "artifacts")}, lambda event: None))
    output = json.loads(__import__("pathlib").Path(converted["artifacts"][0]["path"]).read_text(encoding="utf-8"))
    assert len(output) == 2
    assert original.read_bytes() == before


def test_followup_does_not_read_output_from_another_chat(workspace):
    root, chats, tasks, thread = workspace
    source = save_source(workspace, root / "private.csv", capability="data.transform")
    other = chats.create()
    with pytest.raises(PermissionError, match="conversation"):
        asyncio.run(execute({"capability": "data.transform", "text": "Convert to JSON", "approved": True,
            "thread_id": other["id"], "source_tasks": [source["id"]], "source_output": True}, lambda event: None))


def test_failed_source_never_falls_back_to_original(workspace):
    root, chats, tasks, thread = workspace
    original = root / "orders.csv"
    original.write_text("item\nbolt\n", encoding="utf-8")
    source = save_source(workspace, original, capability="data.transform", state="failed")
    with pytest.raises(ValueError, match="has not completed"):
        asyncio.run(execute({"capability": "data.transform", "text": "Convert it to JSON", "approved": True,
            "granted_paths": [str(original)], "thread_id": thread["id"],
            "source_tasks": [source["id"]], "source_output": True}, lambda event: None))


def test_source_artifact_cannot_escape_task_directory(workspace):
    root, chats, tasks, thread = workspace
    task = save_source(workspace, root / "orders.csv", capability="data.transform")
    outside = root / "outside.csv"
    outside.write_text("secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        data_output(root, {"task": task, "result": {"artifacts": [{"path": str(outside)}]}})
