import pytest

from sweep.desktop.chat import ChatStore
from sweep.desktop.recovery import recovery_sources, validate_selection
from sweep.desktop.tasks import TaskStore


@pytest.fixture
def workflow(tmp_path):
    chats, store = ChatStore(tmp_path), TaskStore(tmp_path)
    thread = chats.create()["id"]

    def create(text="Read invoice", state="completed", sources=None, retry_of=None, owner=thread):
        task = store.create("files.inspect", text, thread_id=owner, source_tasks=sources, retry_of=retry_of)
        chats.append(owner, {"role": "assistant", "task_id": task["id"]})
        if state != "queued":
            store.update(task["id"], state="running", approved=True)
            if state == "completed":
                store.complete(task["id"], {"text": "Invoice quantity 12"})
            elif state != "running":
                store.update(task["id"], state=state)
        return task["id"]

    return chats, store, thread, create


def test_recovery_keeps_success_and_offers_only_completed_retry_lineage(workflow):
    chats, store, thread, create = workflow
    first = create(state="failed")
    good = create("Read packing list")
    parent = create("Compare quantities", "failed", [first, good])
    failed_retry = create(state="failed", retry_of=first)
    retry = create(retry_of=failed_retry)
    create()  # A newer, similar result is not a retry.
    create("Changed request", retry_of=first)
    create(state="running", retry_of=first)
    original = store.get(parent)
    rows = recovery_sources(store, thread, parent)
    assert [item["id"] for item in rows[0]["options"]] == [retry]
    assert [item["id"] for item in rows[1]["options"]] == [good]
    assert validate_selection(store, thread, parent, [retry, good]) == [retry, good]
    assert store.get(parent) == original


def test_recheck_rejects_evidence_damaged_after_review(workflow):
    chats, store, thread, create = workflow
    source = create(state="failed")
    parent = create("Compare", "failed", [source])
    retry = create(retry_of=source)
    assert recovery_sources(store, thread, parent)[0]["options"]
    (store.directory(retry) / "result.json").write_text("damaged", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        validate_selection(store, thread, parent, [retry])


def test_cross_chat_retry_and_inconsistent_membership_are_not_reused(workflow):
    chats, store, thread, create = workflow
    source = create(state="failed")
    parent = create("Compare", "failed", [source])
    other = chats.create()["id"]
    retry = create(retry_of=source, owner=other)
    chats.append(thread, {"role": "assistant", "task_id": retry})
    assert recovery_sources(store, thread, parent)[0]["options"] == []
    with pytest.raises(ValueError):
        validate_selection(store, thread, parent, [retry])
    with pytest.raises(ValueError, match="conversation"):
        recovery_sources(store, other, parent)


@pytest.mark.parametrize("state", ["running", "queued"])
def test_active_parent_cannot_be_retried(workflow, state):
    chats, store, thread, create = workflow
    source = create()
    parent = create("Compare", state, [source])
    with pytest.raises(ValueError, match="Stop or finish"):
        recovery_sources(store, thread, parent)


def test_selection_requires_each_explicit_source_in_order(workflow):
    chats, store, thread, create = workflow
    first, second = create(), create("Packing")
    parent = create("Compare", "failed", [first, second])
    for selected in ([], [first], [first, first], [second, first], [first, "f" * 32]):
        with pytest.raises(ValueError):
            validate_selection(store, thread, parent, selected)


def test_changed_file_grants_and_cyclic_retry_records_are_excluded(workflow):
    chats, store, thread, create = workflow
    source = create(state="failed")
    parent = create("Compare", "failed", [source])
    changed = create(retry_of=source)
    record = store.get(changed)
    record["granted_paths"] = ["different-private-file.txt"]
    store._save(record)  # Simulate an inconsistent record from an older client.
    first = create()
    second = create(retry_of=first)
    record = store.get(first)
    record["retry_of"] = second
    store._save(record)
    assert recovery_sources(store, thread, parent)[0]["options"] == []


def test_recovery_bounds_choices_and_always_retains_valid_original(workflow):
    chats, store, thread, create = workflow
    source = create()
    parent = create("Compare", "failed", [source])
    retries = [create(retry_of=source) for _ in range(12)]
    options = recovery_sources(store, thread, parent)[0]["options"]
    assert len(options) == 8 and options[0]["id"] == source
    assert {item["id"] for item in options[1:]} == set(retries[-7:])
