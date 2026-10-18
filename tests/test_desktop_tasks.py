"""Persistence and permission boundaries for the desktop task queue."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sweep.desktop import tasks
from sweep.desktop.tasks import TaskStore


def test_created_task_survives_reopening_without_shared_mutable_values(tmp_path):
    store = TaskStore(tmp_path)
    grants = [str(tmp_path / "selected.txt")]
    history = [{"role": "user", "content": "Earlier question"}]
    task = store.create("files.inspect", grants[0], granted_paths=grants, history=history)
    assert task["state"] == "queued"
    assert task["approved"] is False
    assert store.directory(task["id"]) == tmp_path / "tasks" / task["id"]
    assert TaskStore(tmp_path).get(task["id"]) == task
    grants.append("another-file.txt")
    history[0]["content"] = "Mutated caller value"
    task["history"].append({"role": "assistant", "content": "Mutated returned value"})
    saved = store.get(task["id"])
    assert saved["granted_paths"] == grants[:1]
    assert saved["history"] == [{"role": "user", "content": "Earlier question"}]


def test_source_references_are_snapshots_and_do_not_inherit_authority(tmp_path):
    store = TaskStore(tmp_path)
    thread_id = "a" * 32
    source = store.create("files.inspect", "Read the selected document",
                          granted_paths=[str(tmp_path / "private.txt")], thread_id=thread_id)
    store.update(source["id"], state="running", approved=True)
    sources = [source["id"]]
    grants = [str(tmp_path / "new-selection.txt")]
    history = [{"role": "user", "content": "Summarize the result"}]
    task = store.create("conversation", "Summarize the result", thread_id=thread_id,
                        source_tasks=sources, granted_paths=grants, history=history)
    sources.append("b" * 32)
    grants.append(str(tmp_path / "extra.txt"))
    history[0]["content"] = "Changed after enqueue"
    task["source_tasks"].clear()
    saved = TaskStore(tmp_path).get(task["id"])
    assert saved["thread_id"] == thread_id
    assert saved["source_tasks"] == [source["id"]]
    assert saved["granted_paths"] == grants[:1]
    assert source["granted_paths"][0] not in saved["granted_paths"]
    assert saved["history"] == [{"role": "user", "content": "Summarize the result"}]
    assert saved["state"] == "queued" and saved["approved"] is False
    assert store.get(source["id"])["approved"] is True
    for changes in ({"source_tasks": []}, {"thread_id": "b" * 32}, {"source_output": True}):
        with pytest.raises(ValueError, match="immutable"):
            store.update(task["id"], **changes)
        assert store.get(task["id"]) == saved


@pytest.mark.parametrize("thread_id", ["", "A" * 32, "a" * 31, "a" * 33, "../outside", 1, False, []])
def test_invalid_conversation_id_is_rejected_before_writing(tmp_path, thread_id):
    with pytest.raises(ValueError, match="conversation ID"):
        TaskStore(tmp_path).create("conversation", "Question", thread_id=thread_id)
    assert not (tmp_path / "tasks").exists()


@pytest.mark.parametrize("sources", ["a" * 32, ("a" * 32,), {}, [None], [[]], [False],
                                    ["../outside"], ["A" * 32], ["a" * 32] * 2,
                                    [f"{index:032x}" for index in range(5)]])
def test_invalid_source_references_are_rejected_before_writing(tmp_path, sources):
    with pytest.raises(ValueError):
        TaskStore(tmp_path).create("conversation", "Question", thread_id="f" * 32,
                                   source_tasks=sources)
    assert not (tmp_path / "tasks").exists()


def test_source_references_need_a_thread_and_cannot_reference_self(tmp_path, monkeypatch):
    store = TaskStore(tmp_path)
    with pytest.raises(ValueError, match="require a conversation"):
        store.create("conversation", "Question", source_tasks=["a" * 32])
    monkeypatch.setattr(tasks.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    with pytest.raises(ValueError, match="cannot reference this task"):
        store.create("conversation", "Question", thread_id="b" * 32, source_tasks=["a" * 32])
    assert not (tmp_path / "tasks").exists()


@pytest.mark.parametrize("invalid", [{"thread_id": False}, {"thread_id": "../outside"},
                                    {"source_tasks": None}, {"source_tasks": "a" * 32},
                                    {"source_tasks": ["a" * 32]},
                                    {"thread_id": "a" * 32, "source_tasks": ["b" * 32] * 2}])
def test_invalid_saved_references_are_not_read_or_recovered(tmp_path, invalid):
    store = TaskStore(tmp_path)
    task = store.create("conversation", "Question")
    task.update(invalid)
    path = store.directory(task["id"]) / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        store.get(task["id"])
    assert store.list() == []
    assert store.recover_interrupted() == 0
    assert path.read_bytes() == before


def test_thread_without_sources_and_legacy_records_remain_readable(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create("conversation", "Question", thread_id="a" * 32)
    assert task["source_tasks"] == []
    assert task["source_output"] is False
    path = store.directory(task["id"]) / "task.json"
    del task["source_tasks"]
    path.write_text(json.dumps(task), encoding="utf-8")
    assert store.get(task["id"])["thread_id"] == "a" * 32
    del task["thread_id"]
    del task["source_output"]
    path.write_text(json.dumps(task), encoding="utf-8")
    assert store.get(task["id"]) == task


@pytest.mark.parametrize("capability", ["data.inspect", "data.transform"])
def test_data_output_reference_survives_recovery_without_becoming_a_file_grant(tmp_path, capability):
    store = TaskStore(tmp_path)
    task = store.create(capability, "Convert it to JSON", thread_id="a" * 32,
                        source_tasks=["b" * 32], source_output=True)
    assert task["granted_paths"] == [] and task["source_output"] is True
    store.update(task["id"], state="running", approved=True)
    assert TaskStore(tmp_path).recover_interrupted() == 1
    saved = store.get(task["id"])
    assert saved["source_output"] is True and saved["source_tasks"] == ["b" * 32]
    assert saved["granted_paths"] == [] and saved["approved"] is False
    with pytest.raises(ValueError, match="immutable"):
        store.update(task["id"], source_output=False)


@pytest.mark.parametrize("source_output", [None, 0, 1, "true", [], {}])
def test_source_output_requires_boolean_on_create_and_read(tmp_path, source_output):
    store = TaskStore(tmp_path)
    with pytest.raises(ValueError, match="source output"):
        store.create("data.transform", "Convert it to JSON", thread_id="a" * 32,
                     source_tasks=["b" * 32], source_output=source_output)
    task = store.create("data.transform", "Convert it to JSON", thread_id="a" * 32,
                        source_tasks=["b" * 32])
    task["source_output"] = source_output
    (store.directory(task["id"]) / "task.json").write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(ValueError, match="source output"):
        store.get(task["id"])


@pytest.mark.parametrize("capability,sources", [("conversation", ["b" * 32]),
                                               ("files.inspect", ["b" * 32]),
                                               ("data.inspect", []),
                                               ("data.transform", ["b" * 32, "c" * 32])])
def test_source_output_only_selects_one_data_task_source(tmp_path, capability, sources):
    with pytest.raises(ValueError, match="exactly one"):
        TaskStore(tmp_path).create(capability, "Use the previous output", thread_id="a" * 32,
                                   source_tasks=sources, source_output=True)
    assert not (tmp_path / "tasks").exists()


def test_recovery_and_retry_keep_references_but_require_fresh_approval(tmp_path):
    store = TaskStore(tmp_path)
    thread_id = "a" * 32
    sources = [f"{index:032x}" for index in range(4)]
    original = store.create("conversation", "Compare these results", thread_id=thread_id,
                            source_tasks=sources, granted_paths=[str(tmp_path / "selected.txt")])
    store.update(original["id"], state="running", approved=True)
    assert TaskStore(tmp_path).recover_interrupted() == 1
    recovered = store.get(original["id"])
    assert recovered["thread_id"] == thread_id and recovered["source_tasks"] == sources
    assert recovered["approved"] is False and recovered["state"] == "interrupted"
    retry = store.create(recovered["capability"], recovered["text"], retry_of=recovered["id"],
                         thread_id=recovered["thread_id"], source_tasks=recovered["source_tasks"],
                         granted_paths=recovered["granted_paths"])
    assert retry["thread_id"] == thread_id and retry["source_tasks"] == sources
    assert retry["approved"] is False and retry["state"] == "queued"
    assert retry["retry_of"] == recovered["id"]
    assert store.get(original["id"]) == recovered


def test_history_sorts_by_creation_then_id_and_obeys_limit(tmp_path, monkeypatch):
    ids = iter(["1" * 32, "3" * 32, "2" * 32])
    times = iter(["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00",
                  "2026-01-02T00:00:00+00:00"])
    monkeypatch.setattr(tasks.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    monkeypatch.setattr(tasks, "_now", lambda: next(times))
    store = TaskStore(tmp_path)
    for number in range(3):
        store.create("computer.command", f"calculate {number}")
    assert [task["id"] for task in store.list()] == ["2" * 32, "3" * 32, "1" * 32]
    assert len(store.list(1)) == 1
    assert store.list(0) == []
    with pytest.raises(ValueError, match="limit"):
        store.list(-1)


def test_queued_task_cannot_gain_approval_before_dispatch(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create("computer.command", "run echo reviewed")
    with pytest.raises(ValueError, match="running task"):
        store.update(task["id"], approved=True)
    assert store.get(task["id"])["approved"] is False
    running = store.update(task["id"], state="running", approved=True)
    assert running["approved"] is True
    assert running["started"]


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled", "interrupted"])
def test_terminal_transitions_clear_approval_and_cannot_resurrect(tmp_path, terminal):
    store = TaskStore(tmp_path)
    task = store.create("computer.command", "calculate 10 * 3")
    store.update(task["id"], state="running", approved=True)
    finished = store.update(task["id"], state=terminal)
    assert finished["state"] == terminal
    assert finished["approved"] is False
    assert finished["finished"]
    assert store.update(task["id"], state=terminal, error="Kept activity log")["error"]
    for invalid in ("queued", "running"):
        with pytest.raises(ValueError, match="transition"):
            store.update(task["id"], state=invalid)
    with pytest.raises(ValueError, match="running task"):
        store.update(task["id"], approved=True)


def test_cancel_queued_task_without_starting_it(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create("web.search", "Python packaging")
    cancelled = store.update(task["id"], state="cancelled")
    assert cancelled["approved"] is False
    assert "started" not in cancelled
    assert cancelled["finished"]


def test_invalid_transition_and_request_mutation_leave_record_intact(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create("web.search", "Python packaging")
    for changes in ({"state": "completed"}, {"state": []}, {"state": "unknown"},
                    {"text": "Changed command"}, {"id": "1" * 32}):
        with pytest.raises(ValueError):
            store.update(task["id"], **changes)
        assert store.get(task["id"]) == task


def test_startup_recovers_every_unfinished_task_without_changing_completed(tmp_path, monkeypatch):
    store = TaskStore(tmp_path)
    queued = [store.create("web.search", f"Question {index}") for index in range(4)]
    store.update(queued[1]["id"], state="running", approved=True)
    done = store.create("computer.command", "calculate 2 + 2")
    store.update(done["id"], state="running", approved=True)
    store.update(done["id"], state="completed")
    done_path = store.directory(done["id"]) / "task.json"
    completed_bytes = done_path.read_bytes()
    monkeypatch.setattr(tasks, "MAX_LIST_TASKS", 2)
    assert len(store.list()) == 2
    assert TaskStore(tmp_path).recover_interrupted() == 4
    for task in queued:
        recovered = store.get(task["id"])
        assert recovered["state"] == "interrupted"
        assert recovered["approved"] is False
        assert recovered["finished"]
        assert "retry" in recovered["error"]
    assert done_path.read_bytes() == completed_bytes
    assert store.recover_interrupted() == 0


def test_retry_requires_fresh_approval_and_preserves_original(tmp_path):
    store = TaskStore(tmp_path)
    original = store.create("files.inspect", str(tmp_path / "selected.txt"),
                            granted_paths=[str(tmp_path / "selected.txt")])
    store.update(original["id"], state="running", approved=True)
    failed = store.update(original["id"], state="failed", error="File is unavailable.")
    retry = store.create(failed["capability"], failed["text"],
                         granted_paths=failed["granted_paths"], retry_of=failed["id"])
    assert retry["id"] != failed["id"]
    assert retry["retry_of"] == failed["id"]
    assert retry["state"] == "queued"
    assert retry["approved"] is False
    assert "error" not in retry
    assert store.get(original["id"]) == failed


@pytest.mark.parametrize("task_id", ["../outside", "a/../b", "a\\..\\b", "..", "", "A" * 32,
                                    "x" * 32, "0" * 31, "0" * 33, "C:\\outside", None])
def test_task_ids_cannot_select_arbitrary_paths(tmp_path, task_id):
    store = TaskStore(tmp_path)
    for operation in (store.get, store.directory):
        with pytest.raises(ValueError, match="ID"):
            operation(task_id)
    with pytest.raises(ValueError, match="ID"):
        store.create("web.search", "example", retry_of=task_id if task_id is not None else "../outside")
    assert not (tmp_path / "tasks").exists()


def test_missing_and_invalid_records_do_not_hide_valid_history(tmp_path):
    store = TaskStore(tmp_path)
    valid = store.create("computer.command", "calculate 1 + 1")
    broken = [b'{"id":', b'[]', b'{"id":"wrong"}', b'\xff', b'[' * 1500]
    for index, content in enumerate(broken):
        directory = tmp_path / "tasks" / f"{index:032x}"
        directory.mkdir()
        (directory / "task.json").write_bytes(content)
    assert store.list() == [valid]
    assert store.recover_interrupted() == 1
    with pytest.raises(FileNotFoundError):
        store.get("f" * 32)
    assert TaskStore(tmp_path / "missing").list() == []


def test_oversize_metadata_is_never_read_or_written(tmp_path, monkeypatch):
    store = TaskStore(tmp_path)
    task = store.create("web.search", "A normal request")
    monkeypatch.setattr(tasks, "MAX_TASK_BYTES", 1024)
    with pytest.raises(ValueError, match="exceeds"):
        store.create("web.search", "x" * 1500)
    path = store.directory(task["id"]) / "task.json"
    path.write_bytes(b"x" * 1025)
    with pytest.raises(ValueError, match="under 1 MB"):
        store.get(task["id"])
    assert store.list() == []


def test_old_completed_task_remains_readable_but_authority_is_not_reused(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create("computer.command", "calculate 2 + 2")
    path = store.directory(task["id"]) / "task.json"
    task.update(state="completed", approved=True, created="2026-01-01T12:00:00")
    del task["history"]
    path.write_text(json.dumps(task), encoding="utf-8")
    original_bytes = path.read_bytes()
    restored = store.get(task["id"])
    assert restored["state"] == "completed"
    assert restored["approved"] is False
    assert restored["history"] == []
    assert store.recover_interrupted() == 0
    assert path.read_bytes() == original_bytes


def _symlink_or_skip(link: Path, destination: Path, directory: bool = False):
    try:
        link.symlink_to(destination, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        if directory and os.name == "nt":
            # Junctions do not need Windows Developer Mode/admin privileges.
            import _winapi
            _winapi.CreateJunction(str(destination), str(link))
            return
        pytest.skip(f"Symlink creation is unavailable: {exc}")


def test_linked_task_directory_is_skipped_and_cannot_be_updated(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "task.json"
    outside_file.write_text("unchanged", encoding="utf-8")
    store = TaskStore(tmp_path / "desktop")
    store.tasks_root.mkdir(parents=True)
    task_id = "a" * 32
    _symlink_or_skip(store.tasks_root / task_id, outside, directory=True)
    with pytest.raises(ValueError, match="redirect"):
        store.update(task_id, state="interrupted")
    assert store.list() == []
    assert store.recover_interrupted() == 0
    assert outside_file.read_text(encoding="utf-8") == "unchanged"


def test_linked_metadata_file_is_rejected_without_modifying_target(tmp_path):
    store = TaskStore(tmp_path / "desktop")
    task = store.create("web.search", "Some sources")
    path = store.directory(task["id"]) / "task.json"
    target = tmp_path / "outside.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    _symlink_or_skip(path, target)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="regular file"):
        store.get(task["id"])
    assert store.list() == []
    assert store.recover_interrupted() == 0
    assert target.read_bytes() == before


def test_redirected_tasks_root_is_rejected_before_writing(tmp_path):
    root = tmp_path / "desktop"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(root / "tasks", outside, directory=True)
    with pytest.raises(ValueError, match="redirect"):
        TaskStore(root).create("web.search", "Do not write through a link")
    assert list(outside.iterdir()) == []
