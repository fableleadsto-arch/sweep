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
