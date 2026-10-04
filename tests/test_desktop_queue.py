"""Exercise the native queue and lifecycle through actual worker processes."""
import json
import sys
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from sweep.desktop import window as module
from sweep.desktop.tasks import TaskStore


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_CONTROLLER_DIR", str(tmp_path / "controller"))
    app = QApplication.instance() or QApplication([])
    window = module.SweepWindow(tmp_path / "desktop")
    window.settings.setValue("notifications", False)
    yield app, window
    window.shutdown()
    window.deleteLater()
    app.processEvents()


def finish(app, window, timeout=30):
    deadline = time.monotonic() + timeout
    while (window.process is not None or window.pending) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.process is None and not window.pending, "Queue did not finish"


def select(window, task_id):
    item = next(window.history.item(i) for i in range(window.history.count())
                if window.history.item(i).data(Qt.ItemDataRole.UserRole) == task_id)
    window.history.setCurrentItem(item)
    return item


def test_fifo_queue_executes_without_overlap_and_persists_results(desktop, monkeypatch):
    app, window = desktop
    starts = []
    original_launch = window.launch_task

    def record_launch():
        assert window.process is None
        starts.append(window.task["text"])
        original_launch()

    monkeypatch.setattr(window, "launch_task", record_launch)
    for request in ("calculate 2 + 2", "calculate 5 * 3", "calculate 8 - 1"):
        assert window.start_task("computer.command", request)
    pending = list(window.pending)
    assert len(pending) == 2 and window.run_button.isEnabled()
    assert all(window.task_store.get(task_id)["approved"] is False for task_id in pending)
    finish(app, window)
    assert starts == ["calculate 2 + 2", "calculate 5 * 3", "calculate 8 - 1"]
    tasks = window.task_store.list()
    assert len(tasks) == 3 and all(task["state"] == "completed" for task in tasks)
    assert all((window.task_store.directory(task["id"]) / "result.json").exists() for task in tasks)
    assert "7" in window.last_result["message"]


def test_cancel_is_durable_and_late_result_cannot_complete_it(desktop, monkeypatch):
    app, window = desktop
    monkeypatch.setattr(module, "launch_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    assert window.start_task("computer.command", "calculate 2 + 2")
    task_id = window.task["id"]
    assert window.process.waitForStarted(5000)
    window.cancel_task()
    window.accept_event({"kind": "result", "data": {"message": "too late"}})
    window.accept_event({"kind": "error", "message": "also too late"})
    finish(app, window)
    assert window.task_store.get(task_id)["state"] == "cancelled"
    assert not (window.task_store.directory(task_id) / "result.json").exists()


def test_queued_actions_ask_only_before_dispatch_and_can_be_denied(desktop, monkeypatch):
    app, window = desktop
    approvals = []

    def decline(dialog):
        approvals.append(dialog.text())
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "exec", decline)
    window.start_task("computer.command", "calculate 2 + 2")
    window.start_task("computer.command", "run echo should-not-run")
    queued_id = window.pending[0]
    assert approvals == []
    finish(app, window)
    assert len(approvals) == 1 and "should-not-run" in approvals[0]
    assert window.task_store.get(queued_id)["state"] == "cancelled"
    assert not (window.task_store.directory(queued_id) / "events.jsonl").exists()


def test_cancel_waiting_task_never_launches_its_worker(desktop):
    app, window = desktop
    window.start_task("computer.command", "calculate 2 + 2")
    window.start_task("computer.command", "calculate 10 + 10")
    task_id = window.pending[0]
    select(window, task_id)
    assert window.cancel_queued_button.isEnabled()
    window.cancel_selected_queued()
    finish(app, window)
    assert window.task_store.get(task_id)["state"] == "cancelled"
    assert not (window.task_store.directory(task_id) / "events.jsonl").exists()


def test_restart_interrupts_unfinished_tasks_without_running_them(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    store = TaskStore(tmp_path)
    queued = store.create("computer.command", "run echo pending-action")
    running = store.create("computer.command", "run echo interrupted-action")
    store.update(running["id"], state="running", approved=True)
    window = module.SweepWindow(tmp_path)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No)
    try:
        app.processEvents()
        assert window.process is None and window.pending == []
        assert {task["state"] for task in store.list()} == {"interrupted"}
        assert all(task["approved"] is False for task in store.list())
        select(window, queued["id"])
        window.retry_selected()
        assert len(store.list()) == 2  # Denial does not create another task.
        assert window.process is None
    finally:
        window.shutdown()
        window.deleteLater()
        app.processEvents()


def test_retry_creates_new_task_and_preserves_original_artifact(desktop):
    app, window = desktop
    window.start_task("computer.command", "calculate 3 * 3")
    finish(app, window)
    original = window.task.copy()
    original_result = (window.current_directory / "result.json").read_bytes()
    select(window, original["id"])
    window.retry_selected()
    finish(app, window)
    assert window.task["id"] != original["id"]
    assert window.task["retry_of"] == original["id"]
    assert window.task_store.get(original["id"]) == original
    assert (window.task_store.directory(original["id"]) / "result.json").read_bytes() == original_result


def test_history_selection_cannot_retarget_running_result(desktop, monkeypatch):
    app, window = desktop
    window.start_task("computer.command", "calculate 2 + 2")
    finish(app, window)
    original = window.task.copy()
    original_result = window.last_result.copy()
    monkeypatch.setattr(module, "launch_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    window.start_task("computer.command", "calculate 4 + 5")
    active_directory = window.current_directory
    assert window.process.waitForStarted(5000)
    window.open_history(select(window, original["id"]))
    window.accept_event({"kind": "result", "data": {"message": "9"}})
    assert window.current_directory == active_directory
    assert json.loads((active_directory / "result.json").read_text()) == {"message": "9"}
    assert window.last_result == original_result
    assert window.viewed_task_id == original["id"]


def test_new_running_task_and_history_actions_refer_to_same_record(desktop, monkeypatch):
    app, window = desktop
    window.start_task("computer.command", "calculate 2 + 2")
    finish(app, window)
    select(window, window.task["id"])
    monkeypatch.setattr(module, "launch_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    window.start_task("computer.command", "calculate 3 + 3")
    assert window.history.currentItem().data(Qt.ItemDataRole.UserRole) == window.task["id"]
    assert window.viewed_task_id == window.task["id"]
    assert not window.retry_button.isEnabled()


def test_cancellation_stops_worker_even_if_saving_state_fails(desktop, monkeypatch):
    app, window = desktop
    monkeypatch.setattr(module, "launch_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    window.start_task("computer.command", "calculate 2 + 2")
    assert window.process.waitForStarted(5000)

    def disk_full(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(window.task_store, "update", disk_full)
    window.cancel_task()
    finish(app, window)
    assert window.task["state"] == "failed"
    assert "could not save" in window.task["error"]
    assert "could not save" in window.result_view.toPlainText()


def test_blocking_worker_is_stopped_by_parent_deadline(desktop, monkeypatch):
    app, window = desktop
    monkeypatch.setattr(module, "launch_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    window.start_task("computer.command", "calculate 2 + 2")
    assert window.process.waitForStarted(5000)
    window.started -= 126
    window.poll_events()
    finish(app, window)
    assert window.task["state"] == "failed"
    assert "time budget" in window.task["error"]


def test_failed_worker_launch_does_not_block_next_task(desktop, monkeypatch):
    app, window = desktop
    original = module.launch_command
    calls = 0

    def first_missing(*args):
        nonlocal calls
        calls += 1
        return [str(window.root / "missing-worker.exe")] if calls == 1 else original(*args)

    monkeypatch.setattr(module, "launch_command", first_missing)
    window.start_task("computer.command", "calculate 1 + 1")
    first_id = window.task["id"]
    window.start_task("computer.command", "calculate 2 + 2")
    finish(app, window)
    assert window.task_store.get(first_id)["state"] == "failed"
    assert window.task["state"] == "completed"
    assert window.task["id"] != first_id
