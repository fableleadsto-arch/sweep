import asyncio
import json
import time
from zipfile import ZipFile

import pytest

from sweep.desktop.runtime import execute, inspect_file, needs_approval, route, run_worker
from scripts.desktop_installer import unpack


def test_routes_reuse_existing_controller_and_web_tools():
    assert route("research Python packaging") == ("web.research", "Python packaging")
    assert route("https://example.com/Case") == ("web.scrape", "https://example.com/Case")
    assert route("calculate 2 + 2")[0] == "computer.command"
    assert needs_approval("computer.command", "run echo test")
    assert not needs_approval("computer.command", "calculate 2 + 2")
    assert needs_approval("conversation", "Hello")


def test_worker_rejects_unapproved_action_before_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_CONTROLLER_DIR", str(tmp_path / "controller"))
    request = {"capability": "computer.command", "text": "run echo not-approved"}
    with pytest.raises(PermissionError):
        asyncio.run(execute(request, lambda event: None))
    assert not (tmp_path / "controller").exists()


def test_selected_file_grant_and_bounded_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\n" + "hello,42\n" * 250)
    with pytest.raises(PermissionError):
        inspect_file(str(path), [])
    result = inspect_file(str(path), [str(path)])
    assert len(result["rows"]) == 100
    assert result["columns"] == ["name", "value"]
    assert path.read_text().count("hello") == 250


def test_worker_protocol_emits_real_result(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_CONTROLLER_DIR", str(tmp_path / "controller"))
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"capability": "computer.command", "text": "calculate 2 + 2"}))
    events = tmp_path / "events.jsonl"
    assert run_worker(str(task), str(events)) == 0
    messages = [json.loads(line) for line in events.read_text().splitlines()]
    assert messages[0]["kind"] == "progress"
    assert messages[-1]["kind"] == "result"
    assert "4" in messages[-1]["data"]["message"]


def test_installer_rejects_traversal_before_writing(tmp_path):
    payload = tmp_path / "payload.zip"
    with ZipFile(payload, "w") as archive:
        archive.writestr("good.txt", "good")
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe path"):
        unpack(payload, tmp_path / "app")
    assert not (tmp_path / "app" / "good.txt").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_installer_extracts_valid_payload(tmp_path):
    payload = tmp_path / "payload.zip"
    with ZipFile(payload, "w") as archive:
        archive.writestr("_internal/data.txt", "hello")
    unpack(payload, tmp_path / "app")
    assert (tmp_path / "app/_internal/data.txt").read_text() == "hello"


def test_installer_registers_per_user_app_and_preserves_user_data(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scripts import desktop_installer as installer
    local = tmp_path / "user space"
    roaming = tmp_path / "roaming"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("APPDATA", str(roaming))
    stored = {}
    fake_registry = SimpleNamespace(HKEY_CURRENT_USER=1, REG_SZ=1, REG_DWORD=4,
        CreateKey=lambda *args: nullcontext("key"),
        SetValueEx=lambda key, name, reserved, kind, value: stored.update({name: value}))
    monkeypatch.setitem(sys.modules, "winreg", fake_registry)
    shortcuts = []
    monkeypatch.setattr(installer, "create_shortcut", lambda target, shortcut: shortcuts.append((target, shortcut)))
    personal = local / "Sweep" / "tasks"
    personal.mkdir(parents=True)
    (personal / "keep.json").write_text("user data")
    payload = tmp_path / "payload.zip"
    with ZipFile(payload, "w") as archive:
        archive.writestr("Sweep.exe", "test executable")
        archive.writestr("_internal/resource.txt", "data")
    installed = installer.install(payload, False, lambda message: None)
    assert installed == local / "Programs" / "Sweep"
    assert stored["DisplayName"] == "Sweep"
    assert "uninstall.ps1" in stored["UninstallString"]
    assert shortcuts[0][0] == installed / "Sweep.exe"
    assert (personal / "keep.json").read_text() == "user data"
    assert installer.install(payload, False, lambda message: None) == installed
    assert not list((local / "Programs").glob("Sweep-previous-*"))
    assert (personal / "keep.json").exists()


def test_installer_refuses_unrecognized_existing_directory(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts import desktop_installer as installer
    monkeypatch.setitem(sys.modules, "winreg", SimpleNamespace())
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    unrelated = tmp_path / "Programs/Sweep"
    unrelated.mkdir(parents=True)
    (unrelated / "personal.txt").write_text("keep")
    with pytest.raises(RuntimeError, match="not a recognized Sweep installation"):
        installer.install(tmp_path / "unused.zip", False, lambda message: None)
    assert (unrelated / "personal.txt").read_text() == "keep"


def test_native_window_executes_real_task_and_restores_artifact(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from sweep.desktop.window import SweepWindow
    monkeypatch.setenv("SWEEP_CONTROLLER_DIR", str(tmp_path / "controller"))
    app = QApplication.instance() or QApplication([])
    window = SweepWindow(tmp_path / "desktop")
    try:
        assert window.start_task("computer.command", "calculate 2 + 2")
        deadline = time.monotonic() + 20
        while window.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        assert window.process is None, "Worker did not finish"
        assert window.task["state"] == "completed", window.task
        assert "4" in window.last_result["message"]
        assert (window.current_directory / "result.json").exists()
        window.open_history(window.history.item(0))
        assert "4" in window.result_view.toPlainText()
    finally:
        window.shutdown()
        window.deleteLater()
        app.processEvents()


def test_native_permission_denial_does_not_start_worker(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from sweep.desktop.window import SweepWindow
    app = QApplication.instance() or QApplication([])
    window = SweepWindow(tmp_path)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No)
    try:
        assert not window.start_task("computer.command", "run echo declined")
        assert window.process is None
        assert not list(tmp_path.glob("tasks/*/task.json"))
    finally:
        window.shutdown()
        window.deleteLater()
        app.processEvents()


def test_native_cancel_stops_worker_and_records_state(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import sys
    from PySide6.QtWidgets import QApplication
    from sweep.desktop import window as module
    app = QApplication.instance() or QApplication([])
    window = module.SweepWindow(tmp_path)
    monkeypatch.setattr(module, "launch_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert window.start_task("computer.command", "calculate 2 + 2")
        assert window.process.waitForStarted(5000)
        window.cancel_task()
        deadline = time.monotonic() + 5
        while window.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        assert window.process is None
        saved = json.loads((window.current_directory / "task.json").read_text())
        assert saved["state"] == "cancelled"
    finally:
        window.shutdown()
        window.deleteLater()
        app.processEvents()
