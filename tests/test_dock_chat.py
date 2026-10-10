import json
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
)

from sweep.desktop.chat import ChatStore, understand, validate_proposal
from sweep.desktop.dock import DockWindow


@pytest.mark.parametrize("text,capability", [
    ("open youtube in brave", "computer.command"),
    ("could you open YouTube in Brave", "computer.command"),
    ("What is a binary tree?", "conversation"),
    ("I had a difficult day", "conversation"),
    ("please search Python documentation", "web.search"),
    ("find Ada Lovelace public profiles", "web.search"),
    ("find this person's social media", "chat.clarify"),
    ("inspect this image", "chat.clarify"),
    ("scrape https://example.com/Case", "web.scrape"),
])
def test_single_chat_understands_conversation_and_tasks(text, capability):
    assert understand(text)[0] == capability


def test_model_proposals_cannot_run_shell_or_invent_capabilities():
    assert validate_proposal({"capability": "computer.command", "text": "run whoami"}) is None
    assert validate_proposal({"capability": "computer.command", "text": "shutdown"}) is None
    assert validate_proposal({"capability": "execute_python", "text": "print(1)"}) is None
    assert validate_proposal({"capability": "computer.command", "text": "open youtube in brave"})


@pytest.fixture
def dock(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "desktop"
    monkeypatch.setenv("SWEEP_DESKTOP_DIR", str(root))
    monkeypatch.setenv("SWEEP_CONTROLLER_DIR", str(tmp_path / "controller"))
    window = DockWindow(root)
    window.settings.setValue("notifications", False)
    yield app, window
    window.shutdown()
    window.deleteLater()
    app.processEvents()


def finish(app, window):
    deadline = time.monotonic() + 30
    while (window.process is not None or window.pending) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert window.process is None and not window.pending


def test_dock_expands_and_collapses_at_screen_top(dock):
    app, window = dock
    assert window.height() == 48
    assert not window.body.isVisible()
    window.reveal()
    app.processEvents()
    assert window.expanded and window.body.isVisible()
    assert window.height() >= 360
    assert 0 <= window.y() - window.screen().availableGeometry().y() <= 12
    window.collapse()
    assert window.height() == 48 and not window.expanded


def test_chat_sends_real_calculation_and_restores_thread(dock):
    app, window = dock
    window.reveal()
    window.prompt.setPlainText("calculate 7 * 6")
    window.submit()
    finish(app, window)
    assert "42" in window.transcript.toPlainText()
    assert window.task["state"] == "completed"
    thread = ChatStore(window.root).get(window.thread_id)
    assert thread["turns"][0]["text"] == "calculate 7 * 6"
    assert thread["turns"][1]["task_id"] == window.task["id"]
    assert window.context_messages()[-1]["role"] == "assistant"
    assert "42" in window.context_messages()[-1]["content"]


def test_enter_sends_and_shift_enter_inserts_newline(dock):
    app, window = dock
    window.reveal()
    window.prompt.setPlainText("calculate 1 + 2")
    QTest.keyClick(window.prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert "\n" in window.prompt.toPlainText()
    window.prompt.setPlainText("calculate 1 + 2")
    QTest.keyClick(window.prompt, Qt.Key.Key_Return)
    finish(app, window)
    assert "3" in window.transcript.toPlainText()


def test_new_chat_during_worker_does_not_leak_result_into_other_thread(dock):
    app, window = dock
    first = window.thread_id
    window.prompt.setPlainText("calculate 8 * 8")
    window.submit()
    window.new_workspace()
    second = window.thread_id
    assert second != first
    finish(app, window)
    assert window.chat_store.get(second)["turns"] == []
    assert "64" not in window.transcript.toPlainText()
    window.thread_id = first
    window.refresh_chat()
    assert "64" in window.transcript.toPlainText()


def test_image_transfer_denial_never_creates_worker(dock, monkeypatch, tmp_path):
    from PIL import Image
    app, window = dock
    path = tmp_path / "image.png"
    Image.new("RGB", (20, 20), "green").save(path)
    monkeypatch.setattr("sweep.desktop.dock.load_config", lambda root: {
        "vision_provider": "ollama", "vision_model": "vision:test", "key_present": {}})
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.No)
    window.attachments = [str(path)]
    window.prompt.setPlainText("Describe this image")
    window.submit()
    assert window.process is None
    assert window.task_store.list() == []
    assert "cancelled" in window.transcript.toPlainText()


def test_missing_image_stays_in_chat_and_asks_for_attachment(dock):
    app, window = dock
    window.prompt.setPlainText("where is this image from?")
    window.submit()
    assert window.process is None
    assert "Attach the image" in window.transcript.toPlainText()


def test_chat_store_blocks_path_escape_and_survives_malformed_file(tmp_path):
    store = ChatStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../outside")
    good = store.create()
    (store.root / ("a" * 32 + ".json")).write_text("[]")
    assert [thread["id"] for thread in store.list()] == [good["id"]]


def test_queued_chat_cannot_use_permission_from_another_thread(dock, monkeypatch):
    from sweep.desktop.providers import load_config
    app, window = dock
    origin = window.thread_id
    window.prompt.setPlainText("calculate 4 + 4")
    window.submit()
    window.prompt.setPlainText("Hello, how are you?")
    window.submit()
    queued_id = window.pending[0]
    assert window._task_threads[queued_id] == origin
    window.new_workspace()
    config = load_config(window.root)
    signature = json.dumps({key: config[key] for key in ("provider", "model", "ollama_base_url")}, sort_keys=True)
    window.settings.setValue("chat_permission/" + window.thread_id, signature)
    dialogs = []
    def decline(dialog):
        dialogs.append(dialog.text())
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "exec", decline)
    finish(app, window)
    assert len(dialogs) == 1 and "Hello, how are you?" in dialogs[0]
    assert window.task_store.get(queued_id)["state"] == "cancelled"
    assert not (window.task_store.directory(queued_id) / "events.jsonl").exists()


def test_new_chat_phrase_starts_separate_local_thread(dock):
    app, window = dock
    previous = window.thread_id
    window.prompt.setPlainText("new chat")
    window.submit()
    assert window.thread_id != previous
    assert window.process is None and not window.task_store.list()


def test_new_answers_scroll_into_view_without_moving_a_reader(dock):
    app, window = dock
    window.reveal()
    window.chat_store.append(window.thread_id, {"role": "assistant", "text": "A line of evidence\n" * 150})
    window.refresh_chat()
    app.processEvents()
    app.processEvents()
    bar = window.transcript.verticalScrollBar()
    assert bar.maximum() > 100 and bar.value() == bar.maximum()
    bar.setValue(0)
    window.refresh_chat()
    app.processEvents()
    assert bar.value() == 0


def test_chat_cleans_data_in_worker_and_previews_created_artifact(dock, tmp_path, monkeypatch):
    app, window = dock
    selected = tmp_path / "observations.csv"
    original = "name,total\n  Alpha  ,3\nAlpha,3\nBeta,7\n,\n"
    selected.write_text(original, encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "exec", lambda dialog: QMessageBox.StandardButton.Yes)
    window.set_attachment(selected)
    window.prompt.setPlainText("Clean this dataset and save as JSON")
    window.submit()
    finish(app, window)
    task = window.task_store.get(window.last_enqueued_id)
    assert task["capability"] == "data.transform" and task["state"] == "completed"
    result = json.loads((window.task_store.directory(task["id"]) / "result.json").read_text(encoding="utf-8"))
    assert result["input_rows"] == 4 and result["output_rows"] == 2
    assert selected.read_text(encoding="utf-8") == original
    assert window.transcript.findChild(QTableWidget) is not None
    assert any(button.text() == "View file ↗" for button in window.transcript.findChildren(QPushButton))
    seen = []

    def inspect(dialog):
        preview = dialog.findChild(QPlainTextEdit)
        assert preview.isReadOnly()
        assert json.loads(preview.toPlainText()) == [{"name": "Alpha", "total": "3"}, {"name": "Beta", "total": "7"}]
        seen.append(True)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.chat_link(QUrl(f"sweep:artifact/{task['id']}/0"))
    assert seen
