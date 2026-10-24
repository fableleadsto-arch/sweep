import json
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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


def test_queued_data_followup_uses_previous_output_in_real_worker(dock, tmp_path, monkeypatch):
    app, window = dock
    selected = tmp_path / "orders.csv"
    selected.write_text("item,quantity\nbolt,2\nbolt,2\nnut,3\n", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "exec", lambda dialog: QMessageBox.StandardButton.Yes)
    window.set_attachment(selected)
    window.prompt.setPlainText("Remove duplicate rows")
    window.submit()
    source_id = window.last_enqueued_id
    # Submit while extraction is still running; the references must survive the queue.
    window.prompt.setPlainText("Now convert it to JSON")
    window.submit()
    followup_id = window.last_enqueued_id
    assert followup_id != source_id
    finish(app, window)
    task = window.task_store.get(followup_id)
    assert task["state"] == "completed" and task["source_output"] is True
    assert task["source_tasks"] == [source_id] and task["thread_id"] == window.thread_id
    result = json.loads((window.task_store.directory(followup_id) / "result.json").read_text(encoding="utf-8"))
    assert result["input_rows"] == 2 and result["output_rows"] == 2


def test_multiple_documents_queue_real_extractions_and_scoped_answer(dock, tmp_path, monkeypatch):
    app, window = dock
    paths = [tmp_path / "invoice.txt", tmp_path / "packing.txt"]
    for path, text in zip(paths, ["Quantity: 12", "Quantity: 10"], strict=True):
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(window, "approve_task", lambda capability, *args, **kwargs: capability != "context.answer")
    window.set_attachments(paths)
    window.prompt.setPlainText("Compare the quantities in these documents")
    window.submit()
    answer_id = window.last_enqueued_id
    finish(app, window)
    answer = window.task_store.get(answer_id)
    assert answer["capability"] == "context.answer" and answer["state"] == "cancelled"
    assert len(answer["source_tasks"]) == 2
    results = []
    for source_id in answer["source_tasks"]:
        source = window.task_store.get(source_id)
        assert source["state"] == "completed" and source["thread_id"] == window.thread_id
        results.append(json.loads((window.task_store.directory(source_id) / "result.json").read_text(encoding="utf-8"))["text"])
    assert results == ["Quantity: 12", "Quantity: 10"]


def test_mixed_multiple_attachments_are_kept_for_correction(dock, tmp_path):
    app, window = dock
    window.set_attachments([tmp_path / "invoice.txt", tmp_path / "photo.jpg"])
    window.prompt.setPlainText("Compare these")
    window.submit()
    assert len(window.attachments) == 2
    assert not window.task_store.list()
    assert "one file" in window.transcript.toPlainText()


@pytest.mark.parametrize("payload", [b'[]', b'{"location":null}', b'{"message":"x","message":"y"}',
                                     b' ' * 2_000_001, None], ids=["array", "invalid-field", "duplicate", "oversized", "missing"])
def test_damaged_evidence_is_recoverable_and_not_used_as_chat_context(dock, monkeypatch, payload):
    app, window = dock
    task = window.task_store.create("computer.command", "calculate 6 * 7", thread_id=window.thread_id)
    window.task_store.update(task["id"], state="running", approved=True)
    window.task_store.complete(task["id"], {"message": "old evidence"})
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": task["id"]})
    path = window.task_store.directory(task["id"]) / "result.json"
    if payload is None:
        path.unlink()
    else:
        path.write_bytes(payload)
    window.refresh_chat()
    assert "missing or damaged" in window.transcript.toPlainText()
    assert "Try again" in [button.text() for button in window.transcript.findChildren(QPushButton)]
    assert window.context_messages() == []
    warnings, exports = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(window, "export_result", lambda: exports.append(True))
    window.chat_link(QUrl(f"sweep:export/{task['id']}"))
    assert warnings and not exports
    window.chat_link(QUrl(f"sweep:retry/{task['id']}"))
    retry_id = window.last_enqueued_id
    finish(app, window)
    assert retry_id != task["id"]
    assert window.task_store.result(retry_id)["message"].endswith("42")
    assert window.task_store.get(retry_id)["retry_of"] == task["id"]
    assert (path.read_bytes() if path.exists() else None) == payload


@pytest.mark.parametrize("action", ["retry", "export", "text", "propose", "cancel", "speech-setup"])
def test_chat_actions_cannot_target_another_conversation(dock, monkeypatch, action):
    app, window = dock
    other = window.chat_store.create()
    task = window.task_store.create("computer.command", "calculate 1 + 1", thread_id=other["id"])
    window.chat_store.append(other["id"], {"role": "assistant", "task_id": task["id"]})
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    window.chat_link(QUrl(f"sweep:{action}/{task['id']}"))
    assert warnings == ["This task does not belong to the open conversation."]
    assert window.task_store.get(task["id"])["state"] == "queued"
    assert len(window.task_store.list()) == 1 and window.process is None


def test_inconsistent_chat_record_cannot_expose_another_threads_evidence(dock, monkeypatch):
    app, window = dock
    other = window.chat_store.create()
    task = window.task_store.create("conversation", "Private synthetic note", thread_id=other["id"])
    window.task_store.update(task["id"], state="running", approved=True)
    window.task_store.complete(task["id"], {"message": "other-thread-secret"})
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": task["id"]})
    window.refresh_chat()
    assert "other-thread-secret" not in window.transcript.toPlainText()
    assert window.context_messages() == []
    assert "Try again" not in [button.text() for button in window.transcript.findChildren(QPushButton)]
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    window.chat_link(QUrl(f"sweep:text/{task['id']}"))
    assert warnings == ["This task belongs to a different conversation."]


@pytest.mark.parametrize("parent_state", ["failed", "interrupted"])
def test_review_recovery_retries_source_then_runs_followup_with_explicit_selection(dock, tmp_path, monkeypatch, parent_state):
    app, window = dock
    path = tmp_path / "orders.csv"
    path.write_text("item,quantity\nbolt,2\nbolt,2\nnut,3\n", encoding="utf-8")
    original = path.read_bytes()
    source = window.task_store.create("data.transform", "Remove duplicate rows", granted_paths=[str(path)], thread_id=window.thread_id)
    window.task_store.update(source["id"], state="running", approved=True)
    window.task_store.update(source["id"], state="failed", error="Earlier extraction failed")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": source["id"]})
    parent = window.task_store.create("data.transform", "Convert to JSON", thread_id=window.thread_id,
                                      source_tasks=[source["id"]], source_output=True)
    window.task_store.update(parent["id"], state="running", approved=True)
    window.task_store.update(parent["id"], state=parent_state)
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": parent["id"]})
    approvals = []
    monkeypatch.setattr(window, "approve_task", lambda *args, **kwargs: approvals.append(args[0]) or True)

    def choose_retry_source(dialog):
        run = next(button for button in dialog.findChildren(QPushButton) if button.text() == "Retry with these sources")
        assert not run.isEnabled()
        assert dialog.findChild(QComboBox).currentData() is None
        next(button for button in dialog.findChildren(QPushButton) if button.text() == "Retry source").click()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", choose_retry_source)
    window.chat_link(QUrl(f"sweep:retry/{parent['id']}"))
    source_retry = window.last_enqueued_id
    finish(app, window)
    assert window.task_store.result(source_retry)["output_rows"] == 2

    def choose_replacement(dialog):
        combo = dialog.findChild(QComboBox)
        assert combo.currentData() is None  # No silent selection of the newest result.
        combo.setCurrentIndex(combo.findData(source_retry))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", choose_replacement)
    window.chat_link(QUrl(f"sweep:retry/{parent['id']}"))
    retry = window.last_enqueued_id
    finish(app, window)
    record = window.task_store.get(retry)
    assert record["source_tasks"] == [source_retry] and record["retry_of"] == parent["id"]
    assert record["source_output"] is True and record["approved"] is False
    result = window.task_store.result(retry)
    assert result["input_rows"] == 2 and result["output_rows"] == 2
    assert approvals == ["data.transform", "data.transform"]
    assert window.task_store.get(source["id"])["state"] == "failed"
    assert window.task_store.get(parent["id"])["source_tasks"] == [source["id"]]
    assert path.read_bytes() == original


def test_recovery_cancel_does_not_queue_or_replay_a_task(dock, monkeypatch):
    app, window = dock
    source = window.task_store.create("computer.command", "open calculator", thread_id=window.thread_id)
    window.task_store.update(source["id"], state="running", approved=True)
    window.task_store.update(source["id"], state="failed")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": source["id"]})
    parent = window.task_store.create("context.answer", "Explain the result", thread_id=window.thread_id, source_tasks=[source["id"]])
    window.task_store.update(parent["id"], state="interrupted")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": parent["id"]})
    monkeypatch.setattr(QDialog, "exec", lambda dialog: QDialog.DialogCode.Rejected)
    window.chat_link(QUrl(f"sweep:retry/{parent['id']}"))
    assert len(window.task_store.list()) == 2 and window.process is None and not window.pending

    def choose_source(dialog):
        next(button for button in dialog.findChildren(QPushButton) if button.text() == "Retry source").click()
        return QDialog.DialogCode.Accepted

    approvals = []
    monkeypatch.setattr(QDialog, "exec", choose_source)
    monkeypatch.setattr(window, "approve_task", lambda *args, **kwargs: approvals.append(args[0]) or False)
    window.chat_link(QUrl(f"sweep:retry/{parent['id']}"))
    assert approvals == ["computer.command"]
    assert len(window.task_store.list()) == 2 and window.process is None and not window.pending
