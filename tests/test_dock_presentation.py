import json

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QWidget,
)

from sweep.desktop.dock import DockWindow
from sweep.desktop.presentation import ChatTranscript, granted_thumbnail, public_source, rich_text


@pytest.fixture
def dock(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "desktop"
    monkeypatch.setenv("SWEEP_DESKTOP_DIR", str(root))
    window = DockWindow(root)
    window.settings.setValue("reduce_motion", True)
    window.settings.setValue("notifications", False)
    yield app, window
    window.shutdown()
    window.deleteLater()
    app.processEvents()


def test_message_markup_never_loads_source_html():
    rendered = rich_text('<img src="file:///private"> **summary** `code`')
    assert "<img" not in rendered and "&lt;img" in rendered
    assert "<b>summary</b>" in rendered


@pytest.mark.parametrize("url", ["https://[invalid", "file:///private", "https://user:password@example.com", "https:///missing-host"])
def test_source_cards_reject_malformed_and_credential_links(url):
    assert public_source(url) is None


def test_thumbnails_require_exact_grant_and_stay_bounded(dock, tmp_path):
    app, window = dock
    path = tmp_path / "selected.png"
    image = QImage(1600, 800, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.green)
    assert image.save(str(path))
    assert granted_thumbnail(str(path), []) is None
    assert granted_thumbnail(str(path), [str(tmp_path)]) is None
    preview = granted_thumbnail(str(path), [str(path)], 300, 200)
    assert preview.width() == 300 and preview.height() == 150
    window.set_attachment(path)
    assert not window.attachment_preview.pixmap().isNull()
    window.clear_attachment()
    assert window.attachments == [] and window.attachment_preview.isHidden()


def test_cards_include_sources_data_and_created_files(dock):
    app, window = dock
    transcript = ChatTranscript()
    transcript.render([{"role": "assistant", "text": "Prepared the table.",
                        "sources": [{"title": "Documentation", "url": "https://example.com/docs"}],
                        "columns": ["name", "total"], "rows": [["A", 3]],
                        "artifacts": [{"name": "cleaned.csv", "type": "csv", "url": "sweep:artifact/abc/0"}]}])
    labels = " ".join(item.text() for item in transcript.findChildren(QLabel))
    assert "Documentation" in labels and "example.com" in labels
    assert "cleaned.csv" in labels and "Created on this device" in labels
    assert "Prepared the table." in transcript.toPlainText()
    table = transcript.findChild(QTableWidget)
    assert table.height() < 120
    copy = next(button for button in transcript.findChildren(QPushButton) if button.text() == "Copy")
    copy.click()
    assert QApplication.clipboard().text() == "Prepared the table."
    transcript.deleteLater()


def test_settings_hides_implementation_and_has_no_cloud_controls(dock, monkeypatch):
    app, window = dock
    checked = []

    def inspect(dialog):
        dialog.show()
        app.processEvents()
        advanced = dialog.findChild(QWidget, "advancedLocalSettings")
        assert advanced.isHidden()
        assert all(not field.isVisible() for field in advanced.findChildren(QLineEdit))
        labels = " ".join(item.text() for item in dialog.findChildren(QLabel) if item.isVisible())
        assert "On this device" in labels and "API key" not in labels
        assert "Ollama" not in labels and "Qwen" not in labels
        checked.append(True)
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.show_settings()
    assert checked


def test_history_searches_message_content_and_single_click_opens(dock, monkeypatch):
    app, window = dock
    original = window.thread_id
    window.chat_store.append(original, {"role": "user", "text": "Planning"})
    window.chat_store.append(original, {"role": "assistant", "text": "Bring the telescope."})
    window.new_workspace()

    def inspect(dialog):
        field = dialog.findChild(QLineEdit)
        listing = dialog.findChild(QListWidget)
        field.setText("telescope")
        visible = [listing.item(index) for index in range(listing.count()) if not listing.item(index).isHidden()]
        assert len(visible) == 1
        listing.itemClicked.emit(visible[0])
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.show_history()
    assert window.thread_id == original and "telescope" in window.transcript.toPlainText()


def test_artifact_links_reject_outside_files_and_preview_csv_safely(dock, tmp_path, monkeypatch):
    app, window = dock
    task = window.task_store.create("data.transform", "clean this table")
    directory = window.task_store.directory(task["id"])
    outside = tmp_path / "private.csv"
    outside.write_text("secret", encoding="utf-8")
    result_path = directory / "result.json"
    result_path.write_text(json.dumps({"artifacts": [{"path": str(outside)}]}), encoding="utf-8")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.chat_link(QUrl(f"sweep:artifact/{task['id']}/0"))
    assert warnings and "cannot be opened" in warnings[0]
    (directory / "artifacts").mkdir()
    created = directory / "artifacts" / "cleaned.csv"
    created.write_text('name,value\nexample,=1+1\n', encoding="utf-8")
    result_path.write_text(json.dumps({"artifacts": [{"path": str(created)}]}), encoding="utf-8")
    seen = []

    def inspect(dialog):
        preview = dialog.findChild(QPlainTextEdit)
        assert preview.isReadOnly() and "=1+1" in preview.toPlainText()
        seen.append(True)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.chat_link(QUrl(f"sweep:artifact/{task['id']}/0"))
    assert seen


def test_data_transform_permission_describes_local_copy(dock, monkeypatch):
    app, window = dock
    seen = []

    def inspect(dialog):
        seen.append(dialog.text())
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "exec", inspect)
    assert not window.approve_task("data.transform", "Clean this dataset")
    assert "local storage" in seen[0] and "original file stays unchanged" in seen[0]


def test_empty_chat_starts_at_top_and_fits_default_dock(dock):
    app, window = dock
    window.reveal()
    app.processEvents()
    app.processEvents()
    bar = window.transcript.verticalScrollBar()
    assert bar.value() == 0
    assert bar.maximum() == 0


def test_replaced_cards_are_hidden_before_deferred_deletion(dock):
    app, window = dock
    window.reveal()
    app.processEvents()
    welcome = window.transcript.findChild(QWidget, "welcome")
    assert welcome.isVisible()
    window.transcript.render([{"role": "assistant", "text": "Ready to help."}])
    assert not welcome.isVisible()


def test_document_summary_keeps_extracted_text_in_read_only_viewer(dock, monkeypatch):
    app, window = dock
    task = window.task_store.create("documents.inspect", "Summarize this document")
    raw = '<img src="file:///private">\n' * 100
    result = {"title": "Report", "summary": "The project starts Friday.",
              "message": "The project starts Friday.", "text": raw}
    value = window._result_message(task, result)
    assert value["text"] == result["summary"]
    assert any(action[0] == "Read extracted text" for action in value["actions"])
    seen = []

    def inspect(dialog):
        view = dialog.findChild(QPlainTextEdit)
        assert view.isReadOnly() and view.toPlainText() == raw
        seen.append(True)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.show_extracted_text(result)
    assert seen


def test_streamed_response_stays_with_its_task_and_ignores_late_events(dock):
    app, window = dock
    origin = window.thread_id
    task = window.task_store.create("conversation", "Explain the project")
    window.task = window.task_store.update(task["id"], state="running")
    window.viewed_task_id = task["id"]
    window.chat_store.append(origin, {"role": "assistant", "task_id": task["id"]})
    window.accept_event({"kind": "response.preview", "text": "A response is arriving"})
    assert "A response is arriving" in window.transcript.toPlainText()
    assert window.owl.state == "responding"
    window.new_workspace()
    window.accept_event({"kind": "response.preview", "text": "Updated response"})
    assert "Updated response" not in window.transcript.toPlainText()
    window.thread_id = origin
    window.refresh_chat()
    assert "Updated response" in window.transcript.toPlainText()
    window.task = window.task_store.update(task["id"], state="cancelled")
    window.accept_event({"kind": "response.preview", "text": "This late event must not appear"})
    assert window._response_previews[task["id"]] == "Updated response"
    window.refresh_chat()
    assert window.owl.state == "idle"


def test_owl_reflects_actual_task_capability_and_terminal_state(dock):
    app, window = dock
    task = window.task_store.create("web.search", "Python documentation")
    window.task = window.task_store.update(task["id"], state="running")
    window.update_owl_state()
    assert window.owl.active and "searching the web" in window.owl.accessibleName()
    window.task = window.task_store.update(task["id"], state="failed", error="No sources")
    window.update_owl_state()
    assert not window.owl.active and window.owl.state == "attention"


def test_completed_answer_replaces_preview_and_restores_from_task_record(dock):
    app, window = dock
    task = window.task_store.create("conversation", "Explain a binary tree")
    window.task = window.task_store.update(task["id"], state="running")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": task["id"]})
    window.accept_event({"kind": "response.preview", "text": "An unfinished draft"})
    result = {"message": "A binary tree has at most two children per node."}
    (window.task_store.directory(task["id"]) / "result.json").write_text(
        json.dumps(result), encoding="utf-8")
    window.task = window.task_store.update(task["id"], state="completed")
    window.refresh_chat()
    assert result["message"] in window.transcript.toPlainText()
    assert "unfinished draft" not in window.transcript.toPlainText()
    assert window.owl.state == "success"
    window._response_previews.clear()
    window.refresh_chat()
    assert result["message"] in window.transcript.toPlainText()
    assert window.context_messages()[-1]["content"] == result["message"]


def test_markdown_artifacts_open_as_inert_text(dock, monkeypatch, tmp_path):
    app, window = dock
    task = window.task_store.create("web.research", "Research Python")
    directory = window.task_store.directory(task["id"])
    (directory / "artifacts").mkdir()
    report = directory / "artifacts" / "report.md"
    raw = '# Research\n<img src="https://example.com/tracker">'
    report.write_text(raw, encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"artifacts": [{"path": str(report)}]}), encoding="utf-8")
    seen = []

    def inspect(dialog):
        preview = dialog.findChild(QPlainTextEdit)
        assert preview.isReadOnly() and preview.toPlainText() == raw
        seen.append(True)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.chat_link(QUrl(f"sweep:artifact/{task['id']}/0"))
    assert seen


def test_inspect_locally_keeps_image_request_in_metadata_mode(dock, monkeypatch, tmp_path):
    app, window = dock
    path = tmp_path / "sample.png"
    picture = QImage(10, 10, QImage.Format.Format_RGB32)
    picture.fill(Qt.GlobalColor.green)
    picture.save(str(path))
    monkeypatch.setattr("sweep.desktop.dock.load_config", lambda root: {"vision_model": "installed"})
    started = []
    monkeypatch.setattr(window, "start_task", lambda capability, *args, **kwargs: started.append(capability) and False)
    window.set_attachment(path)
    window.prompt.setPlainText("Inspect locally")
    window.submit()
    assert started == ["images.inspect"]
