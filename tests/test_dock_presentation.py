import json

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
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


def test_pending_scroll_callback_is_cancelled_when_transcript_is_destroyed(dock, monkeypatch):
    import sys

    import shiboken6

    app, window = dock
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: errors.append(args))
    transcript = ChatTranscript()
    transcript.render([{"role": "assistant", "text": "Saved evidence"}])
    shiboken6.delete(transcript)
    app.processEvents()
    assert not errors


def test_composer_handles_empty_multiline_and_attachment_messages(dock, tmp_path):
    app, window = dock
    window.reveal()
    app.processEvents()
    assert not window.run_button.isEnabled() and not window.cancel_button.isVisible()
    base = window.prompt.height()
    window.prompt.setPlainText("First line\nSecond line\nThird line")
    assert window.run_button.isEnabled() and window.prompt.height() > base
    window.prompt.clear()
    assert window.prompt.height() == base and not window.run_button.isEnabled()
    path = tmp_path / "note.txt"
    path.write_text("A local note", encoding="utf-8")
    window.set_attachment(path)
    assert window.run_button.isEnabled()
    window.clear_attachment()
    assert not window.run_button.isEnabled()


def test_narrow_cards_keep_secondary_actions_accessible(dock):
    app, window = dock
    window.reveal()
    window.resize(420, 620)
    actions = [("Read extracted text", "sweep:text/one", False),
               ("Export result", "sweep:export/one", False),
               ("Try again", "sweep:retry/one", False)]
    window.transcript.render([{"role": "assistant", "text": "A completed document", "actions": actions}])
    app.processEvents()
    buttons = window.transcript.findChildren(QPushButton)
    more = next(button for button in buttons if button.text() == "More")
    received = []
    window.transcript.anchorClicked.disconnect()
    window.transcript.anchorClicked.connect(lambda url: received.append(url.toString()))
    assert [action.text() for action in more.menu().actions()] == ["Export result", "Try again"]
    more.menu().actions()[1].trigger()
    assert received == ["sweep:retry/one"]
    assert window.transcript.horizontalScrollBar().maximum() == 0


def test_streaming_preserves_completed_cards_and_text_selection(dock):
    app, window = dock
    transcript = ChatTranscript()
    completed = {"role": "assistant", "text": "Keep this selected passage.",
                 "columns": ["item"], "rows": [["bolt"]]}
    transcript.render([completed, {"role": "assistant", "state": "running", "text": "Starting"}])
    original = transcript.layout.itemAt(0).layout()
    table = transcript.findChild(QTableWidget)
    label = next(item for item in transcript.findChildren(QLabel) if "Keep this" in item.text())
    label.setSelection(0, 4)
    for index in range(25):
        transcript.render([completed, {"role": "assistant", "state": "running", "text": f"Answer {index}"}])
        app.processEvents()
    assert transcript.layout.itemAt(0).layout() is original
    assert transcript.findChild(QTableWidget) is table
    assert label.selectedText() == "Keep"
    assert transcript.toPlainText().endswith("Answer 24")
    # A new conversation must remove all previous cards, even after streaming.
    transcript.render([])
    app.processEvents()
    assert "Keep this" not in transcript.toPlainText() and "Answer 24" not in transcript.toPlainText()
    transcript.render([{"role": "user", "text": "New chat"}])
    assert transcript.layout.count() == 2
    transcript.deleteLater()


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


@pytest.mark.parametrize("count", [1, 2, 3])
def test_small_generated_file_groups_remain_visible(dock, count):
    app, window = dock
    transcript = ChatTranscript()
    artifacts = [{"name": f"report-{index}.txt", "type": "txt", "url": f"sweep:artifact/abc/{index}"}
                 for index in range(count)]
    transcript.render([{"role": "assistant", "artifacts": artifacts}])
    cards = transcript.findChildren(QWidget, "artifactCard")
    assert len(cards) == count
    assert all(card.isVisibleTo(transcript) for card in cards)
    assert not any(button.property("artifactToggle") for button in transcript.findChildren(QPushButton))
    transcript.deleteLater()


def test_large_generated_file_group_expands_folds_and_keeps_action_urls(dock):
    app, window = dock
    transcript = ChatTranscript()
    artifacts = [{"name": f"frame-{index}.jpg", "type": "jpg", "url": f"sweep:artifact/abc/{index}"}
                 for index in range(10)]
    artifacts[5]["name"] = '<img src="file:///private">.jpg'
    message = {"role": "assistant", "text": "Sampled frames are ready.", "artifacts": artifacts}
    transcript.render([message])
    cards = transcript.findChildren(QWidget, "artifactCard")
    toggle = next(button for button in transcript.findChildren(QPushButton) if button.property("artifactToggle"))
    assert len(cards) == 10
    assert sum(card.isVisibleTo(transcript) for card in cards) == 2
    assert toggle.isCheckable() and not toggle.isChecked()
    assert toggle.text() == toggle.accessibleName() == "Show 8 more files"
    emitted = []
    transcript.anchorClicked.connect(lambda url: emitted.append(url.toString()))
    toggle.click()
    assert toggle.isChecked() and toggle.text() == "Hide 8 extra files"
    assert all(card.isVisibleTo(transcript) for card in cards)
    for card, artifact in zip(cards, artifacts, strict=True):
        name = card.findChildren(QLabel)[0]
        assert name.text() == artifact["name"] and name.textFormat() == Qt.TextFormat.PlainText
        card.findChild(QPushButton).click()
    assert emitted == [artifact["url"] for artifact in artifacts]
    toggle.click()
    assert not toggle.isChecked() and toggle.text() == "Show 8 more files"
    assert sum(card.isVisibleTo(transcript) for card in cards) == 2
    assert emitted == [artifact["url"] for artifact in artifacts]
    transcript.deleteLater()


def test_expanded_generated_files_stay_open_when_chat_refreshes(dock):
    app, window = dock
    transcript = ChatTranscript()
    artifacts = [{"name": f"file-{index}.txt", "url": f"sweep:artifact/abc/{index}"} for index in range(4)]
    message = {"role": "assistant", "artifacts": artifacts}
    transcript.render([message])
    toggle = next(button for button in transcript.findChildren(QPushButton) if button.property("artifactToggle"))
    toggle.click()
    transcript.render([message, {"role": "user", "text": "Explain the first frame."}])
    current_toggle = next(button for button in transcript.findChildren(QPushButton)
                          if button.property("artifactToggle") and button.isVisibleTo(transcript))
    assert current_toggle.isChecked()
    assert sum(card.isVisibleTo(transcript) for card in transcript.findChildren(QWidget, "artifactCard")) == 4
    current_toggle.click()
    assert sum(card.isVisibleTo(transcript) for card in transcript.findChildren(QWidget, "artifactCard")) == 2
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
    window.task_store.update(task["id"], state="running", approved=True)
    window.task_store.update(task["id"], state="completed")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": task["id"]})
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
    assert window.owl.state == "idle"
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
    window.task_store.update(task["id"], state="running", approved=True)
    window.task_store.update(task["id"], state="completed")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": task["id"]})
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


@pytest.mark.parametrize("capability", ["conversation", "context.answer"])
def test_chat_answers_have_copy_and_sources_without_task_controls(dock, capability):
    app, window = dock
    task = window.task_store.create(capability, "Can we talk this through?")
    result = {"message": "Of course. " + "Let’s explore it. " * 100}
    value = window._result_message(task, result)
    assert "state_label" not in value and not value["actions"]
    window.transcript.render([value])
    card = window.transcript.layout.itemAt(0).layout().itemAt(0).widget()
    buttons = [button.text() for button in card.findChildren(QPushButton)]
    assert buttons == ["Copy"]
    assert not window.transcript.findChildren(QLabel, "taskState")
    result["context_sources"] = [{"task_id": "source", "excerpt": "Saved evidence"}]
    result["proposal"] = {"capability": "computer.command", "text": "open calculator"}
    value = window._result_message(task, result)
    assert [action[0] for action in value["actions"]] == ["Review proposed task", "Sources used"]


def test_native_chat_stream_keeps_its_card_without_task_animation(dock):
    app, window = dock
    window.reveal()
    task = window.task_store.create("conversation", "Talk with me", thread_id=window.thread_id)
    window.task = window.task_store.update(task["id"], state="running")
    window.chat_store.append(window.thread_id, {"role": "assistant", "task_id": task["id"]})
    window.activity_summary.setText("Internal task worker started")
    window.refresh_chat()
    assert window.transcript.toPlainText() == "Thinking…"
    card = window.transcript.findChild(QWidget, "assistantMessage")
    text_label = card.findChild(QLabel, "messageText")
    for number in range(25):
        window.accept_event({"kind": "response.preview", "text": f"Reply {number}"})
        app.processEvents()
        assert window.transcript.findChild(QWidget, "assistantMessage") is card
        assert card.findChild(QLabel, "messageText") is text_label
    assert window.transcript.toPlainText() == "Reply 24"
    assert not card.findChildren(QProgressBar)
    assert not card.findChildren(QLabel, "taskTitle")
    assert not card.findChildren(QLabel, "taskState")
    assert not window.body_effect.isEnabled()
    window.task = window.task_store.update(task["id"], state="failed", error="Local processing is unavailable. Open Settings to start it.")
    window.refresh_chat()
    app.processEvents()
    assert "Open Settings" in window.transcript.toPlainText()
    assert "Try again" in [button.text() for button in window.transcript.findChildren(QPushButton) if button.isVisibleTo(window.transcript)]


def test_wrapped_composer_grows_and_attachment_controls_fit_narrow_dock(dock, tmp_path):
    app, window = dock
    window.reveal()
    window.resize(380, 620)
    app.processEvents()
    base = window.prompt.height()
    window.prompt.setPlainText("I want to talk about a difficult day and figure out what to do next. " * 5)
    app.processEvents()
    assert window.prompt.document().blockCount() == 1
    assert window.prompt.height() > base
    assert window.prompt.height() <= 24 + 5 * window.prompt.fontMetrics().lineSpacing()
    path = tmp_path / "note.txt"
    path.write_text("A note", encoding="utf-8")
    window.set_attachment(path)
    app.processEvents()
    assert window.width() == 380
    assert window.clear_attachment_button.isVisible()
    window.prompt.clear()
    app.processEvents()
    assert window.prompt.height() == base


def test_settings_saves_and_restores_quiet_resource_profile(dock, monkeypatch):
    from sweep.desktop.providers import load_config

    app, window = dock

    def choose_quiet(dialog):
        profile = dialog.findChild(QComboBox, "resourceProfile")
        assert profile.currentData() == "balanced"
        profile.setCurrentIndex(profile.findData("efficient"))
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save).click()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", choose_quiet)
    window.show_settings()
    assert load_config(window.root)["resource_profile"] == "efficient"

    def inspect_saved(dialog):
        assert dialog.findChild(QComboBox, "resourceProfile").currentData() == "efficient"
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect_saved)
    window.show_settings()
