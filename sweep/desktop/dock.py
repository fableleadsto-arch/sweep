"""A compact native chat dock; the existing worker queue stays behind the chat."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizeGrip,
    QTableWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .chat import ChatStore, result_text, understand, validate_proposal
from .owl import Owl
from .platform import set_startup, startup_enabled
from .presentation import ChatTranscript, granted_thumbnail, public_source
from .providers import load_config, save_config
from .window import SweepWindow, label

DOCK_STYLE = """
QMainWindow { background: transparent; }
QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #253f36; }
QFrame#dock { background: #fafbf7; border: 1px solid #becdc2; border-radius: 16px; }
QFrame#handle { background: #183c33; border: none; border-radius: 13px; }
QFrame#handle QLabel { color: #f2f7ed; background: transparent; }
QFrame#handle QPushButton { background: transparent; color: #eef6ea; border: none; padding: 7px; }
QFrame#handle QPushButton:hover { background: #305448; }
QFrame#handle QPushButton:checked { background: #416b58; }
QPushButton { background: #eef2e8; border: 1px solid #d7dfd0; border-radius: 7px; padding: 7px 10px; }
QPushButton:hover { background: #e1eadb; }
QPushButton:disabled { color: #8c998e; }
QPushButton#primary { background: #234d3f; color: white; border: none; font-weight: 600; }
QTextBrowser { background: transparent; border: none; padding: 8px; selection-background-color: #cfe0c8; }
QPlainTextEdit { background: #ffffff; border: 1px solid #d1dccd; border-radius: 9px; padding: 8px; font-size: 14px; }
QLabel#muted { color: #718374; }
QDialog { background: #fafaf6; }
QLineEdit, QComboBox { background: white; border: 1px solid #cdd8c8; border-radius: 5px; padding: 7px; }
QListWidget { background: white; border: none; }
QListWidget::item { padding: 12px; }
QListWidget::item:selected { background: #dce8d4; color: #203d36; }
QToolTip { background: #183c33; color: white; padding: 5px; border: none; }
QScrollArea#transcript, QWidget#chatContent { background: transparent; border: none; }
QLabel#messageText { font-size: 14px; line-height: 1.5; color: #263c33; }
QLabel#messageRole, QLabel#eyebrow { color: #728279; font-size: 10px; font-weight: 700; }
QFrame#userMessage { background: #e8efe3; border: 1px solid #dfe8d7; border-radius: 13px; }
QFrame#assistantMessage { background: #ffffff; border: 1px solid #e4e9e1; border-radius: 13px; }
QFrame#sourceCard, QFrame#artifactCard { background: #f5f7f2; border: 1px solid #e6eae0; border-radius: 9px; }
QLabel#taskTitle { font-size: 15px; font-weight: 600; }
QLabel#taskState { color: #557361; font-size: 11px; padding-left: 9px; }
QLabel#taskState[state="failed"], QLabel#taskState[state="interrupted"] { color: #995c42; }
QLabel#attachmentCaption { color: #637769; font-size: 11px; }
QLabel#sourceSnippet { color: #617368; font-size: 12px; }
QLabel#welcomeTitle { font-size: 29px; font-weight: 600; color: #254b3d; }
QLabel#welcomeDescription { font-size: 14px; color: #6a7e71; }
QPushButton#suggestion { text-align: left; background: #f0f4ea; border: 1px solid #e0e7d8; border-radius: 10px; padding: 12px 14px; }
QPushButton#suggestion:hover { background: #e4eddd; border-color: #bbccaf; }
QPushButton#cardAction { background: transparent; color: #446956; border: none; padding: 5px 3px; font-size: 11px; }
QPushButton#cardAction:hover { background: #edf3e8; }
QPushButton#cardPrimary { background: #315b48; color: white; border: none; font-size: 12px; }
QPushButton#advanced { text-align: left; background: transparent; border: none; color: #62766a; }
QLabel#settingsTitle { font-size: 22px; font-weight: 600; }
QLabel#settingsSection { font-size: 14px; font-weight: 600; margin-top: 8px; }
QTableWidget#dataPreview { border: 1px solid #e0e7d8; background: white; font-size: 12px; selection-background-color: #dfead8; }
QHeaderView::section { background: #eef3e8; border: none; padding: 6px; }
QScrollBar:vertical { width: 7px; background: transparent; margin: 3px 0; }
QScrollBar::handle:vertical { background: #cad5c6; border-radius: 3px; min-height: 26px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


class ChatInput(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class DockWindow(SweepWindow):
    def __init__(self, root: Path):
        self.chat_store = ChatStore(root)
        self.attachments = []
        self.expanded = False
        self._drag_offset = None
        self.thread_id = None
        self._task_threads = {}
        self._scroll_to_latest = True
        self._preview_cache = {}
        self._response_previews = {}
        super().__init__(root)
        self.setWindowTitle("Sweep")
        self.setStyleSheet(DOCK_STYLE)
        self.setMinimumSize(380, 48)
        self.pinned = self.settings.value("dock_pinned", False, type=bool)
        self.pin.setChecked(self.pinned)
        threads = self.chat_store.list()
        saved = self.settings.value("active_chat", "")
        self.thread_id = next((thread["id"] for thread in threads if thread["id"] == saved), None)
        if not self.thread_id:
            self.thread_id = threads[0]["id"] if threads else self.chat_store.create()["id"]
        self.settings.setValue("active_chat", self.thread_id)
        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(lambda: self.expand(False))
        self.collapse_timer = QTimer(self)
        self.collapse_timer.setSingleShot(True)
        self.collapse_timer.timeout.connect(self.auto_collapse)
        QShortcut(QKeySequence("Escape"), self, activated=self.collapse)
        self.collapse()
        self.refresh_chat()

    def build_ui(self):
        # Set native styles before the base class creates the window handle for
        # the global shortcut. Changing them afterwards leaves stale Windows
        # frame margins on the first show and invalidates hotkey registration.
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # Keep the tested task engine's detail widgets internal; chat is the only visible workspace.
        hidden = QWidget(self)
        hidden.hide()
        self.history = QListWidget(hidden)
        self.retry_button = QPushButton(hidden)
        self.cancel_queued_button = QPushButton(hidden)
        self.queue_summary = QLabel(hidden)
        self.result_view = QTextBrowser(hidden)
        self.sources = QListWidget(hidden)
        self.table = QTableWidget(hidden)
        self.image_view = QLabel(hidden)
        self.tabs = QTabWidget(hidden)
        for widget in (self.result_view, self.sources, self.table, self.image_view):
            self.tabs.addTab(widget, "")
        self.activity = QListWidget(hidden)
        self.activity_summary = QLabel(hidden)
        self.export_button = QPushButton(hidden)
        self.shortcut_label = QLabel(hidden)
        self.mode = QComboBox(hidden)
        self.mode.addItem("Automatic", "auto")

        self.frame = QFrame()
        self.frame.setObjectName("dock")
        outer = QVBoxLayout(self.frame)
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(0)
        self.setCentralWidget(self.frame)
        self.handle = QFrame()
        self.handle.setObjectName("handle")
        self.handle.setFixedHeight(38)
        header = QHBoxLayout(self.handle)
        header.setContentsMargins(10, 0, 6, 0)
        self.owl = Owl()
        self.owl.setFixedSize(28, 28)
        header.addWidget(self.owl)
        self.open_button = QPushButton("Sweep  ·  Ask anything")
        self.open_button.setAccessibleName("Open Sweep conversation")
        self.open_button.clicked.connect(self.reveal)
        header.addWidget(self.open_button, 1)
        for title, tip, callback in (("＋", "New chat", self.new_workspace), ("History", "Conversation history", self.show_history),
                                      ("Settings", "Local processing and preferences", self.show_settings)):
            button = QPushButton(title)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.clicked.connect(callback)
            header.addWidget(button)
        self.pin = QPushButton("Pin")
        self.pin.setCheckable(True)
        self.pin.setToolTip("Keep the chat expanded")
        self.pin.setAccessibleName("Keep chat expanded")
        self.pin.toggled.connect(self.toggle_pin)
        header.addWidget(self.pin)
        fold = QPushButton("−")
        fold.setToolTip("Collapse dock (Esc)")
        fold.setAccessibleName("Collapse Sweep")
        fold.clicked.connect(self.collapse)
        header.addWidget(fold)
        outer.addWidget(self.handle)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(12, 8, 12, 5)
        body.setSpacing(8)
        self.transcript = ChatTranscript()
        self.transcript.anchorClicked.connect(self.chat_link)
        self.transcript.exampleRequested.connect(self.use_example)
        body.addWidget(self.transcript, 1)
        self.attachment_preview = QLabel()
        self.attachment_preview.setAccessibleName("Selected attachment preview")
        self.attachment_preview.hide()
        body.addWidget(self.attachment_preview)
        self.attachment_label = label("", "muted")
        self.attachment_label.hide()
        body.addWidget(self.attachment_label)
        self.prompt = ChatInput()
        self.prompt.setAccessibleName("Message Sweep")
        self.prompt.setPlaceholderText("Ask anything, or tell Sweep what to do…")
        self.prompt.setFixedHeight(80)
        self.prompt.submitted.connect(self.submit)
        body.addWidget(self.prompt)
        row = QHBoxLayout()
        attach = QPushButton("＋ Attach")
        attach.clicked.connect(self.choose_file)
        row.addWidget(attach)
        self.clear_attachment_button = QPushButton("Clear attachment")
        self.clear_attachment_button.clicked.connect(self.clear_attachment)
        self.clear_attachment_button.hide()
        row.addWidget(self.clear_attachment_button)
        self.status = label("Local processing", "muted")
        row.addWidget(self.status, 1)
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)
        row.addWidget(self.cancel_button)
        self.run_button = QPushButton("Send ↑")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.submit)
        row.addWidget(self.run_button)
        body.addLayout(row)
        footer = QHBoxLayout()
        footer.addWidget(label("Enter to send · Shift+Enter for a new line · Ctrl+Alt+Space to open", "muted"), 1)
        footer.addWidget(QSizeGrip(self))
        body.addLayout(footer)
        outer.addWidget(self.body, 1)
        self.body_effect = QGraphicsOpacityEffect(self.body)
        self.body.setGraphicsEffect(self.body_effect)
        self.body_effect.setOpacity(1)
        self.reveal_animation = QPropertyAnimation(self.body_effect, b"opacity", self)
        self.reveal_animation.setDuration(160)
        self.reveal_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def use_example(self, request):
        if request == "__attach__":
            self.choose_file()
        else:
            self.prompt.setPlainText(request)
            self.prompt.setFocus()
            cursor = self.prompt.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.prompt.setTextCursor(cursor)

    def expand(self, focus=True):
        if not self.expanded:
            self.expanded = True
            self.setMaximumHeight(16777215)
            screen = QApplication.screenAt(self.pos()) or QApplication.primaryScreen()
            area = screen.availableGeometry()
            self.setMinimumHeight(min(360, area.height() - 20))
            self.body.show()
            self.resize(min(self.settings.value("dock_width", 720, type=int), area.width() - 24),
                        min(self.settings.value("dock_height", 620, type=int), area.height() - 24))
            self.place_at_top(screen)
            if not self.settings.value("reduce_motion", False, type=bool):
                self.reveal_animation.stop()
                self.reveal_animation.setStartValue(0)
                self.reveal_animation.setEndValue(1)
                self.reveal_animation.start()
        self.show()
        if focus:
            self.raise_()
            self.activateWindow()
            self.prompt.setFocus()

    def place_at_top(self, screen=None):
        screen = screen or QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        self.move(area.x() + (area.width() - self.width()) // 2, area.y() + 8)

    def collapse(self):
        if self.expanded:
            self.settings.setValue("dock_width", self.width())
            self.settings.setValue("dock_height", self.height())
        self.expanded = False
        self.reveal_animation.stop()
        self.body_effect.setOpacity(1)
        self.body.hide()
        self.setMinimumHeight(48)
        self.setMaximumHeight(48)
        self.resize(540, 48)
        self.place_at_top()

    def reveal(self):
        self.expand(True)

    def toggle_pin(self, checked):
        self.pinned = checked
        self.settings.setValue("dock_pinned", checked)
        if checked and hasattr(self, "body"):
            self.expand(True)

    def enterEvent(self, event):
        if hasattr(self, "hover_timer"):
            self.collapse_timer.stop()
            if not self.expanded:
                self.hover_timer.start(220)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if hasattr(self, "hover_timer"):
            self.hover_timer.stop()
            self.collapse_timer.start(900)
        super().leaveEvent(event)

    def auto_collapse(self):
        if not self.pinned and not self.underMouse() and not self.isActiveWindow() and QApplication.activeModalWidget() is None:
            self.collapse()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 48:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            screen = QApplication.screenAt(event.globalPosition().toPoint()) or QApplication.primaryScreen()
            area = screen.availableGeometry()
            x = event.globalPosition().toPoint().x() - self._drag_offset.x()
            self.move(max(area.left(), min(x, area.right() - self.width())), area.top() + 8)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def clear_attachment(self):
        self.attachments = []
        self.attachment_label.clear()
        self.attachment_label.hide()
        self.attachment_preview.clear()
        self.attachment_preview.hide()
        self.clear_attachment_button.hide()

    def set_attachment(self, path):
        selected = str(Path(path).resolve())
        self.attachments = [selected]
        self.clear_attachment_button.show()
        self.attachment_label.setTextFormat(Qt.TextFormat.PlainText)
        self.attachment_label.setText(Path(selected).name + " · ready for local processing")
        self.attachment_label.show()
        preview = granted_thumbnail(selected, self.attachments, 420, 145)
        self.attachment_preview.setVisible(preview is not None)
        if preview is not None:
            self.attachment_preview.setPixmap(preview)

    def choose_file(self):
        self.reveal()
        path, _ = QFileDialog.getOpenFileName(self, "Attach an image or file")
        if path:
            self.set_attachment(path)
            self.prompt.setFocus()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_file():
                self.set_attachment(url.toLocalFile())
                self.reveal()
                event.acceptProposedAction()
                break

    def context_messages(self):
        messages = []
        for turn in self.chat_store.get(self.thread_id)["turns"][-24:]:
            text = turn.get("text", "")
            if turn.get("task_id"):
                try:
                    task = self.task_store.get(turn["task_id"])
                    if task["state"] != "completed":
                        continue
                    result = json.loads((self.task_store.directory(task["id"]) / "result.json").read_text(encoding="utf-8"))
                    text = result_text(result)
                except (OSError, ValueError):
                    continue
            messages.append({"role": turn["role"], "content": str(text)[:12000]})
        return messages[-12:]

    def submit(self):
        text = self.prompt.toPlainText().strip()
        if not text and self.attachments:
            text = "Inspect this image" if Path(self.attachments[0]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"} else "Inspect this file"
        if not text:
            return
        if len(text) > 60_000:
            QMessageBox.information(self, "Message too long", "Send up to 60,000 characters at a time, or attach a file for local inspection.")
            return
        self.reveal()
        self._scroll_to_latest = True
        context = self.context_messages()
        attachments = list(self.attachments)
        capability, request = understand(text, attachments)
        if capability == "chat.new":
            self.new_workspace()
            return
        self.chat_store.append(self.thread_id, {"role": "user", "text": text, "attachments": [Path(path).name for path in attachments]})
        self.prompt.clear()
        self.clear_attachment()
        if capability == "chat.clarify":
            self.chat_store.append(self.thread_id, {"role": "assistant", "text": request})
            self.refresh_chat()
            return
        if capability == "images.inspect":
            from .images import identity_request
            config = load_config(self.root)
            if identity_request(text):
                self.chat_store.append(self.thread_id, {"role": "assistant", "text": "I can describe the image's visible content, text and metadata. To find public profiles, provide a name or handle; I can't identify a private person or match their accounts from a photo."})
                self.refresh_chat()
                return
            configured = bool(config["vision_model"])
            metadata_only = any(value in text.lower() for value in ("metadata only", "inspect locally"))
            if configured and not metadata_only:
                capability = "images.analyze"
        if capability == "files.inspect":
            request = attachments[0]
        if self.start_task(capability, request, granted_paths=attachments, history=context):
            self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
        else:
            self.chat_store.append(self.thread_id, {"role": "assistant", "text": "Task cancelled or not started. No new action was performed."})
        self.refresh_chat()

    def start_task(self, *args, **kwargs):
        origin = self.thread_id
        started = super().start_task(*args, **kwargs)
        if started:
            self._task_threads[self.last_enqueued_id] = origin
        return started

    def approve_queued_task(self, task, selected_file=False):
        return self.approve_task(task["capability"], task["text"], selected_file,
                                 thread_id=self._task_threads.get(task["id"], ""))

    def approve_task(self, capability, text, selected_file=False, *, thread_id=None):
        if capability in {"images.inspect", "files.inspect", "documents.inspect", "data.inspect"}:
            return True  # Picker/drop grants are independently checked in the worker.
        if capability == "conversation":
            config = load_config(self.root)
            signature = json.dumps({key: config[key] for key in ("provider", "model", "ollama_base_url")}, sort_keys=True)
            origin = self.thread_id if thread_id is None else thread_id
            key = "chat_permission/" + str(origin)
            if origin and self.settings.value(key, "") == signature:
                return True
            dialog = QMessageBox(QMessageBox.Icon.Question, "Allow conversation?",
                f"Use local processing for this conversation?\n\n{text[:3000]}\n\nRecent messages and file excerpts discussed here are processed on this device.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            remember = QCheckBox("Remember for this conversation")
            remember.setChecked(True)
            dialog.setCheckBox(remember)
            dialog.setDefaultButton(QMessageBox.StandardButton.No)
            approved = dialog.exec() == QMessageBox.StandardButton.Yes
            if approved and origin and remember.isChecked():
                self.settings.setValue(key, signature)
            return approved
        if capability == "images.analyze":
            detail = f"Analyze the attached image on this device?\n\n{text}\n\nThe image stays on this computer. Analysis may take a moment."
            dialog = QMessageBox(QMessageBox.Icon.Question, "Allow image analysis?", detail,
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            dialog.setDefaultButton(QMessageBox.StandardButton.No)
            return dialog.exec() == QMessageBox.StandardButton.Yes
        if capability == "data.transform":
            dialog = QMessageBox(QMessageBox.Icon.Question, "Create a transformed copy?",
                f"{text}\n\nSweep will read your selected file and save a new file in this task’s local storage. The original file stays unchanged.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            dialog.setDefaultButton(QMessageBox.StandardButton.No)
            return dialog.exec() == QMessageBox.StandardButton.Yes
        return super().approve_task(capability, text, selected_file)

    def load_history(self):
        super().load_history()
        if self.thread_id:
            self.refresh_chat()

    def accept_event(self, event):
        if event.get("kind") == "response.preview":
            if not self.task or self.task["state"] != "running" or not isinstance(event.get("text"), str):
                return
            if len(self._response_previews) >= 50 and self.task["id"] not in self._response_previews:
                self._response_previews.pop(next(iter(self._response_previews)))
            self._response_previews[self.task["id"]] = event["text"][:12000]
            self.refresh_chat()
            return
        super().accept_event(event)
        self.refresh_chat()

    def render_result(self, result):
        self.last_result = result
        self.export_button.setEnabled(True)
        self.refresh_chat()

    def launch_task(self):
        super().launch_task()
        self.run_button.setText("Send ↑")
        self.update_owl_state()

    def worker_finished(self, code, status):
        super().worker_finished(code, status)
        self.run_button.setText("Send ↑")
        self.refresh_chat()

    def update_owl_state(self):
        if not self.task:
            self.owl.set_state("idle")
            return
        state = self.task["state"]
        if state == "running":
            capability = self.task["capability"]
            if self._response_previews.get(self.task["id"]):
                state = "responding"
            elif capability.startswith("web."):
                state = "searching"
            elif capability in {"images.analyze", "data.inspect", "data.transform"}:
                state = "analyzing"
            elif capability in {"images.inspect", "documents.inspect", "files.inspect"}:
                state = "reading"
            elif capability == "conversation":
                state = "thinking"
            else:
                state = "working"
        else:
            state = {"completed": "success", "failed": "attention", "interrupted": "attention"}.get(state, "idle")
        self.owl.set_state(state)

    def _task_preview(self, task):
        paths = task.get("granted_paths", [])
        if not paths:
            return None
        path = str(paths[0])
        try:
            stamp = Path(path).stat().st_mtime_ns
        except OSError:
            return None
        key = (path, stamp)
        if key not in self._preview_cache:
            if len(self._preview_cache) >= 16:
                self._preview_cache.clear()
            self._preview_cache[key] = granted_thumbnail(path, paths)
        return self._preview_cache[key]

    def _result_message(self, task, result):
        task_id = task["id"]
        value = {"role": "assistant", "state": "completed", "state_label": "Complete",
                 "text": str(result.get("message") or result.get("text") or "Done.")[:30000]}
        if task["capability"] == "conversation":
            value.pop("state_label")
        if result.get("text") and result.get("message") and result["text"] != result["message"]:
            extracted = str(result["text"])
            if not result.get("summary"):
                value["text"] += "\n\n" + extracted[:1200] + ("…" if len(extracted) > 1200 else "")
        if result.get("evidence"):
            value["text"] = result_text({key: item for key, item in result.items() if key not in {"hits", "sources"}})
        value["sources"] = [source for source in result.get("hits", result.get("sources", []))[:8]
                            if isinstance(source, dict) and public_source(str(source.get("url", "")))]
        value["columns"] = result.get("columns", [])[:30]
        value["rows"] = result.get("rows", [])[:15]
        if result.get("exif"):
            value["text"] += "\n\nEmbedded metadata:\n" + "\n".join(f"{key}: {item}" for key, item in result["exif"].items())
        location = result.get("location", {})
        if location.get("status") == "embedded_gps":
            value["text"] += f"\nEmbedded GPS (unverified): {location['latitude']}, {location['longitude']}"
        value["actions"] = []
        if result.get("text") and (result.get("summary") or len(str(result["text"])) > 1200):
            value["actions"].append(("Read extracted text", f"sweep:text/{task_id}", False))
        if result.get("proposal"):
            value["actions"].append(("Review proposed task", f"sweep:propose/{task_id}", True))
        value["actions"].extend([("Export result", f"sweep:export/{task_id}", False),
                                  ("Try again", f"sweep:retry/{task_id}", False)])
        artifacts = result.get("artifacts", [])
        if isinstance(artifacts, list):
            value["artifacts"] = [{"name": str(item.get("name") or item.get("title") or "Created file"),
                                   "type": str(item.get("format") or item.get("type") or "File"),
                                   "url": f"sweep:artifact/{task_id}/{index}"}
                                  for index, item in enumerate(artifacts[:10]) if isinstance(item, dict)]
        return value

    def refresh_chat(self):
        if not self.thread_id or not hasattr(self, "transcript"):
            return
        messages = []
        turns = self.chat_store.get(self.thread_id)["turns"]
        titles = {"web.search": "Searching the web", "web.scrape": "Reading the page",
                  "web.research": "Following the evidence", "images.analyze": "Looking at the image",
                  "images.inspect": "Reading image details", "documents.inspect": "Reading your document",
                  "data.inspect": "Exploring your data", "data.transform": "Preparing your file",
                  "files.inspect": "Reading your file", "conversation": "Thinking",
                  "computer.command": "Taking care of your request", "providers.check": "Checking local processing",
                  "providers.start": "Starting local processing"}
        for index, turn in enumerate(turns):
            value = {"role": turn["role"], "text": str(turn.get("text", "")),
                     "attachments": turn.get("attachments", [])}
            task_id = turn.get("task_id")
            if task_id:
                try:
                    task = self.task_store.get(task_id)
                    state = task["state"]
                    if state == "completed":
                        result = json.loads((self.task_store.directory(task_id) / "result.json").read_text(encoding="utf-8"))
                        value = self._result_message(task, result)
                    else:
                        value.update(state=state, state_label=state.capitalize(), task_title=titles.get(task["capability"], "Working on your request"))
                        if state == "queued":
                            value["text"] = "Ready to begin when the current task finishes."
                            value["actions"] = [("Remove from queue", f"sweep:cancel/{task_id}", False)]
                        elif state == "running":
                            value["text"] = self._response_previews.get(task_id) or (self.activity_summary.text() if self.task and self.task["id"] == task_id else "Working on your request…")
                        else:
                            value["text"] = task.get("error") or "The task stopped before finishing. You can try again."
                            value["actions"] = [("Try again", f"sweep:retry/{task_id}", True)]
                except (OSError, ValueError, KeyError):
                    value["text"] = "This task record is unavailable."
            elif value["attachments"] and index + 1 < len(turns) and turns[index + 1].get("task_id"):
                try:
                    task = self.task_store.get(turns[index + 1]["task_id"])
                    value["preview"] = self._task_preview(task)
                except (OSError, ValueError):
                    pass
            messages.append(value)
        self.transcript.render(messages, scroll_to_latest=self._scroll_to_latest)
        self._scroll_to_latest = False
        self.update_owl_state()

    def chat_link(self, url):
        if url.scheme() != "sweep":
            self.open_link(url)
            return
        try:
            segments = url.path().split("/")
            action, task_id = segments[:2]
            task = self.task_store.get(task_id)
            if action == "cancel" and task["state"] == "queued":
                self.task_store.update(task_id, state="cancelled", error="Cancelled before execution.")
                self.pending = [item for item in self.pending if item != task_id]
                self.load_history()
            elif action == "retry" and task["state"] in {"completed", "failed", "cancelled", "interrupted"}:
                if self.start_task(task["capability"], task["text"], retry_of=task_id,
                                   granted_paths=task.get("granted_paths", []), history=task.get("history", [])):
                    self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
            elif action in {"export", "propose", "artifact", "text"}:
                result = json.loads((self.task_store.directory(task_id) / "result.json").read_text(encoding="utf-8"))
                if action == "export":
                    self.last_result = result
                    self.export_result()
                elif action == "propose":
                    proposal = validate_proposal(result.get("proposal"))
                    if proposal and self.start_task(proposal["capability"], proposal["text"]):
                        self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
                elif action == "text":
                    self.show_extracted_text(result)
                else:
                    index = int(segments[2])
                    artifacts = result.get("artifacts", [])
                    if index < 0 or index >= len(artifacts):
                        raise ValueError("This file is no longer available.")
                    directory = self.task_store.directory(task_id).resolve()
                    artifact_directory = directory / "artifacts"
                    candidate = Path(artifacts[index]["path"])
                    path = (candidate if candidate.is_absolute() else directory / candidate).resolve(strict=True)
                    allowed = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".pdf", ".docx", ".xlsx", ".parquet", ".png", ".jpg", ".jpeg", ".webp"}
                    if not path.is_relative_to(artifact_directory) or not path.is_file() or path.suffix.lower() not in allowed:
                        raise ValueError("This file cannot be opened from the task.")
                    self.show_artifact(path)
            self.refresh_chat()
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            QMessageBox.warning(self, "Sweep", str(exc))

    def show_extracted_text(self, result):
        dialog = QDialog(self)
        dialog.setWindowTitle("Extracted text")
        dialog.resize(660, 520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(22, 20, 22, 20)
        title = label(str(result.get("title") or "Extracted text"), "settingsSection")
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setAccessibleName("Extracted source text")
        view.setPlainText(str(result.get("text", ""))[:80000])
        layout.addWidget(view, 1)
        layout.addWidget(label("Source text · Content is displayed as text only", "muted"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def show_artifact(self, path):
        """Preview generated files without invoking document macros or CSV formulas."""
        dialog = QDialog(self)
        dialog.setWindowTitle(path.name)
        dialog.resize(660, 500)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(label(path.name, "settingsSection"))
        if path.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md"}:
            with path.open("rb") as stream:
                raw = stream.read(500_001)
            preview = QPlainTextEdit()
            preview.setReadOnly(True)
            preview.setAccessibleName("Generated file contents")
            preview.setPlainText(raw[:500_000].decode("utf-8-sig", errors="replace"))
            layout.addWidget(preview, 1)
            if len(raw) > 500_000:
                layout.addWidget(label("Showing the first 500 KB. Save a copy for the complete file.", "muted"))
        else:
            thumbnail = granted_thumbnail(str(path), [str(path)], 600, 340)
            if thumbnail is not None:
                image = QLabel()
                image.setPixmap(thumbnail)
                layout.addWidget(image, 1)
            else:
                layout.addWidget(label("Your file is ready. Save a copy to use it in another application.", "muted"), 1)
        actions = QHBoxLayout()
        actions.addWidget(label("Original source file unchanged", "muted"), 1)
        save = QPushButton("Save a copy…")
        save.setObjectName("primary")

        def save_copy():
            destination, _ = QFileDialog.getSaveFileName(dialog, "Save a copy", path.name)
            if destination:
                try:
                    if Path(destination).resolve() != path:
                        shutil.copyfile(path, destination)
                    dialog.accept()
                except OSError as exc:
                    QMessageBox.warning(dialog, "Could not save file", str(exc))

        save.clicked.connect(save_copy)
        actions.addWidget(save)
        close = QPushButton("Close")
        close.clicked.connect(dialog.reject)
        actions.addWidget(close)
        layout.addLayout(actions)
        dialog.exec()

    def new_workspace(self):
        if not hasattr(self, "chat_store"):
            return
        self.thread_id = self.chat_store.create()["id"]
        self._scroll_to_latest = True
        self.settings.setValue("active_chat", self.thread_id)
        self.clear_attachment()
        self.prompt.clear()
        self.conversation.clear()
        self.refresh_chat()
        self.reveal()

    def show_history(self):
        self.reveal()
        dialog = QDialog(self)
        dialog.setWindowTitle("Your conversations")
        dialog.resize(480, 540)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)
        layout.addWidget(label("Pick up where you left off", "settingsTitle"))
        search = QLineEdit()
        search.setPlaceholderText("Search your conversations…")
        search.setAccessibleName("Search chat history")
        search.setClearButtonEnabled(True)
        layout.addWidget(search)
        listing = QListWidget()
        listing.setAccessibleName("Saved conversations")
        threads = self.chat_store.list()
        linked = set()
        for thread in threads:
            title = thread.get("title", "Chat")
            item = QListWidgetItem(title + "\n" + thread.get("created", "")[:10])
            item.setData(Qt.ItemDataRole.UserRole, thread["id"])
            item.setData(Qt.ItemDataRole.UserRole + 1, (title + " " + " ".join(turn.get("text", "") for turn in thread["turns"])).casefold())
            listing.addItem(item)
            linked.update(turn.get("task_id") for turn in thread["turns"])
        for task in self.task_store.list(limit=50):
            if task["id"] not in linked:
                item = QListWidgetItem(task["text"][:80] + "\nEarlier task · " + task["state"])
                item.setData(Qt.ItemDataRole.UserRole, "task:" + task["id"])
                item.setData(Qt.ItemDataRole.UserRole + 1, task["text"].casefold())
                listing.addItem(item)
        layout.addWidget(listing, 1)
        empty = label("No conversations match your search.", "muted")
        empty.hide()
        layout.addWidget(empty)

        def filter_history(query):
            visible = 0
            for index in range(listing.count()):
                item = listing.item(index)
                matches = query.casefold().strip() in item.data(Qt.ItemDataRole.UserRole + 1)
                item.setHidden(not matches)
                visible += matches
            empty.setVisible(not visible)

        def choose(item):
            selected = item.data(Qt.ItemDataRole.UserRole)
            if selected.startswith("task:"):
                task = self.task_store.get(selected[5:])
                self.thread_id = self.chat_store.create()["id"]
                self.chat_store.append(self.thread_id, {"role": "user", "text": task["text"]})
                self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": task["id"]})
            else:
                self.thread_id = selected
            self.settings.setValue("active_chat", self.thread_id)
            self.clear_attachment()
            self.prompt.clear()
            self._scroll_to_latest = True
            self.refresh_chat()
            dialog.accept()

        search.textChanged.connect(filter_history)
        listing.itemClicked.connect(choose)
        listing.itemActivated.connect(choose)
        layout.addWidget(label("Click to open · Saved on this device", "muted"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        search.setFocus()
        dialog.exec()

    def show_settings(self):
        self.reveal()
        dialog = QDialog(self)
        dialog.setWindowTitle("Sweep settings")
        dialog.resize(560, 570)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(15)
        layout.addWidget(label("Make yourself at home", "settingsTitle"))
        layout.addWidget(label("Local intelligence. Your files, your conversations, your control.", "muted"))
        try:
            config = load_config(self.root)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Local processing", str(exc))
            return
        layout.addWidget(label("On this device", "settingsSection"))
        description = label("Chat and image understanding run on your computer. Web tasks contact public websites only when requested. Conversation history stays here.", "muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        local_actions = QHBoxLayout()
        check = QPushButton("Check readiness")
        start = QPushButton("Start local processing")
        local_actions.addWidget(check)
        local_actions.addWidget(start)
        layout.addLayout(local_actions)
        layout.addWidget(label("Your desktop", "settingsSection"))
        allow = QCheckBox("Remember permission for public web tasks")
        allow.setChecked(self.settings.value("allow_web", False, type=bool))
        layout.addWidget(allow)
        notifications = QCheckBox("Notify me when a background task finishes")
        notifications.setChecked(self.settings.value("notifications", True, type=bool))
        layout.addWidget(notifications)
        startup = QCheckBox("Open Sweep when I sign in")
        startup.setChecked(startup_enabled())
        layout.addWidget(startup)
        reduce_motion = QCheckBox("Reduce motion")
        reduce_motion.setChecked(self.settings.value("reduce_motion", False, type=bool))
        layout.addWidget(reduce_motion)
        layout.addWidget(label("Ctrl + Alt + Space opens Sweep · Esc folds it away", "muted"))
        advanced_button = QPushButton("›  Advanced local settings")
        advanced_button.setObjectName("advanced")
        advanced_button.setCheckable(True)
        layout.addWidget(advanced_button)
        advanced = QWidget()
        advanced.setObjectName("advancedLocalSettings")
        form = QFormLayout(advanced)
        form.setContentsMargins(0, 0, 0, 0)
        fields = {}
        for key, title in (("model", "Conversation model"), ("vision_model", "Image model"),
                           ("ollama_base_url", "Local engine address")):
            field = QLineEdit(config.get(key, ""))
            fields[key] = field
            form.addRow(title, field)
        help_text = label("Installed model identifiers and the loopback engine address. These settings never enable cloud processing.", "muted")
        help_text.setWordWrap(True)
        form.addRow(help_text)
        advanced.hide()
        layout.addWidget(advanced)

        def toggle_advanced(checked):
            advanced.setVisible(checked)
            advanced_button.setText(("⌄" if checked else "›") + "  Advanced local settings")
            dialog.adjustSize()

        advanced_button.toggled.connect(toggle_advanced)
        layout.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)

        def save():
            try:
                values = {key: field.text().strip() for key, field in fields.items()}
                values.update(provider="ollama", vision_provider="ollama")
                save_config(self.root, values)
                for saved_key in self.settings.allKeys():
                    if saved_key.startswith("chat_permission/"):
                        self.settings.remove(saved_key)
                self.settings.setValue("allow_web", allow.isChecked())
                self.settings.setValue("notifications", notifications.isChecked())
                self.settings.setValue("reduce_motion", reduce_motion.isChecked())
                if startup.isChecked() != startup_enabled():
                    set_startup(startup.isChecked())
                self.settings.sync()
                dialog.accept()
                return True
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(dialog, "Could not save settings", str(exc))
                return False

        def local_task(capability, text):
            if save() and self.start_task(capability, text):
                self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
                self._scroll_to_latest = True
                self.refresh_chat()

        check.clicked.connect(lambda: local_task("providers.check", "Check local processing readiness"))
        start.clicked.connect(lambda: local_task("providers.start", "Start local processing on this device"))
        buttons.accepted.connect(save)
        dialog.exec()

    def closeEvent(self, event):
        if self.closing:
            event.accept()
        else:
            self.collapse()
            event.ignore()
