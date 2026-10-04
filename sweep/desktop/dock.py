"""A compact native chat dock; the existing worker queue stays behind the chat."""
from __future__ import annotations

import html
import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
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
from .providers import load_config, save_config
from .window import SweepWindow, label

DOCK_STYLE = """
QMainWindow { background: transparent; }
QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #203d36; }
QFrame#dock { background: #fafaf6; border: 1px solid #bccbc0; border-radius: 15px; }
QFrame#handle { background: #183c33; border: none; border-radius: 13px; }
QFrame#handle QLabel { color: #f2f7ed; background: transparent; }
QFrame#handle QPushButton { background: transparent; color: #eef6ea; border: none; padding: 7px; }
QFrame#handle QPushButton:hover { background: #305448; }
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
        self.open_button.clicked.connect(self.reveal)
        header.addWidget(self.open_button, 1)
        for title, tip, callback in (("＋", "New chat", self.new_workspace), ("History", "Conversation history", self.show_history),
                                      ("Settings", "Models and preferences", self.show_settings)):
            button = QPushButton(title)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            header.addWidget(button)
        self.pin = QPushButton("Pin")
        self.pin.setCheckable(True)
        self.pin.setToolTip("Keep the chat expanded")
        self.pin.toggled.connect(self.toggle_pin)
        header.addWidget(self.pin)
        fold = QPushButton("−")
        fold.setToolTip("Collapse dock (Esc)")
        fold.clicked.connect(self.collapse)
        header.addWidget(fold)
        outer.addWidget(self.handle)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(12, 8, 12, 5)
        body.setSpacing(8)
        self.transcript = QTextBrowser()
        self.transcript.setAccessibleName("Sweep conversation")
        self.transcript.setOpenLinks(False)
        self.transcript.anchorClicked.connect(self.chat_link)
        body.addWidget(self.transcript, 1)
        self.attachment_label = label("", "muted")
        body.addWidget(self.attachment_label)
        self.prompt = ChatInput()
        self.prompt.setAccessibleName("Message Sweep")
        self.prompt.setPlaceholderText("Ask anything, or tell Sweep what to do…")
        self.prompt.setFixedHeight(76)
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
        self.status = label("Ready", "muted")
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
        self.clear_attachment_button.hide()

    def choose_file(self):
        self.reveal()
        path, _ = QFileDialog.getOpenFileName(self, "Attach an image or file")
        if path:
            self.attachments = [str(Path(path).resolve())]
            self.clear_attachment_button.show()
            self.attachment_label.setText("Attached: " + Path(path).name + " · kept local until you approve analysis")
            self.prompt.setFocus()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_file():
                self.attachments = [str(Path(url.toLocalFile()).resolve())]
                self.clear_attachment_button.show()
                self.attachment_label.setText("Attached: " + Path(self.attachments[0]).name)
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
            configured = bool(config["vision_model"] or any(config["key_present"].get(name) for name in ("openai", "gemini")))
            local_only = any(word in text.lower() for word in ("metadata only", "locally", "local only", "don't upload", "do not upload"))
            if configured and not local_only:
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
        if capability in {"images.inspect", "files.inspect"}:
            return True  # Picker/drop grants are independently checked in the worker.
        if capability == "conversation":
            config = load_config(self.root)
            signature = json.dumps({key: config[key] for key in ("provider", "model", "ollama_base_url")}, sort_keys=True)
            origin = self.thread_id if thread_id is None else thread_id
            key = "chat_permission/" + str(origin)
            if origin and self.settings.value(key, "") == signature:
                return True
            provider = config["provider"] if config["provider"] != "auto" else "your configured providers"
            dialog = QMessageBox(QMessageBox.Icon.Question, "Allow conversation?",
                f"Send this message and recent chat context to {provider}?\n\n{text[:3000]}\n\nContext can include excerpts from files and tool results discussed in this chat.",
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
            config = load_config(self.root)
            provider = config["vision_provider"]
            if provider == "auto":
                provider = next((name for name in ("openai", "gemini") if config["key_present"].get(name)), "ollama")
            detail = f"Send the attached image to {provider} for this request?\n\n{text}\n\nSweep resizes the image and removes embedded metadata before sending."
            dialog = QMessageBox(QMessageBox.Icon.Question, "Allow image analysis?", detail,
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
        super().accept_event(event)
        self.refresh_chat()

    def render_result(self, result):
        self.last_result = result
        self.export_button.setEnabled(True)
        self.refresh_chat()

    def launch_task(self):
        super().launch_task()
        self.run_button.setText("Send ↑")

    def worker_finished(self, code, status):
        super().worker_finished(code, status)
        self.run_button.setText("Send ↑")
        self.refresh_chat()

    def refresh_chat(self):
        if not self.thread_id or not hasattr(self, "transcript"):
            return
        parts = []
        turns = self.chat_store.get(self.thread_id)["turns"]
        if not turns:
            parts.append("<br><h2>What can I help with?</h2><p>Talk things through, ask a question, or tell me what to do.</p><p style='color:#718374'>Try “open YouTube in Brave”, “find Python tutorials”, or attach an image and ask about it.</p>")
        for turn in turns:
            user = turn.get("role") == "user"
            title, text = ("You" if user else "Sweep"), str(turn.get("text", ""))
            links, extra = "", ""
            task_id = turn.get("task_id")
            if task_id:
                try:
                    task = self.task_store.get(task_id)
                    state = task["state"]
                    if state == "completed":
                        result = json.loads((self.task_store.directory(task_id) / "result.json").read_text(encoding="utf-8"))
                        text = str(result.get("message") or result.get("text") or "Done.")[:30000]
                        if result.get("text") and result.get("message") and result["text"] != result["message"]:
                            text += "\n\n" + str(result["text"])[:14000]
                        if result.get("evidence"):
                            # Reuse the bounded evidence summary supplied to follow-up chat.
                            text = result_text({key: value for key, value in result.items() if key not in {"hits", "sources"}})
                        for source in result.get("hits", result.get("sources", []))[:8]:
                            url = str(source.get("url", ""))
                            if url.startswith(("https://", "http://")):
                                links += f'<p><a href="{html.escape(url, quote=True)}">{html.escape(str(source.get("title") or url))}</a><br><span style="color:#718374">{html.escape(str(source.get("snippet", ""))[:350])}</span></p>'
                        if result.get("ollama"):
                            text += "\nInstalled local models: " + ", ".join(result["ollama"].get("models", []))
                        if result.get("columns"):
                            text += "\n\n" + " · ".join(result["columns"])
                            text += "\n" + "\n".join(" · ".join(map(str, row)) for row in result.get("rows", [])[:15])
                        if result.get("exif"):
                            text += "\n\nEmbedded metadata:\n" + "\n".join(f"{key}: {value}" for key, value in result["exif"].items())
                        location = result.get("location", {})
                        if location.get("status") == "embedded_gps":
                            text += f"\nEmbedded GPS (unverified): {location['latitude']}, {location['longitude']}"
                        if result.get("proposal"):
                            extra += f'<p><a href="sweep:propose/{task_id}">Review and run proposed task</a></p>'
                        extra += f'<p><a href="sweep:export/{task_id}">Export result</a> · <a href="sweep:retry/{task_id}">Retry</a></p>'
                    elif state == "queued":
                        text = "Waiting for the current task to finish."
                        extra = f'<p><a href="sweep:cancel/{task_id}">Cancel waiting task</a></p>'
                    elif state == "running":
                        text = self.activity_summary.text() if self.task and self.task["id"] == task_id else "Working…"
                    else:
                        text = task.get("error") or state.capitalize()
                        extra = f'<p><a href="sweep:retry/{task_id}">Retry with a new task</a></p>'
                except (OSError, ValueError, KeyError):
                    text = "This task record is unavailable."
            attachments = turn.get("attachments", [])
            if attachments:
                title += " · " + ", ".join(attachments)
            color = "#e7efdf" if user else "#f3f4ef"
            parts.append(f'<table width="100%" cellspacing="0" cellpadding="12" bgcolor="{color}"><tr><td><b>{html.escape(title)}</b><p style="white-space:pre-wrap">{html.escape(text)}</p>{links}{extra}</td></tr></table><br>')
        bar = self.transcript.verticalScrollBar()
        at_bottom = self._scroll_to_latest or bar.value() >= bar.maximum() - 24
        self._scroll_to_latest = False
        position = bar.value()
        self.transcript.setHtml("<style>a { color: #28624a; }</style>" + "".join(parts))
        bar.setValue(bar.maximum() if at_bottom else position)
        if at_bottom:
            # QTextDocument finishes its layout after setHtml returns. Scroll
            # against that final range so new answers remain in view.
            QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def chat_link(self, url):
        if url.scheme() != "sweep":
            self.open_link(url)
            return
        try:
            action, task_id = url.path().split("/", 1)
            task = self.task_store.get(task_id)
            if action == "cancel" and task["state"] == "queued":
                self.task_store.update(task_id, state="cancelled", error="Cancelled before execution.")
                self.pending = [item for item in self.pending if item != task_id]
                self.load_history()
            elif action == "retry" and task["state"] in {"completed", "failed", "cancelled", "interrupted"}:
                if self.start_task(task["capability"], task["text"], retry_of=task_id,
                                   granted_paths=task.get("granted_paths", []), history=task.get("history", [])):
                    self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
            elif action in {"export", "propose"}:
                result = json.loads((self.task_store.directory(task_id) / "result.json").read_text(encoding="utf-8"))
                if action == "export":
                    self.last_result = result
                    self.export_result()
                else:
                    proposal = validate_proposal(result.get("proposal"))
                    if proposal and self.start_task(proposal["capability"], proposal["text"]):
                        self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
            self.refresh_chat()
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Sweep", str(exc))

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
        dialog.setWindowTitle("Chat history")
        dialog.resize(430, 480)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Your conversations"))
        listing = QListWidget()
        threads = self.chat_store.list()
        linked = set()
        for thread in threads:
            item = QListWidgetItem(thread.get("title", "Chat"))
            item.setData(Qt.ItemDataRole.UserRole, thread["id"])
            listing.addItem(item)
            linked.update(turn.get("task_id") for turn in thread["turns"])
        for task in self.task_store.list(limit=50):
            if task["id"] not in linked:
                item = QListWidgetItem("Earlier task · " + task["text"][:60])
                item.setData(Qt.ItemDataRole.UserRole, "task:" + task["id"])
                listing.addItem(item)
        layout.addWidget(listing)
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
            self._scroll_to_latest = True
            self.refresh_chat()
            dialog.accept()
        listing.itemDoubleClicked.connect(choose)
        layout.addWidget(label("Double-click a conversation to open it. History stays on this device.", "muted"))
        dialog.exec()

    def show_settings(self):
        self.reveal()
        dialog = QDialog(self)
        dialog.setWindowTitle("Sweep settings")
        dialog.resize(600, 670)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Conversation and image understanding"))
        try:
            config = load_config(self.root)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Provider settings", str(exc))
            return
        form = QFormLayout()
        fields = {}
        for key, title in (("provider", "Chat provider"), ("model", "Chat model (blank = automatic)"),
                           ("vision_provider", "Image provider"), ("vision_model", "Image-capable model"),
                           ("ollama_base_url", "Ollama address")):
            if "provider" in key:
                widget = QComboBox()
                for value in ("auto", "ollama", "openai", "gemini", "anthropic") if key == "provider" else ("auto", "ollama", "openai", "gemini"):
                    widget.addItem(value.capitalize(), value)
                widget.setCurrentIndex(max(0, widget.findData(config.get(key, "auto"))))
            else:
                widget = QLineEdit(config.get(key, ""))
            fields[key] = widget
            form.addRow(title, widget)
        keys = {}
        for provider in ("openai", "gemini", "anthropic"):
            field = QLineEdit()
            field.setEchoMode(QLineEdit.EchoMode.Password)
            field.setPlaceholderText("Configured · leave blank to keep" if config["key_present"].get(provider) else "API key (optional)")
            keys[provider] = field
            form.addRow(provider.capitalize() + " key", field)
        layout.addLayout(form)
        layout.addWidget(label("Keys saved here are protected by your Windows account. Local Ollama uses installed models; no model downloads happen automatically. Choose an image-capable model for visual descriptions.", "muted"))
        allow = QCheckBox("Remember permission for public web searches and scraping")
        allow.setChecked(self.settings.value("allow_web", False, type=bool))
        layout.addWidget(allow)
        notifications = QCheckBox("Notify me when a task finishes in the background")
        notifications.setChecked(self.settings.value("notifications", True, type=bool))
        layout.addWidget(notifications)
        startup = QCheckBox("Open Sweep when I sign in")
        startup.setChecked(startup_enabled())
        layout.addWidget(startup)
        check = QPushButton("Check installed local models")
        def check_models():
            if not save():
                return
            if self.start_task("providers.check", "Check installed local models"):
                self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
                self.refresh_chat()
        check.clicked.connect(check_models)
        layout.addWidget(check)
        start = QPushButton("Start installed local AI (Ollama)")
        def start_models():
            if not save():
                return
            if self.start_task("providers.start", "Start the installed local Ollama server. No model downloads."):
                self.chat_store.append(self.thread_id, {"role": "assistant", "task_id": self.last_enqueued_id})
                self.refresh_chat()
        start.clicked.connect(start_models)
        layout.addWidget(start)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)
        def save():
            try:
                values = {key: field.currentData() if isinstance(field, QComboBox) else field.text().strip() for key, field in fields.items()}
                save_config(self.root, values, {key: field.text().strip() or None for key, field in keys.items()})
                for saved_key in self.settings.allKeys():
                    if saved_key.startswith("chat_permission/"):
                        self.settings.remove(saved_key)
                self.settings.setValue("allow_web", allow.isChecked())
                self.settings.setValue("notifications", notifications.isChecked())
                if startup.isChecked() != startup_enabled():
                    set_startup(startup.isChecked())
                self.settings.sync()
                dialog.accept()
                return True
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(dialog, "Could not save settings", str(exc))
                return False
        buttons.accepted.connect(save)
        dialog.exec()

    def closeEvent(self, event):
        if self.closing:
            event.accept()
        else:
            self.collapse()
            event.ignore()
