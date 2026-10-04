"""Native workspace: requests, permissions, observable tasks and artifacts."""
from __future__ import annotations

import ctypes
import html
import json
from pathlib import Path
import sys
import time

from PySide6.QtCore import Qt, QProcess, QSettings, QTimer, QUrl, QAbstractNativeEventFilter
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMenu,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QSystemTrayIcon,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from sweep.store import _save_json
from .owl import Owl, owl_icon
from .platform import launch_command, set_startup, startup_enabled
from .runtime import CAPABILITIES, needs_approval, route, task_timeout
from .tasks import TaskStore


STYLE = """
QMainWindow, QDialog { background: #f5f4ef; color: #223b35; }
QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #263e38; }
QFrame#sidebar { background: #eaece5; border-right: 1px solid #dcdfd5; }
QLabel#brand { font-size: 23px; font-weight: 650; }
QLabel#eyebrow { color: #64796c; font-size: 11px; letter-spacing: 2px; }
QLabel#title { font-size: 30px; font-weight: 600; }
QLabel#muted { color: #6f7a70; }
QFrame#card { background: #ffffff; border: 1px solid #e1e4da; border-radius: 12px; }
QPushButton { background: #ffffff; border: 1px solid #d5dbd0; border-radius: 7px; padding: 9px 14px; }
QPushButton:hover { background: #eef2e9; border-color: #a4b6a4; }
QPushButton:pressed { background: #dfe8da; }
QPushButton:disabled { color: #a6aca3; }
QPushButton#primary { background: #264f46; color: #ffffff; border: 1px solid #264f46; font-weight: 600; }
QPushButton#primary:hover { background: #356759; }
QPlainTextEdit { background: #ffffff; border: none; padding: 10px; font-size: 16px; }
QTextBrowser { background: #ffffff; border: none; padding: 16px; selection-background-color: #cbdcca; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 10px 7px; border-radius: 6px; }
QListWidget::item:selected { background: #d8e2d4; color: #244e43; }
QListWidget::item:hover { background: #e4e9df; }
QComboBox { background: white; border: 1px solid #d5dbd0; border-radius: 6px; padding: 6px; min-width: 110px; }
QTabWidget::pane { background: #fff; border: 1px solid #e1e4da; border-radius: 7px; }
QTabBar::tab { background: #f5f4ef; border: none; padding: 10px 17px; color: #6f7a70; }
QTabBar::tab:selected { background: #ffffff; color: #254e43; border-bottom: 2px solid #456e56; }
QSplitter::handle { background: #f5f4ef; width: 12px; }
QTableWidget { background: white; gridline-color: #e8eae3; border: none; }
QHeaderView::section { background: #edf1e9; padding: 8px; border: none; }
QCheckBox { padding: 8px 0; }
QToolTip { background: #263e38; color: white; border: none; padding: 6px; }
"""


def label(text, name=None):
    value = QLabel(text)
    value.setTextFormat(Qt.TextFormat.PlainText)
    value.setWordWrap(True)
    if name:
        value.setObjectName(name)
    return value


class NativeHotkey(QAbstractNativeEventFilter):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def nativeEventFilter(self, event_type, message):
        if sys.platform == "win32":
            from ctypes.wintypes import MSG
            native = MSG.from_address(int(message))
            if native.message == 0x0312 and native.wParam == 0x5357:
                self.window.reveal()
                return True, 0
        return False, 0


class SweepWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = QSettings(str(root / "desktop.ini"), QSettings.Format.IniFormat)
        self.task_store = TaskStore(root)
        self.recovered_count = self.task_store.recover_interrupted()
        self.pending = []
        self.selected_file_tasks = set()
        self.dispatching = False
        self.viewed_task_id = None
        self.process = None
        self.task = None
        self.current_directory = None
        self.event_offset = 0
        self.last_result = None
        self.conversation = []
        self.started = 0.0
        self.closing = False
        self.hotkey_registered = False
        self.setWindowTitle("Sweep — Desktop companion")
        self.setWindowIcon(owl_icon())
        self.resize(1190, 820)
        self.setMinimumSize(920, 680)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        self.build_ui()
        self.build_tray()
        self.load_history()
        if self.recovered_count:
            self.status.setText(f"{self.recovered_count} unfinished task(s) need a deliberate retry")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.submit)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.prompt.setFocus())
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_file)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_events)
        self.timer.start(150)
        self.native_filter = NativeHotkey(self)
        if sys.platform == "win32":
            QApplication.instance().installNativeEventFilter(self.native_filter)
            self.hotkey_registered = bool(ctypes.windll.user32.RegisterHotKey(
                int(self.winId()), 0x5357, 0x0002 | 0x0001 | 0x4000, 0x20))
        self.shortcut_label.setText("Ctrl + Alt + Space to open" if self.hotkey_registered
                                    else "Ctrl + K to focus · Ctrl + Enter to run")

    def build_ui(self):
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(215)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 27, 20, 20)
        side.setSpacing(16)
        brand = QHBoxLayout()
        self.owl = Owl()
        self.owl.setFixedSize(45, 45)
        brand.addWidget(self.owl)
        brand.addWidget(label("sweep", "brand"))
        side.addLayout(brand)
        side.addWidget(label("YOUR DESKTOP COMPANION", "eyebrow"))
        new = QPushButton("＋  New workspace")
        new.clicked.connect(self.new_workspace)
        side.addWidget(new)
        side.addWidget(label("RECENT TASKS", "eyebrow"))
        self.history = QListWidget()
        self.history.setAccessibleName("Recent tasks")
        self.history.itemClicked.connect(self.open_history)
        self.history.currentItemChanged.connect(self.update_history_actions)
        side.addWidget(self.history, 1)
        self.queue_summary = label("No tasks waiting", "muted")
        side.addWidget(self.queue_summary)
        self.retry_button = QPushButton("Retry selected task")
        self.retry_button.setEnabled(False)
        self.retry_button.clicked.connect(self.retry_selected)
        side.addWidget(self.retry_button)
        self.cancel_queued_button = QPushButton("Cancel queued task")
        self.cancel_queued_button.setEnabled(False)
        self.cancel_queued_button.clicked.connect(self.cancel_selected_queued)
        side.addWidget(self.cancel_queued_button)
        side.addWidget(label("Local workspace\nFiles stay local unless you choose a web task.", "muted"))
        settings = QPushButton("Settings & capabilities")
        settings.clicked.connect(self.show_settings)
        side.addWidget(settings)
        outer.addWidget(sidebar)

        body = QVBoxLayout()
        body.setContentsMargins(30, 26, 30, 20)
        body.setSpacing(16)
        top = QHBoxLayout()
        top.addWidget(label("WORKSPACE", "eyebrow"))
        top.addStretch()
        self.status = label("●  Ready", "muted")
        self.status.setWordWrap(False)
        top.addWidget(self.status)
        body.addLayout(top)
        body.addWidget(label("A little clarity. A lot done.", "title"))
        body.addWidget(label("Research a question, work with a file, or take care of your computer.", "muted"))

        suggestions = QHBoxLayout()
        for title, text in (("↗  Research a topic", "research "), ("⌘  Desktop task", "what is 15% of 200"),
                            ("⊞  Inspect a file", None)):
            button = QPushButton(title)
            if text is None:
                button.clicked.connect(self.choose_file)
            else:
                button.clicked.connect(lambda checked=False, value=text: self.fill_prompt(value))
            suggestions.addWidget(button)
        body.addLayout(suggestions)

        composer = QFrame()
        composer.setObjectName("card")
        compose = QVBoxLayout(composer)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("What would you like to do?  Try ‘search Python documentation’")
        self.prompt.setAccessibleName("Your request to Sweep")
        self.prompt.setFixedHeight(80)
        compose.addWidget(self.prompt)
        toolbar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Automatic", "auto")
        for key, capability in CAPABILITIES.items():
            if key != "files.inspect":
                self.mode.addItem(capability.title, key)
        toolbar.addWidget(self.mode)
        attach = QPushButton("Attach file")
        attach.clicked.connect(self.choose_file)
        toolbar.addWidget(attach)
        toolbar.addStretch()
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_task)
        toolbar.addWidget(self.cancel_button)
        self.run_button = QPushButton("Run task  ↵")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.submit)
        toolbar.addWidget(self.run_button)
        compose.addLayout(toolbar)
        body.addWidget(composer)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.result_view = QTextBrowser()
        self.result_view.setOpenLinks(False)
        self.result_view.anchorClicked.connect(self.open_link)
        self.tabs.addTab(self.result_view, "Result")
        self.sources = QListWidget()
        self.sources.itemDoubleClicked.connect(lambda item: self.open_link(QUrl(item.data(Qt.ItemDataRole.UserRole))))
        self.tabs.addTab(self.sources, "Sources")
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table, "Data")
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(self.image_view, "Media")
        split.addWidget(self.tabs)
        activity_frame = QFrame()
        activity_frame.setObjectName("card")
        activity = QVBoxLayout(activity_frame)
        activity.addWidget(label("ACTIVITY", "eyebrow"))
        self.activity_summary = label("A clear view of the work, as it happens.", "muted")
        activity.addWidget(self.activity_summary)
        self.activity = QListWidget()
        self.activity.setWordWrap(True)
        self.activity.setAccessibleName("Task activity log")
        activity.addWidget(self.activity)
        split.addWidget(activity_frame)
        split.setSizes([600, 250])
        body.addWidget(split, 1)
        bottom = QHBoxLayout()
        self.shortcut_label = label("", "muted")
        self.shortcut_label.setWordWrap(False)
        bottom.addWidget(self.shortcut_label)
        bottom.addStretch()
        self.export_button = QPushButton("Export result…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        bottom.addWidget(self.export_button)
        body.addLayout(bottom)
        outer.addLayout(body, 1)
        self.show_welcome()

    def show_welcome(self):
        self.result_view.setHtml("<h2>Your work, in one place.</h2><p>Start with a question, a public URL, or a file from your computer.</p><br>"
            "<p><b>Research with sources</b><br>Search, extract and collect evidence with links you can inspect.</p>"
            "<p><b>Act on your desktop</b><br>Open applications, calculate, manage notes and use existing Sweep commands.</p>"
            "<p><b>Understand a file</b><br>Preview CSV, JSON, text and images locally. Drop a file here to begin.</p>"
            "<p style='color:#748073'>Conversation uses your configured AI provider. Advanced vision, video and maps are not yet connected to this workspace.</p>")

    def build_tray(self):
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("Sweep — Desktop companion")
        menu = QMenu(self)
        for text, callback in (("Open Sweep", self.reveal), ("New workspace", self.new_workspace),
                               ("Settings", self.show_settings), ("Quit Sweep", self.quit_app)):
            action = QAction(text, self)
            action.triggered.connect(callback)
            menu.addAction(action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.reveal() if reason in (
            QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
        self.tray.messageClicked.connect(self.reveal)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def reveal(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.prompt.setFocus()

    def fill_prompt(self, text):
        self.prompt.setPlainText(text)
        self.prompt.setFocus()
        cursor = self.prompt.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.prompt.setTextCursor(cursor)

    def new_workspace(self):
        if self.process is not None:
            self.reveal()
            return
        self.last_result = None
        self.viewed_task_id = None
        self.conversation.clear()
        self.activity.clear()
        self.sources.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.image_view.clear()
        self.prompt.clear()
        self.export_button.setEnabled(False)
        self.mode.setCurrentIndex(0)
        self.status.setText("●  Ready")
        self.activity_summary.setText("A clear view of the work, as it happens.")
        self.tabs.setTabText(1, "Sources")
        self.show_welcome()
        self.reveal()

    def submit(self):
        try:
            capability, text = route(self.prompt.toPlainText(), self.mode.currentData())
        except ValueError as exc:
            QMessageBox.information(self, "Sweep", str(exc))
            return
        self.start_task(capability, text)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a file to inspect locally")
        if path:
            self.start_task("files.inspect", path, selected_file=True)

    def approve_task(self, capability, text, selected_file=False):
        approved = not needs_approval(capability, text) or (selected_file and capability == "files.inspect")
        public_web = capability in {"web.search", "web.scrape", "web.research"}
        if public_web and self.settings.value("allow_web", False, type=bool):
            approved = True
        if not approved:
            detail = ("This sends the request to public web services." if public_web else
                      "This sends your message and the conversation captured when you queued it to your configured AI provider." if capability == "conversation" else
                      "This reads the selected file locally again. The file may have changed since the original task." if capability == "files.inspect" else
                      "This can read or change local files, run commands, or control applications as requested.")
            dialog = QMessageBox(QMessageBox.Icon.Question, "Allow this task?",
                f"{CAPABILITIES[capability].title}\n\n{text[:4000]}\n\n{detail}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            dialog.setDefaultButton(QMessageBox.StandardButton.No)
            approved = dialog.exec() == QMessageBox.StandardButton.Yes
        return approved

    def approve_queued_task(self, task, selected_file=False):
        return self.approve_task(task["capability"], task["text"], selected_file)

    def start_task(self, capability, text, selected_file=False, *, retry_of=None,
                   granted_paths=None, history=None):
        if self.closing:
            return False
        if capability not in CAPABILITIES or not isinstance(text, str) or not text.strip():
            QMessageBox.information(self, "Sweep", "Choose a capability and enter a request first.")
            return False
        if len(self.pending) >= 50:
            QMessageBox.information(self, "Queue full", "Finish or cancel queued tasks before adding more (limit 50).")
            return False
        immediate = self.process is None and not self.dispatching and not self.pending
        if immediate and not self.approve_task(capability, text, selected_file):
            return False
        try:
            task = self.task_store.create(capability, text,
                granted_paths=[str(Path(text).resolve())] if selected_file else granted_paths,
                history=(self.conversation[-12:] if capability == "conversation" else []) if history is None else history,
                retry_of=retry_of)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save task", str(exc))
            return False
        self.pending.append(task["id"])
        self.last_enqueued_id = task["id"]
        if selected_file:
            self.selected_file_tasks.add(task["id"])
        self.load_history()
        if immediate:
            self.run_next(preapproved=task["id"])
        elif self.process is None and not self.dispatching:
            QTimer.singleShot(0, self.run_next)
        return True

    def run_next(self, preapproved=None):
        if self.closing or self.process is not None or self.dispatching:
            return
        self.dispatching = True
        try:
            while self.pending and not self.closing:
                task_id = self.pending.pop(0)
                try:
                    task = self.task_store.get(task_id)
                    if task["state"] != "queued":
                        continue
                    selected = task_id in self.selected_file_tasks
                    self.selected_file_tasks.discard(task_id)
                    approved = task_id == preapproved or self.approve_queued_task(task, selected)
                    # Modal dialogs process native events; recheck cancellation/exit afterward.
                    if self.closing or self.task_store.get(task_id)["state"] != "queued":
                        continue
                    if not approved:
                        self.task_store.update(task_id, state="cancelled", error="Permission was declined before execution.")
                        continue
                    self.task = self.task_store.update(task_id, state="running", approved=True)
                    self.current_directory = self.task_store.directory(task_id)
                    self.launch_task()
                    break
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(self, "Could not start queued task", str(exc))
            self.load_history()
        finally:
            self.dispatching = False

    def clear_result(self):
        self.last_result = None
        self.sources.clear()
        self.table.clear()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.image_view.clear()
        self.tabs.setTabText(1, "Sources")
        self.tabs.setCurrentIndex(0)
        self.export_button.setEnabled(False)

    def launch_task(self):
        capability, text = self.task["capability"], self.task["text"]
        self.viewed_task_id = self.task["id"]
        self.history.setCurrentItem(None)
        self.clear_result()
        self.last_result = None
        self.event_offset = 0
        self.started = time.monotonic()
        self.activity.clear()
        self.sources.clear()
        self.result_view.setHtml(f"<h2>{html.escape(CAPABILITIES[capability].title)}</h2><p>{html.escape(text)}</p><p>Starting the task. Activity and sources will appear as they become available.</p>")
        self.activity.addItem("Starting task")
        self.activity_summary.setText("Starting")
        self.run_button.setText("Add to queue  ↵")
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.owl.active = True
        self.owl.update()
        process = QProcess(self)
        self.process = process
        command = launch_command("--task", str(self.current_directory / "task.json"),
                                 "--events", str(self.current_directory / "events.jsonl"))
        process.setWorkingDirectory(str(self.root))
        process.setStandardOutputFile(str(self.current_directory / "worker.log"))
        process.setStandardErrorFile(str(self.current_directory / "worker.log"), QProcess.OpenModeFlag.Append)
        process.finished.connect(lambda code, status, worker=process: self.worker_finished(code, status)
                                 if self.process is worker else None)
        process.errorOccurred.connect(lambda error, worker=process: self.worker_error(error)
                                      if self.process is worker else None)
        process.start(command[0], command[1:])
        self.load_history()

    def poll_events(self):
        if self.process is None or self.current_directory is None:
            return
        budget = task_timeout(self.task["capability"])
        if time.monotonic() - self.started > budget + 5:
            if self.task["state"] == "running":
                self.fail_running(f"The task exceeded its {budget}-second time budget and its worker was stopped. Completed actions are not undone.")
            else:
                self.process.kill()
            return
        self.status.setText(f"●  Working · {int(time.monotonic() - self.started)}s")
        path = self.current_directory / "events.jsonl"
        if not path.exists():
            return
        try:
            with path.open("rb") as handle:
                handle.seek(self.event_offset)
                # Bound each UI tick so a noisy worker cannot freeze the window.
                for _ in range(200):
                    line = handle.readline(1_000_001)
                    if len(line) > 1_000_000:
                        self.fail_running("The task produced an event larger than the desktop limit.")
                        break
                    if not line or not line.endswith(b"\n"):
                        break
                    self.event_offset = handle.tell()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        self.accept_event(event)
        except OSError:
            self.fail_running("Sweep could not read the task's activity file.")

    def fail_running(self, message):
        if self.process is None or self.task["state"] != "running":
            return
        self.accept_event({"kind": "error", "message": message})
        self.process.kill()

    def storage_failed(self):
        """Fail closed in memory even when durable state cannot be written."""
        message = "Sweep could not save this task's state. Its worker was stopped. Check available disk space and folder permissions; the task may appear interrupted after restart."
        self.task = {**self.task, "state": "failed", "approved": False, "error": message}
        if self.process is not None:
            self.process.kill()
        if self.viewed_task_id == self.task["id"]:
            self.clear_result()
            self.result_view.setPlainText(message)
            self.activity.addItem(message)

    def accept_event(self, event):
        if not self.task or self.task["state"] != "running":
            return
        visible = self.viewed_task_id == self.task["id"]
        kind = event.get("kind")
        if kind == "progress":
            if not visible:
                return
            message = str(event.get("message", ""))
            detail = event.get("detail") or ""
            self.activity.addItem(message + ("\n" + str(detail) if detail else ""))
            self.activity.scrollToBottom()
            self.activity_summary.setText(message[:90])
        elif kind == "result":
            result = event.get("data")
            if not isinstance(result, dict):
                self.fail_running("The task returned an invalid result.")
                return
            try:
                _save_json(self.current_directory / "result.json", result)
                self.task = self.task_store.update(self.task["id"], state="completed")
            except (OSError, ValueError):
                self.storage_failed()
                return
            if visible:
                self.render_result(result)
            if self.task["capability"] == "conversation":
                self.conversation.extend([{"role": "user", "content": self.task["text"]},
                    {"role": "assistant", "content": result.get("message", "")}])
        elif kind == "error":
            message = str(event.get("message") or "The task failed without an error description.")
            try:
                self.task = self.task_store.update(self.task["id"], state="failed", error=message)
            except (OSError, ValueError):
                self.storage_failed()
                return
            if visible:
                self.clear_result()
                self.result_view.setHtml(f"<h2>This task needs attention.</h2><p>{html.escape(message)}</p><p>Edit the request or select Retry. The activity log is preserved.</p>")
                self.activity.addItem(message)

    def render_result(self, result):
        self.last_result = result
        self.export_button.setEnabled(True)
        parts = [f"<h2>{html.escape(str(result.get('title') or 'Task result'))}</h2>"]
        message = result.get("message", "")
        if message:
            parts.append(f"<p style='white-space:pre-wrap'>{html.escape(message)}</p>")
        text = result.get("text", "")
        if text:
            parts.append(f"<p style='white-space:pre-wrap'>{html.escape(text[:60000])}</p>")
        if result.get("url"):
            url = html.escape(result["url"], quote=True)
            parts.append(f'<p><a href="{url}">{url}</a></p>')
        evidence = result.get("evidence", [])
        if evidence:
            parts.append("<h3>Evidence</h3>")
            for item in evidence:
                parts.append(f"<p>{html.escape(item.get('excerpt', ''))}<br><a href='{html.escape(item.get('source_url', ''), quote=True)}'>{html.escape(item.get('source_title', 'Source'))}</a></p>")
        if result.get("path"):
            parts.append(f"<p>{html.escape(result['path'])}<br>{result.get('bytes', 0):,} bytes · {html.escape(result.get('modified', ''))}</p>")
        self.sources.clear()
        sources = result.get("hits", result.get("sources", []))
        for source in sources:
            title, url = source.get("title", "Source"), source.get("url", "")
            item = QListWidgetItem(f"{title}\n{url}")
            item.setData(Qt.ItemDataRole.UserRole, url)
            self.sources.addItem(item)
            parts.append(f"<p><b><a href='{html.escape(url, quote=True)}'>{html.escape(title)}</a></b><br>{html.escape(source.get('snippet', ''))}</p>")
        self.tabs.setTabText(1, f"Sources · {len(sources)}")
        self.result_view.setHtml("".join(parts))
        self.tabs.setCurrentIndex(0)
        columns, rows = result.get("columns", []), result.get("rows", [])
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(columns)
        for i, row in enumerate(rows):
            for j, value in enumerate(row[:len(columns)]):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
        if columns:
            self.table.resizeColumnsToContents()
            self.tabs.setCurrentIndex(2)
        self.image_view.clear()
        if result.get("image"):
            reader = QImageReader(result["image"])
            reader.setAllocationLimit(64)
            size = reader.size()
            if size.isValid():
                reader.setScaledSize(size.scaled(700, 420, Qt.AspectRatioMode.KeepAspectRatio))
                reader.setAutoTransform(True)
                self.image_view.setPixmap(QPixmap.fromImage(reader.read()))
                self.tabs.setCurrentIndex(3)

    def worker_finished(self, code, status):
        if self.process is None:
            return
        # Drain several bounded batches if the worker exited between timer ticks.
        for _ in range(10):
            previous_offset = self.event_offset
            self.poll_events()
            if self.task["state"] != "running" or self.event_offset == previous_offset:
                break
        if self.task["state"] == "running":
            self.accept_event({"kind": "error", "message": f"Task worker stopped (exit {code}). See the local activity log."})
        self.status.setText("●  " + self.task["state"].capitalize())
        if self.viewed_task_id == self.task["id"]:
            self.activity_summary.setText(f"{self.task['state'].capitalize()} · {time.monotonic() - self.started:.1f}s")
            self.activity.addItem(self.task["state"].capitalize())
        self.process.deleteLater()
        self.process = None
        self.run_button.setText("Run task  ↵")
        self.cancel_button.setEnabled(False)
        self.owl.active = False
        self.owl.update()
        self.load_history()
        if not self.isActiveWindow() and self.settings.value("notifications", True, type=bool):
            self.tray.showMessage("Sweep", "Your task " + self.task["state"] + ".", QSystemTrayIcon.MessageIcon.Information, 4000)
        if self.pending and not self.closing:
            QTimer.singleShot(0, self.run_next)

    def worker_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.process is not None:
            self.accept_event({"kind": "error", "message": "Could not start the task worker. Reinstall Sweep or check the Python environment."})
            self.worker_finished(-1, QProcess.ExitStatus.CrashExit)

    def cancel_task(self):
        if self.process is not None and self.task["state"] == "running":
            try:
                self.task = self.task_store.update(self.task["id"], state="cancelled",
                    error="Stopped by you. Completed actions are not undone; externally launched apps may continue running.")
            except (OSError, ValueError):
                self.storage_failed()
                return
            finally:
                if self.process is not None:
                    self.process.kill()
            if self.viewed_task_id == self.task["id"]:
                self.activity.addItem(self.task["error"])
                self.clear_result()
                self.result_view.setPlainText(self.task["error"])

    def load_history(self):
        selected = self.history.currentItem()
        selected_id = selected.data(Qt.ItemDataRole.UserRole) if selected else self.viewed_task_id
        self.history.clear()
        for task in self.task_store.list(limit=1000):
            state = task.get("state", "unknown")
            item = QListWidgetItem(f"{task.get('text', 'Task')[:33]}\n{state}")
            item.setToolTip(task.get("text", ""))
            item.setData(Qt.ItemDataRole.UserRole, task["id"])
            self.history.addItem(item)
            if task["id"] == selected_id:
                self.history.setCurrentItem(item)
        count = len(self.pending)
        self.queue_summary.setText(f"{count} task{'s' if count != 1 else ''} waiting" if count else "No tasks waiting")
        self.update_history_actions()

    def update_history_actions(self, *_):
        if not hasattr(self, "retry_button"):
            return
        item = self.history.currentItem()
        state = None
        if item:
            try:
                state = self.task_store.get(item.data(Qt.ItemDataRole.UserRole))["state"]
            except (OSError, ValueError):
                pass
        self.retry_button.setEnabled(state in {"completed", "failed", "cancelled", "interrupted"})
        self.cancel_queued_button.setEnabled(state == "queued")

    def retry_selected(self):
        item = self.history.currentItem()
        if not item:
            return
        try:
            task = self.task_store.get(item.data(Qt.ItemDataRole.UserRole))
            if task["state"] not in {"completed", "failed", "cancelled", "interrupted"}:
                return
            self.start_task(task["capability"], task["text"], retry_of=task["id"],
                granted_paths=task.get("granted_paths", []), history=task.get("history", []))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not retry task", str(exc))

    def cancel_selected_queued(self):
        item = self.history.currentItem()
        if not item:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            if self.task_store.get(task_id)["state"] != "queued":
                return
            self.task_store.update(task_id, state="cancelled", error="Cancelled before execution.")
            self.pending = [pending for pending in self.pending if pending != task_id]
            self.selected_file_tasks.discard(task_id)
            self.load_history()
            if self.viewed_task_id == task_id:
                self.clear_result()
                self.result_view.setPlainText("Cancelled before execution.")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not cancel task", str(exc))
        finally:
            # Cancel still stops in-session dispatch if the disk cannot be updated.
            self.pending = [pending for pending in self.pending if pending != task_id]
            self.selected_file_tasks.discard(task_id)

    def open_history(self, item):
        try:
            task_id = item.data(Qt.ItemDataRole.UserRole)
            task = self.task_store.get(task_id)
            directory = self.task_store.directory(task_id)
            self.viewed_task_id = task_id
            self.clear_result()
            self.fill_prompt(task["text"])
            mode = self.mode.findData(task["capability"])
            self.mode.setCurrentIndex(max(mode, 0))
            self.activity.clear()
            self.activity_summary.setText(task["state"].capitalize())
            path = directory / "events.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    for line in stream.read(2_000_000).splitlines()[-1000:]:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(event, dict) and event.get("message"):
                            self.activity.addItem(str(event["message"]))
            if task["state"] == "completed" and (directory / "result.json").exists():
                self.render_result(json.loads((directory / "result.json").read_text(encoding="utf-8")))
            else:
                self.result_view.setPlainText(task.get("error", "Waiting in the queue. Permission is checked before execution." if task["state"] == "queued" else
                    "This task is running. Its activity appears here." if task["state"] == "running" else
                    "No completed result. Select Retry to create a new task with fresh permission."))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not open task", str(exc))

    def export_result(self):
        if self.last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export task result", "sweep-result.json", "JSON (*.json)")
        if path:
            try:
                _save_json(Path(path), self.last_result)
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def open_link(self, url):
        if url.scheme().lower() in {"http", "https"}:
            QDesktopServices.openUrl(url)

    def show_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Sweep settings")
        dialog.resize(620, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Make Sweep yours", "title"))
        checks = []
        for key, text, default in (("allow_web", "Allow public web tasks without asking each time", False),
                                    ("notifications", "Notify me when a background task finishes", True),
                                    ("close_to_tray", "Keep Sweep in the system tray when I close its window", True)):
            check = QCheckBox(text)
            check.setChecked(self.settings.value(key, default, type=bool))
            layout.addWidget(check)
            checks.append((key, check))
        startup = QCheckBox("Launch Sweep when I sign in")
        startup.setChecked(startup_enabled())
        startup.setEnabled(sys.platform == "win32")
        layout.addWidget(startup)
        layout.addWidget(label("Shortcut: Ctrl + Alt + Space (Windows). If another app owns it, use the tray icon.\nWindow: Ctrl + K focuses the command bar; Ctrl + Enter runs; Ctrl + O opens a file.", "muted"))
        capabilities = QTextBrowser()
        capabilities.setHtml("<h3>Connected to this workspace</h3>" + "".join(
            f"<p><b>{cap.title}</b> · {cap.execution}<br>{cap.description}</p>" for cap in CAPABILITIES.values()) +
            "<p>Advanced video, maps, tracking and autonomous workflows remain future integrations. Files are never automatically uploaded.</p>")
        layout.addWidget(capabilities, 1)
        layout.addWidget(label(f"Local settings, task history and provider .env: {self.root}\nHistory is stored on this device as plaintext. API keys are never bundled with the app.", "muted"))
        folder = QPushButton("Open local settings folder")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root))))
        layout.addWidget(folder)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                if startup.isChecked() != startup_enabled():
                    set_startup(startup.isChecked())
                for key, check in checks:
                    self.settings.setValue(key, check.isChecked())
                self.settings.sync()
            except OSError as exc:
                QMessageBox.warning(self, "Settings", str(exc))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_file():
                self.start_task("files.inspect", url.toLocalFile(), selected_file=True)
                event.acceptProposedAction()
                break

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        if not self.closing and self.tray.isVisible() and self.settings.value("close_to_tray", True, type=bool):
            self.hide()
            event.ignore()
        else:
            self.quit_app()
            event.accept()

    def shutdown(self):
        self.closing = True
        self.timer.stop()
        if self.process is not None:
            self.cancel_task()
            if self.process is not None:
                self.process.kill()
                self.process.waitForFinished(3000)
        self.pending.clear()
        self.selected_file_tasks.clear()
        try:
            self.task_store.recover_interrupted()
        except (OSError, ValueError):
            pass  # A later startup will retry recovery when storage is available.
        if self.hotkey_registered:
            ctypes.windll.user32.UnregisterHotKey(int(self.winId()), 0x5357)
            self.hotkey_registered = False
        QApplication.instance().removeNativeEventFilter(self.native_filter)
        self.tray.hide()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()

    def quit_app(self):
        self.closing = True
        self.shutdown()
        QApplication.instance().quit()
