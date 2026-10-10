"""Native chat cards and bounded, permission-bound image previews."""
from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def granted_thumbnail(path: str, grants: list[str], width: int = 480, height: int = 260) -> QPixmap | None:
    """Decode only an explicitly selected image, within bounded size limits."""
    try:
        if str(path).startswith(("\\\\", "//")):
            return None
        selected = Path(path).resolve(strict=True)
        if selected not in {Path(value).resolve() for value in grants}:
            return None
        if not selected.is_file() or selected.stat().st_size > 10_000_000:
            return None
        reader = QImageReader(str(selected))
        reader.setAllocationLimit(64)
        size = reader.size()
        if not size.isValid() or size.width() * size.height() > 25_000_000:
            return None
        reader.setScaledSize(size.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio))
        reader.setAutoTransform(True)
        image = reader.read()
        return QPixmap.fromImage(image) if not image.isNull() else None
    except (OSError, ValueError):
        return None


def rich_text(text: str) -> str:
    """A small safe Markdown subset; source HTML and image tags stay literal."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^\n]+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`\n]+)`", r'<span style="font-family: Consolas; color: #49665c">\1</span>', escaped)
    return escaped.replace("\n", "<br>")


def public_source(url: str) -> tuple[str, str] | None:
    """Validate display links without fetching or interpreting source content."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return None
        return url, parsed.hostname
    except ValueError:
        return None


class DataPreview(QTableWidget):
    def showEvent(self, event):
        super().showEvent(event)
        # Inherited styling and header metrics are available once the card is
        # attached. Measure then so short previews do not clip their last row.
        self.setFixedHeight(min(220, self.horizontalHeader().height() +
                                sum(self.rowHeight(row) for row in range(self.rowCount())) + 4))


class ChatTranscript(QScrollArea):
    anchorClicked = Signal(QUrl)
    exampleRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setObjectName("transcript")
        self.setAccessibleName("Sweep conversation")
        self._plain_text = ""
        self._signature = None
        self._following = True
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)
        self.content = QWidget()
        self.content.setObjectName("chatContent")
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(12, 12, 12, 18)
        self.layout.setSpacing(20)
        self.setWidget(self.content)

    def toPlainText(self):
        return self._plain_text

    def _range_changed(self, minimum, maximum):
        if self._following:
            self.verticalScrollBar().setValue(maximum)

    def _text(self, value: str, name="messageText") -> QLabel:
        widget = QLabel(rich_text(value))
        widget.setObjectName(name)
        widget.setTextFormat(Qt.TextFormat.RichText)
        widget.setWordWrap(True)
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        widget.linkActivated.connect(lambda url: self.anchorClicked.emit(QUrl(url)))
        return widget

    def _action(self, title: str, url: str, parent: QHBoxLayout, primary=False):
        button = QPushButton(title)
        button.setObjectName("cardPrimary" if primary else "cardAction")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda: self.anchorClicked.emit(QUrl(url)))
        parent.addWidget(button)

    def _welcome(self):
        frame = QFrame()
        frame.setObjectName("welcome")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(11)
        eyebrow = QLabel("YOUR SPACE TO THINK & DO")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)
        title = QLabel("What’s on your mind?")
        title.setObjectName("welcomeTitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(self._text("Talk it through, explore a question, or get something done.\nYour conversation stays on this device.", "welcomeDescription"))
        layout.addSpacing(4)
        for title, detail, request in (
            ("Open something", "A website or an app, in your words", "Open YouTube in Brave"),
            ("Explore a question", "Find sources and follow the evidence", "Research "),
            ("Look a little closer", "Drop an image or choose a file", "__attach__"),
        ):
            button = QPushButton(title + "   ↗\n" + detail)
            button.setObjectName("suggestion")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, value=request: self.exampleRequested.emit(value))
            layout.addWidget(button)
        self.layout.addWidget(frame)
        self._plain_text = "What’s on your mind?\nTalk it through, explore a question, or get something done."

    def _card(self, message: dict):
        user = message.get("role") == "user"
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if user:
            row.addStretch(1)
        card = QFrame()
        card.setObjectName("userMessage" if user else "assistantMessage")
        if user:
            card.setMaximumWidth(560)
        body = QVBoxLayout(card)
        body.setContentsMargins(17, 14, 17, 15)
        body.setSpacing(10)
        heading = QHBoxLayout()
        role = QLabel("YOU" if user else "SWEEP")
        role.setObjectName("messageRole")
        heading.addWidget(role)
        if message.get("state_label"):
            state = QLabel(message["state_label"])
            state.setObjectName("taskState")
            state.setProperty("state", message.get("state", ""))
            heading.addWidget(state)
        heading.addStretch(1)
        body.addLayout(heading)
        if message.get("task_title"):
            task_title = QLabel(message["task_title"])
            task_title.setObjectName("taskTitle")
            body.addWidget(task_title)
        preview = message.get("preview")
        if preview is not None and not preview.isNull():
            image = QLabel()
            image.setObjectName("chatImage")
            image.setPixmap(preview)
            image.setAlignment(Qt.AlignmentFlag.AlignLeft)
            image.setAccessibleName("Selected image preview")
            body.addWidget(image)
        if message.get("attachments"):
            body.addWidget(self._text(" · ".join(message["attachments"]), "attachmentCaption"))
        value = str(message.get("text", ""))
        if value:
            body.addWidget(self._text(value))
        if message.get("state") == "running":
            progress = QProgressBar()
            progress.setAccessibleName("Task in progress")
            progress.setRange(0, 0)
            progress.setTextVisible(False)
            progress.setFixedHeight(3)
            progress.setStyleSheet("QProgressBar { border: none; background: #eaf0e4; } QProgressBar::chunk { background: #89aa70; }")
            body.addWidget(progress)
        sources = message.get("sources", [])
        if sources:
            source_card = QFrame()
            source_card.setObjectName("sourceCard")
            source_layout = QVBoxLayout(source_card)
            source_layout.setContentsMargins(13, 11, 13, 11)
            source_layout.setSpacing(9)
            caption = QLabel(f"SOURCES  ·  {len(sources)}")
            caption.setObjectName("eyebrow")
            source_layout.addWidget(caption)
            for source in sources:
                url = str(source.get("url", ""))
                link = public_source(url)
                if link is None:
                    continue
                title = str(source.get("title") or url)
                source_label = self._text("")
                source_label.setText(f'<a style="color:#315d50; text-decoration:none" href="{html.escape(url, quote=True)}">{html.escape(title)} ↗</a><br><span style="font-size:11px; color:#67796c">{html.escape(link[1])}</span>')
                source_layout.addWidget(source_label)
                if source.get("snippet"):
                    source_layout.addWidget(self._text(str(source["snippet"])[:220], "sourceSnippet"))
            body.addWidget(source_card)
        columns = message.get("columns", [])
        if columns:
            rows = message.get("rows", [])[:15]
            table = DataPreview(len(rows), len(columns))
            table.setObjectName("dataPreview")
            table.setAccessibleName("File data preview")
            table.setHorizontalHeaderLabels([str(value) for value in columns])
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.verticalHeader().hide()
            table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
            for i, values in enumerate(rows):
                for j, cell in enumerate(values[:len(columns)]):
                    table.setItem(i, j, QTableWidgetItem(str(cell)))
            table.resizeColumnsToContents()
            for column in range(len(columns)):
                table.setColumnWidth(column, min(260, max(90, table.columnWidth(column))))
            if len(columns) <= 4:
                table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setFixedHeight(min(220, table.horizontalHeader().height() + sum(table.rowHeight(row) for row in range(len(rows))) + 4))
            body.addWidget(table)
        for artifact in message.get("artifacts", []):
            frame = QFrame()
            frame.setObjectName("artifactCard")
            line = QHBoxLayout(frame)
            line.setContentsMargins(13, 11, 13, 11)
            description = QVBoxLayout()
            name = QLabel(str(artifact.get("name", "Created file")))
            name.setTextFormat(Qt.TextFormat.PlainText)
            name.setWordWrap(True)
            description.addWidget(name)
            caption = QLabel(str(artifact.get("type", "File")).upper() + " · Created on this device")
            caption.setObjectName("attachmentCaption")
            caption.setTextFormat(Qt.TextFormat.PlainText)
            description.addWidget(caption)
            line.addLayout(description, 1)
            self._action("View file ↗", str(artifact["url"]), line)
            body.addWidget(frame)
        actions = message.get("actions", [])
        if actions or (value and not user and message.get("state") not in {"running", "queued"}):
            buttons = QHBoxLayout()
            buttons.setSpacing(6)
            for title, url, primary in actions:
                self._action(title, url, buttons, primary)
            buttons.addStretch(1)
            if value and not user and message.get("state") not in {"running", "queued"}:
                copy = QPushButton("Copy")
                copy.setObjectName("cardAction")
                copy.setAccessibleName("Copy this response")
                copy.clicked.connect(lambda: QApplication.clipboard().setText(value))
                buttons.addWidget(copy)
            body.addLayout(buttons)
        row.addWidget(card, 5)
        if user:
            row.addSpacing(2)
        self.layout.addLayout(row)

    def render(self, messages: list[dict], *, scroll_to_latest=False):
        # Progress can arrive frequently. Keep native cards and text selection
        # stable unless visible content has actually changed.
        signature = repr([{key: value.cacheKey() if key == "preview" and value is not None else value for key, value in message.items()} for message in messages])
        bar = self.verticalScrollBar()
        at_bottom = bool(messages) and (scroll_to_latest or bar.value() >= bar.maximum() - 24)
        self._following = at_bottom
        position = bar.value() if messages else 0
        if signature == self._signature:
            if scroll_to_latest:
                bar.setValue(bar.maximum())
            return
        self._signature = signature
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().hide()
                        child.widget().deleteLater()
                item.layout().deleteLater()
        self._plain_text = "\n\n".join(str(message.get("text", "")) for message in messages)
        if not messages:
            self._welcome()
        for message in messages:
            self._card(message)
        self.layout.addStretch(1)
        self.content.adjustSize()
        bar.setValue(bar.maximum() if at_bottom else position)
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum() if at_bottom else min(position, bar.maximum())))

    def copy_text(self):
        QApplication.clipboard().setText(self._plain_text)
