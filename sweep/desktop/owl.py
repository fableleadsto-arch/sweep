"""Original vector owl: tiny, resolution-independent and drawn locally."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


def paint_owl(painter, rect, active=False, state="idle"):
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(rect.x(), rect.y())
    painter.scale(rect.width() / 100, rect.height() / 100)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#244e48"))
    shape = QPainterPath()
    shape.moveTo(18, 30)
    shape.lineTo(17, 12)
    shape.quadTo(33, 16, 38, 22)
    shape.quadTo(50, 18, 62, 22)
    shape.quadTo(67, 16, 83, 12)
    shape.lineTo(82, 30)
    shape.cubicTo(104, 63, 80, 91, 50, 92)
    shape.cubicTo(20, 91, -4, 63, 18, 30)
    painter.drawPath(shape)
    painter.setBrush(QColor("#e6dfcb"))
    painter.drawEllipse(QRectF(14, 29, 39, 42))
    painter.drawEllipse(QRectF(47, 29, 39, 42))
    painter.setBrush(QColor("#fffaf0"))
    painter.drawEllipse(QRectF(24, 38, 22, 24))
    painter.drawEllipse(QRectF(54, 38, 22, 24))
    painter.setBrush(QColor("#203c38"))
    shift, down = {"thinking": (0, -3), "searching": (3, 0), "reading": (0, 3),
                   "analyzing": (-2, 1), "responding": (1, 0)}.get(state, (2 if active else 0, 0))
    painter.drawEllipse(QRectF(31 + shift, 44 + down, 10, 13))
    painter.drawEllipse(QRectF(59 + shift, 44 + down, 10, 13))
    painter.setBrush(QColor("#c19148"))
    beak = QPainterPath(QPointF(44, 62))
    beak.lineTo(56, 62)
    beak.lineTo(50, 72)
    beak.closeSubpath()
    painter.drawPath(beak)
    painter.setPen(QPen(QColor("#6e9380"), 2))
    painter.drawArc(QRectF(35, 69, 30, 16), 205 * 16, 130 * 16)
    if state != "idle":
        accent = {"success": "#b3d98d", "attention": "#e4b474", "searching": "#9acbd1",
                  "analyzing": "#c3b2df", "reading": "#dbc891"}.get(state, "#aacb9b")
        painter.setPen(QPen(QColor("#183c33"), 3))
        painter.setBrush(QColor(accent))
        painter.drawEllipse(QRectF(76, 7, 18, 18))
    painter.restore()


def owl_icon():
    pixmap = QPixmap(128, 128)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    paint_owl(painter, QRectF(0, 0, 128, 128))
    painter.end()
    return QIcon(pixmap)


class Owl(QWidget):
    STATES = {"idle": "ready", "thinking": "thinking", "searching": "searching the web",
              "reading": "reading your file", "analyzing": "analyzing your content",
              "responding": "writing a response", "working": "working on your request",
              "success": "finished", "attention": "needs attention"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "idle"
        self.setFixedSize(84, 84)
        self.set_state("idle")

    @property
    def active(self):
        return self.state not in {"idle", "success", "attention"}

    @active.setter
    def active(self, value):
        self.set_state("working" if value else "idle")

    def set_state(self, state):
        self.state = state if state in self.STATES else "idle"
        description = "Sweep · " + self.STATES[self.state]
        self.setAccessibleName(description)
        self.setToolTip(description)
        # State changes reflect task events. There is no continuous animation
        # or idle timer, so reduced-motion mode and idle power stay respected.
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        paint_owl(painter, QRectF(self.rect()), self.active, self.state)
