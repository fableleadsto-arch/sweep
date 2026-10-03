"""Original vector owl: tiny, resolution-independent and drawn locally."""
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPixmap, QIcon, QPen
from PySide6.QtWidgets import QWidget


def paint_owl(painter, rect, active=False):
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
    shift = 2 if active else 0
    painter.drawEllipse(QRectF(31 + shift, 44, 10, 13))
    painter.drawEllipse(QRectF(59 + shift, 44, 10, 13))
    painter.setBrush(QColor("#c19148"))
    beak = QPainterPath(QPointF(44, 62))
    beak.lineTo(56, 62)
    beak.lineTo(50, 72)
    beak.closeSubpath()
    painter.drawPath(beak)
    painter.setPen(QPen(QColor("#6e9380"), 2))
    painter.drawArc(QRectF(35, 69, 30, 16), 205 * 16, 130 * 16)
    painter.restore()


def owl_icon():
    pixmap = QPixmap(128, 128)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    paint_owl(painter, QRectF(0, 0, 128, 128))
    painter.end()
    return QIcon(pixmap)


class Owl(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active = False
        self.setFixedSize(84, 84)
        self.setAccessibleName("Sweep owl companion")

    def paintEvent(self, event):
        painter = QPainter(self)
        paint_owl(painter, QRectF(self.rect()), self.active)
