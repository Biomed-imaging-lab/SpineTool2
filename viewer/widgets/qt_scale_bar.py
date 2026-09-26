import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from utils.themes import get_theme


VIEW_FRACTION = 0.1


def format_micrometers(value: float) -> str:
    """Format a physical length without hiding small changes in zoom."""
    if not math.isfinite(value) or value < 0:
        return "—"
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value < 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value < 100:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


class QtScaleBar(QWidget):
    """A fixed-width reference showing 10% of the visible horizontal span."""

    HEIGHT = 34
    TICK_HEIGHT = 6

    def __init__(self, viewer, parent=None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._theme_id = ""
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._viewer.camera.zoom_.connect(lambda _zoom: self.update())

    @property
    def length_um(self) -> float:
        zoom = float(self._viewer.camera.zoom)
        if not math.isfinite(zoom) or zoom <= 0:
            return math.nan
        return self._canvas_width * VIEW_FRACTION / zoom

    @property
    def _canvas_width(self) -> int:
        return max(0, int(self._viewer.canvas_size[1]))

    def set_theme(self, theme_id: str) -> None:
        self._theme_id = theme_id
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.width() <= 0:
            return

        theme = get_theme(self._theme_id)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.canvas))
        painter.setPen(QPen(QColor(theme.text), 2))

        bar_width = max(1, round(self._canvas_width * VIEW_FRACTION))
        left = (self.width() - bar_width) // 2
        right = left + bar_width
        baseline = self.height() - 8
        painter.drawLine(left, baseline, right, baseline)
        painter.drawLine(left, baseline - self.TICK_HEIGHT, left, baseline)
        painter.drawLine(right, baseline - self.TICK_HEIGHT, right, baseline)

        label = f"10% = {format_micrometers(self.length_um)} µm"
        painter.drawText(
            self.rect().adjusted(0, 1, 0, -12),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            label,
        )
        painter.end()
