import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap


def application_icon(path: str) -> QIcon:
    """Add Dock margins on macOS to the otherwise edge-to-edge app artwork."""
    icon = QIcon(path)
    if sys.platform != "darwin" or icon.isNull():
        return icon

    source = QPixmap(path)
    if source.isNull():
        return icon
    padded_icon = QIcon()
    for size in (16, 32, 64, 128, 256, 512, 1024):
        canvas = QPixmap(size, size)
        canvas.fill(Qt.transparent)
        artwork_size = round(size * 0.82)
        artwork = source.scaled(
            artwork_size, artwork_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter = QPainter(canvas)
        painter.drawPixmap(
            (size - artwork.width()) // 2,
            (size - artwork.height()) // 2,
            artwork,
        )
        painter.end()
        padded_icon.addPixmap(canvas)
    return padded_icon
