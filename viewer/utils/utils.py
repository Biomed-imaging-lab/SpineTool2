from functools import lru_cache

import wrapt
from PyQt5.QtCore import QPoint, QSize, Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap


@lru_cache(maxsize=64)
def square_pixmap(size):
    size = max(int(size), 1)
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.GlobalColor.white)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.setPen(Qt.GlobalColor.black)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.end()
    return pixmap


@lru_cache(maxsize=64)
def crosshair_pixmap():
    size = 25

    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)

    # Base measures
    width = 1
    center = 3  # Must be odd!
    rect_size = center + 2 * width
    square = rect_size + width * 4

    pen = QPen(Qt.GlobalColor.white, 1)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # Horizontal rectangle
    painter.drawRect(0, (size - rect_size) // 2, size - 1, rect_size - 1)

    # Vertical rectangle
    painter.drawRect((size - rect_size) // 2, 0, rect_size - 1, size - 1)

    # Square
    painter.drawRect((size - square) // 2, (size - square) // 2, square - 1, square - 1)

    pen = QPen(Qt.GlobalColor.black, 2)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # Square
    painter.drawRect(
        (size - square) // 2 + 2,
        (size - square) // 2 + 2,
        square - 4,
        square - 4,
    )

    pen = QPen(Qt.GlobalColor.black, 3)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    # # # Horizontal lines
    mid_vpoint = QPoint(2, size // 2)
    painter.drawLine(mid_vpoint, QPoint(((size - center) // 2) - center + 1, size // 2))
    mid_vpoint = QPoint(size - 3, size // 2)
    painter.drawLine(mid_vpoint, QPoint(((size - center) // 2) + center + 1, size // 2))

    # # # Vertical lines
    mid_hpoint = QPoint(size // 2, 2)
    painter.drawLine(QPoint(size // 2, ((size - center) // 2) - center + 1), mid_hpoint)
    mid_hpoint = QPoint(size // 2, size - 3)
    painter.drawLine(QPoint(size // 2, ((size - center) // 2) + center + 1), mid_hpoint)

    painter.end()
    return pixmap


@lru_cache(maxsize=64)
def ellipse_pixmap(w, h):
    w = max(int(w), 1)
    h = max(int(h), 1)
    pixmap = QPixmap(QSize(w, h))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.GlobalColor.white)
    painter.drawEllipse(0, 0, w - 1, h - 1)
    painter.setPen(Qt.GlobalColor.black)
    painter.drawEllipse(1, 1, w - 3, h - 3)
    painter.end()
    return pixmap


class ReadOnlyWrapper(wrapt.ObjectProxy):
    def __init__(self, wrapped, exceptions=()):
        super().__init__(wrapped)
        self._self_exceptions = exceptions

    def __setattr__(self, name, val):
        if (
            name not in ("__wrapped__", "_self_exceptions")
            and name not in self._self_exceptions
        ):
            raise TypeError(
                "cannot set attribute {name}".format(
                    name=name,
                )
            )

        super().__setattr__(name, val)

    def __setitem__(self, name, val):
        if name not in self._self_exceptions:
            raise TypeError("cannot set item {name}".format(name=name))
        super().__setitem__(name, val)
