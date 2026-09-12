from typing import Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from viewer.components._viewer_constants import CursorStyle


class Cursor(QObject):
    position_ = pyqtSignal(object)
    scaled_ = pyqtSignal(bool)
    size_ = pyqtSignal(object)
    style_ = pyqtSignal(object)

    def __init__(self):
        QObject.__init__(self)
        self._scaled = True
        self._size = 1
        self._position = (1.0, 1.0, 1.0)
        self._style = CursorStyle.STANDARD
        self._view_direction = None

    @property
    def scaled(self) -> bool:
        return self._scaled

    @scaled.setter
    def scaled(self, value) -> None:
        self._scaled = value
        self.scaled_.emit(value)

    @property
    def size(self) -> float | np.ndarray:
        return self._size

    @size.setter
    def size(self, value) -> None:
        self._size = value
        self.size_.emit(value)

    @property
    def position(self) -> Tuple[float, ...]:
        return self._position

    @position.setter
    def position(self, value) -> None:
        self._position = value
        self.position_.emit(value)

    @property
    def style(self) -> CursorStyle:
        return self._style

    @style.setter
    def style(self, value) -> None:
        self._style = CursorStyle(value)
        self.style_.emit(self._style)
