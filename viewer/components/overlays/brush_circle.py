from typing import Tuple

from PyQt5.QtCore import pyqtSignal

from viewer.components.overlays.base import Overlay


class BrushCircleOverlay(Overlay):
    size_ = pyqtSignal(int)
    position_ = pyqtSignal(object)

    def __init__(self):
        Overlay.__init__(self)
        self._size = 10
        self._position = (0, 0)

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value) -> None:
        self._size = value
        self.size_.emit(value)

    @property
    def position(self) -> Tuple[int, int]:
        return self._position

    @position.setter
    def position(self, value) -> None:
        self._position = value
        self.position_.emit(value)
