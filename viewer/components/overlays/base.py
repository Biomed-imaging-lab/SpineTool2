from PyQt5.QtCore import QObject, pyqtSignal

from viewer.layers.base._base_constants import Blending


class Overlay(QObject):
    visible_ = pyqtSignal(bool)
    opacity_ = pyqtSignal(float)
    order_ = pyqtSignal(int)
    blending_ = pyqtSignal(object)

    def __init__(self):
        QObject.__init__(self)
        self._visible = False
        self._opacity = 1.0
        self._order = 1e6
        self._blending = Blending.TRANSLUCENT_NO_DEPTH

    def __hash__(self):
        return id(self)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value) -> None:
        self._visible = value
        self.visible_.emit(value)

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value) -> None:
        self._opacity = value
        self.opacity_.emit(value)

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value) -> None:
        self._order = value
        self.order_.emit(value)

    @property
    def blending(self) -> Blending:
        return self._blending

    @blending.setter
    def blending(self, value) -> None:
        self._blending = value
        self.blending_.emit(value)
