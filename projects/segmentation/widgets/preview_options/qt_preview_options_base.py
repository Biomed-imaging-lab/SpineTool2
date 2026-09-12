from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from widgets.qt_custom_button import QtPushButton


class QtPreviewOptions(QWidget):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()

    def __init__(self, fixed: bool = False, parent=None):
        QWidget.__init__(self, parent)
        self.fix_button = QtPushButton("fix parameters", parent=self)
        self.fix_button.clicked.connect(self._fix_parameters)
        self.fixed = fixed
        self._ndisplay = 2
        self._block_dtor_effects = False

    def set_params(self, parameters) -> None:
        pass

    def get_params(self) -> dict:
        pass

    def _block_parameters(self) -> None:
        pass

    def _on_value_changed(self) -> None:
        self.parameters_changed_.emit()

    def _fix_parameters(self) -> None:
        if not self.fixed:
            self.fixed = True
            self.fix_button.hide()
            self._block_parameters()
            self.parameters_fixed_.emit()

    @property
    def ndisplay(self) -> int:
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def block_dtor_effects(self) -> None:
        self._block_dtor_effects = True

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer."""
