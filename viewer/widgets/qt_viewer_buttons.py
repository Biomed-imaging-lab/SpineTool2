from typing import TYPE_CHECKING

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout

from widgets.qt_custom_button import QtPushButton

if TYPE_CHECKING:
    from viewer.components.viewer_model import ViewerModel


class QtViewerButtons(QFrame):
    ndisplay_toggled = pyqtSignal()

    def __init__(self, viewer: "ViewerModel", parent=None) -> None:
        super().__init__(parent)

        self.rollDimsButton = QtPushButton(
            "roll", "roll button description", mode=True, slot=viewer.dims._roll
        )
        self.transposeDimsButton = QtPushButton(
            "transpose",
            "transpose button description",
            mode=True,
            slot=viewer.dims.transpose,
        )
        self.resetViewButton = QtPushButton(
            "home", "home button description", mode=True, slot=viewer.reset_view
        )
        self.ndisplayButton = QtPushButton(
            "ndisplay",
            "ndisplay button description",
            mode=True,
            slot=self.toggle_ndisplay,
        )
        self.ndisplayButton.setCheckable(True)
        self.ndisplayButton.setChecked(viewer.dims.ndisplay == 3)

        viewer.dims.ndisplay_.connect(self.set_ndisplay_mode_checkstate)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ndisplayButton)
        layout.addWidget(self.rollDimsButton)
        layout.addWidget(self.transposeDimsButton)
        layout.addWidget(self.resetViewButton)
        self.setLayout(layout)

    def toggle_ndisplay(self):
        self.ndisplay_toggled.emit()

    def set_ndisplay_mode_checkstate(self, value):
        self.ndisplayButton.setChecked(value == 3)
