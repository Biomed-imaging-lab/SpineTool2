from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from widgets.qt_custom_button import QtPushButton
from widgets.qt_line import QtVLine

if TYPE_CHECKING:
    from projects.segmentation.project_segmentation import SegmentationProject


class QtButtonsContainer(QWidget):
    def __init__(self, buttons, alignment=Qt.AlignmentFlag.AlignCenter, parent=None):
        QWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 0, 8, 6)
        self.setLayout(layout)
        self.buttons = buttons
        layout.addWidget(self.buttons, alignment=alignment)
        self.setMaximumHeight(36)


class QtControlButtons(QFrame):
    def __init__(self, project: "SegmentationProject", parent=None) -> None:
        super().__init__(parent)

        self.imageStageButton = QtPushButton(
            "image_stage", "image stage", mode=True, slot=project._image_stage
        )
        self.addLayerToNonEditableButton = QtPushButton(
            "add_layer_to_non_editable",
            "add non editable",
            mode=True,
            slot=project._add_layer_to_non_editable,
        )
        self.selectedStageButton = QtPushButton(
            "selected_stage", "selected stage", mode=True, slot=project._selected_stage
        )
        self.previousStageButton = QtPushButton(
            "previous_stage", "previous stage", mode=True, slot=project._previous_stage
        )
        self.deleteButton = QtPushButton(
            "delete_button", "delete", mode=True, slot=project._delete_layer
        )
        self.restoreButton = QtPushButton(
            "restore_button", "restore", mode=True, slot=project._restore_last
        )
        self.exportButton = QtPushButton(
            "export_button", "export", mode=True, slot=project._export_layer
        )
        self.backgroundButton = QtPushButton(
            "background_button",
            "background image",
            mode=True,
            slot=project._add_background_image,
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.deleteButton)
        layout.addWidget(self.restoreButton)
        layout.addWidget(QtVLine())
        layout.addWidget(self.exportButton)
        layout.addWidget(self.addLayerToNonEditableButton)
        layout.addWidget(self.backgroundButton)
        layout.addWidget(QtVLine())
        layout.addWidget(self.imageStageButton)
        layout.addWidget(self.selectedStageButton)
        layout.addWidget(self.previousStageButton)
        self.setLayout(layout)
