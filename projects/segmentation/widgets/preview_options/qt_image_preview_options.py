from PyQt5.QtCore import Qt, pyqtSignal
from superqt import QLabeledRangeSlider as QRangeSlider

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtImagePreviewOptions(QtPreviewOptions):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()

    def __init__(self, shape, params: dict = {}, fixed: bool = False, parent=None):
        super().__init__(fixed, parent)

        self._shape = shape
        self.x_slider = QRangeSlider(Qt.Orientation.Horizontal, self)
        self.x_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.x_slider.setEdgeLabelMode(None)
        self.x_slider.label_shift_x = -6
        self.x_slider.setRange(0, shape[2] - 1)
        self.x_slider.setSingleStep(1)
        self.x_slider.valueChanged.connect(self._on_value_changed)
        self.y_slider = QRangeSlider(Qt.Orientation.Horizontal, self)
        self.y_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.y_slider.setEdgeLabelMode(None)
        self.y_slider.label_shift_x = -6
        self.y_slider.setRange(0, shape[1] - 1)
        self.y_slider.setSingleStep(1)
        self.y_slider.valueChanged.connect(self._on_value_changed)
        self.z_slider = QRangeSlider(Qt.Orientation.Horizontal, self)
        self.z_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.z_slider.setEdgeLabelMode(None)
        self.z_slider.label_shift_x = -6
        self.z_slider.setRange(0, shape[0] - 1)
        self.z_slider.setSingleStep(1)
        self.z_slider.valueChanged.connect(self._on_value_changed)
        self.set_params(params)

        layout = QtFormLayout()
        self.setLayout(layout)
        layout.addRow(QtLabel("selected along x", {"en", "ru"}, self), self.x_slider)
        layout.addRow(QtLabel("selected along y", {"en", "ru"}, self), self.y_slider)
        layout.addRow(QtLabel("selected along z", {"en", "ru"}, self), self.z_slider)
        layout.addRow(self.fix_button)

        if self.fixed:
            self.fix_button.hide()
            self._block_parameters()

    def set_params(self, params: dict = {}) -> None:
        self.x_slider.setValue(params.get("x_range", [0, self._shape[2] - 1]))
        self.y_slider.setValue(params.get("y_range", [0, self._shape[1] - 1]))
        self.z_slider.setValue(params.get("z_range", [0, self._shape[0] - 1]))

    def get_params(self) -> dict:
        return {
            "x_range": self.x_slider.value(),
            "y_range": self.y_slider.value(),
            "z_range": self.z_slider.value(),
        }

    def _block_parameters(self) -> None:
        self.x_slider.setEnabled(False)
        self.y_slider.setEnabled(False)
        self.z_slider.setEnabled(False)
