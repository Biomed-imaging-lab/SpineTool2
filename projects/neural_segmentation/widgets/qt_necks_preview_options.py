from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox
from superqt import QLabeledSlider as QSlider

from projects.segmentation.widgets.preview_options.qt_model_path_row import (
    add_model_path_row,
)
from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtNeuralNecksPreviewOptions(QtPreviewOptions):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()

    def __init__(self, params: dict = {}, fixed: bool = False, parent=None):
        super().__init__(fixed, parent)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.threshold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setSingleStep(1)

        self.apply_user_threshold_check_box = QCheckBox(self)
        self.set_params(params)

        self.threshold_slider.valueChanged.connect(self._on_value_changed)
        self.apply_user_threshold_check_box.toggled.connect(self._on_value_changed)

        layout = QtFormLayout()
        self.setLayout(layout)
        add_model_path_row(layout, params, self)
        layout.addRow(QtLabel("threshold", {"en", "ru"}, self), self.threshold_slider)
        layout.addRow(
            QtLabel("apply user threshold", {"en", "ru"}, self),
            self.apply_user_threshold_check_box,
        )
        layout.addRow(self.fix_button)

        if self.fixed:
            self.fix_button.hide()
            self._block_parameters()

    def set_params(self, params: dict = {}) -> None:
        threshold = params.get("threshold")
        if threshold is None and "intensity_factor" in params:
            threshold = round(float(params["intensity_factor"]) * 100 / 3)
        self.threshold_slider.setValue(int(50 if threshold is None else threshold))
        self.apply_user_threshold_check_box.setChecked(
            params.get("apply_user_threshold", False)
        )

    def get_params(self) -> dict:
        return {
            "threshold": self.threshold_slider.value(),
            "apply_user_threshold": self.apply_user_threshold_check_box.isChecked(),
        }

    def _block_parameters(self) -> None:
        self.threshold_slider.setEnabled(False)
        self.apply_user_threshold_check_box.setEnabled(False)
