from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox
from superqt import QLabeledSlider as QSlider

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_model_path_row import (
    add_model_path_row,
)
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtBinarizationPreviewOptions(QtPreviewOptions):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()

    def __init__(self, params: dict = {}, fixed: bool = False, parent=None):
        super().__init__(fixed, parent)
        self._neural = "neural_stage" in params

        self.base_threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.base_threshold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.base_threshold_slider.setMinimum(0)
        self.base_threshold_slider.setMaximum(255)
        self.base_threshold_slider.setSingleStep(1)
        self.base_threshold_slider.valueChanged.connect(self._on_value_changed)
        self.weight_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.weight_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.weight_slider.setMinimum(0)
        self.weight_slider.setMaximum(100)
        self.weight_slider.setSingleStep(1)
        self.weight_slider.valueChanged.connect(self._on_value_changed)
        self.block_size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.block_size_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.block_size_slider.setMinimum(1)
        self.block_size_slider.setMaximum(11)
        self.block_size_slider.setSingleStep(2)
        self.block_size_slider.valueChanged.connect(self._on_value_changed)
        self.apply_user_threshold_check_box = QCheckBox()
        self.apply_user_threshold_check_box.toggled.connect(self._on_value_changed)
        self.fill_holes_check_box = QCheckBox()
        self.fill_holes_check_box.toggled.connect(self._on_value_changed)
        self.set_params(params)

        layout = QtFormLayout()
        self.setLayout(layout)
        add_model_path_row(layout, params, self)
        layout.addRow(
            QtLabel("base threshold", parent=self), self.base_threshold_slider
        )
        if not self._neural:
            self.apply_user_threshold_check_box.hide()
            layout.addRow(QtLabel("weight", {"en", "ru"}, self), self.weight_slider)
            layout.addRow(QtLabel("block size", parent=self), self.block_size_slider)
            layout.addRow(
                QtLabel("fill enclosed holes on fix", {"en", "ru"}, self),
                self.fill_holes_check_box,
            )
        if self._neural:
            self.fill_holes_check_box.hide()
            self.weight_slider.hide()
            self.block_size_slider.hide()
            layout.addRow(
                QtLabel("apply user threshold", {"en", "ru"}, self),
                self.apply_user_threshold_check_box,
            )
        layout.addRow(self.fix_button)

        if self.fixed:
            self.fix_button.hide()
            self._block_parameters()

    def set_params(self, params: dict = {}) -> None:
        self.base_threshold_slider.setValue(params.get("base_threshold", 127))
        self.weight_slider.setValue(params.get("weight", 5))
        self.block_size_slider.setValue(params.get("block_size", 3))
        self.apply_user_threshold_check_box.setChecked(
            params.get("apply_user_threshold", False)
        )
        self.fill_holes_check_box.setChecked(params.get("fill_holes_on_fix", False))

    def get_params(self) -> dict:
        params = {
            "base_threshold": self.base_threshold_slider.value(),
        }
        if self._neural:
            params["apply_user_threshold"] = (
                self.apply_user_threshold_check_box.isChecked()
            )
        else:
            params["weight"] = self.weight_slider.value()
            params["block_size"] = self.block_size_slider.value()
            params["fill_holes_on_fix"] = self.fill_holes_check_box.isChecked()
        return params

    def _block_parameters(self) -> None:
        self.base_threshold_slider.setEnabled(False)
        self.weight_slider.setEnabled(False)
        self.block_size_slider.setEnabled(False)
        self.apply_user_threshold_check_box.setEnabled(False)
        self.fill_holes_check_box.setEnabled(False)
