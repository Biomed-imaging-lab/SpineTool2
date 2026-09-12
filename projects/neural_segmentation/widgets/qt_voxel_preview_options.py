from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox
from superqt import QLabeledSlider as QSlider

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_model_path_row import (
    add_model_path_row,
)
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtNeuralVoxelPreviewOptions(QtPreviewOptions):
    def __init__(self, stage_id: int, params: dict = {}, fixed: bool = False, parent=None):
        super().__init__(fixed, parent)
        self._stage_id = stage_id

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.threshold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.valueChanged.connect(self._on_value_changed)

        self.trunk_threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.trunk_threshold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.trunk_threshold_slider.setMinimum(0)
        self.trunk_threshold_slider.setMaximum(100)
        self.trunk_threshold_slider.setSingleStep(1)
        self.trunk_threshold_slider.valueChanged.connect(self._on_value_changed)

        self.spine_threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.spine_threshold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.spine_threshold_slider.setMinimum(0)
        self.spine_threshold_slider.setMaximum(100)
        self.spine_threshold_slider.setSingleStep(1)
        self.spine_threshold_slider.valueChanged.connect(self._on_value_changed)
        self.apply_user_threshold_check_box = QCheckBox()
        self.apply_user_threshold_check_box.toggled.connect(self._on_value_changed)
        self.threshold_mode = QComboBox(self)
        self.threshold_mode.addItems(["trunk", "spine"])
        self.threshold_mode.currentTextChanged.connect(self._on_threshold_mode_changed)

        self.set_params(params)

        layout = QtFormLayout()
        self.setLayout(layout)
        add_model_path_row(layout, params, self)
        if self._stage_id == 3:
            self.trunk_threshold_slider.hide()
            self.spine_threshold_slider.hide()
            layout.addRow(QtLabel("threshold", {"en", "ru"}, self), self.threshold_slider)
            layout.addRow(
                QtLabel("apply user threshold", {"en", "ru"}, self),
                self.apply_user_threshold_check_box,
            )
        else:
            self.threshold_slider.hide()
            layout.addRow(
                QtLabel("threshold target", {"en", "ru"}, self),
                self.threshold_mode,
            )
            layout.addRow(
                QtLabel("trunk threshold", {"en", "ru"}, self),
                self.trunk_threshold_slider,
            )
            layout.addRow(
                QtLabel("spine threshold", {"en", "ru"}, self),
                self.spine_threshold_slider,
            )
        layout.addRow(self.fix_button)

        if self.fixed:
            self.fix_button.hide()
            self._block_parameters()

    def set_params(self, params: dict = {}) -> None:
        self.threshold_slider.setValue(int(params.get("threshold", 50)))
        self.trunk_threshold_slider.setValue(int(params.get("trunk_threshold", 50)))
        self.spine_threshold_slider.setValue(int(params.get("spine_threshold", 80)))
        self.apply_user_threshold_check_box.setChecked(
            params.get("apply_user_threshold", False)
        )
        mode = str(params.get("threshold_mode", "trunk"))
        self.threshold_mode.setCurrentText(mode if mode in ("trunk", "spine") else "trunk")
        self._on_threshold_mode_changed()

    def get_params(self) -> dict:
        return {
            "threshold": self.threshold_slider.value(),
            "trunk_threshold": self.trunk_threshold_slider.value(),
            "spine_threshold": self.spine_threshold_slider.value(),
            "apply_user_threshold": self.apply_user_threshold_check_box.isChecked(),
            "threshold_mode": self.threshold_mode.currentText(),
        }

    def _on_threshold_mode_changed(self) -> None:
        tune_trunk = self.threshold_mode.currentText() == "trunk"
        self.trunk_threshold_slider.setEnabled(tune_trunk and not self.fixed)
        self.spine_threshold_slider.setEnabled(not tune_trunk and not self.fixed)
        if hasattr(self, "threshold_slider"):
            self._on_value_changed()

    def _block_parameters(self) -> None:
        self.threshold_slider.setEnabled(False)
        self.trunk_threshold_slider.setEnabled(False)
        self.spine_threshold_slider.setEnabled(False)
        self.apply_user_threshold_check_box.setEnabled(False)
        self.threshold_mode.setEnabled(False)
