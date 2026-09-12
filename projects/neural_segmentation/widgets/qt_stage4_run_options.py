from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QSpinBox

from utils.qt_translater import Translater
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtStage4RunOptions(QGroupBox):
    """Editable parameters used when stage 4 inference is started."""

    def __init__(self, params: dict | None = None, parent=None):
        super().__init__(parent)
        self._translater = Translater.instance()
        self._translater.language_changed_signal.connect(self._update_title)
        self._update_title()
        params = params or {}

        self.trunk_threshold = QSpinBox(self)
        self.trunk_threshold.setRange(0, 100)
        self.trunk_threshold.setSuffix(" %")
        self.trunk_threshold.setValue(int(params.get("trunk_threshold", 50)))

        self.spine_threshold = QSpinBox(self)
        self.spine_threshold.setRange(0, 100)
        self.spine_threshold.setSuffix(" %")
        self.spine_threshold.setValue(int(params.get("spine_threshold", 80)))

        self.threshold_mode = QComboBox(self)
        self.threshold_mode.addItems(["trunk", "spine"])
        self.threshold_mode.setCurrentText(str(params.get("threshold_mode", "trunk")))
        self.threshold_mode.currentTextChanged.connect(self._update_threshold_mode)

        self.fill_holes = QCheckBox(self)
        self.fill_holes.setChecked(bool(params.get("fill_holes_before_stage4", True)))

        self.overlap_percent = QComboBox(self)
        self.overlap_percent.addItems(["75 %", "50 %", "20 %"])
        overlap = int(params.get("stage4_overlap_percent", 75))
        self.overlap_percent.setCurrentText(f"{overlap if overlap in (75, 50, 20) else 75} %")

        self.mixed_precision = QCheckBox(self)
        self.mixed_precision.setChecked(
            bool(params.get("stage4_mixed_precision", False))
        )

        self.skeleton_prune_length = QDoubleSpinBox(self)
        self.skeleton_prune_length.setRange(0.0, 6.0)
        self.skeleton_prune_length.setSingleStep(0.1)
        self.skeleton_prune_length.setSuffix(" µm")
        self.skeleton_prune_length.setValue(
            float(params.get("stage4_skeleton_prune_length_um", 1.0))
        )

        layout = QtFormLayout()
        layout.setContentsMargins(3, 4, 5, 6)
        self.setLayout(layout)
        layout.addRow(QtLabel("threshold target", {"en", "ru"}, self), self.threshold_mode)
        layout.addRow(QtLabel("trunk threshold", {"en", "ru"}, self), self.trunk_threshold)
        layout.addRow(QtLabel("spine threshold", {"en", "ru"}, self), self.spine_threshold)
        layout.addRow(QtLabel("fill holes before stage 4", {"en", "ru"}, self), self.fill_holes)
        layout.addRow(QtLabel("stage 4 overlap", {"en", "ru"}, self), self.overlap_percent)
        layout.addRow(QtLabel("mixed precision", {"en", "ru"}, self), self.mixed_precision)
        layout.addRow(
            QtLabel("skeleton prune length", {"en", "ru"}, self),
            self.skeleton_prune_length,
        )
        self._update_threshold_mode()

    def _update_threshold_mode(self) -> None:
        tune_trunk = self.threshold_mode.currentText() == "trunk"
        self.trunk_threshold.setEnabled(tune_trunk)
        self.spine_threshold.setEnabled(not tune_trunk)

    def _update_title(self) -> None:
        self.setTitle(self._translater.get_translation("stage 4 launch parameters"))

    def get_params(self) -> dict:
        return {
            "trunk_threshold": self.trunk_threshold.value(),
            "spine_threshold": self.spine_threshold.value(),
            "threshold_mode": self.threshold_mode.currentText(),
            "fill_holes_before_stage4": self.fill_holes.isChecked(),
            "overlap_percent": int(self.overlap_percent.currentText().split()[0]),
            "mixed_precision": self.mixed_precision.isChecked(),
            "skeleton_prune_length_nm": self.skeleton_prune_length.value() * 1000.0,
        }
