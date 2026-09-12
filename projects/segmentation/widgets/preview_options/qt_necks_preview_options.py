from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox
from superqt import QLabeledDoubleSlider as QDoubleSlider
from superqt import QLabeledSlider as QSlider

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtNecksPreviewOptions(QtPreviewOptions):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()

    def __init__(self, params: dict = {}, fixed: bool = False, parent=None):
        super().__init__(fixed, parent)

        self.intensity_factor_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.intensity_factor_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.intensity_factor_slider.setMinimum(0)
        self.intensity_factor_slider.setMaximum(3)
        self.intensity_factor_slider.setSingleStep(0.1)

        self.floodfill_radius_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.floodfill_radius_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.floodfill_radius_slider.setMinimum(1)
        self.floodfill_radius_slider.setMaximum(20)
        self.floodfill_radius_slider.setSingleStep(1)

        self.floodfill_factor_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.floodfill_factor_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.floodfill_factor_slider.setMinimum(0)
        self.floodfill_factor_slider.setMaximum(1)
        self.floodfill_factor_slider.setSingleStep(0.1)

        self.use_acwe_check_box = QCheckBox()
        self.use_fixed_endpoints_check_box = QCheckBox()

        self.roi_margin_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.roi_margin_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.roi_margin_slider.setMinimum(1)
        self.roi_margin_slider.setMaximum(50)
        self.roi_margin_slider.setSingleStep(1)

        self.corridor_radius_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.corridor_radius_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.corridor_radius_slider.setMinimum(0.01)
        self.corridor_radius_slider.setMaximum(2.0)
        self.corridor_radius_slider.setSingleStep(0.01)

        self.use_acwe_check_box.toggled.connect(self._on_floodfill_method_changed)

        self.acwe_ext_factor_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.acwe_ext_factor_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.acwe_ext_factor_slider.setMinimum(1)
        self.acwe_ext_factor_slider.setMaximum(10)
        self.acwe_ext_factor_slider.setSingleStep(0.5)

        self.acwe_int_factor_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.acwe_int_factor_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.acwe_int_factor_slider.setMinimum(1)
        self.acwe_int_factor_slider.setMaximum(10)
        self.acwe_int_factor_slider.setSingleStep(0.5)

        self.set_params(params)

        self.use_fixed_endpoints_check_box.toggled.connect(self._on_value_changed)
        self.roi_margin_slider.valueChanged.connect(self._on_value_changed)
        self.corridor_radius_slider.valueChanged.connect(self._on_value_changed)

        self.form_layout = QtFormLayout()
        self.setLayout(self.form_layout)
        self.form_layout.addRow(
            QtLabel("intensity factor", {"en", "ru"}, self),
            self.intensity_factor_slider,
        )
        self.form_layout.addRow(
            QtLabel("floodfill radius", {"en", "ru"}, self),
            self.floodfill_radius_slider,
        )
        self.form_layout.addRow(
            QtLabel("floodfill factor", {"en", "ru"}, self),
            self.floodfill_factor_slider,
        )
        self.form_layout.addRow(
            QtLabel("use acwe", {"en", "ru"}, self),
            self.use_acwe_check_box,
        )
        self.form_layout.addRow(
            QtLabel("acwe ext factor", {"en", "ru"}, self),
            self.acwe_ext_factor_slider,
        )
        self.form_layout.addRow(
            QtLabel("acwe int factor", {"en", "ru"}, self),
            self.acwe_int_factor_slider,
        )
        self.form_layout.addRow(
            QtLabel("fixed endpoints", {"en", "ru"}, self),
            self.use_fixed_endpoints_check_box,
        )
        self.form_layout.addRow(
            QtLabel("roi margin", {"en", "ru"}, self),
            self.roi_margin_slider,
        )
        self.form_layout.addRow(
            QtLabel("corridor radius um", {"en", "ru"}, self),
            self.corridor_radius_slider,
        )
        self.form_layout.addRow(self.fix_button)

        self._on_floodfill_method_changed()

        if self.fixed:
            self.fix_button.hide()
            self._block_parameters()

    def set_params(self, params: dict = {}) -> None:
        self.intensity_factor_slider.setValue(params.get("intensity_factor", 3))
        self.floodfill_radius_slider.setValue(params.get("floodfill_radius", 4))
        self.floodfill_factor_slider.setValue(params.get("floodfill_factor", 0.8))
        self.use_acwe_check_box.setChecked(params.get("use_acwe", False))
        self.acwe_ext_factor_slider.setValue(params.get("acwe_ext_factor", 2))
        self.acwe_int_factor_slider.setValue(params.get("acwe_int_factor", 6))
        self.use_fixed_endpoints_check_box.setChecked(
            params.get("use_fixed_endpoints", True)
        )
        self.roi_margin_slider.setValue(params.get("roi_margin", 8))
        self.corridor_radius_slider.setValue(params.get("corridor_radius_um", 0.25))

    def get_params(self) -> dict:
        return {
            "intensity_factor": self.intensity_factor_slider.value(),
            "floodfill_radius": self.floodfill_radius_slider.value(),
            "floodfill_factor": self.floodfill_factor_slider.value(),
            "use_acwe": self.use_acwe_check_box.isChecked(),
            "acwe_ext_factor": self.acwe_ext_factor_slider.value(),
            "acwe_int_factor": self.acwe_int_factor_slider.value(),
            "use_fixed_endpoints": self.use_fixed_endpoints_check_box.isChecked(),
            "roi_margin": self.roi_margin_slider.value(),
            "corridor_radius_um": self.corridor_radius_slider.value(),
        }

    def _on_floodfill_method_changed(self) -> None:
        use_acwe = self.use_acwe_check_box.isChecked()

        self.floodfill_radius_slider.setEnabled(not use_acwe)
        self.floodfill_factor_slider.setEnabled(not use_acwe)
        self.acwe_ext_factor_slider.setEnabled(use_acwe)
        self.acwe_int_factor_slider.setEnabled(use_acwe)

        self._on_value_changed()

    def _block_parameters(self) -> None:
        self.intensity_factor_slider.setEnabled(False)
        self.floodfill_radius_slider.setEnabled(False)
        self.floodfill_factor_slider.setEnabled(False)
        self.use_acwe_check_box.setEnabled(False)
        self.acwe_ext_factor_slider.setEnabled(False)
        self.acwe_int_factor_slider.setEnabled(False)
        self.use_fixed_endpoints_check_box.setEnabled(False)
        self.roi_margin_slider.setEnabled(False)
        self.corridor_radius_slider.setEnabled(False)
