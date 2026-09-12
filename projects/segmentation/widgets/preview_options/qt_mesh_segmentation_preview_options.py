from json import load

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QWidget
from superqt import QLabeledDoubleSlider as QDoubleSlider
from superqt import QLabeledSlider as QSlider
from superqt import QLargeIntSpinBox

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from utils.qt_translater import Translater
from viewer.components.camera import Camera
from viewer.layers.surface.surface import Surface
from widgets.qt_custom_button import QtPushButton
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout
from widgets.qt_line import QtHLine


class QtMeshSegmentationPreviewOptions(QtPreviewOptions):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()
    parameters_fixed_partially_ = pyqtSignal()
    spine_parameters_changed_ = pyqtSignal()
    spine_deleted_ = pyqtSignal()
    spine_restored_ = pyqtSignal()
    current_spine_changed_ = pyqtSignal()
    camera_policy_changed_ = pyqtSignal()

    def __init__(
        self,
        layer: Surface,
        camera: Camera,
        folder: str,
        spines_file: str,
        deleted_spines: set,
        params: dict = {},
        fixed: bool = False,
        partially_fixed: bool = False,
        parent=None,
    ):
        super().__init__(fixed, parent)

        self.partially_fixed = partially_fixed
        self._layer = layer
        self._camera = camera
        self._deleted_spines = deleted_spines
        with open(folder + spines_file) as f:
            self._spines = load(f)

        self.sensitivity_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.sensitivity_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sensitivity_slider.setMinimum(-1.0)
        self.sensitivity_slider.setMaximum(0.0)
        self.sensitivity_slider.setSingleStep(0.01)
        self.sensitivity_slider.valueChanged.connect(self._on_value_changed)
        self.correction_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.correction_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.correction_slider.setMinimum(-20)
        self.correction_slider.setMaximum(20)
        self.correction_slider.setSingleStep(1)
        self.correction_slider.valueChanged.connect(self._on_value_changed)
        self.min_volume_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.min_volume_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.min_volume_slider.setMinimum(-5.0)
        self.min_volume_slider.setMaximum(0.0)
        self.min_volume_slider.setSingleStep(0.1)
        self.min_volume_slider.valueChanged.connect(self._on_value_changed)
        self.partially_fix_button = QtPushButton("fix parameters", parent=self)
        self.partially_fix_button.clicked.connect(self._partially_fix_parameters)
        self.line1 = QtHLine()
        self.spine_label = QtLabel("spine", parent=self)
        self.selection_spin_box = QLargeIntSpinBox()
        self.selection_spin_box.setRange(
            0,
            len(self._spines["spines"]) - 1,
        )
        self.selection_spin_box.setKeyboardTracking(False)
        self.selection_spin_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.move_camera_label = QtLabel("move camera to spine", parent=self)
        self.move_camera_check_box = QCheckBox()
        self.move_camera_check_box.toggled.connect(self._on_camera_policy_changed)
        self.spine_correction_label = QtLabel("spine correction", parent=self)
        self.spine_correction_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.spine_correction_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.spine_correction_slider.setMinimum(-20)
        self.spine_correction_slider.setMaximum(20)
        self.spine_correction_slider.setSingleStep(1)
        self.spine_correction_slider.valueChanged.connect(self._on_spine_value_changed)

        self.warning_area = QWidget(self)
        self.warning_label = QtLabel("", {"en", "ru"}, parent=self.warning_area)
        icon_label = QWidget(self.warning_area)
        icon_label.setObjectName("warning_icon_element")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(icon_label)
        layout.addWidget(self.warning_label)
        self.warning_area.setLayout(layout)

        self.warning_area_2 = QWidget(self)
        self.warning_label_2 = QtLabel("", {"en", "ru"}, parent=self.warning_area_2)
        icon_label = QWidget(self.warning_area_2)
        icon_label.setObjectName("warning_icon_element")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(icon_label)
        layout.addWidget(self.warning_label_2)
        self.warning_area_2.setLayout(layout)

        self.remove_call_button = QtPushButton("remove spine", parent=self)
        self.remove_call_button.clicked.connect(self._remove_spine)
        self.restore_call_button = QtPushButton("restore spine", parent=self)
        self.restore_call_button.clicked.connect(self._restore_spine)
        self.line2 = QtHLine()

        self.set_params(params)
        self.selection_spin_box.valueChanged.connect(self._highlight_spine)

        layout = QtFormLayout()
        self.setLayout(layout)
        layout.addRow(QtLabel("sensitivity", parent=self), self.sensitivity_slider)
        layout.addRow(QtLabel("correction", parent=self), self.correction_slider)
        layout.addRow(QtLabel("min volume", parent=self), self.min_volume_slider)
        layout.addRow(self.partially_fix_button)
        layout.addRow(self.line1)
        layout.addRow(self.spine_label, self.selection_spin_box)
        layout.addRow(self.move_camera_label, self.move_camera_check_box)
        layout.addRow(self.spine_correction_label, self.spine_correction_slider)
        layout.addRow(self.warning_area)
        layout.addRow(self.warning_area_2)
        layout.addRow(self.remove_call_button)
        layout.addRow(self.restore_call_button)
        layout.addRow(self.line2)
        layout.addRow(self.fix_button)

        if self.fixed:
            self.partially_fix_button.hide()
            self.warning_area.hide()
            self.remove_call_button.hide()
            self.restore_call_button.hide()
            self.line2.hide()
            self.fix_button.hide()
            if len(self._spines["spines"]) == 0:
                self.line1.hide()
                self.spine_label.hide()
                self.selection_spin_box.hide()
                self.move_camera_label.hide()
                self.move_camera_check_box.hide()
                self.spine_correction_label.hide()
                self.spine_correction_slider.hide()
                self.warning_area_2.hide()
            else:
                self._highlight_spine()
            self._block_parameters()
        elif self.partially_fixed:
            self.partially_fix_button.hide()
            self._block_segmentation_parameters()
            self._highlight_spine()
        else:
            self.line1.hide()
            self.spine_label.hide()
            self.selection_spin_box.hide()
            self.move_camera_label.hide()
            self.move_camera_check_box.hide()
            self.spine_correction_label.hide()
            self.spine_correction_slider.hide()
            self.warning_area.hide()
            self.warning_area_2.hide()
            self.remove_call_button.hide()
            self.restore_call_button.hide()
            self.line2.hide()
            self.fix_button.hide()

    def _on_ndisplay_changed(self) -> None:
        if self._ndisplay == 3 and self.move_camera_check_box.isChecked():
            spine_info = self._spines["spines"][
                str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
            ]
            self._camera.center = spine_info["average"]

    def _highlight_spine(self) -> None:
        spine_info = self._spines["spines"][
            str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
        ]

        vertex_values = self._layer.vertex_values
        for spine in self._spines["spines"].values():
            for point in spine["indices"]:
                vertex_values[point] = (
                    0.0 if spine["pos"] in self._deleted_spines else 1.0
                )

        if len(spine_info["child_spines"]) > 0:
            for child_id in spine_info["child_spines"]:
                value = (
                    0.25
                    if self._spines["spines"][str(child_id)]["pos"]
                    in self._deleted_spines
                    else 0.75
                )
                for point in self._spines["spines"][str(child_id)]["indices"]:
                    vertex_values[point] = value
        else:
            value = 0.25 if spine_info["pos"] in self._deleted_spines else 0.75
            for point in spine_info["indices"]:
                vertex_values[point] = value
        self._layer.vertex_values = vertex_values

        if not "link" in self._layer.metadata:
            if spine_info["pos"] in self._deleted_spines:
                if not self.fixed:
                    self.restore_call_button.show()
                else:
                    self.restore_call_button.hide()
                self.remove_call_button.hide()
                self.warning_area.hide()
                self.warning_area_2.hide()
            else:
                self.restore_call_button.hide()
                if not self.fixed:
                    self.remove_call_button.show()
                else:
                    self.remove_call_button.hide()
                if spine_info["parent_id"] is not None:
                    if (
                        self._spines["spines"][str(spine_info["parent_id"])]["pos"]
                        not in self._deleted_spines
                    ):
                        self.warning_label.setText(
                            Translater.instance().get_translation("spine depend")
                            + f" {self._spines['spines'][str(spine_info['parent_id'])]['pos']}"
                        )
                        self.warning_area.show()
                    else:
                        self.warning_area.hide()
                elif len(spine_info["child_spines"]) > 0:
                    spines_str = " "
                    for child_spine in spine_info["child_spines"]:
                        spines_str = (
                            spines_str
                            + str(self._spines["spines"][str(child_spine)]["pos"])
                            + ","
                        )
                    self.warning_label.setText(
                        Translater.instance().get_translation("spine affect")
                        + spines_str[:-1]
                    )
                    self.warning_area.show()
                else:
                    self.warning_area.hide()
                if (
                    len(spine_info["child_spines"]) == 0
                    and len(spine_info.get("intersecting_spines", [])) > 0
                ):
                    spines_str = " "
                    for intersecting_spine in spine_info.get("intersecting_spines", []):
                        spines_str = (
                            spines_str
                            + str(
                                self._spines["spines"][str(intersecting_spine)]["pos"]
                            )
                            + ","
                        )
                    self.warning_label_2.setText(
                        Translater.instance().get_translation("spine intersect")
                        + spines_str[:-1]
                    )
                    self.warning_area_2.show()
                else:
                    self.warning_area_2.hide()
        else:
            self.warning_area.hide()
            self.warning_area_2.hide()
            self.remove_call_button.hide()
            self.restore_call_button.hide()

        self.spine_correction_slider.valueChanged.disconnect(
            self._on_spine_value_changed
        )
        self.spine_correction_slider.setValue(spine_info["correction"])
        self.spine_correction_slider.valueChanged.connect(self._on_spine_value_changed)
        if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
            self._camera.center = spine_info["average"]
        self.current_spine_changed_.emit()

    def _restore_spine(self) -> None:
        self._deleted_spines.remove(self.selection_spin_box.value())
        self._layer._changed = True
        self._highlight_spine()
        self.spine_restored_.emit()

    def _remove_spine(self) -> None:
        self._deleted_spines.add(self.selection_spin_box.value())
        self._layer._changed = True
        self._highlight_spine()
        self.spine_deleted_.emit()

    def _on_camera_policy_changed(self) -> None:
        if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
            self._camera.center = self._spines["spines"][
                str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
            ]["average"]
        self.camera_policy_changed_.emit()

    def _on_spine_value_changed(self) -> None:
        self.selection_spin_box.setEnabled(False)
        self.spine_parameters_changed_.emit()

    def _partially_fix_parameters(self) -> None:
        if len(self._spines["spines"]) > 0:
            if not self.partially_fixed:
                self.partially_fixed = True
                self.partially_fix_button.hide()
                self._block_segmentation_parameters()
                self.line1.show()
                self.spine_label.show()
                self.selection_spin_box.show()
                self.move_camera_label.show()
                self.move_camera_check_box.show()
                self.spine_correction_label.show()
                self.spine_correction_slider.show()
                self.line2.show()
                self.fix_button.show()
                self._highlight_spine()
                self.parameters_fixed_partially_.emit()
        else:
            self._fix_parameters()

    def _fix_parameters(self) -> None:
        if not self.fixed:
            self.warning_area.hide()
            self.remove_call_button.hide()
            self.restore_call_button.hide()
            self.line2.hide()
            super()._fix_parameters()

    def _block_segmentation_parameters(self) -> None:
        self.sensitivity_slider.setEnabled(False)
        self.correction_slider.setEnabled(False)
        self.min_volume_slider.setEnabled(False)

    def _block_parameters(self):
        self._block_segmentation_parameters()
        self.spine_correction_slider.setEnabled(False)

    def set_params(self, params: dict = {}) -> None:
        self.sensitivity_slider.setValue(params.get("sensitivity", -1.0))
        self.correction_slider.setValue(params.get("correction", 0))
        self.min_volume_slider.setValue(params.get("min_volume", -1.0))
        self.move_camera_check_box.setChecked(params.get("move_camera", False))
        if self.partially_fixed or self.fixed:
            self.selection_spin_box.setValue(params.get("current_spine", 0))
            if params.get("spine_blocked", None) is not None:
                self.selection_spin_box.setEnabled(False)
            else:
                self.selection_spin_box.setEnabled(True)
            if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
                self._camera.center = self._spines["spines"][
                    str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
                ]["average"]

    def get_params(self) -> dict:
        return {
            "sensitivity": self.sensitivity_slider.value(),
            "correction": self.correction_slider.value(),
            "min_volume": self.min_volume_slider.value(),
        }

    def __del__(self) -> None:
        if self._block_dtor_effects:
            return

        try:
            vertices, facets, vertex_values = self._layer.data
            for spine in self._spines["spines"].values():
                for point in spine["indices"]:
                    vertex_values[point] = (
                        0.0 if spine["pos"] in self._deleted_spines else 1.0
                    )
            self._layer.data = vertices, facets, vertex_values
        except:
            pass
