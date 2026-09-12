from json import load

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox
from superqt import QLargeIntSpinBox

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from viewer.components.camera import Camera
from viewer.layers.surface.surface import Surface
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtFinalSegmentationPreviewOptions(QtPreviewOptions):
    current_spine_changed_ = pyqtSignal()
    camera_policy_changed_ = pyqtSignal()

    def __init__(
        self,
        layer: Surface,
        camera: Camera,
        folder: str,
        spines_file: str,
        params: dict = {},
        parent=None,
    ):
        super().__init__(True, parent)
        self.fix_button.hide()

        self._layer = layer
        self._camera = camera
        self._prev_spine_points = []
        with open(folder + spines_file) as f:
            self._spines = load(f)

        if len(self._spines["spines"]) > 0:
            self.spine_label = QtLabel("spine", parent=self)
            self.selection_spin_box = QLargeIntSpinBox()
            self.selection_spin_box.setRange(
                0,
                len(self._spines["spines"]) - 1,
            )
            self.selection_spin_box.setKeyboardTracking(False)
            self.selection_spin_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.selection_spin_box.valueChanged.connect(self._highlight_spine)
            self.move_camera_label = QtLabel("move camera to spine", parent=self)
            self.move_camera_check_box = QCheckBox()
            self.move_camera_check_box.toggled.connect(self._on_camera_policy_changed)
            self.set_params(params)

            layout = QtFormLayout()
            self.setLayout(layout)
            layout.addRow(self.spine_label, self.selection_spin_box)
            layout.addRow(self.move_camera_label, self.move_camera_check_box)
            self._highlight_spine()

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

        vertices, facets, vertex_values = self._layer.data
        for point in self._prev_spine_points:
            vertex_values[point] = 1.0
        for point in spine_info["indices"]:
            vertex_values[point] = 0.5
        self._prev_spine_points = spine_info["indices"]
        self._layer.data = vertices, facets, vertex_values

        if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
            self._camera.center = spine_info["average"]
        self.current_spine_changed_.emit()

    def _on_camera_policy_changed(self) -> None:
        if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
            self._camera.center = self._spines["spines"][
                str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
            ]["average"]
        self.camera_policy_changed_.emit()

    def set_params(self, params: dict = {}) -> None:
        self.move_camera_check_box.setChecked(params.get("move_camera", False))
        self.selection_spin_box.setValue(params.get("current_spine", 0))
        if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
            self._camera.center = self._spines["spines"][
                str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
            ]["average"]

    def __del__(self) -> None:
        if self._block_dtor_effects:
            return

        try:
            vertices, facets, vertex_values = self._layer.data
            for point in self._prev_spine_points:
                vertex_values[point] = 1.0
            self._layer.data = vertices, facets, vertex_values
        except:
            pass
