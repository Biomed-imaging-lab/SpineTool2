from PyQt5.QtCore import pyqtSignal

from projects.segmentation.widgets.preview_options.qt_final_segmentation_preview_options import (
    QtFinalSegmentationPreviewOptions,
)
from widgets.qt_custom_button import QtPushButton
from widgets.qt_form_layout import QtFormLayout


class QtNeuralSpineMeshPreviewOptions(QtFinalSegmentationPreviewOptions):
    """Editable neural mesh preview shown before the ordinary final layer."""

    spine_deleted_ = pyqtSignal()
    spine_restored_ = pyqtSignal()

    def __init__(self, layer, camera, folder, spines_file, deleted_spines, params=None):
        self._deleted_spines = deleted_spines
        super().__init__(layer, camera, folder, spines_file, params or {})
        self.fixed = False
        self.fix_button.show()
        if not self._spines["spines"]:
            layout = QtFormLayout()
            self.setLayout(layout)
            layout.addRow(self.fix_button)
            return
        self.remove_spine_button = QtPushButton("remove spine", parent=self)
        self.remove_spine_button.clicked.connect(self._remove_spine)
        self.restore_spine_button = QtPushButton("restore spine", parent=self)
        self.restore_spine_button.clicked.connect(self._restore_spine)
        self.layout().addRow(self.remove_spine_button)
        self.layout().addRow(self.restore_spine_button)
        self.layout().addRow(self.fix_button)
        self._highlight_spine()

    def _highlight_spine(self):
        spine = self._spines["spines"][
            str(self._spines["pos_to_id"][str(self.selection_spin_box.value())])
        ]
        vertices, facets, values = self._layer.data
        for item in self._spines["spines"].values():
            value = 0.0 if item["pos"] in self._deleted_spines else 1.0
            values[item["indices"]] = value
        deleted = spine["pos"] in self._deleted_spines
        values[spine["indices"]] = 0.25 if deleted else 0.5
        self._layer.data = vertices, facets, values
        self._prev_spine_points = spine["indices"]
        if hasattr(self, "remove_spine_button"):
            self.remove_spine_button.setVisible(not deleted)
            self.restore_spine_button.setVisible(deleted)
        if self.move_camera_check_box.isChecked() and self._ndisplay == 3:
            self._camera.center = spine["average"]
        self.current_spine_changed_.emit()

    def _remove_spine(self):
        self._deleted_spines.add(self.selection_spin_box.value())
        self._highlight_spine()
        self.spine_deleted_.emit()

    def _restore_spine(self):
        self._deleted_spines.remove(self.selection_spin_box.value())
        self._highlight_spine()
        self.spine_restored_.emit()

    def __del__(self):
        try:
            vertices, facets, values = self._layer.data
            for item in self._spines["spines"].values():
                value = 0.0 if item["pos"] in self._deleted_spines else 1.0
                values[item["indices"]] = value
            self._layer.data = vertices, facets, values
        except Exception:
            pass
