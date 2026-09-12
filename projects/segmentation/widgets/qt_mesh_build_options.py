from PyQt5.QtWidgets import QCheckBox, QComboBox, QGroupBox

from projects.segmentation.utils.constants import DEFAULT_MESH_COMPLEXITY
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtMeshBuildOptions(QGroupBox):
    """User-facing choices shared by all voxel-to-mesh entry points."""

    def __init__(self, params: dict | None = None, parent=None):
        super().__init__("Mesh build options", parent)
        params = params or {}
        self.complexity = QComboBox(self)
        self.complexity.addItems(["high", "medium", "low"])
        value = str(params.get("mesh_complexity", DEFAULT_MESH_COMPLEXITY)).lower()
        self.complexity.setCurrentText(value if value in ("high", "medium", "low") else "high")

        self.fill_holes = QCheckBox(self)
        self.fill_holes.setChecked(bool(params.get("fill_holes_before_mesh", True)))

        layout = QtFormLayout()
        layout.setContentsMargins(3, 4, 5, 6)
        self.setLayout(layout)
        layout.addRow(QtLabel("mesh complexity", {"en", "ru"}, self), self.complexity)
        layout.addRow(QtLabel("fill holes before mesh", {"en", "ru"}, self), self.fill_holes)

    def get_params(self) -> dict:
        return {
            "mesh_complexity": self.complexity.currentText(),
            "fill_holes_before_mesh": self.fill_holes.isChecked(),
        }
