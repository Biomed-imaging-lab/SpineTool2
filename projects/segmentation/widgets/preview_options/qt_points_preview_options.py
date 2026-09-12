# file: projects/segmentation/widgets/preview_options/qt_points_preview_options.py
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QLabel, QPushButton, QSpinBox
from superqt import QLabeledDoubleSlider as QDoubleSlider
from superqt import QLabeledSlider as QSlider

from projects.segmentation.widgets.preview_options.qt_preview_options_base import (
    QtPreviewOptions,
)
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout


class QtPointsPreviewOptions(QtPreviewOptions):
    parameters_changed_ = pyqtSignal()
    parameters_fixed_ = pyqtSignal()
    shaft_point_changed_ = pyqtSignal(int, list, bool)
    pair_active_changed_ = pyqtSignal(int, bool)
    shaft_point_reset_ = pyqtSignal(int)
    selected_pair_changed_ = pyqtSignal(int)
    spine_points_fixed_ = pyqtSignal()

    def __init__(self, params: dict = {}, fixed: bool = False, parent=None):
        super().__init__(fixed, parent)

        self._block_pair_ui = False
        self._edit_phase = str(params.get("neck_edit_phase", "spine"))

        self._image_shape = self._read_image_shape(params)
        self._spine_points = self._read_points_array(params.get("spine_points", []))
        self._shaft_points = self._read_points_array(params.get("shaft_points", []))
        self._auto_shaft_points = self._read_points_array(
            params.get("auto_shaft_points", params.get("shaft_points", []))
        )

        self._pair_count = max(
            len(self._spine_points),
            len(self._shaft_points),
            len(self._auto_shaft_points),
        )

        self._spine_points = self._resize_points_array(
            self._spine_points, self._pair_count
        )
        self._shaft_points = self._resize_points_array(
            self._shaft_points, self._pair_count
        )
        self._auto_shaft_points = self._resize_points_array(
            self._auto_shaft_points,
            self._pair_count,
        )

        self._pair_active = self._normalise_bool_list(
            params.get("pair_active", None),
            self._pair_count,
            True,
        )
        self._pair_source = self._normalise_str_list(
            params.get("pair_source", None),
            self._pair_count,
            "auto_old_medial",
        )

        self.min_spine_head_size_slider = QDoubleSlider(Qt.Orientation.Horizontal, self)
        self.min_spine_head_size_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.min_spine_head_size_slider.setMinimum(-5.0)
        self.min_spine_head_size_slider.setMaximum(0.0)
        self.min_spine_head_size_slider.setSingleStep(0.1)
        self.min_spine_head_size_slider.valueChanged.connect(self._on_value_changed)

        self.max_distance_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.max_distance_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.max_distance_slider.setMinimum(1)
        self.max_distance_slider.setMaximum(15)
        self.max_distance_slider.setSingleStep(1)
        self.max_distance_slider.valueChanged.connect(self._on_value_changed)

        self.pairs_title_label = QLabel("Neck restoration pairs", self)
        self.pairs_title_label.setStyleSheet("font-weight: 600; margin-top: 8px;")

        self.pair_count_label = QLabel(self)
        self.pair_index_spinbox = QSpinBox(self)
        self.pair_index_spinbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        if self._pair_count > 0:
            self.pair_index_spinbox.setMinimum(1)
            self.pair_index_spinbox.setMaximum(self._pair_count)
            self.pair_index_spinbox.setValue(1)
        else:
            self.pair_index_spinbox.setMinimum(0)
            self.pair_index_spinbox.setMaximum(0)
            self.pair_index_spinbox.setValue(0)

        self.pair_active_checkbox = QCheckBox(self)
        self.pair_source_label = QLabel(self)
        self.spine_point_label = QLabel(self)

        self.shaft_hint_label = QLabel(
            "Edit shaft point as Z/Y/X voxel coordinates. "
            "Use snap to shaft before restoring necks.",
            self,
        )
        self.shaft_hint_label.setWordWrap(True)

        self.shaft_z_spinbox = QSpinBox(self)
        self.shaft_y_spinbox = QSpinBox(self)
        self.shaft_x_spinbox = QSpinBox(self)

        self.shaft_z_spinbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shaft_y_spinbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shaft_x_spinbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._configure_coordinate_spinboxes()

        self.save_shaft_button = QPushButton("save shaft point", self)
        self.snap_shaft_button = QPushButton("save and snap to shaft", self)
        self.reset_shaft_button = QPushButton("reset to auto shaft point", self)

        self.pair_status_label = QLabel(self)
        self.pair_status_label.setWordWrap(True)

        self.edit_phase_label = QLabel(self)
        self.edit_phase_label.setWordWrap(True)

        self.set_params(params)

        layout = QtFormLayout()
        self.setLayout(layout)

        layout.addRow(QLabel("Restoration point search", self))
        layout.addRow(self.edit_phase_label)
        layout.addRow(
            QtLabel("min volume", {"en", "ru"}, self),
            self.min_spine_head_size_slider,
        )
        layout.addRow(
            QtLabel("max distance", {"en", "ru"}, self),
            self.max_distance_slider,
        )

        layout.addRow(self.pairs_title_label)
        layout.addRow(QLabel("pair count", self), self.pair_count_label)
        layout.addRow(QLabel("selected pair", self), self.pair_index_spinbox)
        layout.addRow(QLabel("use pair", self), self.pair_active_checkbox)
        layout.addRow(QLabel("point source", self), self.pair_source_label)
        layout.addRow(QLabel("spine point", self), self.spine_point_label)

        layout.addRow(QLabel("Shaft point correction", self))
        layout.addRow(self.shaft_hint_label)
        layout.addRow(QLabel("shaft z", self), self.shaft_z_spinbox)
        layout.addRow(QLabel("shaft y", self), self.shaft_y_spinbox)
        layout.addRow(QLabel("shaft x", self), self.shaft_x_spinbox)
        layout.addRow(self.save_shaft_button)
        layout.addRow(self.snap_shaft_button)
        layout.addRow(self.reset_shaft_button)
        layout.addRow(self.pair_status_label)

        layout.addRow(self.fix_button)

        self.pair_index_spinbox.valueChanged.connect(self._on_pair_index_changed)
        self.pair_active_checkbox.toggled.connect(self._on_pair_active_toggled)

        self.shaft_z_spinbox.valueChanged.connect(self._on_coordinate_changed)
        self.shaft_y_spinbox.valueChanged.connect(self._on_coordinate_changed)
        self.shaft_x_spinbox.valueChanged.connect(self._on_coordinate_changed)

        self.save_shaft_button.clicked.connect(
            lambda: self._emit_shaft_point_changed(False)
        )
        self.snap_shaft_button.clicked.connect(
            lambda: self._emit_shaft_point_changed(True)
        )
        self.reset_shaft_button.clicked.connect(self._on_reset_to_auto)

        # Points use a two-step confirmation workflow.  The first click freezes
        # spine points; only the second one fixes and saves the stage.
        try:
            self.fix_button.clicked.disconnect(self._fix_parameters)
        except TypeError:
            pass
        self.fix_button.clicked.connect(self._on_fix_clicked)

        self._load_current_pair()
        self._apply_edit_phase()

        if self.fixed:
            self.fix_button.hide()
            self._block_parameters()

    def set_params(self, params: dict = {}) -> None:
        self.min_spine_head_size_slider.setValue(
            params.get("min_component_volume", -1.0)
        )
        self.max_distance_slider.setValue(params.get("max_distance", 5))

    def get_params(self) -> dict:
        return {
            "min_component_volume": self.min_spine_head_size_slider.value(),
            "max_distance": self.max_distance_slider.value(),
        }

    def _block_parameters(self) -> None:
        self.min_spine_head_size_slider.setEnabled(False)
        self.max_distance_slider.setEnabled(False)

    def _apply_edit_phase(self) -> None:
        if self.fixed:
            self.edit_phase_label.setText("Spine and shaft points are fixed.")
            self._block_parameters()
            self._set_pair_controls_enabled(False)
            return

        if self._edit_phase == "shaft":
            self.edit_phase_label.setText(
                "Step 2 of 2: drag shaft points in the current plane. "
                "Change Z only with the numeric field below."
            )
            self.fix_button.setText("fix shaft points and save layer")
            self._block_parameters()
            self._set_pair_controls_enabled(self._pair_count > 0)
        else:
            self.edit_phase_label.setText(
                "Step 1 of 2: move, add or delete spine points on the image, "
                "then fix them to generate shaft points."
            )
            self.fix_button.setText("fix spine points and edit shaft points")
            self._set_pair_controls_enabled(False)

    def _on_fix_clicked(self) -> None:
        if self.fixed:
            return
        if self._edit_phase == "spine":
            self.spine_points_fixed_.emit()
            return
        self._fix_parameters()

    def update_shaft_points(self, points) -> None:
        """Keep numeric controls synchronized after a canvas drag."""
        updated = self._read_points_array(points)
        if len(updated) != self._pair_count:
            return
        self._shaft_points = updated
        self._load_current_pair()

    def _read_image_shape(self, params: dict) -> tuple[int, int, int]:
        shape = params.get("image_shape", None)
        if shape is None:
            return (1, 1, 1)

        shape = tuple(int(v) for v in shape)
        if len(shape) != 3:
            return (1, 1, 1)

        return shape

    def _read_points_array(self, value) -> np.ndarray:
        if value is None:
            return np.zeros((0, 3), dtype=np.int64)

        arr = np.asarray(value, dtype=np.int64)
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.int64)

        try:
            return arr.reshape((-1, 3))
        except Exception:
            return np.zeros((0, 3), dtype=np.int64)

    def _resize_points_array(self, arr: np.ndarray, size: int) -> np.ndarray:
        out = np.zeros((size, 3), dtype=np.int64)
        if size == 0 or len(arr) == 0:
            return out

        n = min(size, len(arr))
        out[:n] = arr[:n]
        return out

    def _normalise_bool_list(
        self,
        value,
        size: int,
        default: bool,
    ) -> list[bool]:
        if value is None:
            return [default] * size

        out = [bool(v) for v in list(value)]
        if len(out) < size:
            out.extend([default] * (size - len(out)))
        return out[:size]

    def _normalise_str_list(
        self,
        value,
        size: int,
        default: str,
    ) -> list[str]:
        if value is None:
            return [default] * size

        out = [str(v) for v in list(value)]
        if len(out) < size:
            out.extend([default] * (size - len(out)))
        return out[:size]

    def _configure_coordinate_spinboxes(self) -> None:
        z_max = max(0, self._image_shape[0] - 1)
        y_max = max(0, self._image_shape[1] - 1)
        x_max = max(0, self._image_shape[2] - 1)

        self.shaft_z_spinbox.setMinimum(0)
        self.shaft_y_spinbox.setMinimum(0)
        self.shaft_x_spinbox.setMinimum(0)

        self.shaft_z_spinbox.setMaximum(z_max)
        self.shaft_y_spinbox.setMaximum(y_max)
        self.shaft_x_spinbox.setMaximum(x_max)

    def _current_pair_index(self) -> int | None:
        if self._pair_count <= 0:
            return None

        return int(self.pair_index_spinbox.value()) - 1

    def _load_current_pair(self) -> None:
        idx = self._current_pair_index()

        self._block_pair_ui = True
        try:
            if idx is None:
                self.pair_count_label.setText("0")
                self.pair_active_checkbox.setChecked(False)
                self.pair_source_label.setText("no pairs")
                self.spine_point_label.setText("not available")
                self.shaft_z_spinbox.setValue(0)
                self.shaft_y_spinbox.setValue(0)
                self.shaft_x_spinbox.setValue(0)
                self.pair_status_label.setText(
                    "No restoration pairs found. Adjust min volume / max distance "
                    "and recompute restoration points."
                )
                return

            spine_point = self._spine_points[idx]
            shaft_point = self._shaft_points[idx]
            source = self._pair_source[idx]

            self.pair_count_label.setText(f"{idx + 1} / {self._pair_count}")
            self.pair_active_checkbox.setChecked(self._pair_active[idx])
            self.pair_source_label.setText(self._format_pair_source(source))
            self.spine_point_label.setText(
                f"z={int(spine_point[0])}, y={int(spine_point[1])}, x={int(spine_point[2])}"
            )

            self.shaft_z_spinbox.setValue(int(shaft_point[0]))
            self.shaft_y_spinbox.setValue(int(shaft_point[1]))
            self.shaft_x_spinbox.setValue(int(shaft_point[2]))

            if source == "manual":
                self.pair_status_label.setText(
                    "Current pair uses a manually corrected shaft point."
                )
            else:
                self.pair_status_label.setText(
                    "Current pair uses an automatically suggested shaft point."
                )
        finally:
            self._block_pair_ui = False

    def _format_pair_source(self, source: str) -> str:
        if source == "manual":
            return "manual correction"
        if source == "auto_old_medial":
            return "auto suggestion"
        return source

    def _set_pair_controls_enabled(self, enabled: bool) -> None:
        self.pair_index_spinbox.setEnabled(enabled)
        self.pair_active_checkbox.setEnabled(enabled)
        self.shaft_z_spinbox.setEnabled(enabled)
        self.shaft_y_spinbox.setEnabled(enabled)
        self.shaft_x_spinbox.setEnabled(enabled)
        self.save_shaft_button.setEnabled(enabled)
        self.snap_shaft_button.setEnabled(enabled)
        self.reset_shaft_button.setEnabled(enabled)

    def _on_pair_index_changed(self) -> None:
        self._load_current_pair()
        idx = self._current_pair_index()
        if idx is not None:
            self.selected_pair_changed_.emit(idx)

    def _on_pair_active_toggled(self, checked: bool) -> None:
        if self._block_pair_ui:
            return

        idx = self._current_pair_index()
        if idx is None:
            return

        self._pair_active[idx] = bool(checked)
        self.pair_active_changed_.emit(idx, bool(checked))

    def _on_coordinate_changed(self) -> None:
        if self._block_pair_ui:
            return

        if self._current_pair_index() is None:
            return

        self.pair_status_label.setText(
            "Shaft point coordinates were changed. Press save before restoring necks."
        )

    def _current_shaft_point(self) -> list[int]:
        return [
            int(self.shaft_z_spinbox.value()),
            int(self.shaft_y_spinbox.value()),
            int(self.shaft_x_spinbox.value()),
        ]

    def _emit_shaft_point_changed(self, snap_to_shaft: bool) -> None:
        idx = self._current_pair_index()
        if idx is None:
            return

        shaft_point = self._current_shaft_point()

        self._shaft_points[idx] = np.asarray(shaft_point, dtype=np.int64)
        self._pair_source[idx] = "manual"

        self.shaft_point_changed_.emit(idx, shaft_point, snap_to_shaft)

        if snap_to_shaft:
            self.pair_status_label.setText(
                "Shaft point was saved and sent for snapping to the shaft."
            )
        else:
            self.pair_source_label.setText("manual correction")
            self.pair_status_label.setText("Manual shaft point was saved.")

    def _on_reset_to_auto(self) -> None:
        idx = self._current_pair_index()
        if idx is None:
            return

        auto_point = self._auto_shaft_points[idx].astype(int).tolist()

        self._block_pair_ui = True
        try:
            self.shaft_z_spinbox.setValue(auto_point[0])
            self.shaft_y_spinbox.setValue(auto_point[1])
            self.shaft_x_spinbox.setValue(auto_point[2])
        finally:
            self._block_pair_ui = False

        self._shaft_points[idx] = np.asarray(auto_point, dtype=np.int64)
        self._pair_source[idx] = "auto_old_medial"
        self.pair_source_label.setText("auto suggestion")
        self.pair_status_label.setText("Shaft point was reset to automatic suggestion.")

        self.shaft_point_reset_.emit(idx)
