from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout
from superqt import QLabeledSlider as QSlider

from utils.qt_block_signals import qt_signals_blocked
from utils.qt_translater import Translater
from viewer.layers.labels.labels_constants import Mode, PaintMode
from viewer.widgets.layer_controls.qt_layer_controls_base import QtLayerControls
from viewer.widgets.layer_controls.utils import set_widgets_enabled_with_opacity
from viewer.widgets.qt_color_swatch import QColorSwatchEdit
from viewer.widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from viewer.layers.labels.labels import Labels


class QtLabelsControls(QtLayerControls):
    layer: "Labels"

    def __init__(self, layer, parent=None) -> None:
        super().__init__(layer, parent)

        self.layer.mode_.connect(self._on_mode_change)
        self.layer.colormap_.connect(self._on_color_change)
        self.layer.brush_settings_.connect(self._on_brush_settings_change)
        self.layer.editable_.connect(self._on_editable_ndisplay_or_visible_change)
        self.layer.visible_.connect(self._on_editable_ndisplay_or_visible_change)
        self.layer.set_data_.connect(self._on_editable_ndisplay_or_visible_change)
        self.layer.set_data_.connect(self._on_brush_settings_change)
        self.layer.selected_label_.connect(self._on_selected_label_change)
        self.layer.labels_.connect(self._on_labels_change)
        self.layer.fill_3d_.connect(self._on_fill_3d_change)
        self.layer.show_selected_label_.connect(self._on_show_selected_label_change)
        self.layer.preserve_background_.connect(self._on_preserve_background_change)

        self.labels_label = QtLabel("label", parent=self)
        self.labelsComboBox = QComboBox(self)
        self.labelsComboBox.currentTextChanged.connect(self.changeSelection)

        self.color_label = QtLabel("label color", {"en", "ru"}, parent=self)
        self.colorEdit = QColorSwatchEdit(
            parent=parent,
            initial_color=self.layer.color,
        )
        self.colorEdit.color_changed.connect(self.changeColor)

        self.paint_mode_label = QtLabel("paint mode", {"en", "ru"}, parent=self)
        self.paintModeComboBox = QComboBox(self)
        self.paintModeComboBox.currentTextChanged.connect(self.changeBrushSettings)
        self._update_paint_mode_combo()

        self.brush_x_radius_label = QtLabel("brush x radius", {"en", "ru"}, parent=self)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(30)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeBrushSettings)
        self.brushXRadiusSlider = sld

        self.brush_y_radius_label = QtLabel("brush y radius", {"en", "ru"}, parent=self)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(30)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeBrushSettings)
        self.brushYRadiusSlider = sld

        self.brush_z_radius_label = QtLabel("brush z radius", {"en", "ru"}, parent=self)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(30)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeBrushSettings)
        self.brushZRadiusSlider = sld

        self.brush_all_same_radii_label = QtLabel(
            "all same radii", {"en", "ru"}, parent=self
        )
        self.brushAllSameRadiiCheckBox = QCheckBox(self)
        self.brushAllSameRadiiCheckBox.stateChanged.connect(self.changeBrushSettings)

        self.cylinder_limited_height_depth_label = QtLabel(
            "cylinder limited height depth", {"en", "ru"}, parent=self
        )
        self.cylinderLimitedHeightDepthCheckBox = QCheckBox(self)
        self.cylinderLimitedHeightDepthCheckBox.stateChanged.connect(
            self.changeBrushSettings
        )

        self.cylinder_height_depth_label = QtLabel(
            "cylinder height depth", {"en", "ru"}, parent=self
        )
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(np.max(self.layer.data.shape))
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.changeBrushSettings)
        self.cylinderHeightDepthSlider = sld

        self.preserve_background_label = QtLabel(
            "preserve background", {"en", "ru"}, parent=self
        )
        self.preserveBackgroundCheckBox = QCheckBox(self)
        self.preserveBackgroundCheckBox.stateChanged.connect(
            self.change_preserve_background
        )
        self._on_preserve_background_change()

        self.selected_color_label = QtLabel("selected color", {"en", "ru"}, parent=self)
        self.selectedColorCheckbox = QCheckBox(self)
        self.selectedColorCheckbox.stateChanged.connect(self.toggle_selected_mode)
        self._on_show_selected_label_change()

        self.fill3d_label = QtLabel("fill 3d", {"en", "ru"}, parent=self)
        self.fill3dCheckBox = QCheckBox(self)
        self.fill3dCheckBox.stateChanged.connect(self.toggle_fill_3d)
        self._on_fill_3d_change()

        # shuffle colormap button
        self.colormapUpdate = QtModePushButton(
            layer,
            "shuffle",
            slot=self.changeColors,
            tooltip="update colors",
        )

        self.panzoom_button = QtModeRadioButton(
            layer,
            "pan",
            Mode.PAN_ZOOM,
            checked=(layer.mode == Mode.PAN_ZOOM),
        )
        self.paint_button = QtModeRadioButton(
            layer, "paint", Mode.PAINT, checked=(layer.mode == Mode.PAINT)
        )
        self.fill_button = QtModeRadioButton(
            layer, "fill", Mode.FILL, checked=(layer.mode == Mode.FILL)
        )
        self.erase_button = QtModeRadioButton(
            layer, "erase", Mode.ERASE, checked=(layer.mode == Mode.ERASE)
        )
        self.connected_components_button = QtModeRadioButton(
            layer,
            "connected_components",
            Mode.CONNECTED_COMPONENTS,
            checked=(layer.mode == Mode.CONNECTED_COMPONENTS),
        )

        self._EDIT_BUTTONS = (
            self.paint_button,
            self.fill_button,
            self.erase_button,
        )

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.panzoom_button)
        button_row.addWidget(self.paint_button)
        button_row.addWidget(self.erase_button)
        button_row.addWidget(self.fill_button)
        button_row.addWidget(self.connected_components_button)
        button_row.addWidget(self.colormapUpdate)
        button_row.setSpacing(4)
        button_row.setContentsMargins(0, 0, 0, 5)

        self.layout().addRow(button_row)
        self.layout().addRow(QtLabel("blending", parent=self), self.blendComboBox)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(self.labels_label, self.labelsComboBox)
        self.layout().addRow(self.color_label, self.colorEdit)
        self.layout().addRow(self.paint_mode_label, self.paintModeComboBox)
        self.layout().addRow(
            self.brush_all_same_radii_label, self.brushAllSameRadiiCheckBox
        )
        self.layout().addRow(self.brush_x_radius_label, self.brushXRadiusSlider)
        self.layout().addRow(self.brush_y_radius_label, self.brushYRadiusSlider)
        self.layout().addRow(self.brush_z_radius_label, self.brushZRadiusSlider)
        self.layout().addRow(
            self.cylinder_limited_height_depth_label,
            self.cylinderLimitedHeightDepthCheckBox,
        )
        self.layout().addRow(
            self.cylinder_height_depth_label, self.cylinderHeightDepthSlider
        )
        self.layout().addRow(
            self.preserve_background_label, self.preserveBackgroundCheckBox
        )
        self.layout().addRow(self.selected_color_label, self.selectedColorCheckbox)
        self.layout().addRow(self.fill3d_label, self.fill3dCheckBox)

        self._on_editable_ndisplay_or_visible_change()
        self._on_labels_change()
        self._on_mode_change(self.layer.mode)

    def language_changed(self):
        super().language_changed()
        self._update_labels_combo()
        self._update_paint_mode_combo()

    def _update_labels_combo(self):
        translater = Translater.instance()
        with qt_signals_blocked(self.labelsComboBox):
            self.labelsComboBox.clear()
            for label in self.layer.labels:
                self.labelsComboBox.addItem(translater.get_translation(label), label)

    def _update_paint_mode_combo(self):
        translater = Translater.instance()
        with qt_signals_blocked(self.paintModeComboBox):
            self.paintModeComboBox.clear()
            for index, mode in enumerate(PaintMode):
                self.paintModeComboBox.addItem(
                    translater.get_translation(mode.value), mode
                )
                if mode == self.layer.brush_settings.current_mode:
                    self.paintModeComboBox.setCurrentIndex(index)

    def _on_mode_change(self, mode):
        if mode == Mode.PAN_ZOOM:
            with qt_signals_blocked(self.panzoom_button):
                self.panzoom_button.setChecked(True)
        elif mode == Mode.PAINT:
            with qt_signals_blocked(self.paint_button):
                self.paint_button.setChecked(True)
        elif mode == Mode.FILL:
            with qt_signals_blocked(self.fill_button):
                self.fill_button.setChecked(True)
        elif mode == Mode.ERASE:
            with qt_signals_blocked(self.erase_button):
                self.erase_button.setChecked(True)
        elif mode == Mode.CONNECTED_COMPONENTS:
            pass
        else:
            raise ValueError("Mode not recognized")
        self._on_show_connected_components_change()

    def _on_brush_settings_change(self):
        if self.layer.mode == Mode.CONNECTED_COMPONENTS:
            return
        with qt_signals_blocked(self.paintModeComboBox), qt_signals_blocked(
            self.brushXRadiusSlider
        ), qt_signals_blocked(self.brushYRadiusSlider), qt_signals_blocked(
            self.brushZRadiusSlider
        ), qt_signals_blocked(
            self.brushAllSameRadiiCheckBox
        ), qt_signals_blocked(
            self.cylinderLimitedHeightDepthCheckBox
        ), qt_signals_blocked(
            self.cylinderHeightDepthSlider
        ):
            translater = Translater.instance()
            self.paintModeComboBox.setCurrentText(
                translater.get_translation(self.layer.brush_settings.current_mode.value)
            )
            self.brushAllSameRadiiCheckBox.setChecked(
                self.layer.brush_settings.all_same_radii
            )
            self.cylinderLimitedHeightDepthCheckBox.setChecked(
                self.layer.brush_settings.elliptical_cylinder_limited_height_depth
            )

            self.paint_mode_label.show()
            self.paintModeComboBox.show()
            self.brush_all_same_radii_label.show()
            self.brushAllSameRadiiCheckBox.show()
            if self.layer.brush_settings.current_mode == PaintMode.ELLIPTICAL_CYLINDER:
                self.cylinder_limited_height_depth_label.show()
                self.cylinderLimitedHeightDepthCheckBox.show()
            else:
                self.cylinder_limited_height_depth_label.hide()
                self.cylinderLimitedHeightDepthCheckBox.hide()

            value = np.maximum(1, int(self.layer.brush_settings.radii[2]))
            if value > self.brushXRadiusSlider.maximum():
                self.brushXRadiusSlider.setMaximum(value)
            self.brushXRadiusSlider.setValue(value)

            value = np.maximum(1, int(self.layer.brush_settings.radii[1]))
            if value > self.brushYRadiusSlider.maximum():
                self.brushYRadiusSlider.setMaximum(value)
            self.brushYRadiusSlider.setValue(value)

            value = np.maximum(1, int(self.layer.brush_settings.radii[0]))
            if value > self.brushZRadiusSlider.maximum():
                self.brushZRadiusSlider.setMaximum(value)
            self.brushZRadiusSlider.setValue(value)

            min_, max_ = 1, np.max(self.layer.data.shape)
            value = np.minimum(
                max_,
                np.maximum(
                    min_,
                    int(self.layer.brush_settings.elliptical_cylinder_height_depth),
                ),
            )
            self.cylinderHeightDepthSlider.setMaximum(max_)
            self.cylinderHeightDepthSlider.setValue(value)

            if self.layer.brush_settings.current_mode == PaintMode.ELLIPSOID or (
                2 in self.layer._slice_input.displayed
            ):
                self.brush_x_radius_label.show()
                self.brushXRadiusSlider.show()
            else:
                self.brush_x_radius_label.hide()
                self.brushXRadiusSlider.hide()

            if (
                self.layer.brush_settings.current_mode == PaintMode.ELLIPSOID
                and not self.layer.brush_settings.all_same_radii
            ) or (
                self.layer.brush_settings.current_mode != PaintMode.ELLIPSOID
                and 1 in self.layer._slice_input.displayed
                and not (
                    self.layer.brush_settings.all_same_radii
                    and 2 in self.layer._slice_input.displayed
                )
            ):
                self.brush_y_radius_label.show()
                self.brushYRadiusSlider.show()
            else:
                self.brush_y_radius_label.hide()
                self.brushYRadiusSlider.hide()

            if (
                self.layer.brush_settings.current_mode == PaintMode.ELLIPSOID
                and not self.layer.brush_settings.all_same_radii
            ) or (
                self.layer.brush_settings.current_mode != PaintMode.ELLIPSOID
                and 0 in self.layer._slice_input.displayed
                and not self.layer.brush_settings.all_same_radii
            ):
                self.brush_z_radius_label.show()
                self.brushZRadiusSlider.show()
            else:
                self.brush_z_radius_label.hide()
                self.brushZRadiusSlider.hide()

            if (
                self.layer.brush_settings.elliptical_cylinder_limited_height_depth
                and self.layer.brush_settings.current_mode
                == PaintMode.ELLIPTICAL_CYLINDER
            ):
                self.cylinder_height_depth_label.show()
                self.cylinderHeightDepthSlider.show()
            else:
                self.cylinder_height_depth_label.hide()
                self.cylinderHeightDepthSlider.hide()

    def changeBrushSettings(self):
        self.layer.brush_settings_.disconnect(self._on_brush_settings_change)
        brush_settings = self.layer.brush_settings
        brush_settings.current_mode = self.paintModeComboBox.currentData()
        brush_settings.elliptical_cylinder_limited_height_depth = (
            self.cylinderLimitedHeightDepthCheckBox.isChecked()
        )
        brush_settings.all_same_radii = self.brushAllSameRadiiCheckBox.isChecked()
        brush_settings.radii[2] = self.brushXRadiusSlider.value()
        brush_settings.radii[1] = self.brushYRadiusSlider.value()
        brush_settings.radii[0] = self.brushZRadiusSlider.value()
        brush_settings.elliptical_cylinder_height_depth = (
            self.cylinderHeightDepthSlider.value()
        )
        self.layer.brush_settings = brush_settings
        self._on_brush_settings_change()
        self.layer.brush_settings_.connect(self._on_brush_settings_change)

    def changeColors(self):
        self.layer.new_connected_components_colormap()

    def changeSelection(self):
        self.layer.selected_label = self.labelsComboBox.currentIndex()

    def toggle_selected_mode(self, state):
        self.layer.show_selected_label_.disconnect(self._on_show_selected_label_change)
        self.layer.show_selected_label = Qt.CheckState(state) == Qt.CheckState.Checked
        self.layer.show_selected_label_.connect(self._on_show_selected_label_change)

    def toggle_fill_3d(self, state):
        self.layer.fill_3d_.disconnect(self._on_fill_3d_change)
        self.layer.fill_3d = Qt.CheckState(state) == Qt.CheckState.Checked
        self.layer.fill_3d_.connect(self._on_fill_3d_change)

    def change_preserve_background(self, state):
        self.layer.preserve_background_.disconnect(self._on_preserve_background_change)
        self.layer.preserve_background = Qt.CheckState(state) == Qt.CheckState.Checked
        self.layer.preserve_background_.connect(self._on_preserve_background_change)

    def changeColor(self, color: np.ndarray):
        self.layer.colormap_.disconnect(self._on_color_change)
        self.layer.color = color
        self.layer.colormap_.connect(self._on_color_change)

    def _on_labels_change(self) -> None:
        if self.layer.mode == Mode.CONNECTED_COMPONENTS:
            return
        self._update_labels_combo()
        self._on_selected_label_change()
        if len(self.layer.labels) <= 2:
            self.preserve_background_label.hide()
            self.preserveBackgroundCheckBox.hide()
            self.selected_color_label.hide()
            self.selectedColorCheckbox.hide()
        else:
            self.preserve_background_label.show()
            self.preserveBackgroundCheckBox.show()
            self.selected_color_label.show()
            self.selectedColorCheckbox.show()

    def _on_show_connected_components_change(self) -> None:
        with qt_signals_blocked(self.connected_components_button):
            self.connected_components_button.setChecked(
                self.layer.mode == Mode.CONNECTED_COMPONENTS
            )
            if self.layer.mode == Mode.CONNECTED_COMPONENTS:
                self.colormapUpdate.show()
                self.labels_label.hide()
                self.labelsComboBox.hide()
                self.color_label.hide()
                self.colorEdit.hide()
                self.paint_mode_label.hide()
                self.paintModeComboBox.hide()
                self.brush_x_radius_label.hide()
                self.brushXRadiusSlider.hide()
                self.brush_y_radius_label.hide()
                self.brushYRadiusSlider.hide()
                self.brush_z_radius_label.hide()
                self.brushZRadiusSlider.hide()
                self.brush_all_same_radii_label.hide()
                self.brushAllSameRadiiCheckBox.hide()
                self.cylinder_limited_height_depth_label.hide()
                self.cylinderLimitedHeightDepthCheckBox.hide()
                self.cylinder_height_depth_label.hide()
                self.cylinderHeightDepthSlider.hide()
                self.preserve_background_label.hide()
                self.preserveBackgroundCheckBox.hide()
                self.selected_color_label.hide()
                self.selectedColorCheckbox.hide()
                self.fill3d_label.hide()
                self.fill3dCheckBox.hide()
            else:
                self.colormapUpdate.hide()
                self.labels_label.show()
                self.labelsComboBox.show()
                self.color_label.show()
                self.colorEdit.show()
                self.fill3d_label.show()
                self.fill3dCheckBox.show()
                self._on_brush_settings_change()
                self._on_labels_change()
                self._on_color_change()

    def _on_color_change(self):
        if self.layer.mode == Mode.CONNECTED_COMPONENTS:
            return
        with qt_signals_blocked(self.colorEdit):
            if self.layer.selected_label > 0:
                self.color_label.show()
                self.colorEdit.show()
                self.colorEdit.setColor(self.layer.color)
            else:
                self.color_label.hide()
                self.colorEdit.hide()

    def _on_selected_label_change(self):
        with qt_signals_blocked(self.labelsComboBox):
            self.labelsComboBox.setCurrentIndex(self.layer.selected_label)
            self._on_color_change()

    def _on_preserve_background_change(self):
        with qt_signals_blocked(self.preserveBackgroundCheckBox):
            self.preserveBackgroundCheckBox.setChecked(self.layer.preserve_background)

    def _on_show_selected_label_change(self):
        with qt_signals_blocked(self.selectedColorCheckbox):
            self.selectedColorCheckbox.setChecked(self.layer.show_selected_label)

    def _on_fill_3d_change(self):
        with qt_signals_blocked(self.fill3dCheckBox):
            self.fill3dCheckBox.setChecked(self.layer.fill_3d)

    def _on_ndisplay_changed(self):
        self._on_editable_ndisplay_or_visible_change()

    def _on_editable_ndisplay_or_visible_change(self):
        enabled = (
            self.layer.editable
            and self.layer.visible
            and self.layer._slice_input.ndisplay == 2
        )
        if not enabled and self.layer.mode in {
            Mode.ERASE,
            Mode.FILL,
            Mode.PAINT,
        }:
            self.layer.mode = Mode.PAN_ZOOM
        set_widgets_enabled_with_opacity(self, self._EDIT_BUTTONS, enabled)
