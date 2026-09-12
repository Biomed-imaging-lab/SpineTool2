import contextlib
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QHBoxLayout
from superqt import QLabeledSlider as QSlider

from utils.qt_block_signals import qt_signals_blocked
from viewer.layers.points._points_constants import (
    SYMBOL_TRANSLATION,
    SYMBOL_TRANSLATION_INVERTED,
    Mode,
)
from viewer.widgets.layer_controls.qt_layer_controls_base import QtLayerControls
from viewer.widgets.layer_controls.utils import set_widgets_enabled_with_opacity
from viewer.widgets.qt_color_swatch import QColorSwatchEdit
from viewer.widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from viewer.layers.points.points import Points


class QtPointsControls(QtLayerControls):
    layer: "Points"

    def __init__(self, layer, parent=None) -> None:
        super().__init__(layer, parent)

        self.layer.mode_.connect(self._on_mode_change)
        self.layer.out_of_slice_display_.connect(self._on_out_of_slice_display_change)
        self.layer.symbol_.connect(self._on_symbol_change)
        self.layer.size_.connect(self._on_size_change)
        self.layer.edge_color_.connect(self._on_edge_color_change)
        self.layer.face_color_.connect(self._on_face_color_change)
        self.layer.editable_.connect(self._on_editable_data_or_visible_change)
        self.layer.visible_.connect(self._on_editable_data_or_visible_change)
        self.layer.data_.connect(self._on_editable_data_or_visible_change)
        self.layer.set_data_.connect(self._on_editable_data_or_visible_change)

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        value = self.layer.size
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeSize)
        self.sizeSlider = sld

        self.faceColorEdit = QColorSwatchEdit(
            parent=parent,
            initial_color=self.layer.face_color,
        )
        self.edgeColorEdit = QColorSwatchEdit(
            parent=parent,
            initial_color=self.layer.edge_color,
        )
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        sym_cb = QComboBox()
        current_index = 0
        for index, (symbol_string, text) in enumerate(SYMBOL_TRANSLATION.items()):
            symbol_string = symbol_string.value
            sym_cb.addItem(text, symbol_string)

            if symbol_string == self.layer.symbol:
                current_index = index

        sym_cb.setCurrentIndex(current_index)
        sym_cb.currentTextChanged.connect(self.changeSymbol)
        self.symbolComboBox = sym_cb

        self.outOfSliceCheckBox = QCheckBox()
        self.outOfSliceCheckBox.setChecked(self.layer.out_of_slice_display)
        self.outOfSliceCheckBox.stateChanged.connect(self.change_out_of_slice)

        self.select_button = QtModeRadioButton(
            layer, "select_points", Mode.SELECT, checked=(layer.mode == Mode.SELECT)
        )
        self.addition_button = QtModeRadioButton(
            layer, "add_points", Mode.ADD, checked=(layer.mode == Mode.ADD)
        )
        self.panzoom_button = QtModeRadioButton(
            layer,
            "pan",
            Mode.PAN_ZOOM,
            checked=(layer.mode == Mode.PAN_ZOOM),
        )
        self.delete_button = QtModePushButton(
            layer, "delete_shape", slot=self._on_delete
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.panzoom_button)
        button_row.addWidget(self.addition_button)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.delete_button)
        button_row.setContentsMargins(0, 0, 0, 5)
        button_row.setSpacing(4)

        self.layout().addRow(button_row)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(
            QtLabel("point size", {"en", "ru"}, parent=self), self.sizeSlider
        )
        self.layout().addRow(QtLabel("blending", parent=self), self.blendComboBox)
        self.layout().addRow(QtLabel("symbol", parent=self), self.symbolComboBox)
        self.layout().addRow(
            QtLabel("face color", {"en", "ru"}, parent=self), self.faceColorEdit
        )
        self.layout().addRow(
            QtLabel("edge color", {"en", "ru"}, parent=self), self.edgeColorEdit
        )
        self.layout().addRow(
            QtLabel("out of slice", {"en", "ru"}, parent=self), self.outOfSliceCheckBox
        )

        self._on_editable_data_or_visible_change()

    def _on_delete(self):
        self.layer.remove_selected()

    def _on_mode_change(self, mode):
        if mode == Mode.ADD:
            with qt_signals_blocked(self.addition_button):
                self.addition_button.setChecked(True)
        elif mode == Mode.SELECT:
            with qt_signals_blocked(self.select_button):
                self.select_button.setChecked(True)
        elif mode == Mode.PAN_ZOOM:
            with qt_signals_blocked(self.panzoom_button):
                self.panzoom_button.setChecked(True)
        else:
            raise ValueError("Mode not recognized {mode}".format(mode=mode))

    def changeSymbol(self, text):
        self.layer.symbol_.disconnect(self._on_symbol_change)
        self.layer.symbol = SYMBOL_TRANSLATION_INVERTED[text]
        self.layer.symbol_.connect(self._on_symbol_change)

    def changeSize(self, value):
        self.layer.size_.disconnect(self._on_size_change)
        self.layer.size = value
        self.layer.size_.connect(self._on_size_change)

    def change_out_of_slice(self, state):
        self.layer.out_of_slice_display_.disconnect(
            self._on_out_of_slice_display_change
        )
        self.layer.out_of_slice_display = bool(state)
        self.layer.out_of_slice_display_.connect(self._on_out_of_slice_display_change)

    def _on_out_of_slice_display_change(self):
        with qt_signals_blocked(self.outOfSliceCheckBox):
            self.outOfSliceCheckBox.setChecked(self.layer.out_of_slice_display)

    def _on_symbol_change(self):
        with qt_signals_blocked(self.symbolComboBox):
            self.symbolComboBox.setCurrentIndex(
                self.symbolComboBox.findData(self.layer.symbol.value)
            )

    def _on_size_change(self):
        with qt_signals_blocked(self.sizeSlider):
            value = self.layer.size
            min_val = min(value) if isinstance(value, list) else value
            max_val = max(value) if isinstance(value, list) else value
            if min_val < self.sizeSlider.minimum():
                self.sizeSlider.setMinimum(max(1, int(min_val - 1)))
            if max_val > self.sizeSlider.maximum():
                self.sizeSlider.setMaximum(int(max_val + 1))
            with contextlib.suppress(TypeError):
                self.sizeSlider.setValue(int(value))

    def changeFaceColor(self, color: np.ndarray):
        self.layer.face_color_.disconnect(self._on_face_color_change)
        self.layer.face_color = color
        self.layer.face_color_.connect(self._on_face_color_change)

    def changeEdgeColor(self, color: np.ndarray):
        self.layer.edge_color_.disconnect(self._on_edge_color_change)
        self.layer.edge_color = color
        self.layer.edge_color_.connect(self._on_edge_color_change)

    def _on_face_color_change(self):
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.face_color)

    def _on_edge_color_change(self):
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.edge_color)

    def _on_ndisplay_changed(self):
        self._on_editable_data_or_visible_change()

    def _on_editable_data_or_visible_change(self):
        if not (self.layer.editable and self.layer.visible):
            if self.layer.mode in {Mode.ADD, Mode.SELECT}:
                self.layer.mode = Mode.PAN_ZOOM
            set_widgets_enabled_with_opacity(
                self,
                [
                    self.select_button,
                    self.addition_button,
                    self.delete_button,
                ],
                False,
            )
        else:
            shaft_phase = self.layer.metadata.get("neck_edit_phase") == "shaft"
            set_widgets_enabled_with_opacity(
                self,
                [self.select_button],
                True,
            )
            set_widgets_enabled_with_opacity(
                self,
                [self.addition_button, self.delete_button],
                not shaft_phase,
            )
