from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QComboBox, QHBoxLayout

from utils.qt_block_signals import qt_signals_blocked
from viewer.layers.surface._surface_constants import SHADING_TRANSLATION
from viewer.widgets.layer_controls.qt_image_controls_base import QtBaseImageControls
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from viewer.layers.surface.surface import Surface


class QtSurfaceControls(QtBaseImageControls):
    layer: "Surface"

    def __init__(self, layer, parent=None) -> None:
        super().__init__(layer, parent)

        self.layer.shading_.connect(self._on_shading_changed)

        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(self.colorbarLabel)
        colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        shading_comboBox = QComboBox(self)
        for display_name, shading in SHADING_TRANSLATION.items():
            shading_comboBox.addItem(display_name, shading)
        index = shading_comboBox.findData(SHADING_TRANSLATION[self.layer.shading])
        shading_comboBox.setCurrentIndex(index)
        shading_comboBox.currentTextChanged.connect(self.changeShading)
        self.shadingComboBox = shading_comboBox

        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(QtLabel("gamma", parent=self), self.gammaSlider)
        self.layout().addRow(QtLabel("colormap", {"ru"}, parent=self), colormap_layout)
        self.layout().addRow(QtLabel("blending", parent=self), self.blendComboBox)
        self.layout().addRow(QtLabel("shading", parent=self), self.shadingComboBox)

    def changeShading(self, text):
        self.layer.shading_.disconnect(self._on_shading_changed)
        self.layer.shading = self.shadingComboBox.currentData()
        self.layer.shading_.connect(self._on_shading_changed)

    def _on_shading_changed(self, value):
        with qt_signals_blocked(self.shadingComboBox):
            index = self.shadingComboBox.findData(value)
            if index == -1:
                self.shadingComboBox.addItem(self.layer.shading, value)
            self.shadingComboBox.setCurrentIndex(index)
