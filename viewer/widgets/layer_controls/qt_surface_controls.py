from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout

from utils.qt_block_signals import qt_signals_blocked
from viewer.layers.surface._surface_constants import SHADING_TRANSLATION
from viewer.widgets.layer_controls.qt_image_controls_base import QtBaseImageControls
from viewer.widgets.qt_color_swatch import QColorSwatchEdit
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from viewer.layers.surface.surface import Surface


class QtSurfaceControls(QtBaseImageControls):
    layer: "Surface"

    def __init__(self, layer, parent=None) -> None:
        super().__init__(layer, parent)

        self.layer.shading_.connect(self._on_shading_changed)
        self.layer.show_edges_.connect(self._on_show_edges_changed)
        self.layer.edge_color_.connect(self._on_edge_color_changed)

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

        self.showEdgesCheckBox = QCheckBox(self)
        self.showEdgesCheckBox.setChecked(self.layer.show_edges)
        self.showEdgesCheckBox.stateChanged.connect(self.changeShowEdges)

        self.edgeColorEdit = QColorSwatchEdit(
            parent=self,
            initial_color=self.layer.edge_color,
        )
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)
        self.edgeColorEdit.setEnabled(self.layer.show_edges)

        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(QtLabel("gamma", parent=self), self.gammaSlider)
        self.layout().addRow(QtLabel("colormap", {"ru"}, parent=self), colormap_layout)
        self.layout().addRow(QtLabel("blending", parent=self), self.blendComboBox)
        self.layout().addRow(QtLabel("shading", parent=self), self.shadingComboBox)
        self.layout().addRow(
            QtLabel("show mesh edges", {"en", "ru"}, parent=self),
            self.showEdgesCheckBox,
        )
        self.layout().addRow(
            QtLabel("edge color", {"en", "ru"}, parent=self), self.edgeColorEdit
        )

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

    def changeShowEdges(self, state):
        self.layer.show_edges = bool(state)

    def _on_show_edges_changed(self, value):
        with qt_signals_blocked(self.showEdgesCheckBox):
            self.showEdgesCheckBox.setChecked(value)
        self.edgeColorEdit.setEnabled(value)

    def changeEdgeColor(self, color):
        self.layer.edge_color = color

    def _on_edge_color_changed(self, value):
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(value)
