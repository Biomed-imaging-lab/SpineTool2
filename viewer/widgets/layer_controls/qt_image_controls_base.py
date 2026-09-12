from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
from superqt import QLabeledDoubleSlider as QDoubleSlider

from utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS
from utils.qt_block_signals import qt_signals_blocked
from viewer.widgets.layer_controls.qt_colormap_combobox import QtColormapComboBox
from viewer.widgets.layer_controls.qt_layer_controls_base import QtLayerControls

if TYPE_CHECKING:
    from viewer.layers.image.image import Image


class QtBaseImageControls(QtLayerControls):
    def __init__(self, layer: Image, parent=None) -> None:
        super().__init__(layer, parent)

        self.layer.colormap_.connect(self._on_colormap_change)
        self.layer.gamma_.connect(self._on_gamma_change)

        comboBox = QtColormapComboBox(self)
        comboBox.setObjectName("colormapComboBox")
        comboBox._allitems = set(self.layer.colormaps)

        for name, cm in AVAILABLE_COLORMAPS.items():
            if name in self.layer.colormaps:
                comboBox.addItem(cm._display_name, name)

        comboBox.currentTextChanged.connect(self.changeColor)
        self.colormapComboBox = comboBox

        # gamma slider
        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self.layer.gamma)
        self.gammaSlider = sld
        self.gammaSlider.valueChanged.connect(self.changeGamma)

        self.colorbarLabel = QLabel(parent=self)
        self.colorbarLabel.setObjectName("colorbar")

        self._on_colormap_change()

    def changeGamma(self, value):
        self.layer.gamma_.disconnect(self._on_gamma_change)
        self.layer.gamma = self.gammaSlider.value()
        self.layer.gamma_.connect(self._on_gamma_change)

    def changeColor(self, text):
        self.layer.colormap = self.colormapComboBox.currentData()

    def _on_colormap_change(self):
        with qt_signals_blocked(self.colormapComboBox):
            name = self.layer.colormap.name
            if name not in self.colormapComboBox._allitems and (
                cm := AVAILABLE_COLORMAPS.get(name)
            ):
                self.colormapComboBox._allitems.add(name)
                self.colormapComboBox.addItem(cm._display_name, name)

            if name != self.colormapComboBox.currentData():
                index = self.colormapComboBox.findData(name)
                self.colormapComboBox.setCurrentIndex(index)

            cbar = self.layer.colormap.colorbar
            image = QImage(
                cbar,
                cbar.shape[1],
                cbar.shape[0],
                QImage.Format_RGBA8888,
            )
            self.colorbarLabel.setPixmap(QPixmap.fromImage(image))

    def _on_gamma_change(self):
        with qt_signals_blocked(self.gammaSlider):
            self.gammaSlider.setValue(self.layer.gamma)
