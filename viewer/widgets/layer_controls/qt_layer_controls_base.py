from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QFrame
from superqt import QLabeledDoubleSlider as QDoubleSlider

from utils.qt_block_signals import qt_signals_blocked
from utils.qt_translater import Translater
from viewer.layers.base._base_constants import Blending
from viewer.layers.base.base import Layer
from widgets.qt_custom_label import QtLabel
from widgets.qt_form_layout import QtFormLayout

# opaque blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {str(Blending.OPAQUE)}


class QtLayerControls(QFrame):
    def __init__(self, layer: Layer, parent=None) -> None:
        super().__init__(parent)

        self._ndisplay: int = 2

        self.layer = layer
        self.layer.blending_.connect(self._on_blending_change)
        self.layer.opacity_.connect(self._on_opacity_change)

        self.setObjectName("layer")
        self.setMouseTracking(True)

        self.setLayout(QtFormLayout(self))

        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.valueChanged.connect(self.changeOpacity)
        self.opacitySlider = sld
        self.opacityLabel = QtLabel("opacity", parent=self)

        self._on_opacity_change()

        blend_comboBox = QComboBox(self)
        translater = Translater.instance()
        for index, mode in enumerate(Blending):
            blend_comboBox.addItem(translater.get_translation(mode.value), mode)
            if mode == self.layer.blending:
                blend_comboBox.setCurrentIndex(index)
        translater.language_changed_signal.connect(self.language_changed)

        blend_comboBox.currentTextChanged.connect(self.changeBlending)
        self.blendComboBox = blend_comboBox
        # opaque blending do not support changing alpha
        if self.layer.blending not in NO_OPACITY_BLENDING_MODES:
            self.opacitySlider.show()
            self.opacityLabel.show()
        else:
            self.opacitySlider.hide()
            self.opacityLabel.hide()

    def language_changed(self):
        translater = Translater.instance()
        with qt_signals_blocked(self.blendComboBox):
            self.blendComboBox.clear()
            for index, mode in enumerate(Blending):
                self.blendComboBox.addItem(translater.get_translation(mode.value), mode)
                if mode == self.layer.blending:
                    self.blendComboBox.setCurrentIndex(index)

    def changeOpacity(self, value):
        self.layer.opacity_.disconnect(self._on_opacity_change)
        self.layer.opacity = value
        self.layer.opacity_.connect(self._on_opacity_change)

    def changeBlending(self, text):
        self.layer.blending_.disconnect(self._on_blending_change)
        self.layer.blending = self.blendComboBox.currentData()
        # opaque blending do not support changing alpha
        if self.layer.blending not in NO_OPACITY_BLENDING_MODES:
            self.opacitySlider.show()
            self.opacityLabel.show()
        else:
            self.opacitySlider.hide()
            self.opacityLabel.hide()

        self.layer.blending_.connect(self._on_blending_change)

    def _on_opacity_change(self):
        with qt_signals_blocked(self.opacitySlider):
            self.opacitySlider.setValue(self.layer.opacity)

    def _on_blending_change(self):
        with qt_signals_blocked(self.blendComboBox):
            self.blendComboBox.setCurrentIndex(
                self.blendComboBox.findData(self.layer.blending)
            )

    @property
    def ndisplay(self) -> int:
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer.

        This is needed because some layer controls may have options that are specific
        to 2D or 3D visualization only.
        """
