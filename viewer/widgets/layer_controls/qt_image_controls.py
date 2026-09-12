from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout
from superqt import QLabeledDoubleSlider as QDoubleSlider
from superqt import QLabeledRangeSlider as QRangeSlider

from utils.qt_block_signals import qt_signals_blocked
from utils.qt_translater import Translater
from viewer.layers.image._image_constants import Interpolation
from viewer.widgets.layer_controls.qt_image_controls_base import QtBaseImageControls
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from viewer.layers.image.image import Image


class QtImageControls(QtBaseImageControls):
    layer: "Image"

    def __init__(self, layer, parent=None) -> None:
        super().__init__(layer, parent)

        self.layer.interpolation2d_.connect(self._on_interpolation_change)
        self.layer.interpolation3d_.connect(self._on_interpolation_change)
        self.layer.iso_threshold_.connect(self._on_iso_threshold_change)
        self.layer.max_projection_.connect(self._on_max_projection_change)
        self.layer.contrast_limits_.connect(self._on_contrast_limits_change)

        # Create contrast_limits slider
        self.contrastLimitsSlider = QRangeSlider(Qt.Orientation.Horizontal, self)
        self.contrastLimitsSlider.setEdgeLabelMode(None)
        self.contrastLimitsSlider.label_shift_x = -2
        data_range = [np.min(self.layer.data), np.max(self.layer.data)]
        decimals = range_to_decimals(data_range, self.layer.dtype)
        self.contrastLimitsSlider.setRange(*data_range)
        self.contrastLimitsSlider.setSingleStep(10**-decimals)
        self.contrastLimitsSlider.setValue(self.layer.contrast_limits)

        self.clim_popup = None

        self.contrastLimitsSlider.valueChanged.connect(self.changeContrastLimits)

        self.interpComboBox = QComboBox(self)
        self.interpComboBox.currentTextChanged.connect(self.changeInterpolation)
        self.maxProjectionLabel = QtLabel("max intensity", {"ru"}, parent=self)

        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(data_range[0])
        sld.setMaximum(data_range[1])
        sld.setValue(self.layer.iso_threshold)
        sld.valueChanged.connect(self.changeIsoThreshold)
        self.isoThresholdSlider = sld
        self.isoThresholdLabel = QtLabel("iso threshold", {"ru"}, parent=self)

        max_projection_cb = QCheckBox(self)
        max_projection_cb.setToolTip(
            Translater.instance().get_translation("max intensity description")
        )
        max_projection_cb.stateChanged.connect(self.changeMaxProjection)
        self.maxProjectionCheckBox = max_projection_cb
        self._on_max_projection_change()

        self._on_ndisplay_changed()

        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(self.colorbarLabel)
        colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(
            QtLabel("contrast limits", {"ru"}, parent=self), self.contrastLimitsSlider
        )
        self.layout().addRow(QtLabel("gamma", parent=self), self.gammaSlider)
        self.layout().addRow(QtLabel("colormap", {"ru"}, parent=self), colormap_layout)
        self.layout().addRow(QtLabel("blending", parent=self), self.blendComboBox)
        self.layout().addRow(QtLabel("interpolation", parent=self), self.interpComboBox)
        self.layout().addRow(self.maxProjectionLabel, self.maxProjectionCheckBox)
        self.layout().addRow(self.isoThresholdLabel, self.isoThresholdSlider)

    def language_changed(self):
        super().language_changed()
        self._update_interpolation_combo()
        self.maxProjectionCheckBox.setToolTip(
            Translater.instance().get_translation("max intensity description")
        )

    def changeContrastLimits(self, value):
        self.layer.contrast_limits_.disconnect(self._on_contrast_limits_change)
        self.layer.contrast_limits = value
        self.layer.contrast_limits_.connect(self._on_contrast_limits_change)

    def changeInterpolation(self, text):
        self.layer.interpolation2d_.disconnect(self._on_interpolation_change)
        self.layer.interpolation3d_.disconnect(self._on_interpolation_change)
        if self.ndisplay == 2:
            self.layer.interpolation2d = self.interpComboBox.currentData()
        else:
            self.layer.interpolation3d = self.interpComboBox.currentData()
        self.layer.interpolation2d_.connect(self._on_interpolation_change)
        self.layer.interpolation3d_.connect(self._on_interpolation_change)

    def changeIsoThreshold(self, value):
        self.layer.iso_threshold_.disconnect(self._on_iso_threshold_change)
        self.layer.iso_threshold = value
        self.layer.iso_threshold_.connect(self._on_iso_threshold_change)

    def changeMaxProjection(self, value):
        self.layer.max_projection_.disconnect(self._on_max_projection_change)
        self.layer.max_projection = value
        self.layer.max_projection_.connect(self._on_max_projection_change)

    def _on_contrast_limits_change(self):
        cmin, cmax = self.layer.contrast_limits_range
        self.isoThresholdSlider.setMinimum(cmin)
        self.isoThresholdSlider.setMaximum(cmax)
        with qt_signals_blocked(self.contrastLimitsSlider):
            self.contrastLimitsSlider.setValue(self.layer.contrast_limits)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setValue(self.layer.contrast_limits)

    def _on_iso_threshold_change(self):
        with qt_signals_blocked(self.isoThresholdSlider):
            self.isoThresholdSlider.setValue(self.layer.iso_threshold)

    def _on_max_projection_change(self):
        with qt_signals_blocked(self.maxProjectionCheckBox):
            self.maxProjectionCheckBox.setChecked(self.layer.max_projection)

    def _on_interpolation_change(self, value):
        with qt_signals_blocked(self.interpComboBox):
            self.interpComboBox.setCurrentIndex(self.interpComboBox.findData(value))

    def _update_interpolation_combo(self):
        translater = Translater.instance()
        interp = (
            self.layer.interpolation2d
            if self.ndisplay == 2
            else self.layer.interpolation3d
        )
        with qt_signals_blocked(self.interpComboBox):
            self.interpComboBox.clear()
            for index, mode in enumerate(Interpolation.view_subset()):
                self.interpComboBox.addItem(
                    translater.get_translation(mode.value), mode
                )
                if mode == interp:
                    self.interpComboBox.setCurrentIndex(index)

    def _on_ndisplay_changed(self):
        self._update_interpolation_combo()
        if self.ndisplay == 2:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
            self.maxProjectionLabel.show()
            self.maxProjectionCheckBox.show()
        else:
            self.isoThresholdSlider.show()
            self.isoThresholdLabel.show()
            self.maxProjectionLabel.hide()
            self.maxProjectionCheckBox.hide()


def range_to_decimals(range_, dtype):
    if hasattr(dtype, "numpy_dtype"):
        # retrieve the corresponding numpy.dtype from a tensorstore.dtype
        dtype = dtype.numpy_dtype

    if np.issubdtype(dtype, np.integer):
        return 0

    # scale precision with the log of the data range order of magnitude
    # eg.   0 - 1   (0 order of mag)  -> 3 decimal places
    #       0 - 10  (1 order of mag)  -> 2 decimals
    #       0 - 100 (2 orders of mag) -> 1 decimal
    #       ≥ 3 orders of mag -> no decimals
    # no more than 64 decimals
    d_range = np.subtract(*range_[::-1])
    return min(64, max(int(3 - np.log10(d_range)), 0))
