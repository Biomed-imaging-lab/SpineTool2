from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from utils.colormaps.colormap_utils import ensure_colormap
from utils.dtype import normalize_dtype
from viewer.layers._validators import validate_increasing, validate_n_seq
from viewer.layers.status_messages import format_float

if TYPE_CHECKING:
    from viewer.layers.image.image import Image


validate_2_tuple = validate_n_seq(2)


class IntensityVisualizationMixin(QObject):
    contrast_limits_ = pyqtSignal()
    contrast_limits_range_ = pyqtSignal()
    gamma_ = pyqtSignal()
    colormap_ = pyqtSignal()

    def __init__(self):
        QObject.__init__(self)

        self._gamma = 1
        self._colormap_name = ""
        self._contrast_limits_msg = ""
        self._contrast_limits = [None, None]
        self._contrast_limits_range = [None, None]
        self._auto_contrast_source = "slice"

    def reset_contrast_limits(self: "Image", mode=None):
        mode = mode or self._auto_contrast_source
        self.contrast_limits = self._calc_data_range(mode)

    def reset_contrast_limits_range(self, mode=None):
        dtype = normalize_dtype(self.dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            self.contrast_limits_range = (info.min, info.max)
        else:
            mode = mode or self._auto_contrast_source
            self.contrast_limits_range = self._calc_data_range(mode)

    @property
    def colormap(self):
        return self._colormap

    def _set_colormap(self, colormap):
        self._colormap = ensure_colormap(colormap)
        self._update_thumbnail()
        self.colormap_.emit()

    @colormap.setter
    def colormap(self, colormap):
        self._set_colormap(colormap)

    @property
    def colormaps(self):
        """tuple of str: names of available colormaps."""
        return tuple(self._colormaps_names)

    @property
    def contrast_limits(self):
        """list of float: Limits to use for the colormap."""
        return list(self._contrast_limits)

    @contrast_limits.setter
    def contrast_limits(self, contrast_limits):
        validate_2_tuple(contrast_limits)
        validate_increasing(contrast_limits)
        self._contrast_limits_msg = (
            format_float(contrast_limits[0]) + ", " + format_float(contrast_limits[1])
        )
        self._contrast_limits = contrast_limits
        # make sure range slider is big enough to fit range
        newrange = list(self.contrast_limits_range)
        newrange[0] = min(newrange[0], contrast_limits[0])
        newrange[1] = max(newrange[1], contrast_limits[1])
        self.contrast_limits_range = newrange
        self._update_thumbnail()
        self.contrast_limits_.emit()

    @property
    def contrast_limits_range(self):
        return list(self._contrast_limits_range)

    @contrast_limits_range.setter
    def contrast_limits_range(self, value):
        validate_2_tuple(value)
        validate_increasing(value)
        if list(value) == self.contrast_limits_range:
            return

        # if either value is "None", it just preserves the current range
        current_range = self.contrast_limits_range
        value = list(value)  # make sure it is mutable
        for i in range(2):
            value[i] = current_range[i] if value[i] is None else value[i]
        self._contrast_limits_range = value
        self.contrast_limits_range_.emit()

        # make sure that the contrast limits fit within the new range
        # this also serves the purpose of emitting contrast_limits_
        # and updating the views/controllers
        if hasattr(self, "_contrast_limits") and any(self._contrast_limits):
            clipped_limits = np.clip(self.contrast_limits, *value)
            if clipped_limits[0] < clipped_limits[1]:
                self.contrast_limits = tuple(clipped_limits)
            else:
                self.contrast_limits = tuple(value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._update_thumbnail()
        self.gamma_.emit()
