from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal
from scipy import ndimage as ndi

from utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS_NAMES, ensure_colormap
from utils.misc import reorder_after_dim_reduction
from viewer.layers.base.base import Layer
from viewer.layers.image._image_constants import Interpolation
from viewer.layers.image._image_slice import ImageSlice
from viewer.layers.image._image_slice_data import ImageSliceData
from viewer.layers.intensity_mixin import IntensityVisualizationMixin
from viewer.layers.utils.layer_utils import calc_data_range


class _ImageBase(Layer):
    _colormaps_names = AVAILABLE_COLORMAPS_NAMES

    interpolation2d_ = pyqtSignal(str)
    interpolation3d_ = pyqtSignal(str)
    iso_threshold_ = pyqtSignal()

    def __init__(
        self,
        layer_id,
        name,
        data: np.ndarray,
        *,
        blending="translucent",
        metadata=None,
        opacity=1.0,
        scale=None,
        visible=True,
    ):
        if len(data.shape) != 3:
            raise ValueError("Incorrect data dimension. Expected 3")

        Layer.__init__(
            self,
            layer_id,
            name,
            metadata=metadata,
            scale=scale,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self._centered = True

        # Set data
        self._data = data

        self._new_empty_slice()

    def _new_empty_slice(self):
        wrapper = _weakref_hide(self)
        self._slice = ImageSlice(
            self._get_empty_image(),
            wrapper._raw_to_displayed,
        )
        self._empty = True

    def _get_empty_image(self):
        return np.zeros((1,) * self._slice_input.ndisplay, dtype=np.uint8)

    def _get_order(self) -> Tuple[int]:
        return reorder_after_dim_reduction(self._slice_input.displayed)

    @property
    def _data_view(self):
        return self._slice.image.view

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def _extent_data(self) -> np.ndarray:
        shape = self.data.shape
        return np.vstack([np.zeros(len(shape)), shape])

    def _raw_to_displayed(self, raw):
        image = raw
        return image

    def set_view_slice(self):
        self._new_empty_slice()
        not_disp = self._slice_input.not_displayed

        # Check if requested slice outside of data range
        indices = np.array(self._slice_indices)
        extent = self._extent_data
        if np.any(
            np.less(
                [indices[ax] for ax in not_disp],
                [extent[0, ax] for ax in not_disp],
            )
        ) or np.any(
            np.greater_equal(
                [indices[ax] for ax in not_disp],
                [extent[1, ax] for ax in not_disp],
            )
        ):
            return
        self._empty = False

        image_indices = self._slice_indices
        image = self.data[image_indices]

        data = ImageSliceData(self, image_indices, image)
        data.transpose(self._get_order())
        self._slice.load(data)

    def _get_value(self, position):
        coord = position

        coord = np.round(coord).astype(int)

        raw = self._slice.image.raw
        shape = raw.shape

        coord = coord[self._slice_input.displayed]

        if all(0 <= c < s for c, s in zip(coord, shape)):
            value = raw[tuple(coord)]
        else:
            value = None

        return value

    def _get_offset_data_position(self, position: List[float]) -> List[float]:
        return [p + 0.5 for p in position]


class Image(IntensityVisualizationMixin, _ImageBase):
    set_data_ = pyqtSignal()
    blending_ = pyqtSignal()
    opacity_ = pyqtSignal()
    visible_ = pyqtSignal()
    scale_ = pyqtSignal()
    data_ = pyqtSignal(object)
    name_ = pyqtSignal(str)
    thumbnail_ = pyqtSignal()
    help_ = pyqtSignal(str)
    mouse_pan_ = pyqtSignal(object, bool)
    mouse_zoom_ = pyqtSignal(object, bool)
    cursor_ = pyqtSignal(object)
    cursor_size_ = pyqtSignal(object)
    editable_ = pyqtSignal()
    extent_ = pyqtSignal()
    mode_ = pyqtSignal(object)
    interpolation2d_ = pyqtSignal(str)
    interpolation3d_ = pyqtSignal(str)
    iso_threshold_ = pyqtSignal()
    max_projection_ = pyqtSignal()
    changed_ = pyqtSignal(bool)

    def __init__(
        self,
        layer_id,
        name,
        data: np.ndarray,
        *,
        blending="translucent",
        colormap="gray",
        contrast_limits=None,
        gamma=1.0,
        interpolation2d="nearest",
        interpolation3d="linear",
        iso_threshold=None,
        metadata=None,
        opacity=1.0,
        scale=None,
        max_projection=False,
        visible=True,
    ):
        IntensityVisualizationMixin.__init__(self)
        _ImageBase.__init__(
            self,
            layer_id,
            name,
            data,
            blending=blending,
            metadata=metadata,
            opacity=opacity,
            scale=scale,
            visible=visible,
        )

        self._colormap = ensure_colormap(colormap)
        self._gamma = gamma
        self._interpolation2d = Interpolation.NEAREST
        self._interpolation3d = Interpolation.NEAREST
        self.interpolation2d = interpolation2d
        self.interpolation3d = interpolation3d
        self._max_projection = max_projection

        self._max_projection_images = [
            np.max(self._data, axis=0),
            np.max(self._data, axis=1),
            np.max(self._data, axis=2),
        ]

        # Set contrast limits, colormaps parameters
        if contrast_limits is None:
            self.contrast_limits_range = self._calc_data_range()
        else:
            self.contrast_limits_range = contrast_limits
        self._contrast_limits: Tuple[float, float] = self.contrast_limits_range
        self.contrast_limits = self._contrast_limits

        if iso_threshold is None:
            cmin, cmax = self.contrast_limits_range
            self._iso_threshold = cmin + (cmax - cmin) / 2
        else:
            self._iso_threshold = iso_threshold
        self.refresh()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        if len(data.shape) != 3:
            raise ValueError("Incorrect data dimension. Expected 3")
        self._data = data
        self._clear_extent()
        self.data_.emit(self.data)

    @property
    def interpolation2d(self):
        return str(self._interpolation2d)

    @interpolation2d.setter
    def interpolation2d(self, value):
        self._interpolation2d = Interpolation(value)
        self.interpolation2d_.emit(self._interpolation2d.value)

    @property
    def interpolation3d(self):
        return str(self._interpolation3d)

    @interpolation3d.setter
    def interpolation3d(self, value):
        self._interpolation3d = Interpolation(value)
        self.interpolation3d_.emit(self._interpolation3d.value)

    @property
    def iso_threshold(self) -> float:
        return self._iso_threshold

    @iso_threshold.setter
    def iso_threshold(self, value: float):
        self._iso_threshold = value
        self.iso_threshold_.emit()

    @property
    def max_projection(self) -> bool:
        return self._max_projection

    @max_projection.setter
    def max_projection(self, value: bool):
        self._max_projection = value
        self.max_projection_.emit()
        self.refresh()

    def set_view_slice(self):
        self._new_empty_slice()
        not_disp = self._slice_input.not_displayed

        # Check if requested slice outside of data range
        indices = np.array(self._slice_indices)
        extent = self._extent_data
        if np.any(
            np.less(
                [indices[ax] for ax in not_disp],
                [extent[0, ax] for ax in not_disp],
            )
        ) or np.any(
            np.greater_equal(
                [indices[ax] for ax in not_disp],
                [extent[1, ax] for ax in not_disp],
            )
        ):
            return
        self._empty = False

        image_indices = self._slice_indices
        if self._slice_input.ndisplay == 2 and self._max_projection:
            image = self._max_projection_images[not_disp[0]]
        else:
            image = self.data[image_indices]

        data = ImageSliceData(self, image_indices, image)
        data.transpose(self._get_order())
        self._slice.load(data)

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        image = self._slice.image.raw

        if self._slice_input.ndisplay == 3:
            image = np.max(image, axis=0)

        # float16 not supported by ndi.zoom
        try:
            dtype = np.dtype(image.dtype)
        except TypeError:
            # tensorstore case
            dtype = np.dtype(image.dtype.type)
        dtype = np.dtype(image.dtype)
        if dtype in [np.dtype(np.float16)]:
            image = image.astype(np.float32)

        raw_zoom_factor = np.divide(self._thumbnail_shape[:2], image.shape[:2]).min()
        new_shape = np.clip(
            raw_zoom_factor * np.array(image.shape[:2]),
            1,  # smallest side should be 1 pixel wide
            self._thumbnail_shape[:2],
        )
        zoom_factor = tuple(new_shape / image.shape[:2])
        downsampled = ndi.zoom(image, zoom_factor, prefilter=False, order=0)
        low, high = self.contrast_limits
        downsampled = np.clip(downsampled, low, high)
        color_range = high - low
        if color_range != 0:
            downsampled = (downsampled - low) / color_range
        downsampled = downsampled**self.gamma
        color_array = self.colormap.map(downsampled.ravel())
        colormapped = color_array.reshape((*downsampled.shape, 4))
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def _calc_data_range(self, mode="data") -> Tuple[float, float]:
        if mode == "data":
            input_data = self.data
        elif mode == "slice":
            input_data = self._slice.image.view
        else:
            raise ValueError(
                "mode must be either 'data' or 'slice', got {mode!r}".format(
                    mode=mode,
                )
            )
        return calc_data_range(input_data)


class _weakref_hide:
    def __init__(self, obj) -> None:
        import weakref

        self.obj = weakref.ref(obj)

    def _raw_to_displayed(self, *args, **kwarg):
        return self.obj()._raw_to_displayed(*args, **kwarg)
