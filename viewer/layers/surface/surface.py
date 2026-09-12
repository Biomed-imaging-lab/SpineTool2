import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PyQt5.QtCore import pyqtSignal

from utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS_NAMES
from viewer.layers.base.base import Layer
from viewer.layers.intensity_mixin import IntensityVisualizationMixin
from viewer.layers.surface._surface_constants import Shading
from viewer.layers.utils.layer_utils import calc_data_range


class Surface(IntensityVisualizationMixin, Layer):
    _colormaps_names = AVAILABLE_COLORMAPS_NAMES

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
    contrast_limits_ = pyqtSignal()
    contrast_limits_range_ = pyqtSignal()
    gamma_ = pyqtSignal()
    colormap_ = pyqtSignal()
    shading_ = pyqtSignal(object)
    texture_ = pyqtSignal(object)
    texcoords_ = pyqtSignal(object)
    changed_ = pyqtSignal(bool)

    def __init__(
        self,
        layer_id,
        name,
        data,
        *,
        colormap="gray",
        contrast_limits=None,
        gamma=1,
        metadata=None,
        scale=None,
        opacity=1,
        blending="translucent",
        shading="flat",
        visible=True,
        texture=None,
        texcoords=None,
        vertex_colors=None,
        colormaps_names=None,
    ):
        if not data:
            data = (np.empty((0, 3)), np.empty((0, 3)))
        if len(data) not in (2, 3):
            raise ValueError(
                "Surface data tuple must be 2 or 3, specifying vertices, faces, and optionally vertex values, instead got length {length}.".format(
                    length=len(data),
                )
            )
        if data[0].shape[1] != 3:
            raise ValueError("Incorrect data dimension. Expected 3")

        IntensityVisualizationMixin.__init__(self)
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

        if colormaps_names is not None:
            self._colormaps_names = colormaps_names

        self._vertices = data[0]
        self._faces = data[1]
        if len(data) == 3:
            self._vertex_values = data[2]
        else:
            self._vertex_values = np.ones(len(self._vertices))

        self._texture = texture
        self._texcoords = texcoords
        self._vertex_colors = vertex_colors

        # Set contrast_limits and colormaps
        self._gamma = gamma
        if contrast_limits is None:
            self._contrast_limits_range = calc_data_range(self._vertex_values)
        else:
            self._contrast_limits_range = contrast_limits
        self._contrast_limits = tuple(self._contrast_limits_range)
        self.colormap = colormap
        self.contrast_limits = self._contrast_limits

        # Data containing vectors in the currently viewed slice
        self._data_view = np.zeros((0, self._slice_input.ndisplay))
        self._view_faces = np.zeros((0, 3))
        self._view_vertex_values = []
        self._view_vertex_colors = []

        # Trigger generation of view slice and thumbnail.
        self._clear_extent()

        # Shading mode
        self._shading = shading

    def _calc_data_range(self, mode="data"):
        return calc_data_range(self.vertex_values)

    @property
    def dtype(self):
        return self.vertex_values.dtype

    @property
    def data(self):
        return (self.vertices, self.faces, self.vertex_values)

    @data.setter
    def data(self, data):
        if len(data) not in (2, 3):
            raise ValueError(
                "Surface data tuple must be 2 or 3, specifying vertices, faces, and optionally vertex values, instead got length {data_length}.".format(
                    data_length=len(data),
                )
            )
        if data[0].shape[1] != 3:
            raise ValueError("Incorrect data dimension. Expected 3")
        self._vertices = data[0]
        self._faces = data[1]
        if len(data) == 3:
            self._vertex_values = data[2]
        else:
            self._vertex_values = np.ones(len(self._vertices))

        self._clear_extent()
        self.data_.emit(self.data)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        if vertices.shape[1] != 3:
            ValueError("Incorrect data dimension. Expected 3")
        self._vertices = vertices
        self._clear_extent()
        self.data_.emit(self.data)

    @property
    def vertex_values(self) -> np.ndarray:
        return self._vertex_values

    @vertex_values.setter
    def vertex_values(self, vertex_values: np.ndarray):
        if vertex_values is None:
            vertex_values = np.ones(len(self._vertices))
        if len(vertex_values.shape) != 1:
            ValueError("Incorrect data dimension. Expected 1")

        self._vertex_values = vertex_values
        self._clear_extent()
        self.data_.emit(self.data)

    @property
    def vertex_colors(self) -> Optional[np.ndarray]:
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, vertex_colors: Optional[np.ndarray]):
        if vertex_colors is not None and not isinstance(vertex_colors, np.ndarray):
            msg = f"texture should be None or ndarray; got {type(vertex_colors)}"
            raise ValueError(msg)
        self._vertex_colors = vertex_colors
        self._clear_extent()
        self.data_.emit(self.data)

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @faces.setter
    def faces(self, faces: np.ndarray):
        self.faces = faces
        self.refresh()
        self.data_.emit(self.data)

    @property
    def _extent_data(self) -> np.ndarray:
        if len(self.vertices) == 0:
            extrema = np.full((2, 3), np.nan)
        else:
            maxs = np.max(self.vertices, axis=0)
            mins = np.min(self.vertices, axis=0)

            # The full dimensionality and shape of the layer is determined by
            # the number of additional vertex value dimensions and the
            # dimensionality of the vertices themselves
            if self.vertex_values.ndim > 1:
                mins = [0] * (self.vertex_values.ndim - 1) + list(mins)
                maxs = [n - 1 for n in self.vertex_values.shape[:-1]] + list(maxs)
            extrema = np.vstack([mins, maxs])
        return extrema

    @property
    def shading(self):
        return str(self._shading)

    @shading.setter
    def shading(self, shading):
        if isinstance(shading, Shading):
            self._shading = shading
        else:
            self._shading = Shading(shading)
        self.shading_.emit(self._shading)

    @property
    def texture(self) -> Optional[np.ndarray]:
        return self._texture

    @texture.setter
    def texture(self, texture: np.ndarray):
        if texture is not None and not isinstance(texture, np.ndarray):
            msg = f"texture should be None or ndarray; got {type(texture)}"
            raise ValueError(msg)
        self._texture = texture
        self.texture_.emit(self._texture)

    @property
    def texcoords(self) -> Optional[np.ndarray]:
        return self._texcoords

    @texcoords.setter
    def texcoords(self, texcoords: np.ndarray):
        if texcoords is not None and not isinstance(texcoords, np.ndarray):
            msg = f"texcoords should be None or ndarray; got {type(texcoords)}"
            raise ValueError(msg)
        self._texcoords = texcoords
        self.texcoords_.emit(value=self._texcoords)

    @property
    def _has_texture(self) -> bool:
        return bool(
            self.texture is not None
            and self.texcoords is not None
            and len(self.texcoords)
        )

    def _slice_associated_data(
        self,
        data: np.ndarray,
        vertex_ndim: int,
        dims: int = 1,
    ) -> Union[List[Any], np.ndarray]:
        if data is None:
            return []

        data_ndim = data.ndim - 1
        if data_ndim >= dims:
            # Get indices for axes corresponding to data dimensions
            data_indices = self._slice_indices[:-vertex_ndim]
            data = data[data_indices]
            if data.ndim > dims:
                warnings.warn(
                    "Assigning multiple data per vertex after slicing "
                    "is not allowed. All dimensions corresponding to "
                    "vertex data must be non-displayed dimensions. Data "
                    "may not be visible.",
                    category=UserWarning,
                    stacklevel=2,
                )
                return []
        return data

    def set_view_slice(self):
        N, vertex_ndim = self.vertices.shape
        values_ndim = self.vertex_values.ndim - 1

        self._view_vertex_values = self._slice_associated_data(
            self.vertex_values,
            vertex_ndim,
        )

        self._view_vertex_colors = self._slice_associated_data(
            self.vertex_colors,
            vertex_ndim,
            dims=2,
        )

        if len(self._view_vertex_values) == 0:
            self._data_view = np.zeros((0, self._slice_input.ndisplay))
            self._view_faces = np.zeros((0, 3))
            return

        if values_ndim > 0:
            disp = [
                d
                for d in np.subtract(self._slice_input.displayed, values_ndim)
                if d >= 0
            ]
        else:
            disp = list(self._slice_input.displayed)

        self._data_view = self.vertices[:, disp]
        if len(self.vertices) == 0:
            self._view_faces = np.zeros((0, 3))
        elif vertex_ndim > self._slice_input.ndisplay:
            self._view_faces = np.zeros((0, 3))
        else:
            self._view_faces = self.faces

    def _update_thumbnail(self):
        """Update thumbnail with current surface."""

    def _get_value(self, position):
        return

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[None, float, int], Optional[int]]:
        return None, None
