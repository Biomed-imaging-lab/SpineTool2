from collections import deque
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from PyQt5.QtCore import pyqtSignal
from scipy import ndimage as ndi
from skimage.measure import label

from utils.colormaps.colormap import CyclicLabelColormap
from utils.colormaps.colormap_utils import label_colormap, shuffle_and_extend_colormap
from utils.dtype import normalize_dtype, vispy_texture_dtype
from utils.geometry import clamp_point_to_bounding_box
from viewer.components._viewer_constants import CursorStyle
from viewer.layers.base.base import Layer, no_op
from viewer.layers.image._image_slice_data import ImageSliceData
from viewer.layers.image.image import _ImageBase
from viewer.layers.labels._indexing import index_in_slice
from viewer.layers.labels._labels_mouse_bindings import draw
from viewer.layers.labels._labels_utils import (
    ellipse_indices,
    indices_in_shape,
    interpolate_coordinates,
)
from viewer.layers.labels.brush_settings import BrushSettings
from viewer.layers.labels.labels_constants import Mode, PaintMode
from viewer.layers.status_messages import generate_layer_coords_status


class Labels(_ImageBase):
    _colormap: CyclicLabelColormap
    _connected_components_colormap: CyclicLabelColormap
    _modeclass = Mode
    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.PAINT: draw,
        Mode.FILL: draw,
        Mode.ERASE: draw,
        Mode.CONNECTED_COMPONENTS: no_op,
    }
    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.PAINT: no_op,
        Mode.FILL: no_op,
        Mode.ERASE: no_op,
        Mode.CONNECTED_COMPONENTS: no_op,
    }
    _cursor_modes = {
        Mode.PAN_ZOOM: CursorStyle.STANDARD,
        Mode.PAINT: CursorStyle.CIRCLE,
        Mode.FILL: CursorStyle.CROSS,
        Mode.ERASE: CursorStyle.CIRCLE,
        Mode.CONNECTED_COMPONENTS: CursorStyle.STANDARD,
    }
    _history_limit = 100

    brush_settings_ = pyqtSignal()
    colormap_ = pyqtSignal()
    paint_ = pyqtSignal(object)
    selected_label_ = pyqtSignal(str)
    show_selected_label_ = pyqtSignal(bool)
    interpolation2d_ = pyqtSignal(str)
    interpolation3d_ = pyqtSignal(str)
    iso_threshold_ = pyqtSignal()
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
    changed_ = pyqtSignal(bool)
    extent_ = pyqtSignal()
    mode_ = pyqtSignal(object)
    fill_3d_ = pyqtSignal()
    preserve_background_ = pyqtSignal()
    labels_ = pyqtSignal()

    def __init__(
        self,
        layer_id,
        name,
        data: np.ndarray,
        *,
        blending="translucent_no_depth",
        colormap=None,
        connected_components_colormap=None,
        labels=["background", "general_label"],
        metadata=None,
        opacity=0.7,
        brush_settings=None,
        fill_3d=False,
        scale=None,
        visible=True,
        colors_num=49,
    ) -> None:
        if len(data.shape) != 3:
            raise ValueError("Incorrect data dimension. Expected 3")

        self._labels = labels
        super().__init__(
            layer_id,
            name,
            self._ensure_data(data),
            metadata=metadata,
            scale=scale,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self._show_selected_label = False
        self._fill_3d = fill_3d
        self._connected_components = None
        self.brush_settings = BrushSettings(brush_settings)

        self._selected_label = 1
        if colormap is not None:
            self._set_colormap(colormap)
        else:
            self._set_colormap(
                label_colormap(len(self._labels) - 1, 0.5, background_value=0)
            )

        if connected_components_colormap is not None:
            self._set_connected_components_colormap(connected_components_colormap)
        else:
            self._set_connected_components_colormap(
                label_colormap(colors_num, 0.5, background_value=0)
            )

        self._preserve_background = False
        self._reset_history()

        self.refresh()

    @property
    def brush_settings(self):
        return self._brush_settings

    @brush_settings.setter
    def brush_settings(self, brush_settings: BrushSettings):
        self._brush_settings = brush_settings
        if self._brush_settings.all_same_radii:
            self._brush_settings.radii[1] = self._brush_settings.radii[2]
            self._brush_settings.radii[0] = self._brush_settings.radii[2]
        self.cursor_size = self._calculate_cursor_size()
        self.brush_settings_.emit()

    def _calculate_cursor_size(self):
        scale = np.array(
            [abs(self._scale.scale[d]) for d in self._slice_input.displayed]
        )
        radii = np.array(
            [self._brush_settings.radii[d] for d in self._slice_input.displayed]
        )
        return (2 * radii * scale - 1)[::-1]

    def _update_cursor(self):
        if self._mode in [self._modeclass.PAINT, self._modeclass.ERASE]:
            self.cursor_size = self._calculate_cursor_size()

    def new_connected_components_colormap(self, seed: Optional[int] = None):
        if seed is None:
            seed = np.random.default_rng().integers(2**32 - 1)

        self._set_connected_components_colormap(
            shuffle_and_extend_colormap(self._connected_components_colormap, seed)
        )

    @property
    def colormap(self) -> CyclicLabelColormap:
        if self._mode == self._modeclass.CONNECTED_COMPONENTS:
            return self._connected_components_colormap
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: CyclicLabelColormap):
        if self._mode == self._modeclass.CONNECTED_COMPONENTS:
            self._set_connected_components_colormap(colormap)
        else:
            self._set_colormap(colormap)

    def _set_colormap(self, colormap):
        self._colormap = colormap
        self._colormap._clear_cache()
        self._selected_color = self.get_color(self.selected_label)
        self._colormap.selection = self._selected_label
        self._colormap.use_selection = self._show_selected_label
        self.colormap_.emit()
        self.selected_label_.emit("selected_label")
        if self._mode != self._modeclass.CONNECTED_COMPONENTS:
            self.refresh()

    def _set_connected_components_colormap(self, colormap):
        self._connected_components_colormap = colormap
        self.colormap_.emit()
        if self._mode == self._modeclass.CONNECTED_COMPONENTS:
            self.refresh()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if len(data.shape) != 3:
            raise ValueError("Incorrect data dimension. Expected 3")
        data = self._ensure_data(data)
        self._data = data
        if self._mode == self._modeclass.CONNECTED_COMPONENTS:
            tmp_data = np.zeros_like(self._data)
            tmp_data[self._data > 0] = 1
            self._connected_components = label(tmp_data, connectivity=1)
        self._clear_extent()
        self.data_.emit(self.data)
        self.changed = False
        if self._brush_settings.current_mode == PaintMode.ELLIPTICAL_CYLINDER:
            self.brush_settings_.emit()

    def _ensure_data(self, data):
        int_data = None
        if np.issubdtype(normalize_dtype(data.dtype), np.floating):
            raise TypeError(
                "Only integer types are supported for Labels layers, but data contains {data_level_type}.".format(
                    data_level_type=data.dtype,
                )
            )
        if data.dtype == bool:
            int_data = data.astype(np.int8)
        else:
            int_data = data
        data = int_data
        if len(self._labels) == 2:
            data[int_data > 0] = 1
        return data

    @property
    def labels(self) -> list:
        return self._labels

    @labels.setter
    def labels(self, labels) -> None:
        self._labels = labels
        self._set_colormap(
            label_colormap(len(self._labels) - 1, 0.5, background_value=0)
        )
        self.labels_.emit()

    @property
    def color(self) -> np.ndarray:
        return self._colormap.colors[self._selected_label]

    @color.setter
    def color(self, value):
        self._colormap.colors[self._selected_label] = value
        self._set_colormap(self._colormap)

    @property
    def selected_label(self):
        return self._selected_label

    @selected_label.setter
    def selected_label(self, selected_label):
        if selected_label == self.selected_label:
            return

        self.colormap.selection = selected_label
        self._selected_label = selected_label
        self._selected_color = self.get_color(selected_label)

        self.selected_label_.emit("selected_label")

        if self.show_selected_label:
            self.refresh()

    @property
    def show_selected_label(self):
        return self._show_selected_label

    @show_selected_label.setter
    def show_selected_label(self, show_selected):
        self._show_selected_label = show_selected
        self.colormap.use_selection = show_selected
        self.colormap.selection = self.selected_label
        self.show_selected_label_.emit(show_selected)
        self.refresh()

    @Layer.mode.getter
    def mode(self):
        return str(self._mode)

    def _mode_setter_helper(self, mode):
        if not self.editable and mode not in [
            self._modeclass.PAN_ZOOM,
            self._modeclass.CONNECTED_COMPONENTS,
        ]:
            mode = self._modeclass.PAN_ZOOM
        mode = super()._mode_setter_helper(mode, False)
        if mode == self._mode:
            return mode

        self.mouse_pan = mode in [
            self._modeclass.PAN_ZOOM,
            self._modeclass.CONNECTED_COMPONENTS,
        ]

        if mode == self._modeclass.CONNECTED_COMPONENTS:
            tmp_data = np.zeros_like(self._data)
            tmp_data[self._data > 0] = 1
            self._connected_components = label(tmp_data, connectivity=1)

        return mode

    @Layer.mode.setter
    def mode(self, mode):
        mode = self._mode_setter_helper(mode)
        if mode == self._mode:
            return
        self._mode = mode

        self.colormap_.emit()
        self.refresh()

        self.mode_.emit(mode)

    @property
    def preserve_background(self) -> bool:
        return self._preserve_background

    @preserve_background.setter
    def preserve_background(self, preserve_background: bool):
        self._preserve_background = preserve_background
        self.preserve_background_.emit()

    @property
    def fill_3d(self) -> bool:
        return self._fill_3d

    @fill_3d.setter
    def fill_3d(self, fill_3d: bool):
        self._fill_3d = fill_3d
        self.fill_3d_.emit()

    def _on_editable_changed(self) -> None:
        if not self.editable and self.mode not in [
            self._modeclass.PAN_ZOOM,
            self._modeclass.CONNECTED_COMPONENTS,
        ]:
            self.mode = self._modeclass.PAN_ZOOM

    @staticmethod
    def _to_vispy_texture_dtype(data):
        return vispy_texture_dtype(data)

    def _raw_to_displayed(
        self, raw, data_slice: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        if data_slice is None:
            data_slice = tuple(slice(0, size) for size in raw.shape)

        labels = raw  # for readability

        sliced_labels = labels[data_slice]

        return self.colormap._data_to_texture(sliced_labels)

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
        if self._mode == self._modeclass.CONNECTED_COMPONENTS:
            image = self._connected_components[image_indices]
        else:
            image = self.data[image_indices]

        data = ImageSliceData(self, image_indices, image)
        data.transpose(self._get_order())
        self._slice.load(data)

    def _update_thumbnail(self):
        image = self._slice.image.raw
        if self._slice_input.ndisplay == 3 and image.ndim == 3:
            image = np.max(image, axis=0)
        imshape = np.array(image.shape[:2])
        thumbshape = np.array(self._thumbnail_shape[:2])

        raw_zoom_factor = np.min(thumbshape / imshape)
        new_shape = np.clip(raw_zoom_factor * imshape, a_min=1, a_max=thumbshape)
        zoom_factor = tuple(new_shape / imshape)
        downsampled = ndi.zoom(image, zoom_factor, prefilter=False, order=0)
        color_array = self.colormap.map(downsampled)
        color_array[..., 3] *= self.opacity

        self.thumbnail = color_array

    def get_color(self, label):
        if label == self._colormap.background_value:
            col = None
        elif label is None or (
            self.show_selected_label and label != self.selected_label
        ):
            col = self._colormap.map(self._colormap.background_value)
        else:
            col = self._colormap.map(label)
        return col

    def _get_value_ray(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        if start_point is None or end_point is None:
            return None
        if len(dims_displayed) == 3:
            # only use get_value_ray on 3D for now
            # we use dims_displayed because the image slice
            # has its dimensions  in th same order as the vispy
            # Volume
            start_point = start_point[dims_displayed]
            end_point = end_point[dims_displayed]
            start_point = cast(np.ndarray, start_point)
            end_point = cast(np.ndarray, end_point)
            sample_ray = end_point - start_point
            length_sample_vector = np.linalg.norm(sample_ray)
            n_points = int(2 * length_sample_vector)
            sample_points = np.linspace(start_point, end_point, n_points, endpoint=True)
            im_slice = self._slice.image.raw

            bounding_box = self._display_bounding_box(dims_displayed)

            clamped = clamp_point_to_bounding_box(sample_points, bounding_box).astype(
                int
            )

            values = im_slice[tuple(clamped.T)]
            nonzero_indices = np.flatnonzero(values)
            if len(nonzero_indices > 0):
                # if a nonzer0 value was found, return the first one
                return values[nonzero_indices[0]]

        return None

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        return (
            self._get_value_ray(
                start_point=start_point,
                end_point=end_point,
                dims_displayed=dims_displayed,
            )
            or 0
        )

    def _reset_history(self):
        self._undo_history = deque(maxlen=self._history_limit)
        self._redo_history = deque(maxlen=self._history_limit)
        self._staged_history = []
        self._block_history = False

    @contextmanager
    def block_history(self):
        prev = self._block_history
        self._block_history = True
        try:
            yield
            self._commit_staged_history()
        finally:
            self._block_history = prev

    def _commit_staged_history(self):
        if self._staged_history:
            self._append_to_undo_history(self._staged_history)
            self._staged_history = []

    def _append_to_undo_history(self, item):
        self._undo_history.append(item)
        self.paint_.emit(item)

    def _save_history(self, value):
        self._redo_history.clear()
        if self._block_history:
            self._staged_history.append(value)
        else:
            self._append_to_undo_history([value])

    def _load_history(self, before, after, undoing=True):
        if len(before) == 0:
            return

        history_item = before.pop()
        after.append(list(reversed(history_item)))
        for prev_indices, prev_values, next_values in reversed(history_item):
            values = prev_values if undoing else next_values
            self.data[prev_indices] = values

        self.refresh()
        self.changed = True

    def undo(self):
        self._load_history(self._undo_history, self._redo_history, undoing=True)

    def redo(self):
        self._load_history(self._redo_history, self._undo_history, undoing=False)

    def fill(self, coord, new_label, refresh=True):
        int_coord = tuple(np.round(coord).astype(int))
        # If requested fill location is outside data shape then return
        if np.any(np.less(int_coord, 0)) or np.any(
            np.greater_equal(int_coord, self.data.shape)
        ):
            return

        # If requested new label doesn't change old label then return
        old_label = np.asarray(self.data[int_coord]).item()
        if old_label == new_label:
            return

        if self.fill_3d:
            dims_to_fill = self._slice_input.order
        else:
            dims_to_fill = sorted(self._slice_input.order[-2:])
        data_slice_list = list(int_coord)
        for dim in dims_to_fill:
            data_slice_list[dim] = slice(None)
        data_slice = tuple(data_slice_list)
        labels = np.asarray(self.data[data_slice])
        slice_coord = tuple(int_coord[d] for d in dims_to_fill)

        matches = labels == old_label
        labeled_matches, num_features = ndi.label(
            matches, structure=ndi.generate_binary_structure(len(dims_to_fill), 1)
        )
        if num_features != 1:
            match_label = labeled_matches[slice_coord]
            matches = np.logical_and(matches, labeled_matches == match_label)

        match_indices_local = np.nonzero(matches)
        n_idx = len(match_indices_local[0])
        match_indices = []
        j = 0
        for d in data_slice:
            if isinstance(d, slice):
                match_indices.append(match_indices_local[j])
                j += 1
            else:
                match_indices.append(np.full(n_idx, d, dtype=np.intp))
        del match_indices_local
        match_indices = tuple(match_indices)

        self.data_setitem(match_indices, new_label, refresh)

    def _draw(self, new_label, last_cursor_coord, coordinates):
        if coordinates is None or self._slice_input.ndisplay == 3:
            return
        min_brush_radius = np.min(
            [self._brush_settings.radii[d] for d in self._slice_input.displayed]
        )
        interp_coord = interpolate_coordinates(
            last_cursor_coord, coordinates, min_brush_radius
        )
        for c in interp_coord:
            if self._mode in [self._modeclass.PAINT, self._modeclass.ERASE]:
                self.paint(c, new_label, refresh=False)
            elif self._mode == self._modeclass.FILL:
                self.fill(c, new_label, refresh=False)
        self.refresh()

    def paint(self, coord, new_label, refresh=True):
        shape = self.data.shape
        if self._brush_settings.current_mode == PaintMode.ELLIPSOID:
            dims_to_paint = self._slice_input.order
            dim_not_painted = None
        else:
            dims_to_paint = self._slice_input.displayed
            dim_not_painted = self._slice_input.not_displayed[0]

        radii = [max(1, int(self._brush_settings.radii[i])) for i in dims_to_paint]

        slice_coord = [int(np.round(c)) for c in coord]
        coord_paint = [coord[i] for i in dims_to_paint]
        shape = [shape[i] for i in dims_to_paint]

        mask_indices = ellipse_indices(tuple(radii))
        mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(int)

        # discard candidate coordinates that are out of bounds
        mask_indices = indices_in_shape(mask_indices, shape)

        if self._brush_settings.current_mode == PaintMode.ELLIPTICAL_CYLINDER:
            if self.brush_settings.elliptical_cylinder_limited_height_depth:
                height_depth = self.brush_settings.elliptical_cylinder_height_depth
            else:
                height_depth = self.data.shape[dim_not_painted]

        # Transfer valid coordinates to slice_coord
        slice_coord_temp = list(mask_indices.T)
        for j, i in enumerate(dims_to_paint):
            if self._brush_settings.current_mode == PaintMode.ELLIPTICAL_CYLINDER:
                slice_coord[i] = np.reshape(
                    np.repeat(
                        np.array([slice_coord_temp[j]], dtype=np.uint16),
                        min(
                            self.data.shape[dim_not_painted],
                            slice_coord[dim_not_painted] + height_depth + 1,
                        )
                        - max(0, slice_coord[dim_not_painted] - height_depth),
                        axis=0,
                    ),
                    -1,
                )
            else:
                slice_coord[i] = slice_coord_temp[j]
        if dim_not_painted is not None:
            if self._brush_settings.current_mode == PaintMode.ELLIPTICAL_CYLINDER:
                slice_coord[dim_not_painted] = np.repeat(
                    np.array(
                        range(
                            max(0, slice_coord[dim_not_painted] - height_depth),
                            min(
                                self.data.shape[dim_not_painted],
                                slice_coord[dim_not_painted] + height_depth + 1,
                            ),
                        ),
                        dtype=np.uint16,
                    ),
                    mask_indices.shape[0],
                )
            else:
                slice_coord[dim_not_painted] = slice_coord[dim_not_painted] * np.ones(
                    mask_indices.shape[0], dtype=np.uint16
                )
        del slice_coord_temp
        slice_coord = tuple(slice_coord)

        if self.preserve_background:
            if new_label != self.colormap.background_value:
                keep_coords = self.data[slice_coord] != self.colormap.background_value
                slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

        self.data_setitem(slice_coord, new_label, refresh)

    def _get_pt_not_disp(self) -> Dict[int, int]:
        point = np.round(self.world_to_data(self._slice_input.point)).astype(int)
        return {dim: point[dim] for dim in self._slice_input.not_displayed}

    def data_setitem(self, indices, value, refresh=True):
        changed_indices = self.data[indices] != value
        indices = tuple(x[changed_indices] for x in indices)
        del changed_indices
        value = self._slice.image.raw.dtype.type(value)

        if not indices or indices[0].size == 0:
            return

        values: np.ndarray = np.unique(self.data[indices])
        self._save_history(
            (
                indices,
                (
                    np.array(self.data[indices], copy=True)
                    if values.shape[0] > 1
                    else values[0]
                ),
                value,
            )
        )

        # update the labels image
        self.data[indices] = value

        pt_not_disp = self._get_pt_not_disp()
        displayed_indices = index_in_slice(
            indices, pt_not_disp, self._slice_input.order
        )

        # update data view
        self._slice.image.view[displayed_indices] = self.colormap._data_to_texture(
            value
        )

        if refresh is True:
            self.refresh()
        self.changed = True

    def get_status(
        self,
        position: Optional[Tuple] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world: bool = False,
    ) -> dict:
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = {}
        source_info["source"] = self.name
        source_info["additional_info"] = generate_layer_coords_status(
            position[-3:] / self.scale, value
        )

        return source_info
