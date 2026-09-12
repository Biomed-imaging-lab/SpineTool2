import warnings
from collections import deque
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from psygnal.containers import Selection
from PyQt5.QtCore import pyqtSignal

from utils.colormaps.standardize_color import hex_to_name, rgb_to_hex, transform_color
from utils.geometry import project_points_onto_plane, rotate_points
from viewer.components._viewer_constants import CursorStyle
from viewer.layers.base.base import Layer, no_op
from viewer.layers.points._points_constants import ActionType, Mode, Shading, Symbol
from viewer.layers.points._points_mouse_bindings import add, select
from viewer.layers.points._points_utils import coerce_symbol, fix_data_points
from viewer.layers.status_messages import generate_layer_coords_status
from viewer.layers.utils.interactivity_utils import displayed_plane_from_nd_line_segment


class Points(Layer):
    _modeclass = Mode
    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.ADD: add,
        Mode.SELECT: select,
    }
    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.ADD: no_op,
        Mode.SELECT: no_op,
    }
    _cursor_modes = {
        Mode.PAN_ZOOM: CursorStyle.STANDARD,
        Mode.ADD: CursorStyle.CROSSHAIR,
        Mode.SELECT: CursorStyle.STANDARD,
    }
    _max_points_thumbnail = 1024
    _history_limit = 100

    size_ = pyqtSignal()
    edge_width_ = pyqtSignal()
    edge_width_is_relative_ = pyqtSignal()
    face_color_ = pyqtSignal()
    edge_color_ = pyqtSignal()
    symbol_ = pyqtSignal()
    out_of_slice_display_ = pyqtSignal()
    highlight_ = pyqtSignal()
    shading_ = pyqtSignal()
    antialiasing_ = pyqtSignal(float)
    changed_ = pyqtSignal(bool)
    canvas_size_limits_ = pyqtSignal()

    def __init__(
        self,
        layer_id,
        name,
        data: np.ndarray,
        *,
        symbol="o",
        size=10,
        edge_width=0.1,
        edge_width_is_relative=True,
        edge_color="dimgray",
        face_color="white",
        out_of_slice_display=False,
        metadata=None,
        scale=None,
        opacity=1,
        blending="translucent",
        visible=True,
        shading="none",
        canvas_size_limits=(2, 10000),
        antialiasing=1,
        highlight_thickness=1,
        shown=True,
    ):
        self._data = fix_data_points(data).astype(np.int16)

        super().__init__(
            layer_id,
            name,
            metadata=metadata,
            scale=scale,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        # Index of hovered point
        self._highlight_index = []
        self._highlight_box = None
        self._selected = True

        self._drag_box = None
        self._drag_start = None
        self._drag_normal = None
        self._drag_up = None

        # initialize view data
        self.__indices_view = np.empty(0, int)
        self._view_size_scale = []

        self._is_selecting = False
        self._round_index = False

        self._edge_width_is_relative = False
        self._shown = np.empty(0).astype(bool)

        # Indices of selected points
        self._selected_data: Selection[int] = Selection()
        # Indices of selected points within the currently viewed slice
        self._selected_view = []

        self._size = 10
        self._edge_width = 0.1
        self._symbol = "o"

        # Index of hovered point
        self._value = None
        self._value_stored = None

        self._edge_color = transform_color("dimgray")
        self._face_color = transform_color("white")

        self._out_of_slice_display = out_of_slice_display

        # Save the point style params
        self.size = size
        self.shown = shown
        self.symbol = symbol
        self.edge_width = edge_width
        self.edge_width_is_relative = edge_width_is_relative
        self.edge_color = edge_color
        self.face_color = face_color

        self.canvas_size_limits = canvas_size_limits
        self.shading = shading
        self.antialiasing = antialiasing

        self._highlight_thickness = highlight_thickness

        # Trigger generation of view slice and thumbnail
        self.refresh()
        self._reset_history()

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

    def _save_history(self, value):
        self._redo_history.clear()
        if self._block_history:
            self._staged_history.append(value)
        else:
            self._append_to_undo_history([value])

    def _load_history(self, before, after):
        if len(before) == 0:
            return

        history_items = before.pop()
        new_history_items = []
        for history_item in reversed(history_items):
            new_history_item = {}
            action = history_item["action"]
            data = history_item["data"]
            if action == ActionType.ADDED:
                new_history_item["action"] = ActionType.REMOVED
                new_history_item["data"] = self._data[-data:]
                self._data = self._data[:-data]
                self._shown = self._shown[:-data]
            elif action == ActionType.REMOVED:
                new_history_item["action"] = ActionType.ADDED
                shown = np.repeat([True], data.shape[0], axis=0)
                self._shown = np.concatenate((self._shown, shown), axis=0)
                self._data = np.concatenate((self._data, data), axis=0)
                new_history_item["data"] = data.shape[0]
            else:
                new_history_item["action"] = ActionType.CHANGED
                new_history_item["data"] = (data[0], self._data[data[0]])
                self._data[data[0]] = data[1]
            new_history_items.append(new_history_item)
        after.append(new_history_items)

        self.refresh()
        self.changed = True

    def undo(self):
        self._load_history(self._undo_history, self._redo_history)

    def redo(self):
        self._load_history(self._redo_history, self._undo_history)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: Optional[np.ndarray]):
        self._set_data(data)
        self.data_.emit(self._data)
        self._reset_history()
        self.changed = False

    def _set_data(self, data: Optional[np.ndarray]):
        data = fix_data_points(data)
        cur_npoints = len(self._data)
        self._data = data.astype(np.int16)

        if len(data) < cur_npoints:
            self._shown = self._shown[: len(data)]
        elif len(data) > cur_npoints:
            adding = len(data) - cur_npoints
            shown = np.repeat([True], adding, axis=0)
            self._shown = np.concatenate((self._shown, shown), axis=0)

        self._clear_extent()

    def _on_selection(self, selected):
        if selected:
            self._selected = True
            self._set_highlight()
        else:
            self._selected = False
            self._highlight_box = None
            self._highlight_index = []
            self.highlight_.emit()

    @property
    def _extent_data(self) -> np.ndarray:
        if len(self.data) == 0:
            extrema = np.full((2, 3), np.nan)
        else:
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema.astype(float)

    @property
    def out_of_slice_display(self) -> bool:
        return self._out_of_slice_display

    @out_of_slice_display.setter
    def out_of_slice_display(self, out_of_slice_display: bool) -> None:
        self._out_of_slice_display = bool(out_of_slice_display)
        self.out_of_slice_display_.emit()
        self.refresh()

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol) -> None:
        self._symbol = coerce_symbol(symbol)
        self.symbol_.emit()
        self.highlight_.emit()

    @property
    def highlight_thickness(self) -> int:
        return self._highlight_thickness

    @highlight_thickness.setter
    def highlight_thickness(self, highlight_thickness: int) -> None:
        if highlight_thickness <= 0:
            raise ValueError("Thickness value must be positive")
        self._highlight_thickness = highlight_thickness
        self.highlight_.emit()

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Size value must be positive")
        self._size = size
        self.size_.emit()
        self.refresh()

    @property
    def antialiasing(self) -> float:
        return self._antialiasing

    @antialiasing.setter
    def antialiasing(self, value: float):
        if value < 0:
            warnings.warn(
                message="antialiasing value must be positive, value will be set to 0.",
                category=RuntimeWarning,
            )
        self._antialiasing = max(0, value)
        self.antialiasing_.emit(self._antialiasing)

    @property
    def shading(self) -> Shading:
        return self._shading

    @shading.setter
    def shading(self, value):
        self._shading = Shading(value)
        self.shading_.emit()

    @property
    def canvas_size_limits(self) -> Tuple[float, float]:
        return self._canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value):
        self._canvas_size_limits = float(value[0]), float(value[1])
        self.canvas_size_limits_.emit()

    @property
    def shown(self):
        return self._shown

    @shown.setter
    def shown(self, shown):
        self._shown = np.broadcast_to(shown, self.data.shape[0]).copy().astype(bool)
        self.refresh()

    @property
    def edge_width(self) -> Union[int, float]:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: Union[int, float]) -> None:
        if edge_width < 0:
            raise ValueError("Edge_width value must be positive or 0")
        if self.edge_width_is_relative and edge_width > 1:
            raise ValueError(
                "Edge_width must be between 0 and 1 if edge_width_is_relative is enabled"
            )

        self._edge_width = edge_width
        self.edge_width_.emit()
        self.refresh()

    @property
    def edge_width_is_relative(self) -> bool:
        return self._edge_width_is_relative

    @edge_width_is_relative.setter
    def edge_width_is_relative(self, edge_width_is_relative: bool) -> None:
        if edge_width_is_relative and (self.edge_width > 1) | (self.edge_width < 0):
            raise ValueError(
                "edge_width_is_relative can only be enabled if edge_width is between 0 and 1"
            )
        self._edge_width_is_relative = edge_width_is_relative
        self.edge_width_is_relative_.emit()

    @property
    def edge_color(self) -> np.ndarray:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = transform_color(edge_color)
        self.edge_color_.emit()

    @property
    def edge_color_as_string(self) -> str:
        hex_ = rgb_to_hex(self._edge_color)
        return hex_to_name.get(hex_, hex_)

    @property
    def face_color(self) -> np.ndarray:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._face_color = transform_color(face_color)
        self.face_color_.emit()

    @property
    def face_color_as_string(self) -> str:
        hex_ = rgb_to_hex(self._face_color)
        return hex_to_name.get(hex_, hex_)

    @property
    def selected_data(self) -> Selection[int]:
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data: Sequence[int]) -> None:
        self._selected_data.clear()
        self._selected_data.update(set(selected_data))
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )

        self._set_highlight()

    @Layer.mode.getter
    def mode(self) -> str:
        return str(self._mode)

    def _mode_setter_helper(self, mode):
        mode = super()._mode_setter_helper(mode)
        if mode == self._mode:
            return mode

        if mode == Mode.ADD:
            self.selected_data = set()
            self.mouse_pan = True

        self._set_highlight()
        return mode

    @property
    def _indices_view(self):
        return self.__indices_view

    @_indices_view.setter
    def _indices_view(self, value):
        if len(self._shown) == 0:
            self.__indices_view = np.empty(0, int)
        else:
            self.__indices_view = value[self.shown[value]]

    @property
    def _view_data(self) -> np.ndarray:
        if len(self._indices_view) > 0:
            data = self.data[np.ix_(self._indices_view, self._slice_input.displayed)]
        else:
            data = np.zeros((0, self._slice_input.ndisplay))

        return data

    @property
    def _view_size(self) -> np.ndarray:
        if len(self._indices_view) > 0:
            sizes = (
                np.broadcast_to([self.size], len(self._indices_view)).copy()
                * self._view_size_scale
            )
        else:
            sizes = np.array([])
        return sizes

    @property
    def _view_symbol(self) -> np.ndarray:
        return np.broadcast_to([self.symbol], len(self._indices_view)).copy()

    @property
    def _view_edge_width(self) -> np.ndarray:
        return np.broadcast_to([self.edge_width], len(self._indices_view)).copy()

    @property
    def _view_face_color(self) -> np.ndarray:
        return np.repeat(self.face_color, len(self._indices_view), axis=0).copy()

    @property
    def _view_edge_color(self) -> np.ndarray:
        return np.repeat(self.edge_color, len(self._indices_view), axis=0).copy()

    def _check_editable(self) -> bool:
        preview_editing = bool(self.metadata.get("allow_preview_point_editing", False))
        return (self._editable or preview_editing) and self._slice_input.ndisplay == 2

    def _on_editable_changed(self) -> None:
        if not self.editable:
            self.mode = Mode.PAN_ZOOM

    def _update_draw(self, scale_factor):
        super()._update_draw(scale_factor)
        self._set_highlight()

    def _slice_data(self, dims_indices) -> Tuple[List[int], Union[float, np.ndarray]]:
        # Get a list of the data for the points in this slice
        not_disp = list(self._slice_input.not_displayed)
        # We want a numpy array so we can use fancy indexing with the non-displayed
        # indices, but as dims_indices can (and often/always does) contain slice
        # objects, the array has dtype=object which is then very slow for the
        # arithmetic below. As Points._round_index is always False, we can safely
        # convert to float to get a major performance improvement.
        not_disp_indices = np.array(dims_indices)[not_disp].astype(float)
        if len(self.data) > 0:
            if self.out_of_slice_display is True:
                distances = abs(self.data[:, not_disp] - not_disp_indices)
                view_dim = distances.shape[1]
                sizes = (
                    np.repeat(
                        np.broadcast_to(self.size, len(self.data)).copy(), view_dim
                    ).reshape(distances.shape)
                    / 2
                )
                matches = np.all(distances <= sizes, axis=1)
                size_match = sizes[matches]
                size_match[size_match == 0] = 1
                scale_per_dim = (size_match - distances[matches]) / size_match
                scale_per_dim[size_match == 0] = 1
                scale = np.prod(scale_per_dim, axis=1)
                slice_indices = np.where(matches)[0].astype(int)
                return slice_indices, scale

            data = self.data[:, not_disp]
            distances = np.abs(data - not_disp_indices)
            matches = np.all(distances <= 0.5, axis=1)
            slice_indices = np.where(matches)[0].astype(int)
            return slice_indices, 1

        return [], np.empty(0)

    def _get_value(self, position) -> Optional[int]:
        # Display points if there are any in this slice
        view_data = self._view_data
        selection = None
        if len(view_data) > 0:
            displayed_position = [position[i] for i in self._slice_input.displayed]
            # positions are scaled anisotropically by scale, but sizes are not,
            # so we need to calculate the ratio to correctly map to screen coordinates
            scale_ratio = self.scale[self._slice_input.displayed] / self.scale[-1]
            # Get the point sizes
            sizes = np.expand_dims(self._view_size, axis=1) / scale_ratio / 2
            distances = abs(view_data - displayed_position)
            in_slice_matches = np.all(
                distances <= sizes,
                axis=1,
            )
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = self._indices_view[indices[-1]]

        return selection

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        if (start_point is None) or (end_point is None):
            # if the ray doesn't intersect the data volume, no points could have been intersected
            return None
        plane_point, plane_normal = displayed_plane_from_nd_line_segment(
            start_point, end_point, dims_displayed
        )

        # project the in view points onto the plane
        projected_points, projection_distances = project_points_onto_plane(
            points=self._view_data,
            plane_point=plane_point,
            plane_normal=plane_normal,
        )

        # rotate points and plane to be axis aligned with normal [0, 0, 1]
        rotated_points, rotation_matrix = rotate_points(
            points=projected_points,
            current_plane_normal=plane_normal,
            new_plane_normal=[0, 0, 1],
        )
        rotated_click_point = np.dot(rotation_matrix, plane_point)

        # positions are scaled anisotropically by scale, but sizes are not,
        # so we need to calculate the ratio to correctly map to screen coordinates
        scale_ratio = self.scale[self._slice_input.displayed] / self.scale[-1]
        # find the points the click intersects
        sizes = np.expand_dims(self._view_size, axis=1) / scale_ratio / 2
        distances = abs(rotated_points - rotated_click_point)
        in_slice_matches = np.all(
            distances <= sizes,
            axis=1,
        )
        indices = np.where(in_slice_matches)[0]

        if len(indices) > 0:
            # find the point that is most in the foreground
            candidate_point_distances = projection_distances[indices]
            closest_index = indices[np.argmin(candidate_point_distances)]
            selection = self._indices_view[closest_index]
        else:
            selection = None
        return selection

    def get_ray_intersections(
        self,
        position: List[float],
        view_direction: np.ndarray,
        dims_displayed: List[int],
        world: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        if len(dims_displayed) != 3:
            return None, None

        # create the bounding box in data coordinates
        bounding_box = self._display_bounding_box(dims_displayed)

        if bounding_box is None:
            return None, None

        start_point, end_point = self._get_ray_intersections(
            position=position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
            bounding_box=bounding_box,
        )
        return start_point, end_point

    def set_view_slice(self):
        # get the indices of points in view
        indices, scale = self._slice_data(self._slice_indices)

        # Update the _view_size_scale in accordance to the self._indices_view setter.
        # If out_of_slice_display is False, scale is a number and not an array.
        # Therefore we have an additional if statement checking for
        # self._view_size_scale being an integer.
        if not isinstance(scale, np.ndarray):
            self._view_size_scale = scale
        elif len(self._shown) == 0:
            self._view_size_scale = np.empty(0, int)
        else:
            self._view_size_scale = scale[self.shown[indices]]

        self._indices_view = np.array(indices, dtype=int)
        # get the selected points that are in view
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )
        self._set_highlight(False)

    def _set_highlight(self, signal=True):
        self._highlight_index = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )
        if signal:
            self.highlight_.emit()

    def _update_thumbnail(self):
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        view_data = self._view_data
        if len(view_data) > 0:
            # Get the zoom factor required to fit all data in the thumbnail.
            de = self._extent_data
            min_vals = [de[0, i] for i in self._slice_input.displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in self._slice_input.displayed]
            ).astype(int)
            zoom_factor = np.divide(self._thumbnail_shape[:2], shape[-2:]).min()

            # Maybe subsample the points.
            if len(view_data) > self._max_points_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(view_data), self._max_points_thumbnail
                )
                points = view_data[thumbnail_indices]
            else:
                points = view_data
                thumbnail_indices = self._indices_view

            # Calculate the point coordinates in the thumbnail data space.
            thumbnail_shape = np.clip(
                np.ceil(zoom_factor * np.array(shape[:2])).astype(int),
                1,  # smallest side should be 1 pixel wide
                self._thumbnail_shape[:2],
            )
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(coords, 0, thumbnail_shape - 1)

            # Draw single pixel points in the colormapped thumbnail.
            colormapped = np.zeros((*thumbnail_shape, 4))
            colormapped[..., 3] = 1
            colormapped[coords[:, 0], coords[:, 1]] = self._face_color

        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def add(self, coords):
        if self.metadata.get("neck_edit_phase") == "shaft":
            return
        cur_points = len(self.data)
        self._set_data(np.append(self.data, np.atleast_2d(coords), axis=0))
        self._save_history({"action": ActionType.ADDED, "data": 1})
        self.data_.emit(self._data)
        self.selected_data = set(np.arange(cur_points, len(self.data)))
        self.changed = True

    def remove_selected(self):
        if self.metadata.get("neck_edit_phase") == "shaft":
            return
        index = list(self.selected_data)
        index.sort()
        if len(index):
            self._save_history({"action": ActionType.REMOVED, "data": self.data[index]})
            self._shown = np.delete(self._shown, index, axis=0)
            if self._value in self.selected_data:
                self._value = None
            else:
                if self._value is not None:
                    indices_removed = np.array(index) < self._value
                    offset = np.sum(indices_removed)
                    self._value -= offset
                    self._value_stored -= offset

            self._set_data(np.delete(self.data, index, axis=0))
            self.data_.emit(self._data)
            self.selected_data = set()
        self.changed = True

    def _move(
        self,
        selection_indices: Sequence[int],
        position: Sequence[Union[int, float]],
        history=True,
        signal=True,
    ) -> None:
        if len(selection_indices) > 0:
            selection_indices = list(selection_indices)
            disp = list(self._slice_input.displayed)
            if self.metadata.get("neck_edit_phase") == "shaft":
                # Shaft Z is intentionally controlled only by the numeric field.
                disp = [dimension for dimension in disp if dimension != 0]
                if not disp:
                    return
            if history:
                self._save_history(
                    {
                        "action": ActionType.CHANGED,
                        "data": (
                            np.ix_(selection_indices, disp),
                            self.data[np.ix_(selection_indices, disp)],
                        ),
                    }
                )
            self._set_drag_start(selection_indices, position)
            center = self.data[np.ix_(selection_indices, disp)].mean(axis=0)
            shift = np.array(position)[disp] - center - self._drag_start
            self.data[np.ix_(selection_indices, disp)] = (
                self.data[np.ix_(selection_indices, disp)] + shift
            )
            self.refresh()
            if signal:
                self.data_.emit(self._data)
            self.changed = True

    def _set_drag_start(
        self,
        selection_indices: Sequence[int],
        position: Sequence[Union[int, float]],
        center_by_data: bool = True,
    ) -> None:
        selection_indices = list(selection_indices)
        dims_displayed = list(self._slice_input.displayed)
        if self.metadata.get("neck_edit_phase") == "shaft":
            dims_displayed = [dimension for dimension in dims_displayed if dimension != 0]
        if self._drag_start is None:
            self._drag_start = np.array(position, dtype=float)[dims_displayed]
            if len(selection_indices) > 0 and center_by_data:
                center = self.data[np.ix_(selection_indices, dims_displayed)].mean(
                    axis=0
                )
                self._drag_start -= center

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
