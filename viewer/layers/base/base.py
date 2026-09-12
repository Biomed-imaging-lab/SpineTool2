from __future__ import annotations

from collections import defaultdict, namedtuple
from functools import cached_property
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from PyQt5.QtCore import QObject, pyqtSignal

from utils.geometry import (
    find_front_back_face,
    intersect_line_with_axis_aligned_bounding_box_3d,
)
from utils.qt_translater import Translater
from viewer.components._viewer_constants import CursorStyle
from viewer.layers.base._base_constants import Blending, Mode
from viewer.layers.status_messages import generate_layer_coords_status
from viewer.layers.utils._slice_input import SliceInput
from viewer.layers.utils.layer_utils import convert_to_uint8, get_extent_world
from viewer.utils.mouse_handler import MouseHandler
from viewer.utils.transforms import Scale

Extent = namedtuple("Extent", "data world step")


def no_op(layer: Layer, event) -> None:
    return


class Layer(MouseHandler, QObject):
    _modeclass = Mode

    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
    }
    _move_modes = {
        Mode.PAN_ZOOM: no_op,
    }
    _cursor_modes = {
        Mode.PAN_ZOOM: CursorStyle.STANDARD,
    }

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
    changed_ = pyqtSignal(bool)
    mode_ = pyqtSignal(object)

    def __init__(
        self,
        layer_id,
        name,
        *,
        metadata=None,
        scale=None,
        opacity=1,
        blending="translucent",
        visible=True,
        mode="pan_zoom",
    ):
        MouseHandler.__init__(self)
        QObject.__init__(self)

        if scale is not None and not np.all(scale):
            raise ValueError(
                "Layer {name} is invalid because it has scale values of 0. The layer's scale is currently {scale}".format(
                    name=repr(name),
                    scale=repr(scale),
                )
            )

        self._id = layer_id
        self._metadata = dict(metadata or {})
        self._opacity = opacity
        self._blending = Blending(blending)
        self._visible = visible
        self._help = ""
        self._cursor = CursorStyle.STANDARD
        self._cursor_size = 1.0
        self._mouse_pan = True
        self._mouse_zoom = True
        self._value = None
        self.scale_factor = 1
        self._mode = self._modeclass("pan_zoom")
        self._changed = False
        self._preview = False

        if scale is None:
            scale = [1] * 3
        self._scale = Scale(scale)
        self._slice_input = SliceInput(
            ndisplay=2,
            point=self.data_to_world((0,) * 3),
            order=tuple(range(3)),
        )

        self._editable = True
        self._centered = False

        self._thumbnail_shape = (32, 32, 4)
        self._thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        self._name = ""

        self.name = name
        self.mode = mode
        Translater.instance().language_changed_signal.connect(self.language_changed)

    def __str__(self):
        """Return self.name."""
        return self.name

    def __repr__(self):
        cls = type(self)
        return f"<{cls.__name__} layer {self.name!r} at {hex(id(self))}>"

    def _mode_setter_helper(self, mode, check_editable=True):
        mode = self._modeclass(mode)
        assert mode is not None
        if not self.editable and check_editable:
            mode = self._modeclass.PAN_ZOOM
        if mode == self._mode:
            return mode

        if mode.value not in self._modeclass.keys():
            raise ValueError("Mode not recognized: {mode}".format(mode=mode))

        for callback_list, mode_dict in [
            (self.mouse_drag_callbacks, self._drag_modes),
            (self.mouse_move_callbacks, self._move_modes),
            (
                self.mouse_double_click_callbacks,
                getattr(self, "_double_click_modes", defaultdict(lambda: no_op)),
            ),
        ]:
            if mode_dict[self._mode] in callback_list:
                callback_list.remove(mode_dict[self._mode])
            callback_list.append(mode_dict[mode])
        self.cursor = self._cursor_modes[mode]

        self.mouse_pan = mode == self._modeclass.PAN_ZOOM

        if mode == self._modeclass.PAN_ZOOM:
            self.help = ""

        return mode

    def language_changed(self) -> None:
        self.name_.emit(self._name)

    @property
    def _type_string(self):
        return self.__class__.__name__.lower()

    @property
    def mode(self) -> str:
        return str(self._mode)

    @mode.setter
    def mode(self, mode):
        mode = self._mode_setter_helper(mode)
        if mode == self._mode:
            return
        self._mode = mode

        self.mode_.emit(mode)

    @classmethod
    def _basename(cls):
        return f"{cls.__name__}"

    @property
    def name(self):
        text = ""
        if self._preview:
            text = Translater.instance().get_translation("preview")
        return text + self._name + " (" + str(self._id) + ")"

    @name.setter
    def name(self, name):
        if name == self._name:
            return
        if not name:
            name = self._basename()
        self._name = str(name)
        self.name_.emit(self._name)

    @property
    def changed(self):
        return self._changed

    @changed.setter
    def changed(self, changed):
        self._changed = changed
        self.changed_.emit(self._changed)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, preview):
        self._preview = preview
        self.editable = not preview
        self.name_.emit(self._name)

    def set_name(self, name) -> None:
        self.name = name

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict) -> None:
        self._metadata.clear()
        self._metadata.update(value)

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                "opacity must be between 0.0 and 1.0; got {opacity}".format(
                    opacity=opacity,
                )
            )

        self._opacity = opacity
        self._update_thumbnail()
        self.opacity_.emit()

    @property
    def blending(self):
        return str(self._blending)

    @blending.setter
    def blending(self, blending):
        self._blending = Blending(blending)
        self.blending_.emit()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        self._visible = visible
        self.refresh()
        self.visible_.emit()

    def _check_editable(self) -> bool:
        return self._editable

    @property
    def editable(self) -> bool:
        return self._check_editable()

    @editable.setter
    def editable(self, editable: bool):
        if self._editable == editable or self._preview and editable:
            return
        self._editable = editable
        self._on_editable_changed()
        self.editable_.emit()

    def _on_editable_changed(self) -> None:
        """Executes side-effects on this layer related to changes of the editable state."""

    @property
    def scale(self):
        return self._scale.scale

    @scale.setter
    def scale(self, scale):
        if scale is None:
            scale = [1] * 3
        self._scale = Scale(scale)
        self._clear_extent()
        self.scale_.emit()

    @property
    def _is_moving(self):
        return self._private_is_moving

    @_is_moving.setter
    def _is_moving(self, value):
        assert value in (True, False)
        if value:
            assert self._moving_coordinates is not None
        self._private_is_moving = value

    @property
    def data(self):
        raise NotImplementedError

    @data.setter
    def data(self, data):
        raise NotImplementedError

    @property
    def _extent_data(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    def extent(self) -> Extent:
        extent_data = self._extent_data
        data_to_world = self._scale
        extent_world = get_extent_world(extent_data, data_to_world, self._centered)
        return Extent(
            data=extent_data,
            world=extent_world,
            step=abs(data_to_world.scale),
        )

    def _clear_extent(self):
        if "extent" in self.__dict__:
            del self.extent
        self.extent_.emit()
        self.refresh()

    @property
    def _slice_indices(self):
        if len(self._slice_input.not_displayed) == 0:
            return (slice(None),) * 3
        return self._slice_input.data_indices(
            self._scale.inverse,
            getattr(self, "_round_index", True),
        )

    @property
    def thumbnail(self):
        return self._thumbnail

    @thumbnail.setter
    def thumbnail(self, thumbnail):
        if 0 in thumbnail.shape:
            thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        if thumbnail.dtype != np.uint8:
            thumbnail = convert_to_uint8(thumbnail)

        padding_needed = np.subtract(self._thumbnail_shape, thumbnail.shape)
        pad_amounts = [(p // 2, (p + 1) // 2) for p in padding_needed]
        thumbnail = np.pad(thumbnail, pad_amounts, mode="constant")

        # blend thumbnail with opaque black background
        background = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        background[..., 3] = 255

        f_dest = thumbnail[..., 3][..., None] / 255
        f_source = 1 - f_dest
        thumbnail = thumbnail * f_dest + background * f_source

        self._thumbnail = thumbnail.astype(np.uint8)
        self.thumbnail_.emit()

    @property
    def help(self):
        return self._help

    @help.setter
    def help(self, help_text):
        if help_text == self.help:
            return
        self._help = help_text
        self.help_.emit(help_text)

    @property
    def mouse_pan(self) -> bool:
        return self._mouse_pan

    @mouse_pan.setter
    def mouse_pan(self, mouse_pan: bool):
        if mouse_pan == self._mouse_pan:
            return
        self._mouse_pan = mouse_pan
        self.mouse_pan_.emit(self, mouse_pan)

    @property
    def mouse_zoom(self) -> bool:
        return self._mouse_zoom

    @mouse_zoom.setter
    def mouse_zoom(self, mouse_zoom: bool):
        if mouse_zoom == self._mouse_zoom:
            return
        self._mouse_zoom = mouse_zoom
        self.mouse_zoom_.emit(self, mouse_zoom)

    @property
    def cursor(self):
        return self._cursor

    @cursor.setter
    def cursor(self, cursor):
        if cursor == self.cursor:
            return
        self._cursor = cursor
        self.cursor_.emit(cursor)

    @property
    def cursor_size(self):
        return self._cursor_size

    @cursor_size.setter
    def cursor_size(self, cursor_size: float | np.ndarray):
        if (
            isinstance(cursor_size, np.ndarray)
            and isinstance(self.cursor_size, np.ndarray)
            and len(cursor_size) == len(self.cursor_size)
            and np.all(cursor_size == self.cursor_size)
            or isinstance(cursor_size, float)
            and isinstance(self.cursor_size, float)
            and cursor_size == self.cursor_size
        ):
            return
        self._cursor_size = cursor_size
        self.cursor_size_.emit(cursor_size)

    def set_view_slice(self):
        raise NotImplementedError

    def _slice_dims(self, point=None, ndisplay=2, order=None):
        point = (0,) * 3 if point is None else tuple(point)

        if order is None:
            order = tuple(range(3))

        slice_input = SliceInput(
            ndisplay=ndisplay,
            point=point,
            order=order,
        )

        if self._slice_input == slice_input:
            return

        self._slice_input = slice_input

        self.refresh()

    def _update_thumbnail(self):
        raise NotImplementedError

    def _get_value(self, position):
        raise NotImplementedError

    def get_value(
        self,
        position: Tuple[float],
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world=False,
    ):
        if self.visible:
            if world:
                position = self.world_to_data(position)

            if (dims_displayed is not None) and (view_direction is not None):
                if len(dims_displayed) == 2:
                    value = self._get_value(position=tuple(position))

                elif len(dims_displayed) == 3:
                    view_direction = self._world_to_data_ray(list(view_direction))
                    start_point, end_point = self.get_ray_intersections(
                        position=position,
                        view_direction=view_direction,
                        dims_displayed=dims_displayed,
                        world=False,
                    )
                    value = self._get_value_3d(
                        start_point=start_point,
                        end_point=end_point,
                        dims_displayed=dims_displayed,
                    )
            else:
                value = self._get_value(position)

        else:
            value = None
        self._value = value
        return value

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Union[float, int, None, Tuple[Union[float, int, None], Optional[int]]]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value
            The data value along the supplied ray.
        """

    def _set_highlight(self):
        """Render layer highlights when appropriate."""

    def _update_cursor(self):
        """Update cursor size."""

    def refresh(self, event=None):
        """Refresh all layer data based on current view slice."""
        if self.visible:
            self.set_view_slice()
            self.set_data_.emit()
            self._update_thumbnail()
            self._set_highlight()
            self._update_cursor()

    def world_to_data(self, position):
        if len(position) >= 3:
            coords = list(position[-3:])
        else:
            coords = [0] * (3 - len(position)) + list(position)

        return tuple(self._scale.inverse(coords))

    def data_to_world(self, position):
        if len(position) >= 3:
            coords = list(position[-3:])
        else:
            coords = [0] * (3 - len(position)) + list(position)

        return tuple(self._scale(coords))

    def _world_to_displayed_data(
        self, position: np.ndarray, dims_displayed: np.ndarray
    ) -> tuple:
        position_nd = self.world_to_data(position)
        position_ndisplay = np.asarray(position_nd)[dims_displayed]
        return tuple(position_ndisplay)

    def _world_to_data_ray(self, vector) -> tuple:
        p = np.asarray(self.world_to_data(vector))
        normalized_vector = p / np.linalg.norm(p)

        return tuple(normalized_vector)

    def _world_to_displayed_data_ray(
        self, vector_world: npt.ArrayLike, dims_displayed: List[int]
    ) -> np.ndarray:
        vector_data_nd = np.asarray(self._world_to_data_ray(vector_world))
        vector_data_ndisplay = vector_data_nd[dims_displayed]
        vector_data_ndisplay /= np.linalg.norm(vector_data_ndisplay)
        return vector_data_ndisplay

    def _display_bounding_box(self, dims_displayed: np.ndarray):
        return self._extent_data[:, dims_displayed].T

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

        start_point, end_point = self._get_ray_intersections(
            position=position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
            bounding_box=bounding_box,
        )
        return start_point, end_point

    def _get_offset_data_position(self, position: List[float]) -> List[float]:
        return position

    def _get_ray_intersections(
        self,
        position: List[float],
        view_direction: np.ndarray,
        dims_displayed: List[int],
        world: bool = True,
        bounding_box: Optional[np.ndarray] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        # get the view direction and click position in data coords
        # for the displayed dimensions only
        if world is True:
            view_dir = self._world_to_displayed_data_ray(view_direction, dims_displayed)
            click_pos_data = self._world_to_displayed_data(position, dims_displayed)
        else:
            # adjust for any offset between viewer and data coordinates
            position = self._get_offset_data_position(position)

            view_dir = np.asarray(view_direction)[dims_displayed]
            click_pos_data = np.asarray(position)[dims_displayed]

        # Determine the front and back faces
        front_face_normal, back_face_normal = find_front_back_face(
            click_pos_data, bounding_box, view_dir
        )
        if front_face_normal is None or back_face_normal is None:
            # click does not intersect the data bounding box
            return None, None

        # Calculate ray-bounding box face intersections
        start_point_displayed_dimensions = (
            intersect_line_with_axis_aligned_bounding_box_3d(
                click_pos_data, view_dir, bounding_box, front_face_normal
            )
        )
        end_point_displayed_dimensions = (
            intersect_line_with_axis_aligned_bounding_box_3d(
                click_pos_data, view_dir, bounding_box, back_face_normal
            )
        )

        # add the coordinates for the axes not displayed
        start_point = np.asarray(position)
        start_point[dims_displayed] = start_point_displayed_dimensions
        end_point = np.asarray(position)
        end_point[dims_displayed] = end_point_displayed_dimensions

        return start_point, end_point

    def _update_draw(self, scale_factor):
        self.scale_factor = scale_factor

    def get_source_str(self):
        return self.name

    def get_status(
        self,
        position: Optional[Tuple[float, ...]] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
        world=False,
    ):
        source_info = {}
        source_info["source"] = self.name
        source_info["additional_info"] = generate_layer_coords_status(
            position[-3:] / self.scale, None
        )
        return source_info

    def _on_selection(self, selected: bool):
        """Reaction to changing the layer selection"""
