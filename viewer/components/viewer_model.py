from typing import Callable, List

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from superqt.utils import QSignalThrottler

from viewer.components._viewer_constants import CursorStyle
from viewer.components._viewer_mouse_bindings import dims_scroll
from viewer.components.camera import Camera
from viewer.components.cursor import Cursor
from viewer.components.dims import Dims
from viewer.components.layer_list.layerlist import LayerList
from viewer.components.overlays.brush_circle import BrushCircleOverlay
from viewer.layers.base.base import Layer
from viewer.layers.points.points import Points
from viewer.utils.mouse_handler import MouseHandler


def no_op() -> None:
    return


class ViewerModel(MouseHandler, QObject):
    mouse_over_canvas_ = pyqtSignal(bool)
    status_ = pyqtSignal(object)
    layer_added_ = pyqtSignal(object)
    layer_removed_ = pyqtSignal(object)
    layers_reordered_ = pyqtSignal()

    def __init__(
        self,
        layer_lists: List[LayerList],
        ndisplay=2,
        order=(),
        axis_labels=(),
        layers_sorter: Callable = None,
        active_layer: Layer = None,
    ):
        MouseHandler.__init__(self)
        QObject.__init__(self)

        self.dims = Dims(ndisplay, order, axis_labels)
        self.camera = Camera()
        self.cursor = Cursor()
        self.layer_lists = layer_lists
        if layers_sorter:
            self._layers_sorter = layers_sorter
        else:
            self._layers_sorter = default_layers_sorter
        self.layers = self._layers_sorter(self.layer_lists)
        self.brush_circle = BrushCircleOverlay()
        self.canvas_size = (600, 800)
        self._active_layer = active_layer
        self._status = {"text": ""}
        self._mouse_over_canvas = False

        self.dims.ndisplay_.connect(self._update_layers)
        self.dims.ndisplay_.connect(lambda _: self.reset_view())
        self.dims.order_.connect(self._update_layers)
        self.dims.order_.connect(lambda _: self.reset_view())
        self.dims.current_step_.connect(self._update_layers)

        self.status_throttler = QSignalThrottler(parent=self)
        self.status_throttler.setTimeout(50)
        self.cursor.position_.connect(self.status_throttler.throttle)
        self.status_throttler.triggered.connect(self.update_status_bar_from_cursor)

        for layer_list in self.layer_lists:
            layer_list.inserted_.connect(self._on_add_layer)
            layer_list.removed_.connect(self._on_remove_layer)
            layer_list.reordered_.connect(self._on_layers_reordered)

        # Add mouse callback
        self.mouse_wheel_callbacks.append(dims_scroll)

    def __hash__(self):
        return id(self)

    @property
    def _sliced_extent_world(self) -> np.ndarray:
        if len(self.layers) == 0:
            return np.vstack([np.zeros(3), np.repeat(512, 3)])

        return self.layers.extent.world[:, self.dims.displayed]

    def reset_view(self, pos=None, zoom_multiplier=1.0):
        extent = self._sliced_extent_world
        scene_size = extent[1] - extent[0]
        corner = extent[0]
        if not pos or isinstance(pos, int):
            center = np.add(corner, np.divide(scene_size, 2))[-self.dims.ndisplay :]
        else:
            center = np.add(corner, pos[-self.dims.ndisplay :])
        center = [0] * (self.dims.ndisplay - len(center)) + list(center)
        self.camera.center = center
        if np.max(scene_size) == 0:
            self.camera.zoom = zoom_multiplier * 0.95 * np.min(self.canvas_size)
        else:
            scale = np.array(scene_size[-2:])
            scale[np.isclose(scale, 0)] = 1
            self.camera.zoom = (
                zoom_multiplier * 0.95 * np.min(np.array(self.canvas_size) / scale)
            )
        self.camera.angles = (0, 0, 90)

    def _update_layers(self, *, layers=None):
        layers = layers or self.layers
        for layer in layers:
            layer._slice_dims(self.dims.point, self.dims.ndisplay, self.dims.order)
        position = list(self.cursor.position)
        for ind in self.dims.order[: -self.dims.ndisplay]:
            position[ind] = self.dims.point[ind]
        self.cursor.position = position

    @property
    def active_layer(self) -> Layer:
        return self._active_layer

    @active_layer.setter
    def active_layer(self, layer: Layer) -> None:
        self._active_layer = layer
        if layer is None:
            self.cursor.style = CursorStyle.STANDARD
        else:
            self.cursor.style = layer.cursor
            self.cursor.size = layer.cursor_size
            self.camera.mouse_pan = layer.mouse_pan
            self.camera.mouse_zoom = layer.mouse_zoom
            self.update_status_bar_from_cursor()

    @property
    def status(self) -> dict:
        return self._status

    @status.setter
    def status(self, status) -> None:
        self._status = status
        self.status_.emit(status)

    @property
    def mouse_over_canvas(self) -> bool:
        return self._mouse_over_canvas

    @mouse_over_canvas.setter
    def mouse_over_canvas(self, value) -> None:
        self._mouse_over_canvas = value
        self.mouse_over_canvas_.emit(value)

    @property
    def layers_sorter(self) -> Callable:
        return self._layers_sorter

    @layers_sorter.setter
    def layers_sorter(self, sorter) -> None:
        self._layers_sorter = sorter
        self.layers = self._layers_sorter(self.layer_lists)
        self.layers_reordered_.emit()

    def _update_mouse_pan(self, layer, mouse_pan):
        if layer is self._active_layer:
            self.camera.mouse_pan = mouse_pan

    def _update_mouse_zoom(self, layer, mouse_zoom):
        if layer is self._active_layer:
            self.camera.mouse_zoom = mouse_zoom

    def _update_cursor(self, cursor):
        self.cursor.style = cursor

    def _update_cursor_size(self, cursor_size):
        self.cursor.size = cursor_size

    def _on_layers_change(self):
        if len(self.layers) == 0:
            self.dims.reset()
        else:
            ranges = self.layers._ranges
            self.dims.set_range(range(3), ranges)

    def _on_add_layer(self, index, layer: Layer):
        layer.mouse_pan_.connect(self._update_mouse_pan)
        layer.mouse_zoom_.connect(self._update_mouse_zoom)
        layer.cursor_.connect(self._update_cursor)
        layer.cursor_size_.connect(self._update_cursor_size)
        layer.data_.connect(self._on_layers_change)
        layer.scale_.connect(self._on_layers_change)

        self.layers = self._layers_sorter(self.layer_lists)

        self._on_layers_change()
        self._update_layers(layers=[layer])

        if len(self.layers) == 1:
            self.reset_view()
            self.dims._go_to_center_step()

        self.layer_added_.emit(layer)

    def _on_remove_layer(self, index, layer: Layer):
        layer.mouse_pan_.disconnect(self._update_mouse_pan)
        layer.mouse_zoom_.disconnect(self._update_mouse_zoom)
        layer.cursor_.disconnect(self._update_cursor)
        layer.cursor_size_.disconnect(self._update_cursor_size)
        layer.data_.disconnect(self._on_layers_change)
        layer.scale_.disconnect(self._on_layers_change)

        self.layers = self._layers_sorter(self.layer_lists)

        self._on_layers_change()

        self.layer_removed_.emit(layer)

    def _on_layers_reordered(self):
        self._on_layers_change()
        self.layers_reordered_.emit()

    def add_layer(self, layer: Layer, layer_list: int) -> Layer:
        self.layer_lists[layer_list].append(layer)
        return layer

    def remove_layer(self, layer: Layer, layer_list: int) -> None:
        self.layer_lists[layer_list].remove(layer)

    def set_highlight_thickness(self, thickness: int) -> None:
        for layer in self.layers:
            if isinstance(layer, Points):
                layer.highlight_thickness = thickness

    def set_scale(self, scale) -> None:
        for layer in self.layers:
            layer.scale = scale
        self.dims.current_step = self.dims.current_step

    def unsaved_layers(self) -> List[Layer]:
        unsaved = []
        for layer in self.layers:
            if layer.changed:
                unsaved.append(layer)
        return unsaved

    def update_status_bar_from_cursor(self, event=None):
        if not self._mouse_over_canvas:
            return
        if self._active_layer is not None:
            self.status = self._active_layer.get_status(
                self.cursor.position,
                view_direction=self.cursor._view_direction,
                dims_displayed=list(self.dims.displayed),
                world=True,
            )
        else:
            self.status = {"text": ""}


def default_layers_sorter(layer_lists: List[LayerList]) -> LayerList:
    layers = []
    for l in layer_lists:
        layers.extend(l._list)
    return LayerList(layers)
