from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget

from utils.colormaps.standardize_color import transform_color
from utils.themes import color_as_hex, get_theme
from viewer.components._viewer_constants import CursorStyle
from viewer.components.overlays.base import Overlay
from viewer.components.overlays.brush_circle import BrushCircleOverlay
from viewer.layers.base.base import Layer
from viewer.utils.utils import (
    ReadOnlyWrapper,
    crosshair_pixmap,
    ellipse_pixmap,
    square_pixmap,
)
from viewer.vispy.camera import VispyCamera
from viewer.vispy.canvas import VispyCanvas
from viewer.vispy.utils.visual import create_vispy_layer, create_vispy_overlay
from viewer.widgets.qt_dims import QtDims
from viewer.widgets.qt_viewer_central_widget import QtViewerCentralWidget

if TYPE_CHECKING:
    from viewer.components.viewer_model import ViewerModel


class QtViewer(QSplitter):
    def __init__(self, viewer: "ViewerModel", parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        QCoreApplication.setAttribute(
            Qt.AA_UseStyleSheetPropagationInWidgetStyles, True
        )

        self.viewer = viewer
        self.reset_when_resize = False
        self.dims = QtDims(self.viewer.dims)

        # This dictionary holds the corresponding vispy visual for each layer
        self.layer_to_visual = {}
        self.overlay_to_visual = {}

        self._create_canvas()

        # Stacked widget to provide a welcome page
        self._central_widget = QtViewerCentralWidget(self, self.canvas.native)
        self._central_widget.leave.connect(self._leave_canvas)
        self._central_widget.enter.connect(self._enter_canvas)
        self._central_widget.resized.connect(self._reset_view_if_needed)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 2, 0, 2)
        main_layout.addWidget(self._central_widget)
        main_layout.addWidget(self.dims)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        self.setOrientation(Qt.Orientation.Vertical)
        self.addWidget(main_widget)

        self._cursors = {
            CursorStyle.CROSS: Qt.CursorShape.CrossCursor,
            CursorStyle.FORBIDDEN: Qt.CursorShape.ForbiddenCursor,
            CursorStyle.POINTING: Qt.CursorShape.PointingHandCursor,
            CursorStyle.STANDARD: Qt.CursorShape.ArrowCursor,
        }

        self.viewer.cursor.style_.connect(self._on_cursor)
        self.viewer.cursor.size_.connect(self._on_cursor)
        self.viewer.camera.zoom_.connect(self._on_cursor)
        self.viewer.layers_reordered_.connect(self._reorder_layers)
        self.viewer.layer_added_.connect(self._add_layer)
        self.viewer.layer_removed_.connect(self._remove_layer)

        self.view = self.canvas.central_widget.add_view(border_width=0)
        self.camera = VispyCamera(self.view, self.viewer.camera, self.viewer.dims)
        self.canvas.events.draw.connect(self.camera.on_draw)

        for layer in self.viewer.layers:
            self._add_layer(layer)
        self._add_overlay(self.viewer.brush_circle)

    def _leave_canvas(self):
        self.viewer.status = {"text": ""}
        self.viewer.mouse_over_canvas = False

    def _enter_canvas(self):
        self.viewer.status = {"text": ""}
        self.viewer.mouse_over_canvas = True

    def _reset_view_if_needed(self):
        if self.reset_when_resize:
            self.viewer.reset_view()

    def set_theme(self, theme_id: str) -> None:
        self.canvas._on_theme_change(theme_id)

    def _create_canvas(self) -> None:
        self.canvas = VispyCanvas(
            keys=None,
            vsync=True,
            parent=self,
            size=self.viewer.canvas_size[::-1],
            autoswap=False,
        )

        self.canvas.events.mouse_double_click.connect(self.on_mouse_double_click)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.canvas.events.draw.connect(self.on_draw)
        self.canvas.events.resize.connect(self.on_resize)
        self.canvas.events.key_press.connect(self._on_canvas_key_press)
        self.canvas.events.key_release.connect(self._on_canvas_key_release)
        self.canvas.bgcolor = transform_color(color_as_hex(get_theme().canvas))[0]

    def _on_canvas_key_press(self, event) -> None:
        self.parent().keyPressEvent(event._native)

    def _on_canvas_key_release(self, event) -> None:
        self.parent().keyReleaseEvent(event._native)

    def _add_overlay(self, overlay: Overlay) -> None:
        vispy_overlay = create_vispy_overlay(overlay, viewer=self.viewer)

        if isinstance(overlay, BrushCircleOverlay):
            vispy_overlay.node.parent = self.view

        self.overlay_to_visual[overlay] = vispy_overlay

    def _add_layer(self, layer: Layer):
        layer._slice_dims(
            self.dims.dims.point, self.dims.dims.ndisplay, self.dims.dims.order
        )
        layer.set_view_slice()
        vispy_layer = create_vispy_layer(layer)

        vispy_layer.node.parent = self.view.scene
        self.layer_to_visual[layer] = vispy_layer

        layer.visible_.connect(self._reorder_layers)

        self._reorder_layers()

    def _remove_layer(self, layer: Layer):
        layer.visible_.disconnect(self._reorder_layers)
        vispy_layer = self.layer_to_visual[layer]
        vispy_layer.close()
        del vispy_layer
        del self.layer_to_visual[layer]
        self._reorder_layers()

    def _reorder_layers(self):
        first_visible_found = False
        for i, layer in enumerate(self.viewer.layers):
            vispy_layer = self.layer_to_visual[layer]
            vispy_layer.order = i

            # the bottommost visible layer needs special treatment for blending
            if layer.visible and not first_visible_found:
                vispy_layer.first_visible = True
                first_visible_found = True
            else:
                vispy_layer.first_visible = False
            vispy_layer._on_blending_change()

        self.canvas._draw_order.clear()
        self.canvas.update()

    def _on_cursor(self):
        cursor = self.viewer.cursor.style
        if cursor == CursorStyle.SQUARE or cursor == CursorStyle.CIRCLE:
            # Scale size by zoom if needed
            if self.viewer.cursor.scaled:
                scaled_size = self.viewer.cursor.size * self.viewer.camera.zoom
            else:
                scaled_size = self.viewer.cursor.size

            if isinstance(scaled_size, np.ndarray):
                min_size = np.min(scaled_size)
                max_size = np.max(scaled_size)
            else:
                min_size = scaled_size
                max_size = scaled_size

            # make sure the square fits within the current canvas
            if min_size < 8 or max_size > (min(*self.canvas.size) - 4):
                q_cursor = self._cursors[CursorStyle.CROSS]
            elif cursor == CursorStyle.CIRCLE:
                if isinstance(scaled_size, np.ndarray):
                    pixmap = ellipse_pixmap(scaled_size[0], scaled_size[1])
                else:
                    pixmap = ellipse_pixmap(scaled_size)
                q_cursor = QCursor(pixmap)
            else:
                q_cursor = QCursor(square_pixmap(scaled_size))
        elif cursor == CursorStyle.CROSSHAIR:
            q_cursor = QCursor(crosshair_pixmap())
        else:
            q_cursor = self._cursors[cursor]

        self.canvas.native.setCursor(q_cursor)

    def _map_canvas2world(self, position):
        nd = self.viewer.dims.ndisplay
        transform = self.view.scene.transform
        mapped_position = transform.imap(list(position))[:nd]
        position_world_slice = mapped_position[::-1]

        # handle position for 3D views of 2D data
        nd_point = len(self.viewer.dims.point)
        if nd_point < nd:
            position_world_slice = position_world_slice[-nd_point:]

        position_world = list(self.viewer.dims.point)
        for i, d in enumerate(self.viewer.dims.displayed):
            position_world[d] = position_world_slice[i]

        return tuple(position_world)

    def on_resize(self, event):
        self.viewer.canvas_size = tuple(self.canvas.size[::-1])

    def _process_mouse_event(self, event) -> Any:
        if event.pos is None:
            return None

        # Add the view ray to the event
        event.view_direction = self.viewer.camera.calculate_nd_view_direction(
            self.viewer.dims.displayed
        )
        event.up_direction = self.viewer.camera.calculate_nd_up_direction(
            self.viewer.dims.displayed
        )

        # Update the cursor position
        self.viewer.cursor._view_direction = event.view_direction
        self.viewer.cursor.position = self._map_canvas2world(list(event.pos))

        # Add the cursor position to the event
        event.position = self.viewer.cursor.position

        # Add the displayed dimensions to the event
        event.dims_displayed = list(self.viewer.dims.displayed)

        # Add the current dims indices
        event.dims_point = list(self.viewer.dims.point)
        return event

    def on_mouse_wheel(self, event):
        event = self._process_mouse_event(event)
        event = ReadOnlyWrapper(event, exceptions=("handled",))
        self.viewer.on_mouse_wheel_scrolled(event)

        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_wheel_scrolled(event)

    def on_mouse_double_click(self, event):
        event = self._process_mouse_event(event)
        event = ReadOnlyWrapper(event, exceptions=("handled",))
        self.viewer.on_mouse_double_clicked(event)

        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_double_clicked(event)

    def on_mouse_press(self, event):
        event = self._process_mouse_event(event)
        event = ReadOnlyWrapper(event, exceptions=("handled",))
        self.viewer.on_mouse_pressed(event)

        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_pressed(event)

    def on_mouse_move(self, event):
        event = self._process_mouse_event(event)
        event = ReadOnlyWrapper(event, exceptions=("handled",))
        self.viewer.on_mouse_moved(event)

        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_moved(event)

    def on_mouse_release(self, event):
        event = self._process_mouse_event(event)
        event = ReadOnlyWrapper(event, exceptions=("handled",))
        self.viewer.on_mouse_released(event)

        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_released(event)

    def on_draw(self, event):
        for layer in self.viewer.layers:
            layer._update_draw(scale_factor=1 / self.viewer.camera.zoom)

    def keyPressEvent(self, event):
        self.canvas._backend._keyEvent(self.canvas.events.key_press, event)

    def keyReleaseEvent(self, event):
        self.canvas._backend._keyEvent(self.canvas.events.key_release, event)

    def closeEvent(self, event):
        self.canvas.native.deleteLater()
        event.accept()
