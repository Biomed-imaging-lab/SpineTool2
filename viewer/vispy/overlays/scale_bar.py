import numpy as np
from vispy.scene.visuals import Compound, Line, Text

from viewer.components.overlays.scale_bar import (
    format_scale_bar_length,
    scale_bar_length,
)
from viewer.vispy.overlays.base import VispyCanvasOverlay


MARGIN = 16
LABEL_GAP = 6


class VispyScaleBarOverlay(VispyCanvasOverlay):
    """White scale bar anchored to the bottom-left corner of the canvas."""

    def __init__(self, *, viewer, overlay, parent=None):
        self.viewer = viewer
        self._line = Line(
            pos=np.zeros((2, 2)),
            color="white",
            width=3,
            method="gl",
        )
        self._text = Text(
            "",
            color="white",
            font_size=10,
            anchor_x="center",
            anchor_y="center",
        )
        super().__init__(
            node=Compound([self._line, self._text]),
            overlay=overlay,
            parent=parent,
        )
        self.viewer.camera.zoom_.connect(self._on_position_change)
        self.viewer.scale_changed_.connect(self._on_position_change)
        self.viewer.dims.order_.connect(self._on_position_change)
        self.viewer.dims.ndisplay_.connect(self._on_position_change)
        self.reset()

    def _on_parent_change(self, event):
        super()._on_parent_change(event)
        if event.new is not None:
            self._on_position_change()

    def _on_position_change(self, event=None):
        canvas = self.node.canvas
        if canvas is None:
            return

        microns_per_world_unit = self.viewer.microns_per_world_unit
        length_um = scale_bar_length(
            float(self.viewer.camera.zoom),
            microns_per_world_unit,
        )
        if length_um is None:
            self.node.visible = False
            return

        length_px = (
            length_um
            / microns_per_world_unit
            * float(self.viewer.camera.zoom)
        )
        canvas_height = float(canvas.size[1])
        self.node.visible = self.overlay.visible
        self.node.transform.translate = [MARGIN, canvas_height - MARGIN, 0, 0]
        self._line.set_data(pos=np.array([[0, 0], [length_px, 0]]))
        self._text.text = format_scale_bar_length(length_um)
        font_height = self._text.font_size * canvas.dpi / 72.0
        self._text.pos = (
            length_px / 2,
            -(font_height / 2 + LABEL_GAP + self._line.width / 2),
        )

    def close(self):
        self.viewer.camera.zoom_.disconnect(self._on_position_change)
        self.viewer.scale_changed_.disconnect(self._on_position_change)
        self.viewer.dims.order_.disconnect(self._on_position_change)
        self.viewer.dims.ndisplay_.disconnect(self._on_position_change)
        super().close()
