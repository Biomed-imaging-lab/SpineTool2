import numpy as np

from utils.colormaps.standardize_color import transform_color
from viewer.layers.points.points import Points
from viewer.vispy.layers.base import VispyBaseLayer
from viewer.vispy.utils.gl import BLENDING_MODES
from viewer.vispy.visuals.points import PointsVisual


class VispyPointsLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    layer: "Points"

    def __init__(self, layer) -> None:
        node = PointsVisual()
        super().__init__(layer, node)

        self.layer.symbol_.connect(self._on_data_change)
        self.layer.edge_width_.connect(self._on_data_change)
        self.layer.edge_width_is_relative_.connect(self._on_data_change)
        self.layer.edge_color_.connect(self._on_data_change)
        self.layer.face_color_.connect(self._on_data_change)
        self.layer.highlight_.connect(self._on_highlight_change)
        self.layer.shading_.connect(self._on_shading_change)
        self.layer.antialiasing_.connect(self._on_antialiasing_change)
        self.layer.canvas_size_limits_.connect(self._on_canvas_size_limits_change)

        self._on_data_change()

    def _on_data_change(self):
        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switched for vispy's x / y ordering
        if len(self.layer._indices_view) == 0:
            # always pass one invisible point to avoid issues
            data = np.zeros((1, self.layer._slice_input.ndisplay))
            size = np.zeros(1)
            edge_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            face_color = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
            edge_width = np.zeros(1)
            symbol = ["o"]
        else:
            data = self.layer._view_data
            size = self.layer._view_size
            edge_color = self.layer._view_edge_color
            face_color = self.layer._view_face_color
            edge_width = self.layer._view_edge_width
            symbol = self.layer._view_symbol

        # use only last dimension to scale point sizes, see #5582
        scale = self.layer.scale[-1]

        if self.layer.edge_width_is_relative:
            edge_kw = {
                "edge_width": None,
                "edge_width_rel": edge_width,
            }
        else:
            edge_kw = {
                "edge_width": edge_width * scale,
                "edge_width_rel": None,
            }

        self.node._subvisuals[0].set_data(
            data[:, ::-1],
            size=size * scale,
            symbol=symbol,
            edge_color=edge_color,
            face_color=face_color,
            **edge_kw,
        )

        self.reset()

    def _on_highlight_change(self):
        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._view_data[self.layer._highlight_index]
            size = self.layer._view_size[self.layer._highlight_index]
            edge_width = self.layer._view_edge_width[self.layer._highlight_index]
            if self.layer.edge_width_is_relative:
                edge_width = (
                    edge_width * self.layer._view_size[self.layer._highlight_index][-1]
                )
            symbol = self.layer._view_symbol[self.layer._highlight_index]
        else:
            data = np.zeros((1, self.layer._slice_input.ndisplay))
            size = 0
            symbol = ["o"]
            edge_width = np.array([0])

        scale = self.layer.scale[-1]
        scaled_highlight = self.layer.highlight_thickness * self.layer.scale_factor

        self.node._subvisuals[1].set_data(
            data[:, ::-1],
            size=(size + edge_width) * scale,
            symbol=symbol,
            edge_width=scaled_highlight * 2,
            edge_color=self._highlight_color,
            face_color=transform_color("transparent"),
        )

        self.node.update()

    def _on_blending_change(self):
        points_blending_kwargs = BLENDING_MODES[self.layer.blending]
        self.node.set_gl_state(**points_blending_kwargs)

        self.node.update()

    def _on_antialiasing_change(self):
        self.node.antialias = self.layer.antialiasing

    def _on_shading_change(self):
        shading = self.layer.shading
        if shading == "spherical":
            self.node.spherical = True
        else:
            self.node.spherical = False

    def _on_canvas_size_limits_change(self):
        self.node.canvas_size_limits = self.layer.canvas_size_limits

    def reset(self):
        super().reset()
        self._on_highlight_change()
        self._on_antialiasing_change()
        self._on_shading_change()
        self._on_canvas_size_limits_change()
