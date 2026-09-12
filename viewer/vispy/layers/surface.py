import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.geometry import MeshData
from vispy.visuals.filters import TextureFilter

from viewer.layers.surface.surface import Surface
from viewer.vispy.layers.base import VispyBaseLayer
from viewer.vispy.visuals.surface import SurfaceVisual


class VispySurfaceLayer(VispyBaseLayer):
    layer: "Surface"

    def __init__(self, layer: Surface) -> None:
        node = SurfaceVisual()
        self._texture_filter = None
        self._meshdata = None
        super().__init__(layer, node)

        self.layer.colormap_.connect(self._on_colormap_change)
        self.layer.contrast_limits_.connect(self._on_contrast_limits_change)
        self.layer.gamma_.connect(self._on_gamma_change)
        self.layer.shading_.connect(self._on_shading_change)
        self.layer.texture_.connect(self._on_texture_change)
        self.layer.texcoords_.connect(self._on_texture_change)
        self.layer.data_.connect(self._on_data_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        vertices = None
        faces = None
        vertex_values = None
        vertex_colors = None
        if len(self.layer._data_view) and len(self.layer._view_faces):
            # Offsetting so pixels now centered
            # coerce to float to solve vispy/vispy#2007
            # reverse order to get zyx instead of xyz
            vertices = np.asarray(self.layer._data_view[:, ::-1], dtype=np.float32)
            # due to above xyz>zyx, also reverse order of faces to fix
            # handedness of normals
            faces = self.layer._view_faces[:, ::-1]

            values = self.layer._view_vertex_values
            if len(values):
                vertex_values = values

            colors = self.layer._view_vertex_colors
            if len(colors):
                vertex_colors = colors

        # making sure the vertex data is 3D prevents shape errors with
        # attached filters, instead of trying to attach/detach each time
        if vertices is not None and vertices.shape[-1] == 2:
            vertices = np.pad(
                vertices,
                ((0, 0), (0, 1)),
                mode="constant",
                constant_values=0,
            )
        assert vertices is None or vertices.shape[-1] == 3

        self.node.set_data(
            vertices=vertices,
            faces=faces,
            vertex_values=vertex_values,
            vertex_colors=vertex_colors,
        )

        # disable normals in 2D to avoid shape errors
        if self.layer._slice_input.ndisplay == 2:
            self._meshdata = MeshData()
        else:
            self._meshdata = self.node.mesh_data

        self._on_texture_change()
        self._on_shading_change()

        self.node.update()

        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_texture_change(self):
        if self.layer._has_texture and self._texture_filter is None:
            self._texture_filter = TextureFilter(
                np.flipud(self.layer.texture),
                self.layer.texcoords,
            )
            self.node.attach(self._texture_filter)
        elif self.layer._has_texture:
            self._texture_filter.texture = np.flipud(self.layer.texture)
            self._texture_filter.texcoords = self.layer.texcoords

        if self._texture_filter is not None:
            self._texture_filter.enabled = self.layer._has_texture
            self.node.update()

    def _on_colormap_change(self):
        if self.layer.gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            colors = self.layer.colormap.map(np.linspace(0, 1, 256) ** self.layer.gamma)
            cmap = VispyColormap(colors)
        else:
            cmap = VispyColormap(*self.layer.colormap)
        if self.layer._slice_input.ndisplay == 3:
            self.node.view_program["texture2D_LUT"] = (
                cmap.texture_lut() if (hasattr(cmap, "texture_lut")) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self):
        self.node.clim = self.layer.contrast_limits

    def _on_gamma_change(self):
        self._on_colormap_change()

    def _on_shading_change(self):
        shading = None if self.layer.shading == "none" else self.layer.shading
        if not self.node.mesh_data.is_empty():
            self.node.shading = shading
        self.node.update()

    def reset(self):
        super().reset()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_shading_change()
        self._on_texture_change()
