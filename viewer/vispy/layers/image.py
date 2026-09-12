from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.scene.node import Node
from vispy.visuals import ImageVisual

from viewer.layers.base._base_constants import Blending
from viewer.layers.image.image import Image, _ImageBase
from viewer.utils.transforms import Scale
from viewer.vispy.layers.base import VispyBaseLayer
from viewer.vispy.utils.gl import fix_data_dtype, get_gl_extensions
from viewer.vispy.visuals.image import Image as ImageNode
from viewer.vispy.visuals.volume import Volume as VolumeNode


class ImageLayerNode:
    def __init__(self, custom_node: Node = None, texture_format=None) -> None:
        if texture_format == "auto" and "texture_float" not in get_gl_extensions():
            texture_format = None

        self._custom_node = custom_node
        self._image_node = ImageNode(
            (
                None
                if (texture_format is None or texture_format == "auto")
                else np.array([[0.0]], dtype=np.float32)
            ),
            method="auto",
            texture_format=texture_format,
        )
        self._image_node.method
        self._volume_node = VolumeNode(
            np.zeros((1, 1, 1), dtype=np.float32),
            clim=[0, 1],
            texture_format=texture_format,
            method="iso",
        )

    def get_node(self, ndisplay: int, dtype: Optional[np.dtype] = None) -> Node:
        # Return custom node if we have one.
        if self._custom_node is not None:
            return self._custom_node

        # Return Image or Volume node based on 2D or 3D.
        res = self._image_node if ndisplay == 2 else self._volume_node
        if (
            res.texture_format not in {"auto", None}
            and dtype is not None
            and _VISPY_FORMAT_TO_DTYPE[res.texture_format] != dtype
        ):
            # it is a bug to hit this error — it is here to catch bugs
            # early when we are creating the wrong nodes or
            # textures for our data
            raise ValueError(
                "dtype {dtype} does not match texture_format={texture_format}".format(
                    dtype=dtype,
                    texture_format=res.texture_format,
                )
            )
        return res


class VispyScalarFieldBaseLayer(VispyBaseLayer):
    def __init__(
        self,
        layer: _ImageBase,
        node=None,
        texture_format="auto",
        layer_node_class=ImageLayerNode,
    ) -> None:
        # Use custom node from caller, or our standard image/volume nodes.
        self._layer_node = layer_node_class(node, texture_format=texture_format)

        # Default to 2D (image) node.
        super().__init__(layer, self._layer_node.get_node(2))

        self._array_like = True

        self.layer.colormap_.connect(self._on_colormap_change)

        # display_change is special (like data_change) because it requires a
        # self.reset(). This means that we have to call it manually. Also,
        # it must be called before reset in order to set the appropriate node
        # first
        self._on_display_change()
        self.reset()
        self._on_data_change()

    def _on_display_change(self, data=None):
        parent = self.node.parent
        self.node.parent = None
        ndisplay = self.layer._slice_input.ndisplay
        self.node = self._layer_node.get_node(ndisplay, getattr(data, "dtype", None))

        if data is None:
            texture_format = self.node.texture_format
            data = np.zeros(
                (1,) * ndisplay,
                dtype=get_dtype_from_vispy_texture_format(texture_format),
            )

        if self.layer._empty:
            self.node.visible = False
        else:
            self.node.visible = self.layer.visible

        self.node.set_data(data)

        self.node.parent = parent
        self.node.order = self.order
        for overlay_visual in self.overlays.values():
            overlay_visual.node.parent = self.node
        self.reset()

    def _on_data_change(self):
        self._set_node_data(self.node, self.layer._data_view)

    def _set_node_data(self, node, data):
        data = fix_data_dtype(data)
        ndisplay = self.layer._slice_input.ndisplay

        node = self._layer_node.get_node(ndisplay, getattr(data, "dtype", None))

        # Check if data exceeds MAX_TEXTURE_SIZE and downsample
        if self.MAX_TEXTURE_SIZE_2D is not None and ndisplay == 2:
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_2D)
        elif self.MAX_TEXTURE_SIZE_3D is not None and ndisplay == 3:
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_3D)

        # Check if ndisplay has changed current node type needs updating
        if (ndisplay == 3 and not isinstance(node, VolumeNode)) or (
            ndisplay == 2 and not isinstance(node, ImageVisual) or node != self.node
        ):
            self._on_display_change(data)
        else:
            node.set_data(data)

        if self.layer._empty:
            node.visible = False
        else:
            node.visible = self.layer.visible

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        node.update()

    def _on_blending_change(self):
        super()._on_blending_change()

    def reset(self):
        super().reset()
        if isinstance(self.node, VolumeNode):
            self.node.raycasting_mode = "volume"

    def downsample_texture(self, data, MAX_TEXTURE_SIZE):
        if np.any(np.greater(data.shape, MAX_TEXTURE_SIZE)):
            warnings.warn(
                "data shape {shape} exceeds GL_MAX_TEXTURE_SIZE {texture_size}"
                " in at least one axis and will be downsampled."
                " Rendering is currently in {ndisplay}D mode.".format(
                    shape=data.shape,
                    texture_size=MAX_TEXTURE_SIZE,
                    ndisplay=self.layer._slice_input.ndisplay,
                )
            )
            downsample = np.ceil(np.divide(data.shape, MAX_TEXTURE_SIZE)).astype(int)
            scale = np.ones(3)
            for i, d in enumerate(self.layer._slice_input.displayed):
                scale[d] = downsample[i]
            self.layer._scale = Scale(scale)
            self._on_matrix_change()
            slices = tuple(slice(None, None, ds) for ds in downsample)
            data = data[slices]
        return data


class VispyImageLayer(VispyScalarFieldBaseLayer):
    layer: "Image"

    def __init__(
        self,
        layer: Image,
        node=None,
        texture_format="auto",
        layer_node_class=ImageLayerNode,
    ) -> None:
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=layer_node_class,
        )

        self.layer.interpolation2d_.connect(self._on_interpolation_change)
        self.layer.interpolation3d_.connect(self._on_interpolation_change)
        self.layer.contrast_limits_.connect(self._on_contrast_limits_change)
        self.layer.gamma_.connect(self._on_gamma_change)
        self.layer.iso_threshold_.connect(self._on_iso_threshold_change)

        # display_change is special (like data_change) because it requires a
        # self.reset(). This means that we have to call it manually. Also,
        # it must be called before reset in order to set the appropriate node
        # first
        self._on_display_change()
        self.reset()
        self._on_data_change()

    def _on_interpolation_change(self) -> None:
        self.node.interpolation = (
            self.layer.interpolation2d
            if self.layer._slice_input.ndisplay == 2
            else self.layer.interpolation3d
        )

    def _on_colormap_change(self, event=None) -> None:
        self.node.cmap = VispyColormap(*self.layer.colormap)

    def _update_mip_minip_cutoff(self) -> None:
        # discard fragments beyond contrast limits, but only with translucent blending
        if isinstance(self.node, VolumeNode):
            if self.layer.blending in {
                Blending.TRANSLUCENT,
                Blending.TRANSLUCENT_NO_DEPTH,
            }:
                self.node.mip_cutoff = self.node._texture.clim_normalized[0]
                self.node.minip_cutoff = self.node._texture.clim_normalized[1]
            else:
                self.node.mip_cutoff = None
                self.node.minip_cutoff = None

    def _on_contrast_limits_change(self) -> None:
        self.node.clim = self.layer.contrast_limits
        # cutoffs must be updated after clims, so we can set them to the new values
        self._update_mip_minip_cutoff()
        # iso also may depend on contrast limit values
        self._on_iso_threshold_change()

    def _on_blending_change(self) -> None:
        super()._on_blending_change()
        # cutoffs must be updated after blending, so we can know if
        # the new blending is a translucent one
        self._update_mip_minip_cutoff()

    def _on_gamma_change(self) -> None:
        self.node.gamma = self.layer.gamma

    def _on_iso_threshold_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            if self.node._texture.is_normalized:
                cmin, cmax = self.layer.contrast_limits_range
                self.node.threshold = (self.layer.iso_threshold - cmin) / (cmax - cmin)
            else:
                self.node.threshold = self.layer.iso_threshold

    def reset(self) -> None:
        super().reset()
        self._on_interpolation_change()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_gamma_change()


_VISPY_FORMAT_TO_DTYPE: Dict[Optional[str], np.dtype] = {
    "r8": np.dtype(np.uint8),
    "r16": np.dtype(np.uint16),
    "r32f": np.dtype(np.float32),
}

_DTYPE_TO_VISPY_FORMAT = {v: k for k, v in _VISPY_FORMAT_TO_DTYPE.items()}

_VISPY_FORMAT_TO_DTYPE[None] = np.dtype(np.float32)


def get_dtype_from_vispy_texture_format(format_str: str) -> np.dtype:
    return _VISPY_FORMAT_TO_DTYPE.get(format_str, np.dtype(np.float32))
