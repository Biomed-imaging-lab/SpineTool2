import math
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from vispy.color import Colormap as VispyColormap
from vispy.gloo import Texture2D
from vispy.scene.node import Node

from utils.colormaps.colormap import CyclicLabelColormap
from viewer.vispy.layers.image import (
    _DTYPE_TO_VISPY_FORMAT,
    _VISPY_FORMAT_TO_DTYPE,
    ImageLayerNode,
    VispyScalarFieldBaseLayer,
    get_dtype_from_vispy_texture_format,
)
from viewer.vispy.visuals.labels import LabelNode
from viewer.vispy.visuals.volume import Volume as VolumeNode

if TYPE_CHECKING:
    from viewer.layers.labels.labels import Labels


ColorTuple = Tuple[float, float, float, float]


auto_lookup_shader_uint8 = """
uniform sampler2D texture2D_values;

vec4 sample_label_color(float t) {
    if (($use_selection) && ($selection != int(t * 255))) {
        return vec4(0);
    }
    return texture2D(
        texture2D_values,
        vec2(0.0, t)
    );
}
"""

auto_lookup_shader_uint16 = """
uniform sampler2D texture2D_values;

vec4 sample_label_color(float t) {
    // uint 16
    t = t * 65535;
    if (($use_selection) && ($selection != int(t))) {
        return vec4(0);
    }
    float v = mod(t, 256);
    float v2 = (t - v) / 256;
    return texture2D(
        texture2D_values,
        vec2((v + 0.5) / 256, (v2 + 0.5) / 256)
    );
}
"""


class LabelVispyColormap(VispyColormap):
    def __init__(
        self,
        colormap: CyclicLabelColormap,
        view_dtype: np.dtype,
        raw_dtype: np.dtype,
    ):
        super().__init__(colors=["w", "w"], controls=None, interpolation="zero")
        if view_dtype.itemsize == 1:
            shader = auto_lookup_shader_uint8
        elif view_dtype.itemsize == 2:
            shader = auto_lookup_shader_uint16
        else:
            raise ValueError(  # pragma: no cover
                f"Cannot use dtype {view_dtype} with LabelVispyColormap"
            )

        selection = colormap._selection_as_minimum_dtype(raw_dtype)

        self.glsl_map = shader.replace(
            "$use_selection", str(colormap.use_selection).lower()
        ).replace("$selection", str(selection))


def build_textures_from_dict(
    color_dict: Dict[int, ColorTuple], max_size: int
) -> np.ndarray:
    if len(color_dict) > 2**23:
        raise ValueError(  # pragma: no cover
            "Cannot map more than 2**23 colors because of float32 precision. "
            f"Got {len(color_dict)}"
        )
    if len(color_dict) > max_size**2:
        raise ValueError(
            "Cannot create a 2D texture holding more than "
            f"{max_size}**2={max_size ** 2} colors."
            f"Got {len(color_dict)}"
        )
    data = np.zeros(
        (
            min(len(color_dict), max_size),
            math.ceil(len(color_dict) / max_size),
            4,
        ),
        dtype=np.float32,
    )
    for key, value in color_dict.items():
        data[key % data.shape[0], key // data.shape[0]] = value
    return data


def _select_colormap_texture(
    colormap: CyclicLabelColormap, view_dtype, raw_dtype
) -> np.ndarray:
    if raw_dtype.itemsize > 2:
        color_texture = colormap._get_mapping_from_cache(view_dtype)
    else:
        color_texture = colormap._get_mapping_from_cache(raw_dtype)

    if color_texture is None:
        raise ValueError(  # pragma: no cover
            f"Cannot build a texture for dtype {raw_dtype=} and {view_dtype=}"
        )
    return color_texture.reshape(256, -1, 4)


class VispyLabelsLayer(VispyScalarFieldBaseLayer):
    layer: "Labels"

    def __init__(self, layer, node=None, texture_format="r8") -> None:
        super().__init__(
            layer,
            node=node,
            texture_format=texture_format,
            layer_node_class=LabelLayerNode,
        )

        self.layer.selected_label_.connect(self._on_colormap_change)
        self.layer.show_selected_label_.connect(self._on_colormap_change)
        self.layer.data_.connect(self._on_colormap_change)

    def _on_colormap_change(self, event=None):
        if (
            event is not None
            and isinstance(event, str)
            and event == "selected_label"
            and not self.layer.show_selected_label
        ):
            return
        colormap = self.layer.colormap
        view_dtype = self.layer._slice.image.view.dtype
        raw_dtype = self.layer._slice.image.raw.dtype
        color_texture = _select_colormap_texture(colormap, view_dtype, raw_dtype)
        self.node.cmap = LabelVispyColormap(
            colormap, view_dtype=view_dtype, raw_dtype=raw_dtype
        )
        self.node.shared_program["texture2D_values"] = Texture2D(
            color_texture,
            internalformat="rgba32f",
            interpolation="nearest",
        )
        self.texture_data = color_texture

    def reset(self) -> None:
        super().reset()
        self._on_colormap_change()


class LabelLayerNode(ImageLayerNode):
    def __init__(self, custom_node: Node = None, texture_format=None):
        self._custom_node = custom_node
        self._setup_nodes(texture_format)

    def _setup_nodes(self, texture_format):
        self._image_node = LabelNode(
            (
                None
                if (texture_format is None or texture_format == "auto")
                else np.zeros(
                    (1, 1),
                    dtype=get_dtype_from_vispy_texture_format(texture_format),
                )
            ),
            method="auto",
            texture_format=texture_format,
        )

        self._volume_node = VolumeNode(
            np.zeros(
                (1, 1, 1),
                dtype=get_dtype_from_vispy_texture_format(texture_format),
            ),
            clim=[0, 2**23 - 1],
            texture_format=texture_format,
            interpolation="nearest",
            method="iso_categorical",
        )

    def get_node(self, ndisplay: int, dtype=None) -> Node:
        res = self._image_node if ndisplay == 2 else self._volume_node

        if (
            res.texture_format != "auto"
            and dtype is not None
            and _VISPY_FORMAT_TO_DTYPE[res.texture_format] != dtype
        ):
            self._setup_nodes(_DTYPE_TO_VISPY_FORMAT[dtype])
            return self.get_node(ndisplay, dtype)
        return res
