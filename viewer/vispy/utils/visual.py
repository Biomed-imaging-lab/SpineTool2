from __future__ import annotations

from typing import List

from viewer.components.overlays.base import Overlay
from viewer.components.overlays.brush_circle import BrushCircleOverlay
from viewer.layers.base.base import Layer
from viewer.layers.image.image import Image
from viewer.layers.labels.labels import Labels
from viewer.layers.points.points import Points
from viewer.layers.surface.surface import Surface
from viewer.vispy.layers.base import VispyBaseLayer
from viewer.vispy.layers.image import VispyImageLayer
from viewer.vispy.layers.labels import VispyLabelsLayer
from viewer.vispy.layers.points import VispyPointsLayer
from viewer.vispy.layers.surface import VispySurfaceLayer
from viewer.vispy.overlays.base import VispyBaseOverlay
from viewer.vispy.overlays.brush_circle import VispyBrushCircleOverlay

layer_to_visual = {
    Image: VispyImageLayer,
    Labels: VispyLabelsLayer,
    Points: VispyPointsLayer,
    Surface: VispySurfaceLayer,
}


overlay_to_visual = {
    BrushCircleOverlay: VispyBrushCircleOverlay,
}


def create_vispy_layer(layer: Layer) -> VispyBaseLayer:
    for layer_type, visual_class in layer_to_visual.items():
        if isinstance(layer, layer_type):
            return visual_class(layer)

    raise TypeError(
        "Could not find VispyLayer for layer of type {dtype}".format(
            dtype=type(layer),
        )
    )


def create_vispy_overlay(overlay: Overlay, **kwargs) -> List[VispyBaseOverlay]:
    for overlay_type, visual_class in overlay_to_visual.items():
        if isinstance(overlay, overlay_type):
            return visual_class(overlay=overlay, **kwargs)

    raise TypeError(
        "Could not find VispyOverlay for overlay of type {dtype}".format(
            dtype=type(overlay),
        )
    )
