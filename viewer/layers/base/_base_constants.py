from enum import auto

from utils.misc import StringEnum


class Blending(StringEnum):
    """BLENDING: Blending mode for the layer.

    Selects a preset blending mode in vispy that determines how
            RGB and alpha values get mixed.
            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
                and blend_equation=('func_add').
            Blending.TRANSLUCENT_NO_DEPTH
                Allows for multiple layers to be blended with different opacity,
                but no depth testing is performed.
                and corresponds to depth_test=False, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
                and blend_equation=('func_add').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
    """

    TRANSLUCENT = auto()
    TRANSLUCENT_NO_DEPTH = auto()
    ADDITIVE = auto()
    OPAQUE = auto()


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.
    """

    PAN_ZOOM = auto()
