from enum import auto

from utils.misc import StringEnum


class Mode(StringEnum):
    """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    In PAINT mode the cursor functions like a paint brush changing any pixels
    it brushes over to the current label. If the background label `0` is
    selected than any pixels will be changed to background and this tool
    functions like an eraser. The size and shape of the cursor can be adjusted
    in the properties widget.

    In FILL mode the cursor functions like a fill bucket replacing pixels
    of the label clicked on with the current label. It can either replace all
    pixels of that label or just those that are contiguous with the clicked on
    pixel. If the background label `0` is selected than any pixels will be
    changed to background and this tool functions like an eraser.

    In ERASE mode the cursor functions similarly to PAINT mode, but to paint
    with background label, which effectively removes the label.

    In CONNECTED_COMPONENTS mode user can only change the transparency of the image.
    The data is colored based on its connectivity components.
    """

    PAN_ZOOM = auto()
    PAINT = auto()
    FILL = auto()
    ERASE = auto()
    CONNECTED_COMPONENTS = auto()


class PaintMode(StringEnum):
    ELLIPSE = auto()
    ELLIPSOID = auto()
    ELLIPTICAL_CYLINDER = auto()


class BrushSettingsKeys(StringEnum):
    CURRENT_MODE = auto()
    RADII = auto()
    ALL_SAME_RADII = auto()
    ELLIPTICAL_CYLINDER_LIMITED_HEIGHT_DEPTH = (
        auto()
    )  # если не установлен, то кисть рисует по всем слоям сразу
    ELLIPTICAL_CYLINDER_HEIGHT_DEPTH = (
        auto()
    )  # сколько соседних слоев будет изменяться при установленном ограничении


EMPTY_BRUSH_SETTINGS = {
    BrushSettingsKeys.CURRENT_MODE.value: PaintMode.ELLIPSE.value,
    BrushSettingsKeys.RADII.value: [10, 10, 10],
    BrushSettingsKeys.ALL_SAME_RADII.value: True,
    BrushSettingsKeys.ELLIPTICAL_CYLINDER_LIMITED_HEIGHT_DEPTH.value: False,
    BrushSettingsKeys.ELLIPTICAL_CYLINDER_HEIGHT_DEPTH.value: 3,
}
