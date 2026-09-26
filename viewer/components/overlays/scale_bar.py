import math

from PyQt5.QtCore import pyqtSignal

from viewer.components.overlays.base import Overlay


TARGET_PIXELS = 100
MIN_LENGTH_UM = 0.1


def scale_bar_length(zoom: float, microns_per_world_unit: float = 1.0):
    """Return a readable scale length in micrometers for the current zoom."""
    if (
        not math.isfinite(zoom)
        or zoom <= 0
        or not math.isfinite(microns_per_world_unit)
        or microns_per_world_unit <= 0
    ):
        return None

    target_um = TARGET_PIXELS / zoom * microns_per_world_unit
    if target_um < MIN_LENGTH_UM:
        return None
    if target_um < 1:
        return round(target_um * 10) / 10

    exponent = math.floor(math.log10(target_um))
    magnitude = 10**exponent
    multiplier = min(
        (1, 2, 5, 10),
        key=lambda value: abs(value * magnitude - target_um),
    )
    return int(multiplier * magnitude)


def format_scale_bar_length(length_um: float) -> str:
    if length_um < 1:
        return f"{length_um:.1f} µm"
    return f"{int(length_um)} µm"


class ScaleBarOverlay(Overlay):
    """Always-visible physical scale reference for the image canvas."""

    position_ = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._visible = True
