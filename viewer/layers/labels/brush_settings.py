from typing import List

from viewer.layers.labels.labels_constants import EMPTY_BRUSH_SETTINGS
from viewer.layers.labels.labels_constants import BrushSettingsKeys as BSK
from viewer.layers.labels.labels_constants import PaintMode


class BrushSettings:
    def __init__(self, settings=None):
        if not settings:
            settings = EMPTY_BRUSH_SETTINGS.copy()
        self.current_mode: PaintMode = PaintMode(settings[BSK.CURRENT_MODE.value])
        self.radii: List[int | float] = settings[BSK.RADII.value]
        self.all_same_radii: bool = settings[BSK.ALL_SAME_RADII.value]
        self.elliptical_cylinder_limited_height_depth: bool = settings[
            BSK.ELLIPTICAL_CYLINDER_LIMITED_HEIGHT_DEPTH.value
        ]
        self.elliptical_cylinder_height_depth: int | float = settings[
            BSK.ELLIPTICAL_CYLINDER_HEIGHT_DEPTH.value
        ]
        if self.all_same_radii and not (
            self.radii[0] == self.radii[1] and self.radii[0] == self.radii[2]
        ):
            self.all_same_radii = False

    def as_dict(self) -> dict:
        settings = {
            BSK.CURRENT_MODE.value: self.current_mode.value,
            BSK.RADII.value: self.radii,
            BSK.ALL_SAME_RADII.value: self.all_same_radii,
            BSK.ELLIPTICAL_CYLINDER_LIMITED_HEIGHT_DEPTH.value: self.elliptical_cylinder_limited_height_depth,
            BSK.ELLIPTICAL_CYLINDER_HEIGHT_DEPTH.value: self.elliptical_cylinder_height_depth,
        }
        return settings
