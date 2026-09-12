from typing import List, Optional, Protocol

from vispy.visuals.filters import Filter
from vispy.visuals.filters.clipping_planes import PlanesClipper


class _PVisual(Protocol):
    _subvisuals: Optional[List["_PVisual"]]

    def attach(self, filt: Filter, view=None): ...


class ClippingPlanesMixin:
    def __init__(self: _PVisual, *args, **kwargs) -> None:
        self._clip_filter = PlanesClipper()
        super().__init__(*args, **kwargs)

        self.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
