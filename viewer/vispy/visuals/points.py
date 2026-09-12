from vispy.scene.visuals import Compound

from viewer.vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin
from viewer.vispy.visuals.markers import Markers


class PointsVisual(ClippingPlanesMixin, Compound):
    def __init__(self) -> None:
        super().__init__(
            [
                Markers(scaling="visual"),
                Markers(scaling="visual"),
            ]
        )
        self.scaling = True

    @property
    def scaling(self):
        return self._subvisuals[0].scaling == "visual"

    @scaling.setter
    def scaling(self, value):
        for marker in self._subvisuals:
            marker.scaling = "visual" if value else "fixed"

    @property
    def antialias(self):
        return self._subvisuals[0].antialias

    @antialias.setter
    def antialias(self, value):
        for marker in self._subvisuals:
            marker.antialias = value

    @property
    def spherical(self):
        return self._subvisuals[0].spherical

    @spherical.setter
    def spherical(self, value):
        self._subvisuals[0].spherical = value

    @property
    def canvas_size_limits(self):
        return self._subvisuals[0].canvas_size_limits

    @canvas_size_limits.setter
    def canvas_size_limits(self, value):
        self._subvisuals[0].canvas_size_limits = value
