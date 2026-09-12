from vispy.scene.visuals import Image as BaseImage

from viewer.vispy.visuals.util import TextureMixin


class Image(TextureMixin, BaseImage):
    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        if axis > 1:
            return (0, 0)

        return (0, self.size[axis])
