from typing import Sequence

import numpy as np


class Scale:
    def __init__(self, scale=(1.0,)) -> None:
        self.scale = np.array(scale)

    def __call__(self, coords):
        coords = np.asarray(coords)
        append_first_axis = coords.ndim == 1
        if append_first_axis:
            coords = coords[np.newaxis, :]

        coords_ndim = coords.shape[1]
        if coords_ndim == len(self.scale):
            scale = self.scale
        else:
            scale = np.concatenate(
                ([1.0] * (coords_ndim - len(self.scale)), self.scale)
            )
        out = scale * coords
        if append_first_axis:
            out = out[0]
        return out

    @property
    def inverse(self) -> "Scale":
        return Scale(1 / self.scale)

    def set_slice(self, axes: Sequence[int]) -> "Scale":
        return Scale(self.scale[axes])

    def expand_dims(self, axes: Sequence[int]) -> "Scale":
        n = len(axes) + len(self.scale)
        not_axes = [i for i in range(n) if i not in axes]
        scale = np.ones(n)
        scale[not_axes] = self.scale
        return Scale(scale)
