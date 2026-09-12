from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from viewer.utils.transforms import Scale


@dataclass(frozen=True)
class SliceInput:
    ndisplay: int
    point: Tuple[float, ...]
    order: Tuple[int, ...]

    @property
    def displayed(self) -> List[int]:
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self) -> List[int]:
        return list(self.order[: -self.ndisplay])

    def data_indices(
        self, world_to_data: Scale, round_index: bool = True
    ) -> Tuple[Union[int, float, slice]]:
        slice_world_to_data = world_to_data.set_slice(self.not_displayed)
        world_pts = [self.point[ax] for ax in self.not_displayed]
        data_pts = slice_world_to_data(world_pts)
        if round_index:
            # A round is taken to convert these values to slicing integers
            data_pts = np.round(data_pts).astype(int)

        indices = [slice(None)] * 3
        for i, ax in enumerate(self.not_displayed):
            indices[ax] = data_pts[i]

        return tuple(indices)
