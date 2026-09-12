from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from viewer.layers.base.base import Layer


class ImageSliceData:
    def __init__(
        self,
        layer: Layer,
        indices: Tuple[Optional[slice], ...],
        image: np.ndarray,
    ) -> None:
        self.layer = layer
        self.indices = indices
        self.image = np.asarray(image)

    def transpose(self, order: tuple) -> None:
        self.image = np.transpose(self.image, order)
