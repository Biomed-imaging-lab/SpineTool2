from __future__ import annotations

from typing import Callable

import numpy as np

from viewer.layers.image._image_slice_data import ImageSliceData
from viewer.layers.image._image_view import ImageView


class ImageSlice:
    def __init__(
        self,
        image: np.ndarray,
        image_converter: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.image: ImageView = ImageView(image, image_converter)

    def load(self, data: ImageSliceData) -> None:
        self.image.raw = data.image
