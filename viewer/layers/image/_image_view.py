from __future__ import annotations

from typing import Callable

import numpy as np


class ImageView:
    def __init__(
        self,
        view_image: np.ndarray,
        image_converter: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.view = view_image
        self.image_converter = image_converter

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, view_image: np.ndarray):
        self._view = view_image
        self._raw = view_image

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, raw_image: np.ndarray):
        self._raw = raw_image

        # Update the view image based on this new raw image.
        self._view = self.image_converter(raw_image)
