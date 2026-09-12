import itertools
import warnings
from collections import namedtuple
from functools import cached_property
from typing import Iterable, List, Tuple

import numpy as np

from viewer.components.layer_list._evented_list import EventedList
from viewer.layers.base.base import Layer
from viewer.layers.image.image import _ImageBase

Extent = namedtuple("Extent", "data world step")


class LayerList(EventedList[Layer]):
    def __init__(self, data=()) -> None:
        super().__init__(
            data=data,
            basetype=Layer,
            lookup={str: lambda e: e.name},
        )

    def _process_delete_item(self, item: Layer):
        super()._process_delete_item(item)
        item.extent_.disconnect(self._clean_cache)
        self._clean_cache()

    def _clean_cache(self):
        cached_properties = (
            "extent",
            "_step_size",
            "_ranges",
        )
        [self.__dict__.pop(p, None) for p in cached_properties]

    def __newlike__(self, data):
        return LayerList(data)

    def _ensure_unique(self, values, allow=()):
        bad = set(self._list) - set(allow)
        values = tuple(values) if isinstance(values, Iterable) else (values,)
        for v in values:
            if v in bad:
                raise ValueError(
                    "Layer '{v}' is already present in layer list".format(
                        v=v,
                    )
                )
        return values

    def __setitem__(self, key, value):
        old = self._list[key]
        if isinstance(key, slice):
            value = self._ensure_unique(value, old)
        elif isinstance(key, int):
            (value,) = self._ensure_unique((value,), (old,))
        super().__setitem__(key, value)

    def insert(self, index: int, value: Layer):
        (value,) = self._ensure_unique((value,))
        new_layer = self._type_check(value)
        self._clean_cache()
        new_layer.extent_.connect(self._clean_cache)
        super().insert(index, new_layer)

    def _get_min_and_max(self, mins_list, maxes_list):
        # Reverse dimensions since it is the last dimensions that are
        # displayed.
        mins_list = [mins[::-1] for mins in mins_list]
        maxes_list = [maxes[::-1] for maxes in maxes_list]

        with warnings.catch_warnings():
            # Taking the nanmin and nanmax of an axis of all nan
            # raises a warning and returns nan for that axis
            # as we have do an explicit nan_to_num below this
            # behaviour is acceptable and we can filter the
            # warning
            warnings.filterwarnings(
                "ignore",
                message=str("All-NaN axis encountered"),
            )
            min_v = np.nanmin(
                list(itertools.zip_longest(*mins_list, fillvalue=np.nan)),
                axis=1,
            )
            max_v = np.nanmax(
                list(itertools.zip_longest(*maxes_list, fillvalue=np.nan)),
                axis=1,
            )

        # 512 element default extent as documented in `_get_extent_world`
        min_v = np.nan_to_num(min_v, nan=-0.5)
        max_v = np.nan_to_num(max_v, nan=511.5)

        # switch back to original order
        return min_v[::-1], max_v[::-1]

    def _get_extent_world(self, layer_extent_list):
        if len(self) == 0:
            min_v = np.asarray([-0.5] * 3)
            max_v = np.asarray([511.5] * 3)
        else:
            extrema = [extent.world for extent in layer_extent_list]
            mins = [e[0] for e in extrema]
            maxs = [e[1] for e in extrema]
            min_v, max_v = self._get_min_and_max(mins, maxs)

        return np.vstack([min_v, max_v])

    @cached_property
    def _step_size(self) -> np.ndarray:
        return self._get_step_size([layer.extent for layer in self])

    def _step_size_from_scales(self, scales):
        # Reverse order so last axes of scale with different ndim are aligned
        scales = [scale[::-1] for scale in scales]
        full_scales = list(
            np.array(list(itertools.zip_longest(*scales, fillvalue=np.nan)))
        )
        # restore original order
        return np.nanmin(full_scales, axis=1)[::-1]

    def _get_step_size(self, layer_extent_list):
        if len(self) == 0:
            return np.ones(3)

        scales = [extent.step for extent in layer_extent_list]
        return self._step_size_from_scales(scales)

    def get_extent(self, layers: Iterable[Layer]) -> Extent:
        extent_list = [layer.extent for layer in layers]
        return Extent(
            data=None,
            world=self._get_extent_world(extent_list),
            step=self._get_step_size(extent_list),
        )

    @cached_property
    def extent(self) -> Extent:
        return self.get_extent(list(self))

    @cached_property
    def _ranges(self) -> List[Tuple[float, float, float]]:
        if len(self) == 0:
            return [(0, 1, 1)] * 3

        # Determine minimum step size across all layers
        layer_extent_list = [layer.extent for layer in self]
        scales = [extent.step for extent in layer_extent_list]
        min_steps = self._step_size_from_scales(scales)

        # Pixel-based layers need to be offset by 0.5 * min_steps to align
        # Dims.range with pixel centers in world coordinates
        pixel_offsets = [
            0.5 * min_steps if isinstance(layer, _ImageBase) else [0] * len(min_steps)
            for layer in self
        ]

        # Non-pixel layers need an offset of the range stop by min_steps since the upper
        # limit of Dims.range is non-inclusive.
        point_offsets = [
            [0] * len(min_steps) if isinstance(layer, _ImageBase) else min_steps
            for layer in self
        ]

        # Determine world coordinate extents similarly to
        # `_get_extent_world`, but including offsets calculated above.
        extrema = [extent.world for extent in layer_extent_list]
        mins = [e[0] + o1[: len(e[0])] for e, o1 in zip(extrema, pixel_offsets)]
        maxs = [
            e[1] + o1[: len(e[0])] + o2[: len(e[0])]
            for e, o1, o2 in zip(extrema, pixel_offsets, point_offsets)
        ]
        min_v, max_v = self._get_min_and_max(mins, maxs)

        # form range tuples, switching back to original dimension order
        return list(zip(min_v, max_v, min_steps))
