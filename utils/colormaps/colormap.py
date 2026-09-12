from enum import auto
from typing import Optional, Union, overload

import numpy as np

from utils.colormaps.colorbars import make_colorbar
from utils.colormaps.standardize_color import transform_color
from utils.custom_types import Array
from utils.misc import StringEnum


class ColormapInterpolationMode(StringEnum):
    LINEAR = auto()
    ZERO = auto()


class Colormap:
    def __init__(self, **data) -> None:
        self.name = data.get("name", "custom")
        self.colors = transform_color(data.get("colors", []))
        self.controls = Array.validate_type(data.get("controls", []))
        self.interpolation = data.get("interpolation", ColormapInterpolationMode.LINEAR)
        self._check_controls()
        self._display_name = data.get("display_name", self.name)

    def _check_controls(self):
        # If no control points provided generate defaults
        if self.controls is None or len(self.controls) == 0:
            n_controls = len(self.colors) + int(
                self.interpolation == ColormapInterpolationMode.ZERO
            )
            self.controls = np.linspace(0, 1, n_controls, dtype=np.float32)
            return

        # Check control end points are correct
        if self.controls[0] != 0 or (len(self.controls) > 1 and self.controls[-1] != 1):
            raise ValueError(
                "Control points must start with 0.0 and end with 1.0. "
                "Got {start_control_point} and {end_control_point}".format(
                    start_control_point=self.controls[0],
                    end_control_point=self.controls[-1],
                )
            )

        # Check control points are sorted correctly
        if not np.array_equal(self.controls, sorted(self.controls)):
            raise ValueError("Control points need to be sorted in ascending order")

        # Check number of control points is correct
        n_controls_target = len(self.colors) + int(
            self.interpolation == ColormapInterpolationMode.ZERO
        )
        n_controls = len(self.controls)
        if n_controls != n_controls_target:
            raise ValueError(
                "Wrong number of control points provided. Expected {n_controls_target}, got {n_controls}".format(
                    n_controls_target=n_controls_target,
                    n_controls=n_controls,
                )
            )

    def __iter__(self):
        yield from (self.colors, self.controls, self.interpolation)

    def __len__(self):
        return len(self.colors)

    def map(self, values):
        values = np.atleast_1d(values)
        if self.interpolation == ColormapInterpolationMode.LINEAR:
            # One color per control point
            cols = [
                np.interp(values, self.controls, self.colors[:, i]) for i in range(4)
            ]
            cols = np.stack(cols, axis=-1)
        elif self.interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            # Colors beyond max clipped to final bin
            indices = np.clip(
                np.searchsorted(self.controls, values, side="right") - 1,
                0,
                len(self.colors) - 1,
            )
            cols = self.colors[indices.astype(np.int32)]
        else:
            raise ValueError("Unrecognized Colormap Interpolation Mode")

        return cols

    @property
    def colorbar(self):
        return make_colorbar(self)


class LabelColormapBase(Colormap):
    def __init__(self, **data):
        self.use_selection = data.get("use_selection", False)
        data["interpolation"] = ColormapInterpolationMode.ZERO
        self.selection = data.get("selection", 0)
        self.background_value = data.get("background_value", 0)
        super().__init__(**data)
        self._cache_mapping = {}
        self._cache_other = {}

    @overload
    def _data_to_texture(self, values: np.ndarray) -> np.ndarray: ...

    @overload
    def _data_to_texture(self, values: np.integer) -> np.integer: ...

    def _data_to_texture(
        self, values: Union[np.ndarray, np.integer]
    ) -> Union[np.ndarray, np.integer]:
        """Map input values to values for send to GPU."""
        raise NotImplementedError

    def _cmap_without_selection(self) -> "LabelColormapBase":
        if self.use_selection:
            cmap = self.__class__(**self.dict())
            cmap.use_selection = False
            return cmap
        return self

    def _get_mapping_from_cache(self, data_dtype: np.dtype) -> Optional[np.ndarray]:
        """For given dtype, return precomputed array mapping values to colors.

        Returns None if the dtype itemsize is greater than 2.
        """
        target_dtype = _texture_dtype(self._num_unique_colors, data_dtype)
        key = (data_dtype, target_dtype)
        if key not in self._cache_mapping and data_dtype.itemsize <= 2:
            data = np.arange(np.iinfo(target_dtype).max + 1, dtype=target_dtype).astype(
                data_dtype
            )
            self._cache_mapping[key] = self._map_without_cache(data)
        return self._cache_mapping.get(key)

    def _clear_cache(self):
        """Mechanism to clean cached properties"""
        self._cache_mapping = {}
        self._cache_other = {}

    @property
    def _num_unique_colors(self) -> int:
        """Number of unique colors, not counting transparent black."""
        return len(self.colors) - 1

    def _map_without_cache(self, values: np.ndarray) -> np.ndarray:
        """Function that maps values to colors without selection or cache"""
        raise NotImplementedError

    def _selection_as_minimum_dtype(self, dtype: np.dtype) -> int:
        """Treat selection as given dtype and calculate value with min dtype.

        Parameters
        ----------
        dtype : np.dtype
            The dtype to convert the selection to.

        Returns
        -------
        int
            The selection converted.
        """
        return int(self._data_to_texture(dtype.type(self.selection)))


class CyclicLabelColormap(LabelColormapBase):
    def __init__(self, **data) -> None:
        self.seed = data.get("seed", 0.5)
        super().__init__(**data)
        self._validate_colors()

    def _validate_colors(self):
        if len(self.colors) > 2**16:
            raise ValueError(
                "Only up to 2**16=65535 colors are supported for LabelColormap"
            )

    def _background_as_minimum_dtype(self, dtype: np.dtype) -> int:
        """Treat background as given dtype and calculate value with min dtype.

        Parameters
        ----------
        dtype : np.dtype
            The dtype to convert the background to.

        Returns
        -------
        int
            The background converted.
        """
        return int(self._data_to_texture(dtype.type(self.background_value)))

    @overload
    def _data_to_texture(self, values: np.ndarray) -> np.ndarray: ...

    @overload
    def _data_to_texture(self, values: np.integer) -> np.integer: ...

    def _data_to_texture(
        self, values: Union[np.ndarray, np.integer]
    ) -> Union[np.ndarray, np.integer]:
        """Map input values to values for send to GPU."""
        return _cast_labels_data_to_texture_dtype_auto(values, self)

    def _map_without_cache(self, values) -> np.ndarray:
        texture_dtype_values = _zero_preserving_modulo_numpy(
            values,
            len(self.colors) - 1,
            values.dtype,
            self.background_value,
        )
        mapped = self.colors[texture_dtype_values]
        mapped[texture_dtype_values == 0] = 0
        return mapped

    def map(self, values: Union[np.ndarray, np.integer, int]) -> np.ndarray:
        """Map values to colors.

        Parameters
        ----------
        values : np.ndarray or int
            Values to be mapped.

        Returns
        -------
        np.ndarray of the same shape as values,
            but with the last dimension of size 4
            Mapped colors.
        """
        original_shape = np.shape(values)
        values = np.atleast_1d(values)

        if values.dtype.kind == "f":
            values = values.astype(np.int64)
        mapper = self._get_mapping_from_cache(values.dtype)
        if mapper is not None:
            mapped = mapper[values]
        else:
            mapped = self._map_without_cache(values)
        if self.use_selection:
            mapped[(values != self.selection)] = 0

        return np.reshape(mapped, original_shape + (4,))


@overload
def _convert_small_ints_to_unsigned(
    data: np.ndarray,
) -> np.ndarray: ...


@overload
def _convert_small_ints_to_unsigned(
    data: np.integer,
) -> np.integer: ...


def _convert_small_ints_to_unsigned(
    data: Union[np.ndarray, np.integer],
) -> Union[np.ndarray, np.integer]:
    """Convert (u)int8 to uint8 and (u)int16 to uint16.

    Otherwise, return the original array.

    Parameters
    ----------
    data : np.ndarray | np.integer
        Data to be converted.

    Returns
    -------
    np.ndarray | np.integer
        Converted data.
    """
    if data.dtype.itemsize == 1:
        # for fast rendering of int8
        return data.view(np.uint8)
    if data.dtype.itemsize == 2:
        # for fast rendering of int16
        return data.view(np.uint16)
    return data


@overload
def _cast_labels_data_to_texture_dtype_auto(
    data: np.ndarray,
    colormap: CyclicLabelColormap,
) -> np.ndarray: ...


@overload
def _cast_labels_data_to_texture_dtype_auto(
    data: np.integer,
    colormap: CyclicLabelColormap,
) -> np.integer: ...


def _cast_labels_data_to_texture_dtype_auto(
    data: Union[np.ndarray, np.integer],
    colormap: CyclicLabelColormap,
) -> Union[np.ndarray, np.integer]:
    """Convert labels data to the data type used in the texture.

    - uint8 and uint16 labels data are unchanged. (No copy of the arrays.)
    - int8 and int16 data are converted with a *view* to uint8 and uint16.
      (This again does not involve a copy so is fast, and lossless.)
    - higher precision integer data (u)int{32,64} are hashed to uint8, uint16,
      or float32, depending on the number of colors in the input colormap. (See
      `minimum_dtype_for_labels`.) Since the hashing can result in collisions,
      this conversion *has* to happen in the CPU to correctly map the
      background and selection values.

    Parameters
    ----------
    data : np.ndarray
        Labels data to be converted.
    colormap : CyclicLabelColormap
        Colormap used to display the labels data.

    Returns
    -------
    np.ndarray | np.integer
        Converted labels data.
    """
    original_shape = np.shape(data)
    if data.itemsize <= 2:
        return _convert_small_ints_to_unsigned(data)

    data_arr = np.atleast_1d(data)
    num_colors = len(colormap.colors) - 1
    zero_preserving_modulo_func = _zero_preserving_modulo_numpy

    dtype = minimum_dtype_for_labels(num_colors + 1)

    if colormap.use_selection:
        selection_in_texture = _zero_preserving_modulo_numpy(
            np.array([colormap.selection]), num_colors, dtype
        )
        converted = np.where(
            data_arr == colormap.selection, selection_in_texture, dtype.type(0)
        )
    else:
        converted = zero_preserving_modulo_func(
            data_arr, num_colors, dtype, colormap.background_value
        )

    if isinstance(data, np.integer):
        return dtype.type(converted[0])

    return np.reshape(converted, original_shape)


def _zero_preserving_modulo_numpy(
    values: np.ndarray, n: int, dtype: np.dtype, to_zero: int = 0
) -> np.ndarray:
    """``(values - 1) % n + 1``, but with one specific value mapped to 0.

    This ensures (1) an output value in [0, n] (inclusive), and (2) that
    no nonzero values in the input are zero in the output, other than the
    ``to_zero`` value.

    Parameters
    ----------
    values : np.ndarray
        The dividend of the modulo operator.
    n : int
        The divisor.
    dtype : np.dtype
        The desired dtype for the output array.
    to_zero : int, optional
        A specific value to map to 0. (By default, 0 itself.)

    Returns
    -------
    np.ndarray
        The result: 0 for the ``to_zero`` value, ``values % n + 1``
        everywhere else.
    """
    res = ((values - 1) % n + 1).astype(dtype)
    res[values == to_zero] = 0
    return res


def _texture_dtype(num_colors: int, dtype: np.dtype) -> np.dtype:
    """Compute VisPy texture dtype given number of colors and raw data dtype.

    - for data of type int8 and uint8 we can use uint8 directly, with no copy.
    - for int16 and uint16 we can use uint16 with no copy.
    - for any other dtype, we fall back on `minimum_dtype_for_labels`, which
      will require on-CPU mapping between the raw data and the texture dtype.
    """
    if dtype.itemsize == 1:
        return np.dtype(np.uint8)
    if dtype.itemsize == 2:
        return np.dtype(np.uint16)
    return minimum_dtype_for_labels(num_colors)


def minimum_dtype_for_labels(num_colors: int) -> np.dtype:
    """Return the minimum texture dtype that can hold given number of colors.

    Parameters
    ----------
    num_colors : int
        Number of unique colors in the data.

    Returns
    -------
    np.dtype
        Minimum dtype that can hold the number of colors.
    """
    if num_colors <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if num_colors <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.float32)
