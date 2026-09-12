import numpy as np


def _nanmin(array):
    """
    call np.min but fall back to avoid nan and inf if necessary
    """
    min_value = np.min(array)
    if not np.isfinite(min_value):
        masked = array[np.isfinite(array)]
        if masked.size == 0:
            return 0
        min_value = np.min(masked)
    return min_value


def _nanmax(array):
    """
    call np.max but fall back to avoid nan and inf if necessary
    """
    max_value = np.max(array)
    if not np.isfinite(max_value):
        masked = array[np.isfinite(array)]
        if masked.size == 0:
            return 1
        max_value = np.max(masked)
    return max_value


def calc_data_range(data):
    """Calculate range of data values. If all values are equal return [0, 1].

    Parameters
    ----------
    data : array
        Data to calculate range of values over.

    Returns
    -------
    values : list of float
        Range of values.

    Notes
    -----
    If the data type is uint8, no calculation is performed, and 0-255 is
    returned.
    """
    if data.shape[0] == 0:
        return [0, 1]

    if data.dtype == np.uint8:
        return [0, 255]

    if data.size > 1e7 and data.ndim == 1:
        # If data is very large take the average of start, middle and end.
        center = int(data.shape[0] // 2)
        slices = [
            slice(0, 4096),
            slice(center - 2048, center + 2048),
            slice(-4096, None),
        ]
        reduced_data = [
            [_nanmax(data[sl]) for sl in slices],
            [_nanmin(data[sl]) for sl in slices],
        ]
    else:
        reduced_data = data

    min_val = _nanmin(reduced_data)
    max_val = _nanmax(reduced_data)

    if min_val == max_val:
        min_val = 0
        max_val = 1
    return [float(min_val), float(max_val)]


def convert_to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Convert array content to uint8, always returning a copy.

    Based on skimage.util.dtype._convert but limited to an output type uint8,
    so should be equivalent to skimage.util.dtype.img_as_ubyte.

    If all negative, values are clipped to 0.

    If values are integers and below 256, this simply casts.
    Otherwise the maximum value for the input data type is determined and
    output values are proportionally scaled by this value.

    Binary images are converted so that False -> 0, True -> 255.

    Float images are multiplied by 255 and then cast to uint8.
    """
    out_dtype = np.dtype(np.uint8)
    out_max = np.iinfo(out_dtype).max
    if data.dtype == out_dtype:
        return data
    in_kind = data.dtype.kind
    if in_kind == "b":
        return data.astype(out_dtype) * 255
    if in_kind == "f":
        image_out = np.multiply(data, out_max, dtype=data.dtype)
        np.rint(image_out, out=image_out)
        np.clip(image_out, 0, out_max, out=image_out)
        image_out = np.nan_to_num(image_out, copy=False)
        return image_out.astype(out_dtype)

    if in_kind in "ui":
        if in_kind == "u":
            if data.max() < out_max:
                return data.astype(out_dtype)
            return np.right_shift(data, (data.dtype.itemsize - 1) * 8).astype(out_dtype)

        np.maximum(data, 0, out=data, dtype=data.dtype)
        if data.dtype == np.int8:
            return (data * 2).astype(np.uint8)
        if data.max() < out_max:
            return data.astype(out_dtype)
        return np.right_shift(data, (data.dtype.itemsize - 1) * 8 - 1).astype(out_dtype)
    return None


def get_extent_world(data_extent, data_to_world, centered=False):
    """Range of layer in world coordinates base on provided data_extent

    Parameters
    ----------
    data_extent : array, shape (2, D)
        Extent of layer in data coordinates.
    data_to_world : viewer.utils.transforms.Scale
        The transform from data to world coordinates.
    centered : bool
        If pixels should be centered. By default False.

    Returns
    -------
    extent_world : array, shape (2, D)
    """
    D = data_extent.shape[1]
    # subtract 0.5 to get from pixel center to pixel edge
    offset = 0.5 * bool(centered)
    pixel_extents = tuple(d - offset for d in data_extent.T)

    full_data_extent = np.array(np.meshgrid(*pixel_extents)).T.reshape(-1, D)
    full_world_extent = data_to_world(full_data_extent)
    world_extent = np.array(
        [
            np.min(full_world_extent, axis=0),
            np.max(full_world_extent, axis=0),
        ]
    )
    return world_extent
