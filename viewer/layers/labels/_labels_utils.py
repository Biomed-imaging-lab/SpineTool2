from functools import lru_cache

import numpy as np


def interpolate_coordinates(old_coord, new_coord, brush_size):
    if old_coord is None:
        old_coord = new_coord
    if new_coord is None:
        new_coord = old_coord
    num_step = round(
        max(abs(np.array(new_coord) - np.array(old_coord))) / brush_size * 4
    )
    coords = [
        np.linspace(old_coord[i], new_coord[i], num=int(num_step + 1))
        for i in range(len(new_coord))
    ]
    coords = np.stack(coords).T
    if len(coords) > 1:
        coords = coords[1:]

    return coords


@lru_cache(maxsize=64)
def ellipse_indices(radii):
    radii = np.array(radii).astype(int) - 1
    slices = [slice(-int(np.ceil(r)), int(np.floor(r)) + 1) for r in radii]
    indices = np.mgrid[slices].T.reshape(-1, radii.shape[0])
    distances_sq = np.sum((indices / (radii + 0.5)) ** 2, axis=1)
    mask_indices = indices[distances_sq <= 1].astype(int)

    return mask_indices


def indices_in_shape(idxs, shape):
    np_index = isinstance(idxs, tuple)
    if np_index:
        idxs = np.transpose(idxs)
    keep_coords = np.logical_and(
        np.all(idxs >= 0, axis=1), np.all(idxs < np.array(shape), axis=1)
    )
    filtered = idxs[keep_coords]
    if np_index:
        filtered = tuple(filtered.T)
    return filtered


def first_nonzero_coordinate(data, start_point, end_point):
    shape = np.asarray(data.shape)
    length = np.linalg.norm(end_point - start_point)
    length_int = np.round(length).astype(int)
    coords = np.linspace(start_point, end_point, length_int + 1, endpoint=True)
    clipped_coords = np.clip(np.round(coords), 0, shape - 1).astype(int)
    nonzero = np.flatnonzero(data[tuple(clipped_coords.T)])
    return None if len(nonzero) == 0 else clipped_coords[nonzero[0]]


def mouse_event_to_labels_coordinate(layer, event):
    ndim = len(layer._slice_input.displayed)
    if ndim == 2:
        coordinates = layer.world_to_data(event.position)
    else:  # 3d
        start, end = layer.get_ray_intersections(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=layer._slice_input.displayed,
            world=True,
        )
        if start is None and end is None:
            return None
        coordinates = first_nonzero_coordinate(layer.data, start, end)
    return coordinates
