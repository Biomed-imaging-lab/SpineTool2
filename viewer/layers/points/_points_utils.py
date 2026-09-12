from typing import List, Optional, Union

import numpy as np

from utils.geometry import project_points_onto_plane
from viewer.layers.points._points_constants import SYMBOL_ALIAS, Symbol


def _create_box_from_corners_3d(
    box_corners: np.ndarray, box_normal: np.ndarray, up_vector: np.ndarray
) -> np.ndarray:
    horizontal_vector = np.cross(box_normal, up_vector)

    diagonal_vector = box_corners[1] - box_corners[0]

    up_displacement = np.dot(diagonal_vector, up_vector) * up_vector
    horizontal_displacement = (
        np.dot(diagonal_vector, horizontal_vector) * horizontal_vector
    )

    corner_1 = box_corners[0] + horizontal_displacement
    corner_3 = box_corners[0] + up_displacement

    box = np.array([box_corners[0], corner_1, box_corners[1], corner_3])
    return box


def create_box(data):
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    box = np.array([tl, tr, br, bl])
    return box


def points_to_squares(points, sizes):
    rect = np.concatenate(
        [
            points + 0.5 * np.array([sizes, sizes]).T,
            points + 0.5 * np.array([sizes, -sizes]).T,
            points + 0.5 * np.array([-sizes, sizes]).T,
            points + 0.5 * np.array([-sizes, -sizes]).T,
        ],
        axis=0,
    )
    return rect


def _points_in_box_3d(
    box_corners: np.ndarray,
    points: np.ndarray,
    sizes: np.ndarray,
    box_normal: np.ndarray,
    up_direction: np.ndarray,
) -> List[int]:
    # the the corners for a bounding box that is has one axis aligned
    # with the camera up direction and is normal to the view direction.
    bbox_corners = _create_box_from_corners_3d(box_corners, box_normal, up_direction)

    # project points onto the same plane as the box
    projected_points, _ = project_points_onto_plane(
        points=points,
        plane_point=bbox_corners[0],
        plane_normal=box_normal,
    )

    # create a new basis in which the bounding box is
    # axis aligned
    horz_direction = np.cross(box_normal, up_direction)
    plane_basis = np.column_stack([up_direction, horz_direction, box_normal])

    # transform the points and bounding box into a new basis
    # such that tha boudning box is axis aligned
    bbox_corners_axis_aligned = bbox_corners @ plane_basis
    bbox_corners_axis_aligned = bbox_corners_axis_aligned[:, :2]
    points_axis_aligned = projected_points @ plane_basis
    points_axis_aligned = points_axis_aligned[:, :2]

    # determine which points are in the box using the
    # axis-aligned basis
    return points_in_box(bbox_corners_axis_aligned, points_axis_aligned, sizes)


def points_in_box(
    corners: np.ndarray, points: np.ndarray, sizes: np.ndarray
) -> List[int]:
    box = create_box(corners)[[0, 2]]
    # Check all four corners in a square around a given point. If any corner
    # is inside the box, then that point is considered inside
    point_corners = points_to_squares(points, sizes)
    below_top = np.all(box[1] >= point_corners, axis=1)
    above_bottom = np.all(point_corners >= box[0], axis=1)
    point_corners_in_box = np.where(np.logical_and(below_top, above_bottom))[0]
    # Determine indices of points which have at least one corner inside box
    inside = np.unique(point_corners_in_box % len(points))
    return list(inside)


def fix_data_points(points: Optional[np.ndarray]) -> np.ndarray:
    if points is None or len(points) == 0:
        points = np.empty((0, 3))
    else:
        points = np.atleast_2d(points)
        if 3 != points.shape[1]:
            raise ValueError("Points dimensions must be equal to 3")
    return points


def coerce_symbol(symbol: Union[str, Symbol]) -> Symbol:
    if isinstance(symbol, Symbol):
        return symbol
    for k, v in SYMBOL_ALIAS.items():
        if (symbol == k) | (symbol == k.upper()):
            return v
    return Symbol(symbol)
