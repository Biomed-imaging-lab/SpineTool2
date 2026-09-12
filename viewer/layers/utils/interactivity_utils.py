from typing import List, Tuple, Union

import numpy as np


def displayed_plane_from_nd_line_segment(
    start_point: np.ndarray,
    end_point: np.ndarray,
    dims_displayed: Union[List[int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the plane defined by start_point and the normal vector that goes
    from start_point to end_point.

    Note the start_point and end_point are nD and
    the returned plane is in the displayed dimensions (i.e., 3D).

    Parameters
    ----------
    start_point : np.ndarray
        The start point of the line segment in nD coordinates.
    end_point : np.ndarray
        The end point of the line segment in nD coordinates..
    dims_displayed : Union[List[int], np.ndarray]
        The dimensions of the data array currently in view.

    Returns
    -------
    plane_point : np.ndarray
        The point on the plane that intersects the click ray. This is returned
        in data coordinates with only the dimensions that are displayed.
    plane_normal : np.ndarray
        The normal unit vector for the plane. It points in the direction of the click
        in data coordinates.
    """
    plane_point = start_point[dims_displayed]
    end_position_view = end_point[dims_displayed]
    ray_direction = end_position_view - plane_point
    plane_normal = ray_direction / np.linalg.norm(ray_direction)
    return plane_point, plane_normal
