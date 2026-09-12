from typing import Dict, Optional, Tuple

import numpy as np

# normal vectors for a 3D axis-aligned box
# coordinates are ordered [z, y, x]
FACE_NORMALS = {
    "x_pos": np.array([0, 0, 1]),
    "x_neg": np.array([0, 0, -1]),
    "y_pos": np.array([0, 1, 0]),
    "y_neg": np.array([0, -1, 0]),
    "z_pos": np.array([1, 0, 0]),
    "z_neg": np.array([-1, 0, 0]),
}


def project_points_onto_plane(
    points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project points on to a plane.

    Plane is defined by a point and a normal vector. This function
    is designed to work with points and planes in 3D.

    Parameters
    ----------
    points : np.ndarray
        The coordinate of the point to be projected. The points
        should be 3D and have shape shape (N,3) for N points.
    plane_point : np.ndarray
        The point on the plane used to define the plane.
        Should have shape (3,).
    plane_normal : np.ndarray
        The normal vector used to define the plane.
        Should be a unit vector and have shape (3,).

    Returns
    -------
    projected_point : np.ndarray
        The point that has been projected to the plane.
        This is always an Nx3 array.
    signed_distance_to_plane : np.ndarray
        The signed projection distance between the points and the plane.
        Positive values indicate the point is on the positive normal side of the plane.
        Negative values indicate the point is on the negative normal side of the plane.
    """
    points = np.atleast_2d(points)
    plane_point = np.asarray(plane_point)
    # make the plane normals have the same shape as the points
    plane_normal = np.tile(plane_normal, (points.shape[0], 1))

    # get the vector from point on the plane
    # to the point to be projected
    point_vector = points - plane_point

    # find the distance to the plane along the normal direction
    signed_distance_to_plane = np.multiply(point_vector, plane_normal).sum(axis=1)

    # project the point
    projected_points = points - (signed_distance_to_plane[:, np.newaxis] * plane_normal)

    return projected_points, signed_distance_to_plane


def rotation_matrix_from_vectors_3d(vec_1, vec_2):
    """Calculate the rotation matrix that aligns vec1 to vec2.

    Parameters
    ----------
    vec_1 : np.ndarray
        The vector you want to rotate
    vec_2 : np.ndarray
        The vector you would like to align to.

    Returns
    -------
    rotation_matrix : np.ndarray
        The rotation matrix that aligns vec_1 with vec_2.
        That is rotation_matrix.dot(vec_1) == vec_2
    """
    vec_1 = (vec_1 / np.linalg.norm(vec_1)).reshape(3)
    vec_2 = (vec_2 / np.linalg.norm(vec_2)).reshape(3)
    cross_prod = np.cross(vec_1, vec_2)
    dot_prod = np.dot(vec_1, vec_2)
    if any(cross_prod):  # if not all zeros then
        s = np.linalg.norm(cross_prod)
        kmat = np.array(
            [
                [0, -cross_prod[2], cross_prod[1]],
                [cross_prod[2], 0, -cross_prod[0]],
                [-cross_prod[1], cross_prod[0], 0],
            ]
        )
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - dot_prod) / (s**2))

    else:
        if np.allclose(dot_prod, 1):
            # if the vectors are already aligned, return the identity
            rotation_matrix = np.eye(3)
        else:
            # if the vectors are in opposite direction, rotate 180 degrees
            rotation_matrix = np.diag([-1, -1, 1])
    return rotation_matrix


def rotate_points(
    points: np.ndarray,
    current_plane_normal: np.ndarray,
    new_plane_normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate points using a rotation matrix defined by the rotation from
    current_plane to new_plane.

    Parameters
    ----------
    points : np.ndarray
        The points to rotate. They should all lie on the same plane with the
        normal vector current_plane_normal. Should be (NxD) array.
    current_plane_normal : np.ndarray
        The normal vector for the plane the points currently reside on.
    new_plane_normal : np.ndarray
        The normal vector for the plane the points will be rotated to.

    Returns
    -------
    rotated_points : np.ndarray
        The points that have been rotated
    rotation_matrix : np.ndarray
        The rotation matrix used for rotating the points.
    """
    rotation_matrix = rotation_matrix_from_vectors_3d(
        current_plane_normal, new_plane_normal
    )
    rotated_points = points @ rotation_matrix.T

    return rotated_points, rotation_matrix


def clamp_point_to_bounding_box(point: np.ndarray, bounding_box: np.ndarray):
    """Ensure that a point is inside of the bounding box. If the point has a
    coordinate outside of the bounding box, the value is clipped to the max
    extent of the bounding box.

    Parameters
    ----------
    point : np.ndarray
        n-dimensional point as an (n,) ndarray. Multiple points can
        be passed as an (n, D) array.
    bounding_box : np.ndarray
        n-dimensional bounding box as a (n, 2) ndarray

    Returns
    -------
    clamped_point : np.ndarray
        `point` clamped to the limits of `bounding_box`
    """
    clamped_point = np.clip(point, bounding_box[:, 0], bounding_box[:, 1] - 1)
    return clamped_point


def face_coordinate_from_bounding_box(
    bounding_box: np.ndarray, face_normal: np.ndarray
) -> float:
    """Get the coordinate for a given face in an axis-aligned bounding box.
    For example, if the bounding box has extents [[0, 10], [0, 20], [0, 30]]
    (ordered zyx), then the face with normal [0, 1, 0] is described by
    y=20. Thus, the face_coordinate in this case is 20.

    Parameters
    ----------
    bounding_box : np.ndarray
        n-dimensional bounding box as a (n, 2) ndarray.
        Each row should contain the [min, max] extents for the
        axis.
    face_normal : np.ndarray
        normal vector of the face as an (n,)  ndarray

    Returns
    -------
    face_coordinate : float
        The value where the bounding box face specified by face_normal intersects
        the axis its normal is aligned with.
    """
    axis = np.argwhere(face_normal)
    if face_normal[axis] > 0:
        # face is pointing in the positive direction,
        # take the max extent
        face_coordinate = bounding_box[axis, 1]
    else:
        # face is pointing in the negative direction,
        # take the min extent
        face_coordinate = bounding_box[axis, 0]
    return face_coordinate


def intersect_line_with_axis_aligned_plane(
    plane_intercept: float,
    plane_normal: np.ndarray,
    line_start: np.ndarray,
    line_direction: np.ndarray,
) -> np.ndarray:
    """Find the intersection of a line with an axis aligned plane.

    Parameters
    ----------
    plane_intercept : float
        The coordinate that the plane intersects on the axis to which plane is
        normal.
        For example, if the plane is described by y=42, plane_intercept is 42.
    plane_normal : np.ndarray
        normal vector of the plane as an (n,)  ndarray
    line_start : np.ndarray
        start point of the line as an (n,) ndarray
    line_direction : np.ndarray
        direction vector of the line as an (n,) ndarray

    Returns
    -------
    intersection_point : np.ndarray
        point where the line intersects the axis aligned plane
    """
    # find the axis the plane exists in
    plane_axis = np.squeeze(np.argwhere(plane_normal))

    # get the intersection coordinate
    t = (plane_intercept - line_start[plane_axis]) / line_direction[plane_axis]
    return line_start + t * line_direction


def bounding_box_to_face_vertices(
    bounding_box: np.ndarray,
) -> Dict[str, np.ndarray]:
    """From a layer bounding box (N, 2), N=ndim, return a dictionary containing
    the vertices of each face of the bounding_box.

    Parameters
    ----------
    bounding_box : np.ndarray
        (N, 2), N=ndim array with the min and max value for each dimension of
        the bounding box. The bounding box is take form the last
        three rows, which are assumed to be in order (z, y, x).

    Returns
    -------
    face_coords : Dict[str, np.ndarray]
        A dictionary containing the coordinates for the vertices for each face.
        The keys are strings: 'x_pos', 'x_neg', 'y_pos', 'y_neg', 'z_pos', 'z_neg'.
        'x_pos' is the face with the normal in the positive x direction and
        'x_neg' is the face with the normal in the negative direction.
        Coordinates are ordered (z, y, x).
    """
    x_min, x_max = bounding_box[-1, :]
    y_min, y_max = bounding_box[-2, :]
    z_min, z_max = bounding_box[-3, :]

    face_coords = {
        "x_pos": np.array(
            [
                [z_min, y_min, x_max],
                [z_min, y_max, x_max],
                [z_max, y_max, x_max],
                [z_max, y_min, x_max],
            ]
        ),
        "x_neg": np.array(
            [
                [z_min, y_min, x_min],
                [z_min, y_max, x_min],
                [z_max, y_max, x_min],
                [z_max, y_min, x_min],
            ]
        ),
        "y_pos": np.array(
            [
                [z_min, y_max, x_min],
                [z_min, y_max, x_max],
                [z_max, y_max, x_max],
                [z_max, y_max, x_min],
            ]
        ),
        "y_neg": np.array(
            [
                [z_min, y_min, x_min],
                [z_min, y_min, x_max],
                [z_max, y_min, x_max],
                [z_max, y_min, x_min],
            ]
        ),
        "z_pos": np.array(
            [
                [z_max, y_min, x_min],
                [z_max, y_min, x_max],
                [z_max, y_max, x_max],
                [z_max, y_max, x_min],
            ]
        ),
        "z_neg": np.array(
            [
                [z_min, y_min, x_min],
                [z_min, y_min, x_max],
                [z_min, y_max, x_max],
                [z_min, y_max, x_min],
            ]
        ),
    }
    return face_coords


def inside_triangles(triangles):
    """Checks which triangles contain the origin

    Parameters
    ----------
    triangles : (N, 3, 2) array
        Array of N triangles that should be checked

    Returns
    -------
    inside : (N,) array of bool
        Array with `True` values for triangles containing the origin
    """
    AB = triangles[:, 1, :] - triangles[:, 0, :]
    AC = triangles[:, 2, :] - triangles[:, 0, :]
    BC = triangles[:, 2, :] - triangles[:, 1, :]

    s_AB = -AB[:, 0] * triangles[:, 0, 1] + AB[:, 1] * triangles[:, 0, 0] >= 0
    s_AC = -AC[:, 0] * triangles[:, 0, 1] + AC[:, 1] * triangles[:, 0, 0] >= 0
    s_BC = -BC[:, 0] * triangles[:, 1, 1] + BC[:, 1] * triangles[:, 1, 0] >= 0

    inside = np.all(np.array([s_AB != s_AC, s_AB == s_BC]), axis=0)

    return inside


def point_in_quadrilateral_2d(point: np.ndarray, quadrilateral: np.ndarray) -> bool:
    """Determines whether a point is inside a 2D quadrilateral.

    Parameters
    ----------
    point : np.ndarray
        (2,) array containing coordinates of a point.
    quadrilateral : np.ndarray
        (4, 2) array containing the coordinates for the 4 corners
        of a quadrilateral. The vertices should be in clockwise order
        such that indexing with [0, 1, 2], and [0, 2, 3] results in
        the two non-overlapping triangles that divide the
        quadrilateral.

    Returns
    -------

    """
    triangle_vertices = np.stack((quadrilateral[[0, 1, 2]], quadrilateral[[0, 2, 3]]))
    in_triangles = inside_triangles(triangle_vertices - point)
    return in_triangles.sum() >= 1


def line_in_quadrilateral_3d(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    quadrilateral: np.ndarray,
) -> bool:
    """Determine if a line goes tbrough any of a  set of quadrilaterals.

    For example, this could be used to determine if a click was
    in a specific face of a bounding box.

    Parameters
    ----------
    line_point : np.ndarray
        (3,) array containing the location that was clicked. This
        should be in the same coordinate system as the vertices.
    line_direction : np.ndarray
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as
        the vertices.
    quadrilateral : np.ndarray
        (4, 3) array containing the coordinates for the 4 corners
        of a quadrilateral. The vertices should be in clockwise order
        such that indexing with [0, 1, 2], and [0, 2, 3] results in
        the two non-overlapping triangles that divide the quadrilateral.


    Returns
    -------
    in_region : bool
        True if the click is in the region specified by vertices.
    """

    # project the vertices of the bound region on to the view plane
    vertices_plane, _ = project_points_onto_plane(
        points=quadrilateral,
        plane_point=line_point,
        plane_normal=line_direction,
    )

    # rotate the plane to make the triangles 2D
    rotated_vertices, rotation_matrix = rotate_points(
        points=vertices_plane,
        current_plane_normal=line_direction,
        new_plane_normal=[0, 0, 1],
    )
    quadrilateral_2D = rotated_vertices[:, :2]
    click_pos_2D = rotation_matrix.dot(line_point)[:2]

    return point_in_quadrilateral_2d(click_pos_2D, quadrilateral_2D)


def find_front_back_face(
    click_pos: np.ndarray, bounding_box: np.ndarray, view_dir: np.ndarray
):
    """Find the faces of an axis aligned bounding box a
    click intersects with.

    Parameters
    ----------
    click_pos : np.ndarray
        (3,) array containing the location that was clicked.
    bounding_box : np.ndarray
        (N, 2), N=ndim array with the min and max value for each dimension of
        the bounding box. The bounding box is take form the last
        three rows, which are assumed to be in order (z, y, x).
        This should be in the same coordinate system as click_pos.
    view_dir
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as click_pos.

    Returns
    -------
    front_face_normal : np.ndarray
        The (3,) normal vector of the face closest to the camera the click
        intersects with.
    back_face_normal : np.ndarray
        The (3,) normal vector of the face farthest from the camera the click
        intersects with.
    """
    front_face_normal = None
    back_face_normal = None

    bbox_face_coords = bounding_box_to_face_vertices(bounding_box)
    for k, v in FACE_NORMALS.items():
        if np.dot(view_dir, v) < -0.001:
            if line_in_quadrilateral_3d(click_pos, view_dir, bbox_face_coords[k]):
                front_face_normal = v
        elif line_in_quadrilateral_3d(click_pos, view_dir, bbox_face_coords[k]):
            back_face_normal = v
        if front_face_normal is not None and back_face_normal is not None:
            # stop looping if both the front and back faces have been found
            break

    return front_face_normal, back_face_normal


def intersect_line_with_axis_aligned_bounding_box_3d(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    bounding_box: np.ndarray,
    face_normal: np.ndarray,
):
    """Find the intersection of a ray with the specified face of an
    axis-aligned bounding box.

    Parameters
    ----------
    face_normal : np.ndarray
        The (3,) normal vector of the face the click intersects with.
    line_point : np.ndarray
        (3,) array containing the location that was clicked.
    bounding_box : np.ndarray
        (N, 2), N=ndim array with the min and max value for each dimension of
        the bounding box. The bounding box is take form the last
        three rows, which are assumed to be in order (z, y, x).
        This should be in the same coordinate system as click_pos.
    line_direction
        (3,) array describing the direction camera is pointing in
        the scene. This should be in the same coordinate system as click_pos.

    Returns
    -------
    intersection_point : np.ndarray
        (3,) array containing the coordinate for the intersection of the click on
        the specified face.
    """
    front_face_coordinate = face_coordinate_from_bounding_box(bounding_box, face_normal)
    intersection_point = np.squeeze(
        intersect_line_with_axis_aligned_plane(
            front_face_coordinate,
            face_normal,
            line_point,
            -line_direction,
        )
    )

    return intersection_point
