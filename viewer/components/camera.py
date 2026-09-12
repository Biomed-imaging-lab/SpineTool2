from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from scipy.spatial.transform import Rotation as R

from utils.misc import ensure_n_tuple


class Camera(QObject):
    center_ = pyqtSignal(object)
    zoom_ = pyqtSignal(float)
    angles_ = pyqtSignal(object)
    perspective_ = pyqtSignal(float)
    mouse_pan_ = pyqtSignal(bool)
    mouse_zoom_ = pyqtSignal(bool)

    def __init__(self):
        QObject.__init__(self)
        self._center = (0.0, 0.0, 0.0)
        self._zoom = 1.0
        self._angles = (0.0, 0.0, 90.0)
        self._perspective = 0.0
        self._mouse_pan = True
        self._mouse_zoom = True

    @property
    def center(self) -> Tuple[float, float, float]:
        return self._center

    @center.setter
    def center(self, value) -> None:
        self._center = ensure_n_tuple(value, n=3)
        self.center_.emit(self._center)

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value) -> None:
        self._zoom = value
        self.zoom_.emit(value)

    @property
    def angles(self) -> Tuple[float, float, float]:
        return self._angles

    @angles.setter
    def angles(self, value) -> None:
        self._angles = ensure_n_tuple(value, n=3)
        self.angles_.emit(self._angles)

    @property
    def perspective(self) -> float:
        return self._perspective

    @perspective.setter
    def perspective(self, value) -> None:
        self._perspective = value
        self.perspective_.emit(value)

    @property
    def mouse_pan(self) -> bool:
        return self._mouse_pan

    @mouse_pan.setter
    def mouse_pan(self, value) -> None:
        self._mouse_pan = value
        self.mouse_pan_.emit(value)

    @property
    def mouse_zoom(self) -> bool:
        return self._mouse_zoom

    @mouse_zoom.setter
    def mouse_zoom(self, value) -> None:
        self._mouse_zoom = value
        self.mouse_zoom_.emit(value)

    @property
    def view_direction(self) -> Tuple[float, float, float]:
        ang = np.deg2rad(self.angles)
        view_direction = (
            np.sin(ang[2]) * np.cos(ang[1]),
            np.cos(ang[2]) * np.cos(ang[1]),
            -np.sin(ang[1]),
        )
        return view_direction

    @property
    def up_direction(self) -> Tuple[float, float, float]:
        rotation_matrix = R.from_euler(
            seq="yzx", angles=self.angles, degrees=True
        ).as_matrix()
        return tuple(rotation_matrix[:, 2][::-1])

    def set_view_direction(
        self,
        view_direction: Tuple[float, float, float],
        up_direction: Tuple[float, float, float] = (0, -1, 0),
    ):
        # default behaviour of up direction
        view_direction_along_y_axis = (
            view_direction[0],
            view_direction[2],
        ) == (0, 0)
        up_direction_along_y_axis = (up_direction[0], up_direction[2]) == (
            0,
            0,
        )
        if view_direction_along_y_axis and up_direction_along_y_axis:
            up_direction = (-1, 0, 0)  # align up direction along z axis

        # xyz ordering for vispy, normalise vectors for rotation matrix
        view_direction = np.asarray(view_direction, dtype=float)[::-1]
        view_direction /= np.linalg.norm(view_direction)

        up_direction = np.asarray(up_direction, dtype=float)[::-1]
        up_direction = np.cross(view_direction, up_direction)
        up_direction /= np.linalg.norm(up_direction)

        # explicit check for parallel view direction and up direction
        if np.allclose(np.cross(view_direction, up_direction), 0):
            raise ValueError("view direction and up direction are parallel")

        x_direction = np.cross(up_direction, view_direction)
        x_direction /= np.linalg.norm(x_direction)

        # construct rotation matrix, convert to euler angles
        rotation_matrix = np.column_stack((up_direction, view_direction, x_direction))
        euler_angles = R.from_matrix(rotation_matrix).as_euler(seq="yzx", degrees=True)
        self.angles = euler_angles

    def calculate_nd_view_direction(self, dims_displayed: Tuple[int]) -> np.ndarray:
        if len(dims_displayed) != 3:
            return None
        view_direction_nd = np.zeros(3)
        view_direction_nd[list(dims_displayed)] = self.view_direction
        return view_direction_nd

    def calculate_nd_up_direction(
        self, dims_displayed: Tuple[int]
    ) -> Optional[np.ndarray]:
        if len(dims_displayed) != 3:
            return None
        up_direction_nd = np.zeros(3)
        up_direction_nd[list(dims_displayed)] = self.up_direction
        return up_direction_nd
