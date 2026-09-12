from numbers import Integral
from typing import Sequence, Tuple, Union

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from utils.misc import argsort


class Dims(QObject):
    ndisplay_ = pyqtSignal(int)
    last_used_ = pyqtSignal(int)
    range_ = pyqtSignal(object)
    current_step_ = pyqtSignal(object)
    order_ = pyqtSignal(object)
    axis_labels_ = pyqtSignal(object)

    def __init__(self, ndisplay, order, axis_labels):
        QObject.__init__(self)
        self._ndisplay = ndisplay
        self._last_used = 0
        self._range = ((0, 1, 1), (0, 1, 1), (0, 1, 1))
        self._current_step = (0, 0, 0)
        self._order = order
        self._axis_labels = axis_labels
        self._scroll_progress = 0

    @property
    def ndisplay(self) -> int:
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, value) -> None:
        self._ndisplay = value
        self.ndisplay_.emit(value)

    @property
    def last_used(self) -> int:
        return self._last_used

    @last_used.setter
    def last_used(self, value) -> None:
        self._last_used = value
        self.last_used_.emit(value)

    @property
    def range(self) -> Tuple[Tuple[float, float, float], ...]:
        return self._range

    @range.setter
    def range(self, value) -> None:
        self._range = value
        self.range_.emit(value)

    @property
    def current_step(self) -> Tuple[int, ...]:
        return self._current_step

    @current_step.setter
    def current_step(self, value) -> None:
        self._current_step = value
        self.current_step_.emit(value)

    @property
    def order(self) -> Tuple[int, ...]:
        return self._order

    @order.setter
    def order(self, value) -> None:
        if not set(value) == set(range(3)):
            raise ValueError(
                "Invalid ordering {order} for {ndim} dimensions".format(
                    order=value,
                    ndim=3,
                )
            )
        self._order = value
        self.order_.emit(value)

    @property
    def axis_labels(self) -> Tuple[str, ...]:
        return self._axis_labels

    @axis_labels.setter
    def axis_labels(self, value) -> None:
        self._axis_labels = value
        self.axis_labels_.emit(value)

    @property
    def nsteps(self) -> Tuple[int, ...]:
        return tuple(
            int(np.round((max_val - min_val) / step_size))
            for min_val, max_val, step_size in self.range
        )

    @property
    def point(self) -> Tuple[int, ...]:
        point = tuple(
            min_val + step_size * value
            for (min_val, _, step_size), value in zip(self.range, self.current_step)
        )
        return point

    @property
    def displayed(self) -> Tuple[int, ...]:
        return self.order[-self.ndisplay :]

    @property
    def not_displayed(self) -> Tuple[int, ...]:
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self) -> Tuple[int, ...]:
        return tuple(argsort(self.displayed))

    def set_range(
        self,
        axis: Union[int, Sequence[int]],
        _range: Union[
            Sequence[Union[int, float]], Sequence[Sequence[Union[int, float]]]
        ],
    ):
        if isinstance(axis, Integral):
            axis = assert_axis_in_bounds(axis)  # type: ignore
            if self.range[axis] != _range:
                full_range = list(self.range)
                full_range[axis] = _range
                self.range = full_range
        else:
            full_range = list(self.range)
            # cast range to list for list comparison below
            _range = list(_range)  # type: ignore
            axis = tuple(axis)  # type: ignore
            if len(axis) != len(_range):
                raise ValueError("axis and _range sequences must have equal length")
            if _range != full_range:
                for ax, r in zip(axis, _range):
                    ax = assert_axis_in_bounds(int(ax))
                    full_range[ax] = r
                self.range = full_range

    def set_current_step(
        self,
        axis: Union[int, Sequence[int]],
        value: Union[Union[int, float], Sequence[Union[int, float]]],
    ):
        if isinstance(axis, Integral):
            axis = assert_axis_in_bounds(axis)
            step = round(min(max(value, 0), self.nsteps[axis] - 1))
            if self.current_step[axis] != step:
                full_current_step = list(self.current_step)
                full_current_step[axis] = step
                self.current_step = full_current_step
        else:
            full_current_step = list(self.current_step)
            # cast value to list for list comparison below
            value = list(value)  # type: ignore
            axis = tuple(axis)  # type: ignore
            if len(axis) != len(value):
                raise ValueError("axis and value sequences must have equal length")
            if value != full_current_step:
                # (computed) nsteps property outside of the loop for efficiency
                nsteps = self.nsteps
                for ax, val in zip(axis, value):
                    ax = assert_axis_in_bounds(int(ax))
                    step = round(min(max(val, 0), nsteps[ax] - 1))
                    full_current_step[ax] = step
                self.current_step = full_current_step

    def reset(self):
        # Don't reset axis labels
        self.range = ((0, 2, 1),) * 3
        self.current_step = (0,) * 3
        self.order = tuple(range(3))

    def transpose(self):
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order

    def _increment_dims_right(self, axis: int = None):
        if axis is None:
            axis = self.last_used
        self.set_current_step(axis, self.current_step[axis] + 1)

    def _increment_dims_left(self, axis: int = None):
        if axis is None:
            axis = self.last_used
        self.set_current_step(axis, self.current_step[axis] - 1)

    def _focus_up(self):
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        index = (sliders.index(self.last_used) + 1) % len(sliders)
        self.last_used = sliders[index]

    def _focus_down(self):
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        index = (sliders.index(self.last_used) - 1) % len(sliders)
        self.last_used = sliders[index]

    def _roll(self):
        order = np.array(self.order)
        nsteps = np.array(self.nsteps)
        order[nsteps > 1] = np.roll(order[nsteps > 1], 1)
        self.order = order.tolist()

    def _go_to_center_step(self):
        self.current_step = [int((ns - 1) / 2) for ns in self.nsteps]


def assert_axis_in_bounds(axis: int) -> int:
    if axis not in range(-3, 3):
        msg = "Axis {axis} not defined for dimensionality {ndim}. Must be in [{ndim_lower}, {ndim}).".format(
            axis=axis,
            ndim=3,
            ndim_lower=-3,
        )
        raise ValueError(msg)

    return axis % 3
