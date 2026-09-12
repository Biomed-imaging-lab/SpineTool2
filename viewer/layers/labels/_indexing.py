from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt


def elements_in_slice(
    index: Tuple[npt.NDArray[np.int_], ...], position_in_axes: Dict[int, int]
) -> npt.NDArray[np.bool_]:
    queries = [index[ax] == position for ax, position in position_in_axes.items()]
    return np.logical_and.reduce(queries, axis=0)


def index_in_slice(
    index: Tuple[npt.NDArray[np.int_], ...],
    position_in_axes: Dict[int, int],
    indices_order: Tuple[int, ...],
) -> Tuple[npt.NDArray[np.int_], ...]:
    index_in_slice = elements_in_slice(index, position_in_axes)
    return tuple(
        index[i][index_in_slice] for i in indices_order if i not in position_in_axes
    )
