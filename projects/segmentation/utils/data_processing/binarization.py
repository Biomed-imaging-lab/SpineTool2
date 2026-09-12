from math import ceil
from multiprocessing import Pool, Queue
from time import sleep

import numpy as np
from scipy.ndimage import median_filter

from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)

POOL_SIZE = 8


def _binarize_fragment(fragment) -> np.ndarray:
    if np.max(fragment[1]) == 0.0:
        return (fragment[0], fragment[1])

    local_median = median_filter(fragment[1], size=fragment[2])
    threshold = fragment[3] - fragment[4] * (fragment[3] - local_median)
    return (fragment[0], fragment[1] > threshold)


def binarize(
    data: np.ndarray,
    area: np.ndarray,
    parameters: dict,
    metadata: dict,
    queue_in: Queue,
    queue_out: Queue,
):
    try:
        fragment_size = 64
        block_size = parameters.get("block_size", 3)
        base_threshold = parameters.get("base_threshold", 127)
        weight = parameters.get("weight", 5)
        fill_holes_on_fix = bool(parameters.get("fill_holes_on_fix", False))
        metadata.update(
            {
                "params": {
                    "block_size": block_size,
                    "base_threshold": base_threshold,
                    "weight": weight,
                    "fill_holes_on_fix": fill_holes_on_fix,
                }
            }
        )
        weight /= 100
        shape = data.shape
        additional = int(block_size / 2) + 2
        x_count = ceil(shape[2] / fragment_size)
        y_count = ceil(shape[1] / fragment_size)

        if np.max(area) > 0:
            data = data * area

        fragments = []
        for j in range(y_count):
            for i in range(x_count):
                fragments.append(
                    [
                        (j, i),
                        data[
                            :,
                            max(0, j * fragment_size - additional) : min(
                                shape[1], (j + 1) * fragment_size + additional
                            ),
                            max(0, i * fragment_size - additional) : min(
                                shape[2], (i + 1) * fragment_size + additional
                            ),
                        ],
                        block_size,
                        base_threshold,
                        weight,
                    ]
                )

        if not queue_in.empty():
            queue_in.get()
            queue_out.put(Result(np.zeros_like(data), metadata, True))
            return

        pool = Pool(POOL_SIZE)
        result = pool.map_async(_binarize_fragment, fragments)

        sleep(CHECK_QUEUE_TIME_INTERVAL)
        while not result.ready() and queue_in.empty():
            sleep(CHECK_QUEUE_TIME_INTERVAL)

        if not queue_in.empty():
            queue_in.get()
            pool.terminate()
            pool.close()
            pool.join()
            queue_out.put(Result(np.zeros_like(data), metadata, True))
            return

        pool.close()
        pool.join()
        output = np.zeros_like(data)
        for r in result.get():
            y_l_shift = additional if r[0][0] > 0 else 0
            x_l_shift = additional if r[0][1] > 0 else 0
            y_r_shift = y_l_shift if r[0][0] < y_count - 1 else 0
            x_r_shift = x_l_shift if r[0][1] < x_count - 1 else 0
            output[
                :,
                r[0][0] * fragment_size : min(shape[1], (r[0][0] + 1) * fragment_size),
                r[0][1] * fragment_size : min(shape[2], (r[0][1] + 1) * fragment_size),
            ] = r[1][
                :,
                y_l_shift : min(r[1].shape[1] - y_r_shift, fragment_size + y_l_shift),
                x_l_shift : min(r[1].shape[2] - x_r_shift, fragment_size + x_l_shift),
            ]
        queue_out.put(Result(output, metadata))
    except Exception as e:
        queue_out.put(Result(np.zeros_like(data), metadata, error=str(e)))
