from multiprocessing import Queue

import numpy as np

from projects.segmentation.utils.data_processing.result import Result


def build_voxel_from_binary(
    data: np.ndarray,
    area: np.ndarray,
    metadata: dict,
    queue_in: Queue,
    queue_out: Queue,
) -> None:
    try:
        binary = (data > 0).astype(np.uint8)
        if np.max(area) > 0:
            roi = (area > 0).astype(np.uint8)
            binary = binary * roi

        if not queue_in.empty():
            queue_in.get()
            queue_out.put(Result(np.zeros_like(binary), metadata, True))
            return

        queue_out.put(Result(binary, metadata))
    except Exception as e:
        queue_out.put(Result(np.zeros_like(data, dtype=np.uint8), metadata, error=str(e)))
