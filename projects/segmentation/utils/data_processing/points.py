from datetime import datetime
from multiprocessing import Process, Queue
from time import sleep

import cv2
import numpy as np
from scipy import spatial
from skimage.measure import label

from projects.segmentation.utils.constants import TMP_AUXILIARY_PATH
from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)
from projects.segmentation.utils.data_processing.segmentation_utils import (
    find_medial,
    get_skelet,
)

NO_CONNECTED_COMPONENTS = "empty"


def _find_connected_components(data, scale, min_component_volume):
    labels = label(data, connectivity=1)

    unique_labels, counts = np.unique(labels, return_counts=True)
    unique_counts = {u: c for u, c in zip(unique_labels, counts)}
    unique_counts = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)

    voxel_volume = scale[0] * scale[1] * scale[2]
    min_component_size = min_component_volume // voxel_volume

    used_labels = []
    shaft_label = None
    for unique, count in unique_counts:
        if count >= min_component_size and unique != 0 and shaft_label is not None:
            used_labels.append(unique)
        elif count < min_component_size:
            break
        elif unique != 0:
            shaft_label = unique
    return labels, used_labels, shaft_label


def _get_medial(
    data: np.ndarray,
    scale: list,
    min_component_volume: float,
    folder: str,
    additional_files: dict,
    queue: Queue,
) -> None:
    try:
        labels, used_labels, shaft_label = _find_connected_components(
            data, scale, min_component_volume
        )

        if len(used_labels) == 0:
            queue.put((None, None, None, [1, 1, 1], False, NO_CONNECTED_COMPONENTS))
            return

        zoom = [1, 1, 1]
        if data.shape[1] > 512:
            zoom[1] = 2
        if data.shape[2] > 512:
            zoom[2] = 2

        medial_file = additional_files.get("medial", None)
        if medial_file:
            medial = np.fromfile(folder + medial_file, dtype=np.uint16)
            medial = np.reshape(medial, (-1, 3))
            queue.put((labels, used_labels, medial, zoom, False, ""))
        else:
            shaft = np.zeros_like(data)
            shaft[labels == shaft_label] = 1
            new_shape = [shaft.shape[2] // zoom[2], shaft.shape[1] // zoom[1]]
            if max(zoom) > 1:
                zoomed_shaft = np.zeros(
                    (shaft.shape[0], new_shape[1], new_shape[0]), dtype=np.uint8
                )
                for i in range(shaft.shape[0]):
                    zoomed_shaft[i] = cv2.resize(shaft[i], new_shape)
            else:
                zoomed_shaft = shaft.copy()

            skel = get_skelet(zoomed_shaft)
            del shaft
            del zoomed_shaft

            medial = np.array(find_medial(skel), dtype=np.uint16)
            del skel

            queue.put((labels, used_labels, medial, zoom, True, ""))
    except Exception as e:
        queue.put((None, None, None, [1, 1, 1], False, str(e)))


def _get_points(
    medial: np.ndarray,
    labels: np.ndarray,
    used_labels: list,
    max_distance: int,
    zoom: list,
    scale: list,
    queue: Queue,
) -> None:
    try:
        scale = np.asarray(scale, dtype=float)
        zoom = np.asarray(zoom, dtype=np.uint16)

        medial_voxels = medial.copy()
        medial_physical = medial_voxels * scale * zoom

        points = []
        shaft_points = []
        spine_component_ids = []

        for lbl in used_labels:
            spine_points = np.argwhere(labels == lbl) * scale
            average_spine_point = np.average(spine_points, axis=0)

            dists = spatial.distance.cdist([average_spine_point], medial_physical)
            _, idx2 = np.unravel_index(np.argmin(dists), dists.shape)

            if dists[0, idx2] > max_distance:
                continue

            spine_point = np.round(average_spine_point / scale).astype(np.uint16)
            shaft_point = np.round(medial_voxels[idx2] * zoom).astype(np.uint16)

            points.append(spine_point)
            shaft_points.append(shaft_point)
            spine_component_ids.append(int(lbl))

        points_arr = np.asarray(points, dtype=np.uint16).reshape((-1, 3))
        shaft_points_arr = np.asarray(shaft_points, dtype=np.uint16).reshape((-1, 3))

        queue.put((points_arr, shaft_points_arr, spine_component_ids, ""))

    except Exception as e:
        queue.put((None, None, None, str(e)))


def find_points(
    data: np.ndarray,
    scale: list,
    parameters: dict,
    folder: str,
    metadata: dict,
    additional_files: dict,
    queue_in: Queue,
    queue_out: Queue,
) -> None:
    try:
        min_component_volume = parameters.get("min_component_volume", -1.0)
        max_distance = parameters.get("max_distance", 5)

        metadata.update(
            {
                "params": {
                    "min_component_volume": min_component_volume,
                    "max_distance": max_distance,
                }
            }
        )

        min_component_volume = 10**min_component_volume

        if np.max(data) == 0:
            queue_out.put(Result(None, metadata))
            return

        queue = Queue()
        process = Process(
            target=_get_medial,
            args=(
                data,
                scale,
                min_component_volume,
                folder,
                additional_files,
                queue,
            ),
        )
        process.start()

        sleep(CHECK_QUEUE_TIME_INTERVAL)
        while queue.empty() and queue_in.empty():
            sleep(CHECK_QUEUE_TIME_INTERVAL)

        if not queue_in.empty():
            queue_in.get()
            process.terminate()
            process.join()
            queue.close()
            queue.join_thread()
            queue_out.put(Result(None, metadata, True))
            return

        labels, used_labels, medial, zoom, new_medial, error = queue.get()
        process.join()
        queue.close()
        queue.join_thread()

        if error != "":
            if error == NO_CONNECTED_COMPONENTS:
                queue_out.put(Result(None, metadata))
                return

            else:
                queue_out.put(Result(None, metadata, error=error))
                return

        if new_medial:
            medial_file = (
                TMP_AUXILIARY_PATH
                + f"/medial_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".bin"
            )
            medial.tofile(folder + medial_file)
        else:
            medial_file = additional_files["medial"]

        if not queue_in.empty():
            queue_in.get()
            queue_out.put(Result(None, metadata, True, {"medial": medial_file}))
            return

        queue = Queue()
        process = Process(
            target=_get_points,
            args=(
                medial,
                labels,
                used_labels,
                max_distance,
                zoom,
                scale,
                queue,
            ),
        )
        process.start()

        sleep(CHECK_QUEUE_TIME_INTERVAL)
        while queue.empty() and queue_in.empty():
            sleep(CHECK_QUEUE_TIME_INTERVAL)

        if not queue_in.empty():
            queue_in.get()
            process.terminate()
            process.join()
            queue.close()
            queue.join_thread()
            queue_out.put(Result(None, metadata, True, {"medial": medial_file}))
            return

        points, shaft_points, spine_component_ids, error = queue.get()
        process.join()
        queue.close()
        queue.join_thread()

        if error != "":
            queue_out.put(Result(None, metadata, False, {"medial": medial_file}, error))
            return

        if points is None:
            queue_out.put(Result(None, metadata, False, {"medial": medial_file}, error))
            return

        points = np.asarray(points, dtype=np.uint16).reshape((-1, 3))

        if shaft_points is None:
            shaft_points = np.zeros((0, 3), dtype=np.uint16)
        else:
            shaft_points = np.asarray(shaft_points, dtype=np.uint16).reshape((-1, 3))

        if spine_component_ids is None:
            spine_component_ids = []

        pair_metadata = {
            "neck_restoration_pairs_version": 1,
            "coordinate_order": "zyx",
            "spine_points": points.tolist(),
            "shaft_points": shaft_points.tolist(),
            "auto_shaft_points": shaft_points.tolist(),
            "pair_active": [True] * len(points),
            "pair_source": ["auto_old_medial"] * len(points),
            "pair_edit_state": "auto",
            "spine_component_ids": spine_component_ids,
            "shaft_component_ids": [None] * len(points),
        }

        metadata.update(pair_metadata)
        metadata.setdefault("params", {})
        metadata["params"].update(pair_metadata)

        queue_out.put(Result(points, metadata, False, {"medial": medial_file}, error))

    except Exception as e:
        queue_out.put(Result(None, metadata, error=str(e)))
