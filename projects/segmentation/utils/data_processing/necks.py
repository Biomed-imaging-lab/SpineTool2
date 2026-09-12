from datetime import datetime
from itertools import chain, product
from multiprocessing import Pool, Process, Queue
from time import sleep

import cv2
import numpy as np
from scipy import spatial
from skimage.segmentation import morphological_chan_vese

from projects.segmentation.utils.constants import TMP_AUXILIARY_PATH
from projects.segmentation.utils.data_processing.graph import find_path
from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)
from projects.segmentation.utils.data_processing.segmentation_utils import (
    find_main_label,
    find_medial,
    get_skelet,
)

from projects.segmentation.utils.data_processing.paired_necks import (
    apply_neck_preview,
    build_local_roi,
    shortest_path_between_points,
    snap_to_valid_shaft_point,
    snap_to_valid_spine_point,
    validate_neck_mask,
)

POOL_SIZE = 8


def _get_medial_and_zoomed_image(
    image: np.ndarray,
    binarization: np.ndarray,
    folder: str,
    additional_files: dict,
    queue: Queue,
) -> None:
    try:
        zoom = [1, 1, 1]
        if image.shape[1] > 512:
            zoom[1] = 2
        if image.shape[2] > 512:
            zoom[2] = 2
        new_shape = [image.shape[2] // zoom[2], image.shape[1] // zoom[1]]

        medial_file = additional_files.get("medial", None)
        new_medial = not bool(medial_file)
        if medial_file:
            medial = np.fromfile(folder + medial_file, dtype=np.uint16)
            medial = np.reshape(medial, (-1, 3))
        else:
            labels, shaft_label = find_main_label(binarization)

            shaft = np.zeros_like(binarization)
            shaft[labels == shaft_label] = 1
            del labels

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

        if max(zoom) > 1:
            zoomed_image = np.zeros(
                (image.shape[0], new_shape[1], new_shape[0]), dtype=np.uint8
            )
            for i in range(image.shape[0]):
                zoomed_image[i] = cv2.resize(image[i], new_shape)
        else:
            zoomed_image = image.copy()

        queue.put((medial, zoomed_image, zoom, new_medial, ""))
    except Exception as e:
        queue.put((None, None, None, False, str(e)))


def _floodfill(img, path, size, factor):
    flooded = np.array([path[0]])
    box_coords = list(
        product(range(-size, size + 1), range(-size, size + 1), range(-size, size + 1))
    )
    z, h, w = img.shape
    for p in path:
        box = box_coords + p
        filter1 = np.all(box >= (0, 0, 0), axis=1)
        filter2 = np.all(box < (z, h, w), axis=1)
        box = box[filter1 & filter2]
        flooded = np.vstack(
            [
                flooded,
                box[
                    img[box[:, 0], box[:, 1], box[:, 2]]
                    >= img[p[0], p[1], p[2]] * factor
                ],
            ]
        )
    return flooded


def _acwe_floodfill(img, path, acwe_ext_factor, acwe_int_factor, radius):
    init_contour = np.zeros(img.shape, dtype=np.uint8)
    pipe = np.zeros(img.shape, dtype=np.uint8)
    box_coords = list(
        product(
            range(-radius, radius + 1),
            range(-radius, radius + 1),
            range(-radius, radius + 1),
        )
    )
    z, h, w = img.shape
    left, top, right, bot = 1e10, 1e10, -1, -1
    for p in path:
        init_contour[p[0]][p[1]][p[2]] = 1
        for coord in box_coords:
            k = p[0] + coord[0]
            i = p[1] + coord[1]
            j = p[2] + coord[2]
            if 0 <= i and i < h and 0 <= j and j < w and 0 <= k < z:
                pipe[k][i][j] = img[k][i][j]
                top = min(top, i)
                bot = max(bot, i)
                left = min(left, j)
                right = max(right, j)
    pipe = pipe[:, top : bot + 1, left : right + 1]
    init_contour = init_contour[:, top : bot + 1, left : right + 1]
    seg = morphological_chan_vese(
        pipe,
        num_iter=100,
        init_level_set=init_contour,
        lambda1=acwe_ext_factor,
        lambda2=acwe_int_factor,
        smoothing=0,
    )
    res = np.argwhere(seg > 0)
    return res + np.tile(np.array([0, top, left]), (len(res), 1))


def _get_neck(params) -> None:
    (
        zoomed_image,
        image,
        scale,
        zoom,
        medial,
        point,
        intensity_factor,
        use_acwe,
        floodfill_radius,
        floodfill_factor,
        acwe_ext_factor,
        acwe_int_factor,
    ) = params

    dists = spatial.distance.cdist([point * zoom * scale], medial * zoom * scale)
    _, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
    del dists
    medial_point = medial[idx2]
    path = find_path(zoomed_image, point, medial_point, intensity_factor)[:-1]
    path = np.array(path) * zoom
    if not use_acwe:
        neck = _floodfill(image, path, floodfill_radius, floodfill_factor)
    else:
        neck = _acwe_floodfill(
            image,
            path,
            acwe_ext_factor,
            acwe_int_factor,
            floodfill_radius,
        )

    return neck


def _get_fixed_endpoint_neck(
    image: np.ndarray,
    binarization: np.ndarray,
    spine_point: np.ndarray,
    shaft_point: np.ndarray,
    scale: list,
    intensity_factor: float,
    roi_margin: int,
    corridor_radius_um: float,
    use_acwe: bool,
    floodfill_radius: int,
    floodfill_factor: float,
    acwe_ext_factor: float,
    acwe_int_factor: float,
) -> tuple[np.ndarray, dict]:
    scale_zyx = np.asarray(scale, dtype=float)

    spine_point, spine_component_id = snap_to_valid_spine_point(
        spine_point,
        binarization,
        scale_zyx,
        max_distance_um=None,
    )
    shaft_point, shaft_component_id = snap_to_valid_shaft_point(
        shaft_point,
        binarization,
        scale_zyx,
        max_distance_um=None,
    )

    roi_slices, offset = build_local_roi(
        spine_point,
        shaft_point,
        image.shape,
        scale_zyx,
        roi_margin_voxels=roi_margin,
    )

    image_roi = image[roi_slices]
    bin_roi = binarization[roi_slices]

    local_start = spine_point.astype(np.int64) - offset
    local_target = shaft_point.astype(np.int64) - offset
    allowed_mask = np.ones_like(bin_roi, dtype=bool)

    path_local = shortest_path_between_points(
        image_roi=image_roi,
        start_zyx=local_start,
        target_zyx=local_target,
        scale_zyx=scale_zyx,
        allowed_mask=allowed_mask,
        forbidden_mask=None,
        intensity_factor=float(intensity_factor),
        corridor_radius_um=float(corridor_radius_um),
    )

    if use_acwe:
        neck_local = _acwe_floodfill(
            image_roi,
            path_local,
            acwe_ext_factor,
            acwe_int_factor,
            floodfill_radius,
        )
    else:
        neck_local = _floodfill(
            image_roi,
            path_local,
            floodfill_radius,
            floodfill_factor,
        )

    path_global = path_local.astype(np.int64) + offset
    neck_global = neck_local.astype(np.int64) + offset
    neck_mask = np.zeros_like(binarization, dtype=bool)
    neck_mask[tuple(neck_global.transpose())] = True

    validation = validate_neck_mask(
        binarization=binarization,
        neck_mask=neck_mask,
        spine_component_id=spine_component_id,
        shaft_component_id=shaft_component_id,
    )

    info = {
        "spine_point": spine_point.tolist(),
        "shaft_point": shaft_point.tolist(),
        "spine_component_id": int(spine_component_id),
        "shaft_component_id": int(shaft_component_id),
        "validation_ok": bool(validation.ok),
        "validation_warnings": validation.warnings,
        "path_length_voxels": int(len(path_global)),
    }

    return neck_mask, info


def run_two_points_restoration(
    image: np.ndarray,
    binarization: np.ndarray,
    points: np.ndarray,
    shaft_points: np.ndarray,
    pair_active: list | None,
    scale: list,
    intensity_factor: float,
    use_acwe: bool,
    floodfill_radius: int,
    floodfill_factor: float,
    acwe_ext_factor: float,
    acwe_int_factor: float,
    roi_margin: int,
    corridor_radius_um: float,
    metadata: dict,
    additional_files: dict,
    queue_in: Queue,
) -> Result:
    points_arr = np.asarray(points, dtype=np.uint16).reshape((-1, 3))
    shaft_arr = np.asarray(shaft_points, dtype=np.uint16).reshape((-1, 3))

    if points_arr.shape != shaft_arr.shape:
        return Result(
            np.zeros_like(binarization),
            metadata,
            error=(
                "shaft_points shape must match points shape: "
                f"{shaft_arr.shape} != {points_arr.shape}"
            ),
        )

    if pair_active is None:
        pair_active = [True] * len(points_arr)

    if len(pair_active) != len(points_arr):
        return Result(
            np.zeros_like(binarization),
            metadata,
            error=(
                "pair_active length must match points length: "
                f"{len(pair_active)} != {len(points_arr)}"
            ),
        )

    necks_mask = np.zeros_like(binarization, dtype=bool)
    pair_infos = []

    for i, (spine_point, shaft_point, active) in enumerate(
        zip(points_arr, shaft_arr, pair_active)
    ):
        if not queue_in.empty():
            queue_in.get()
            return Result(np.zeros_like(binarization), metadata, True)

        if not active:
            continue

        try:
            one_mask, info = _get_fixed_endpoint_neck(
                image=image,
                binarization=binarization,
                spine_point=spine_point,
                shaft_point=shaft_point,
                scale=scale,
                intensity_factor=intensity_factor,
                roi_margin=roi_margin,
                corridor_radius_um=corridor_radius_um,
                use_acwe=use_acwe,
                floodfill_radius=floodfill_radius,
                floodfill_factor=floodfill_factor,
                acwe_ext_factor=acwe_ext_factor,
                acwe_int_factor=acwe_int_factor,
            )
            info["pair_index"] = i
            pair_infos.append(info)

            if info["validation_ok"]:
                necks_mask |= one_mask
        except Exception as e:
            pair_infos.append(
                {
                    "pair_index": i,
                    "spine_point": np.asarray(spine_point).tolist(),
                    "shaft_point": np.asarray(shaft_point).tolist(),
                    "validation_ok": False,
                    "validation_warnings": [str(e)],
                }
            )

    metadata["neck_restoration_pairs"] = pair_infos
    metadata["coordinate_order"] = "zyx"

    result = apply_neck_preview(
        binarization=binarization,
        neck_mask=necks_mask,
        neck_label=2,
    )

    return Result(result, metadata, files=additional_files)


def run_one_point_restoration(
    image: np.ndarray,
    binarization: np.ndarray,
    points: np.ndarray,
    scale: list,
    intensity_factor: float,
    use_acwe: bool,
    floodfill_radius: int,
    floodfill_factor: float,
    acwe_ext_factor: float,
    acwe_int_factor: float,
    folder: str,
    metadata: dict,
    additional_files: dict,
    queue_in: Queue,
) -> Result:
    queue = Queue()
    process = Process(
        target=_get_medial_and_zoomed_image,
        args=(
            image,
            binarization,
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
        return Result(np.zeros_like(binarization), metadata, True)

    medial, zoomed_image, zoom, new_medial, error = queue.get()
    process.join()
    queue.close()
    queue.join_thread()

    if error != "":
        return Result(np.zeros_like(binarization), metadata, error=error)

    if new_medial:
        medial_file = (
            TMP_AUXILIARY_PATH
            + f"/medial_"
            + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
            + ".bin"
        )
        medial.tofile(folder + medial_file)
    else:
        medial_file = additional_files["medial"]

    if not queue_in.empty():
        queue_in.get()
        return Result(
            np.zeros_like(binarization),
            metadata,
            True,
            {"medial": medial_file},
        )

    points = points // zoom

    pool = Pool(POOL_SIZE)
    result = pool.map_async(
        _get_neck,
        [
            [
                zoomed_image,
                image,
                scale,
                zoom,
                medial,
                point,
                intensity_factor,
                use_acwe,
                floodfill_radius,
                floodfill_factor,
                acwe_ext_factor,
                acwe_int_factor,
            ]
            for point in points
        ],
    )

    sleep(CHECK_QUEUE_TIME_INTERVAL)
    while not result.ready() and queue_in.empty():
        sleep(CHECK_QUEUE_TIME_INTERVAL)

    if not queue_in.empty():
        queue_in.get()
        pool.terminate()
        pool.close()
        pool.join()
        return Result(
            np.zeros_like(binarization),
            metadata,
            True,
            {"medial": medial_file},
        )

    pool.close()
    pool.join()

    necks = np.zeros_like(binarization)
    for neck in result.get():
        necks[tuple(neck.transpose())] = 1

    return Result(
        binarization + (necks - necks * binarization) * 2,
        metadata,
        files={"medial": medial_file},
    )


def necks_reconnection(
    image: np.ndarray,
    binarization: np.ndarray,
    points: np.ndarray,
    scale: list,
    parameters: dict,
    folder: str,
    metadata: dict,
    additional_files: dict,
    queue_in: Queue,
    queue_out: Queue,
) -> None:
    try:
        intensity_factor = parameters.get("intensity_factor", 3)
        floodfill_radius = parameters.get("floodfill_radius", 4)
        floodfill_factor = parameters.get("floodfill_factor", 0.8)
        use_acwe = parameters.get("use_acwe", False)
        acwe_ext_factor = parameters.get("acwe_ext_factor", 2)
        acwe_int_factor = parameters.get("acwe_int_factor", 6)
        use_fixed_endpoints = parameters.get("use_fixed_endpoints", True)
        shaft_points = parameters.get("shaft_points", None)
        pair_active = parameters.get("pair_active", None)
        roi_margin = parameters.get("roi_margin", 8)
        corridor_radius_um = parameters.get("corridor_radius_um", 0.25)

        metadata.update(
            {
                "params": {
                    "intensity_factor": intensity_factor,
                    "floodfill_radius": floodfill_radius,
                    "floodfill_factor": floodfill_factor,
                    "use_acwe": use_acwe,
                    "acwe_ext_factor": acwe_ext_factor,
                    "acwe_int_factor": acwe_int_factor,
                    "use_fixed_endpoints": use_fixed_endpoints,
                    "roi_margin": roi_margin,
                    "corridor_radius_um": corridor_radius_um,
                }
            }
        )

        intensity_factor = 10**intensity_factor

        if np.max(binarization) == 0:
            queue_out.put(Result(np.zeros_like(binarization), metadata))
            return

        if len(points) == 0:
            queue_out.put(Result(binarization, metadata))
            return

        if use_fixed_endpoints and shaft_points is not None:
            result = run_two_points_restoration(
                image=image,
                binarization=binarization,
                points=points,
                shaft_points=shaft_points,
                pair_active=pair_active,
                scale=scale,
                intensity_factor=intensity_factor,
                use_acwe=use_acwe,
                floodfill_radius=floodfill_radius,
                floodfill_factor=floodfill_factor,
                acwe_ext_factor=acwe_ext_factor,
                acwe_int_factor=acwe_int_factor,
                roi_margin=roi_margin,
                corridor_radius_um=corridor_radius_um,
                metadata=metadata,
                additional_files=additional_files,
                queue_in=queue_in,
            )
        else:
            result = run_one_point_restoration(
                image=image,
                binarization=binarization,
                points=points,
                scale=scale,
                intensity_factor=intensity_factor,
                use_acwe=use_acwe,
                floodfill_radius=floodfill_radius,
                floodfill_factor=floodfill_factor,
                acwe_ext_factor=acwe_ext_factor,
                acwe_int_factor=acwe_int_factor,
                folder=folder,
                metadata=metadata,
                additional_files=additional_files,
                queue_in=queue_in,
            )

        queue_out.put(result)
    except Exception as e:
        queue_out.put(
            Result(
                np.zeros_like(binarization),
                metadata,
                error=str(e),
            )
        )
