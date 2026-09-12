import os
from datetime import datetime
from multiprocessing import Pool, Queue
from time import sleep

import easy3d
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, binary_fill_holes, generate_binary_structure
from skimage.measure import label
from tifffile import imread, imwrite
from trimesh import Trimesh

from projects.segmentation.utils.constants import LAYERS_PATH, TMP_AUXILIARY_PATH
from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)

POOL_SIZE = 6


def _voxelizing(params) -> np.ndarray:
    spine_file, folder, shape, scale, off_file, min_coords = params
    min_z, min_h, min_w = shape

    emesh = easy3d.SurfaceMeshIO.load(folder + spine_file)
    copied_emesh = easy3d.SurfaceMesh(emesh)
    filler = easy3d.SurfaceMeshHoleFilling(copied_emesh)
    filler.fill_holes(20000)

    easy3d.SurfaceMeshIO.save(file_name=folder + off_file, mesh=copied_emesh)

    mesh: Trimesh = trimesh.load_mesh(folder + off_file)
    os.remove(folder + off_file)
    for v in mesh.vertices:
        v[0] = min(shape[0] - 1, max((v[0] - min_coords[0]) / scale[0], 0))
        v[1] = min(shape[1] - 1, max((v[1] - min_coords[1]) / scale[1], 0))
        v[2] = min(shape[2] - 1, max((v[2] - min_coords[2]) / scale[2], 0))
        z = int(v[0])
        h = int(v[1])
        w = int(v[2])
        min_z = z if z < min_z else min_z
        min_h = h if h < min_h else min_h
        min_w = w if w < min_w else min_w

    voxel_grid = mesh.voxelized(pitch=1)
    indices = voxel_grid.points_to_indices(voxel_grid.points)
    z, h, w = np.max(indices, axis=0) + 1
    voxel_image = np.zeros((z, h, w), dtype=np.uint8)
    for idx in indices:
        voxel_image[idx[0], idx[1], idx[2]] = 1

    padded = np.pad(
        voxel_image,
        (
            (min_z, max(shape[0] - min_z - z, 0)),
            (min_h, max(shape[1] - min_h - h, 0)),
            (min_w, max(shape[2] - min_w - w, 0)),
        ),
    )
    spine = binary_fill_holes(
        binary_dilation(padded, structure=generate_binary_structure(3, 3))
    )
    del padded

    return spine


def _voxelize_spine(params) -> np.ndarray:
    binary, spine_file, folder, shape, scale, min_coords, index = params
    off_file = (
        TMP_AUXILIARY_PATH
        + f"/spine_{index}_tmp_"
        + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        + ".off"
    )
    spine = _voxelizing([spine_file, folder, shape, scale, off_file, min_coords])
    spine = spine * binary

    return np.nonzero(spine)


def _voxelize_dendrite(params) -> str:
    mesh_file_path, folder, shape, scale, min_coords = params
    off_file = (
        TMP_AUXILIARY_PATH
        + f"/mesh_tmp_"
        + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        + ".off"
    )
    voxel_result = _voxelizing([mesh_file_path, folder, shape, scale, off_file, min_coords])

    tif_file = (
        LAYERS_PATH
        + f"/mesh_base_"
        + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        + ".tif"
    )
    imwrite(folder + tif_file, data=voxel_result)

    return tif_file


def voxelize(
    binary_filename: str,
    spines_files: list,
    folder: str,
    shape: tuple,
    scale: list,
    min_coord: list,
    metadata: dict,
    queue_in: Queue,
    queue_out: Queue,
) -> None:
    try:
        binary = imread(folder + binary_filename).astype(np.uint8)
        binary[binary > 0] = 1

        pool = Pool(POOL_SIZE)
        result = pool.map_async(
            _voxelize_spine,
            [
                [binary, file, folder, shape, scale, min_coord, i]
                for i, file in enumerate(spines_files)
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
            queue_out.put(Result(np.zeros_like(binary), metadata, True))
            return

        pool.close()
        pool.join()
        spines = np.zeros_like(binary)
        for r in result.get():
            spines[r] = 1
        result = np.zeros_like(binary)
        result[spines > 0] = 1
        result = binary - result
        labels = label(result, connectivity=1)
        unique_labels, counts = np.unique(labels, return_counts=True)
        for l, c in zip(unique_labels, counts):
            if c < 10000:
                result[labels == l] = 2
        result[spines > 0] = 2
        queue_out.put(Result(result, metadata))
    except Exception as e:
        queue_out.put(Result(np.zeros_like(binary), metadata, error=str(e)))
