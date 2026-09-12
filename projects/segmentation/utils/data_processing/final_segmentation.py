import os
from datetime import datetime
from itertools import product
from json import dump
from multiprocessing import Lock, Process, Queue
from time import sleep

import numpy as np
import point_cloud_utils as pcu
from meshlib import mrmeshpy as mm
from projects.segmentation.utils.data_processing.mesh import voxel_to_mesh
from projects.segmentation.utils.data_processing.voxel_cleanup import (
    fill_small_enclosed_holes,
)
from projects.segmentation.utils.constants import DEFAULT_MESH_COMPLEXITY
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from projects.segmentation.utils.constants import TMP_AUXILIARY_PATH, TMP_LAYERS_PATH
from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)
from projects.segmentation.utils.data_processing.segmentation_utils import (
    find_main_label,
    get_spine_meshes,
    hash_point,
    point_to_list,
)
from projects.segmentation.utils.project_info import FinalSegmentationData


def _segmenation_data_to_v_f_vv(
    mesh: Polyhedron_3, spines: dict, shape, scale, min_coord
) -> tuple:
    with np.errstate(all="ignore"):
        spines_indices = {}
        for id in spines.keys():
            spines_indices[id] = []

        vertices = np.ndarray((mesh.size_of_vertices(), 3)).astype(np.uint16)
        vertex_values = np.ndarray(mesh.size_of_vertices()).astype(np.float16)
        for i, vertex in enumerate(mesh.vertices()):
            vertex.set_id(i)
            vertices[i, :] = point_to_list(vertex.point(), shape, scale, min_coords=min_coord)
            hp = hash_point(vertex.point())
            spine_id = -1
            for id, points in spines.items():
                if hp in points:
                    spine_id = id
                    break
            if spine_id != -1:
                vertex_values[i] = 1.0
                spines_indices[spine_id].append(i)
            else:
                vertex_values[i] = 0.0

        facets = np.ndarray((mesh.size_of_facets(), 3)).astype(np.uint32)
        for i, facet in enumerate(mesh.facets()):
            circulator = facet.facet_begin()
            j = 0
            begin = facet.facet_begin()
            while circulator.hasNext():
                halfedge = circulator.next()
                v = halfedge.vertex()
                facets[i, j] = v.id()
                j += 1
                if circulator == begin or j == 3:
                    break

        return (vertices, facets, vertex_values), spines_indices


def _save_off(vertices: np.ndarray, facets: np.ndarray, file_name: str):
    with open(file_name, "w") as f:
        f.write("OFF\n")
        f.write(f"{vertices.shape[0]} {facets.shape[0]} 0\n\n")
        for i in range(vertices.shape[0]):
            f.write(f"{vertices[i][0]} {vertices[i][1]} {vertices[i][2]}\n")
        for i in range(facets.shape[0]):
            f.write(f"3  {facets[i][0]} {facets[i][1]} {facets[i][2]}\n")


def _reconstruct_surface(
    data: np.ndarray, shape: tuple, scale: list, min_coord: list, folder: str,
    queue: Queue, lock, mesh_complexity: str, fill_holes_before_mesh: bool,
):
    try:
        with np.errstate(all="ignore"):
            if fill_holes_before_mesh:
                data = fill_small_enclosed_holes(data)
            _, shaft_label = find_main_label(data, True)

            data_ = np.zeros_like(data, dtype=np.uint8)
            data_[data > 0] = 1
            surface_poly, _ = voxel_to_mesh(
                data_, shape, (1, 1, 1), folder, mesh_complexity
            )

            size = 3
            box_coords = list(
                product(
                    range(-size, size + 1),
                    range(-size, size + 1),
                    range(-size, size + 1),
                )
            )
            segmentation = set()
            for v in surface_poly.vertices():
                p = np.array(
                    [
                        min(shape[0] - 1, max(round(v.point().x()), 0)),
                        min(shape[1] - 1, max(round(v.point().y()), 0)),
                        min(shape[2] - 1, max(round(v.point().z()), 0)),
                    ],
                    dtype=np.uint16,
                )
                v.set_point(
                    Point_3(
                        v.point().x() * scale[0] + min_coord[0],
                        v.point().y() * scale[1] + min_coord[1],
                        v.point().z() * scale[2] + min_coord[2],
                    )
                )
                box = box_coords + p
                filter1 = np.all(box >= (0, 0, 0), axis=1)
                filter2 = np.all(box < tuple(shape), axis=1)
                box = box[filter1 & filter2]
                if (
                    len(
                        box[
                            np.array(
                                data[box[:, 0], box[:, 1], box[:, 2]] == shaft_label
                            )
                        ]
                    )
                    < 3
                    * len(box[np.array(data[box[:, 0], box[:, 1], box[:, 2]] > 0)])
                    / 4
                ):
                    segmentation.add(hash_point(v.point()))

            spine_meshes = get_spine_meshes(surface_poly, segmentation)

            spines = {}
            average = []
            for i, spine in enumerate(spine_meshes):
                spine_vertices = np.ndarray((spine.size_of_vertices(), 3)).astype(
                    np.uint16
                )
                cur_spine_points = set()
                for j, p in enumerate(spine.points()):
                    spine_vertices[j, :] = point_to_list(p, shape, scale, min_coords = min_coord)
                    cur_spine_points.add(hash_point(p))
                spines[i] = cur_spine_points
                average.append(np.average(spine_vertices, axis=0).tolist())
            mesh_v_f_vv, spines_indices = _segmenation_data_to_v_f_vv(
                surface_poly, spines, shape, scale, min_coord
            )

            spines = {
                "spines": {},
                "pos_to_id": {},
            }
            spines_files = []
            spine_ids = list(spines_indices.keys())
            spine_ids.sort()
            files = {}
            for i, id in enumerate(spine_ids):
                filename = (
                    TMP_LAYERS_PATH
                    + f"/final_spine_{id}_"
                    + str(datetime.now())
                    .replace(".", "_")
                    .replace(" ", "_")
                    .replace(":", "_")
                    + ".off"
                )
                spines["spines"][id] = {
                    "id": id,
                    "pos": i,
                    "average": average[id],
                    "indices": spines_indices[id],
                }
                files[id] = filename
                spines["pos_to_id"][i] = id
                spines_files.append(filename)

            mesh_filename = (
                TMP_LAYERS_PATH
                + f"/final_surface_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".off"
            )
            lock.acquire()
            surface_poly.write_to_file(folder + mesh_filename)

            for i, spine in enumerate(spine_meshes):
                spine.write_to_file(folder + files[i])

            additional_file = (
                TMP_AUXILIARY_PATH
                + "/spines_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".json"
            )
            f = open(folder + additional_file, "w")
            dump(spines, f)
            f.close()

            queue.put(
                (
                    FinalSegmentationData(mesh_filename, mesh_v_f_vv, spines_files),
                    {"spines": additional_file},
                    "",
                )
            )
            lock.release()
    except Exception as e:
        queue.put((FinalSegmentationData(), {}, str(e)))


def build_final_segmentation(
    data: np.ndarray,
    shape: tuple,
    scale: list,
    min_coord: list,
    folder: str,
    metadata: dict,
    queue_in: Queue,
    queue_out: Queue,
    mesh_complexity: str = DEFAULT_MESH_COMPLEXITY,
    fill_holes_before_mesh: bool = True,
):
    try:
        lock = Lock()
        queue = Queue()
        process = Process(
            target=_reconstruct_surface,
            args=(
                data, shape, scale, min_coord, folder, queue, lock,
                mesh_complexity, fill_holes_before_mesh,
            ),
        )
        process.start()

        sleep(CHECK_QUEUE_TIME_INTERVAL)
        while queue.empty() and queue_in.empty():
            sleep(CHECK_QUEUE_TIME_INTERVAL)

        if not queue_in.empty():
            queue_in.get()
            lock.acquire()
            process.terminate()
            process.join()
            lock.release()
            if not queue.empty():
                result, files, error = queue.get()
                queue.close()
                queue.join_thread()
                queue_out.put(Result(result, metadata, False, files, error))
                return
            queue.close()
            queue.join_thread()
            queue_out.put(Result(FinalSegmentationData(), metadata, True))
        else:
            result, files, error = queue.get()
            queue.close()
            queue.join_thread()
            process.join()
            queue_out.put(Result(result, metadata, False, files, error))
    except Exception as e:
        queue_out.put(Result(FinalSegmentationData(), metadata, error=str(e)))
