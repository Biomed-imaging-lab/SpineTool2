from datetime import datetime
from json import dump, load
from multiprocessing import Lock, Process, Queue
from time import sleep
from typing import List

import numpy as np

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from projects.segmentation.utils.constants import TMP_AUXILIARY_PATH, TMP_LAYERS_PATH
from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)
from projects.segmentation.utils.data_processing.segmentation_utils import (
    correct_segmentation,
    get_spine_meshes,
    hash_point,
    point_to_list,
    segmentation_by_distance,
)
from projects.segmentation.utils.project_info import SegmentationData


def _mesh_list_to_v_f(meshes: List[Polyhedron_3], ids: list, shape, scale, min_coord) -> tuple:
    with np.errstate(all="ignore"):
        spines_indices = {}
        intersecting_spines = {}

        count = 0
        average = []
        vertices = np.empty((0, 3), dtype=np.uint16)
        facets = np.empty((0, 3), dtype=np.uint32)
        vertex_to_id = {}
        for id, mesh in zip(ids, meshes):
            new_vertices = set()
            intersecting_spines[id] = set()
            spines_indices[id] = set()
            mesh_vertices = np.ndarray((mesh.size_of_vertices(), 3)).astype(np.uint16)
            new_mesh_vertices = np.ndarray((mesh.size_of_vertices(), 3)).astype(
                np.uint16
            )
            for i, vertex in enumerate(mesh.vertices()):
                hashed_point = hash_point(vertex.point())
                if hashed_point in vertex_to_id:
                    vertex_id = vertex_to_id[hashed_point]["id"]
                    vertex.set_id(vertex_id)
                    spines_indices[id].add(vertex_id)
                    for prev_spine_id in vertex_to_id[hashed_point]["spines_id"]:
                        intersecting_spines[id].add(prev_spine_id)
                        intersecting_spines[prev_spine_id].add(id)
                    vertex_to_id[hashed_point]["spines_id"].add(id)
                else:
                    new_id = len(new_vertices) + count
                    vertex_to_id[hashed_point] = {"id": new_id, "spines_id": set([id])}
                    vertex.set_id(new_id)
                    spines_indices[id].add(new_id)
                    new_mesh_vertices[len(new_vertices), :] = point_to_list(
                        vertex.point(), shape, scale, min_coords=min_coord
                    )
                    new_vertices.add(new_id)
                mesh_vertices[i, :] = point_to_list(vertex.point(), shape, scale, min_coords=min_coord)
            average.append(np.average(mesh_vertices, axis=0).tolist())
            mesh_facets = np.ndarray((mesh.size_of_facets(), 3)).astype(np.uint32)
            new_facets_count = 0
            for facet in mesh.facets():
                new_facet = False
                circulator = facet.facet_begin()
                j = 0
                begin = facet.facet_begin()
                while circulator.hasNext():
                    halfedge = circulator.next()
                    v = halfedge.vertex()
                    mesh_facets[new_facets_count, j] = v.id()
                    if v.id() in new_vertices:
                        new_facet = True
                    j += 1
                    if circulator == begin or j == 3:
                        break
                if new_facet:
                    new_facets_count += 1
            if len(new_vertices) > 0:
                vertices = np.concatenate(
                    [vertices, new_mesh_vertices[: len(new_vertices), :]], axis=0
                )
                count += len(new_vertices)
            if new_facets_count > 0:
                facets = np.concatenate(
                    [facets, mesh_facets[:new_facets_count, :]], axis=0
                )

        return (vertices, facets), spines_indices, average, intersecting_spines


def _segmentation(
    mesh_file: str,
    shape: tuple,
    scale: list,
    min_coord: list,
    sensitivity: float,
    correction: int,
    min_volume: float,
    folder: str,
    additional_files: dict,
    queue: Queue,
    lock,
) -> None:
    try:
        surface_poly = Polyhedron_3(folder + mesh_file)

        with open(folder + additional_files["descriptions"]) as f:
            descriptions = load(f)

        segmentation = segmentation_by_distance(
            surface_poly,
            descriptions["correspondence"],
            descriptions["dendrite_skeleton_points"],
            descriptions["all_distance"],
            sensitivity,
        )
        segmentation = correct_segmentation(segmentation, surface_poly, correction)
        spine_meshes = get_spine_meshes(surface_poly, segmentation, min_volume)

        spine_ids = [i for i in range(len(spine_meshes))]
        mesh_v_f, spines_indices, average, _ = _mesh_list_to_v_f(
            spine_meshes, spine_ids, shape, scale, min_coord
        )

        spines = {"spines": {}, "pos_to_id": {}, "new_spine_id": len(spine_meshes)}
        spines_files = []
        files = {}
        for id in spine_ids:
            filename = (
                TMP_LAYERS_PATH
                + f"/spine_{id}_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".off"
            )
            spines["spines"][id] = {
                "id": id,
                "parent_id": None,
                "pos": id,
                "correction": 0,
                "average": average[id],
                "indices": list(spines_indices[id]),
                "intersecting_spines": [],
                "child_spines": [],
            }
            files[id] = filename
            spines["pos_to_id"][id] = id
            spines_files.append(filename)

        lock.acquire()
        for i, spine in enumerate(spine_meshes):
            spine.write_to_file(folder + files[i])

        additional_file = (
            TMP_AUXILIARY_PATH
            + "/spines_"
            + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
            + ".json"
        )
        f = open(folder + additional_file, "w")
        dump(spines, f)
        f.close()

        queue.put(
            (
                SegmentationData(spines_files, spines_files, mesh_v_f),
                {"spines": additional_file},
                "",
            )
        )
        lock.release()
    except Exception as e:
        queue.put((SegmentationData(), {}, str(e)))


def segment_spines(
    mesh_file: str,
    shape: tuple,
    scale: list,
    min_coord: list,
    parameters: dict,
    metadata: dict,
    folder: str,
    additional_files: dict,
    queue_in: Queue,
    queue_out: Queue,
) -> None:
    try:
        sensitivity = parameters.get("sensitivity", -1.0)
        correction = parameters.get("correction", 0)
        min_volume = parameters.get("min_volume", -1.0)
        metadata.update(
            {
                "params": {
                    "sensitivity": sensitivity,
                    "correction": correction,
                    "min_volume": min_volume,
                }
            }
        )
        sensitivity = 1 - 10**sensitivity
        min_volume = 10**min_volume

        lock = Lock()
        queue = Queue()
        process = Process(
            target=_segmentation,
            args=(
                mesh_file,
                shape,
                scale,
                min_coord,
                sensitivity,
                correction,
                min_volume,
                folder,
                additional_files,
                queue,
                lock,
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
            queue_out.put(Result(SegmentationData(), metadata, True))
        else:
            result, files, error = queue.get()
            queue.close()
            queue.join_thread()
            process.join()
            queue_out.put(Result(result, metadata, False, files, error))
    except Exception as e:
        queue_out.put(Result(SegmentationData(), metadata, error=str(e)))


def _spine_correction(
    mesh_file: str,
    spines_files: List[str],
    adjusted_spines_files: List[str],
    spine_id: int,
    deleted_spines: set,
    shape: tuple,
    scale: list,
    min_coord: list,
    correction: int,
    folder: str,
    additional_files: dict,
    queue: Queue,
    lock,
) -> None:
    try:
        with open(folder + additional_files["spines"]) as f:
            spines = load(f)

        spine_id = spines["pos_to_id"][str(spine_id)]

        child_spines = spines["spines"][str(spine_id)]["child_spines"]
        for child_spine in child_spines:
            pos = spines["spines"][str(child_spine)]["pos"]
            if pos in deleted_spines:
                deleted_spines.remove(pos)
            del spines["spines"][str(child_spine)]
        spines["spines"][str(spine_id)]["child_spines"] = []

        new_deleted_spines_id = set()
        for deleted_spine in deleted_spines:
            new_deleted_spines_id.add(spines["pos_to_id"][str(deleted_spine)])

        files = {}
        adjusted_files = {}
        for spine in spines["spines"].values():
            files[spine["id"]] = spines_files[spine["pos"]]
            adjusted_files[spine["id"]] = adjusted_spines_files[spine["pos"]]

        surface_poly = Polyhedron_3(folder + mesh_file)
        spine = Polyhedron_3(folder + files[spine_id])

        segmentation = set()
        for v in spine.vertices():
            segmentation.add(hash_point(v.point()))
        segmentation = correct_segmentation(segmentation, surface_poly, correction)
        new_spine_meshes = get_spine_meshes(surface_poly, segmentation)

        if len(new_spine_meshes) == 1:
            new_deleted_spines = set()
            new_spines_files = []
            new_adjusted_spines_files = []
            spine_meshes = []
            ids = []
            adjusted_files[spine_id] = ""
            for id, file in adjusted_files.items():
                if file != "":
                    spine_meshes.append(Polyhedron_3(folder + file))
                    ids.append(id)
                elif id == spine_id:
                    spine_meshes.append(new_spine_meshes[0])
                    ids.append(id)
            mesh_v_f, spines_indices, average, intersecting_spines = _mesh_list_to_v_f(
                spine_meshes, ids, shape, scale, min_coord
            )

            adjusted_spine_file = (
                TMP_LAYERS_PATH
                + f"/spine_{spine_id}_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".off"
            )
            adjusted_files[spine_id] = adjusted_spine_file

            spines["pos_to_id"] = {}
            spines["spines"][str(spine_id)]["average"] = average[ids.index(spine_id)]
            spine_ids = list(spines["spines"].keys())
            spine_ids.sort(key=lambda x: int(x))
            for i, id in enumerate(spine_ids):
                int_id = int(id)
                if int_id in spines_indices:
                    spines["spines"][id]["indices"] = list(spines_indices[int_id])
                else:
                    spines["spines"][id]["indices"] = []
                if int_id in intersecting_spines:
                    spines["spines"][id]["intersecting_spines"] = list(
                        intersecting_spines[int_id]
                    )
                else:
                    spines["spines"][id]["intersecting_spines"] = []
                spines["spines"][id]["pos"] = i
                spines["pos_to_id"][i] = int_id
                new_spines_files.append(files[int_id])
                new_adjusted_spines_files.append(adjusted_files[int_id])
                if int_id in new_deleted_spines_id:
                    new_deleted_spines.add(i)
            spines["spines"][str(spine_id)]["correction"] = correction

            lock.acquire()
            new_spine_meshes[0].write_to_file(folder + adjusted_spine_file)
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
                    SegmentationData(
                        new_spines_files,
                        new_adjusted_spines_files,
                        mesh_v_f,
                        new_deleted_spines,
                    ),
                    {"spines": additional_file},
                    "",
                )
            )
            lock.release()
        elif len(new_spine_meshes) > 1:
            new_spines_ids = []
            for i in range(len(new_spine_meshes)):
                new_id = spines["new_spine_id"]
                spines["new_spine_id"] += 1
                new_spines_ids.append(new_id)
                spines["spines"][str(new_id)] = {
                    "id": new_id,
                    "parent_id": spine_id,
                    "pos": None,
                    "correction": 0,
                    "average": None,
                    "indices": None,
                    "intersecting_spines": [],
                    "child_spines": [],
                }
                files[new_id] = ""
                adjusted_files[new_id] = ""

            new_deleted_spines = set()
            new_spines_files = []
            new_adjusted_spines_files = []
            spine_meshes = []
            ids = []
            adjusted_files[spine_id] = ""
            for id, file in adjusted_files.items():
                if file != "":
                    spine_meshes.append(Polyhedron_3(folder + file))
                    ids.append(id)
            spine_meshes.extend(new_spine_meshes)
            ids.extend(new_spines_ids)
            mesh_v_f, spines_indices, average, intersecting_spines = _mesh_list_to_v_f(
                spine_meshes, ids, shape, scale, min_coord
            )

            for id in new_spines_ids:
                filename = (
                    TMP_LAYERS_PATH
                    + f"/spine_{id}_"
                    + str(datetime.now())
                    .replace(".", "_")
                    .replace(" ", "_")
                    .replace(":", "_")
                    + ".off"
                )
                files[id] = filename
                adjusted_files[id] = filename
                spines["spines"][str(id)]["average"] = average[ids.index(id)]

            spines["pos_to_id"] = {}
            spine_ids = list(spines["spines"].keys())
            spine_ids.sort(key=lambda x: int(x))
            for i, id in enumerate(spine_ids):
                int_id = int(id)
                if int_id in spines_indices:
                    spines["spines"][id]["indices"] = list(spines_indices[int_id])
                else:
                    spines["spines"][id]["indices"] = []
                if int_id in intersecting_spines:
                    spines["spines"][id]["intersecting_spines"] = list(
                        intersecting_spines[int_id]
                    )
                else:
                    spines["spines"][id]["intersecting_spines"] = []
                spines["spines"][id]["pos"] = i
                spines["pos_to_id"][i] = int_id
                new_spines_files.append(files[int_id])
                new_adjusted_spines_files.append(adjusted_files[int_id])
                if int_id in new_deleted_spines_id:
                    new_deleted_spines.add(i)
            spines["spines"][str(spine_id)]["correction"] = correction
            spines["spines"][str(spine_id)]["child_spines"] = new_spines_ids

            lock.acquire()
            for i, id in enumerate(new_spines_ids):
                new_spine_meshes[i].write_to_file(folder + files[id])
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
                    SegmentationData(
                        new_spines_files,
                        new_adjusted_spines_files,
                        mesh_v_f,
                        new_deleted_spines,
                    ),
                    {"spines": additional_file},
                    "",
                )
            )
            lock.release()
        else:
            new_deleted_spines = set()
            new_spines_files = []
            new_adjusted_spines_files = []
            spine_meshes = []
            ids = []
            adjusted_files[spine_id] = ""
            for id, file in adjusted_files.items():
                if file != "":
                    spine_meshes.append(Polyhedron_3(folder + file))
                    ids.append(id)
            mesh_v_f, spines_indices, _, intersecting_spines = _mesh_list_to_v_f(
                spine_meshes, ids, shape, scale, min_coord
            )
            spines["pos_to_id"] = {}
            spine_ids = list(spines["spines"].keys())
            spine_ids.sort(key=lambda x: int(x))
            for i, id in enumerate(spine_ids):
                int_id = int(id)
                if int_id in spines_indices:
                    spines["spines"][id]["indices"] = list(spines_indices[int_id])
                else:
                    spines["spines"][id]["indices"] = []
                if int_id in intersecting_spines:
                    spines["spines"][id]["intersecting_spines"] = list(
                        intersecting_spines[int_id]
                    )
                else:
                    spines["spines"][id]["intersecting_spines"] = []
                spines["spines"][id]["pos"] = i
                spines["pos_to_id"][i] = int_id
                new_spines_files.append(files[int_id])
                new_adjusted_spines_files.append(adjusted_files[int_id])
                if int_id in new_deleted_spines_id:
                    new_deleted_spines.add(i)
            spines["spines"][str(spine_id)]["indices"] = []
            spines["spines"][str(spine_id)]["correction"] = correction

            lock.acquire()
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
                    SegmentationData(
                        new_spines_files,
                        new_adjusted_spines_files,
                        mesh_v_f,
                        new_deleted_spines,
                    ),
                    {"spines": additional_file},
                    "",
                )
            )
            lock.release()
    except Exception as e:
        queue.put((SegmentationData(), {}, str(e)))


def correct_spine(
    mesh_file: str,
    spines_files: List[str],
    adjusted_spines_files: List[str],
    spine_id: int,
    deleted_spines: set,
    correction: int,
    shape: tuple,
    scale: list,
    min_coord: list,
    metadata: dict,
    folder: str,
    additional_files: dict,
    queue_in: Queue,
    queue_out: Queue,
) -> None:
    try:
        lock = Lock()
        queue = Queue()
        process = Process(
            target=_spine_correction,
            args=(
                mesh_file,
                spines_files,
                adjusted_spines_files,
                spine_id,
                deleted_spines,
                shape,
                scale,
                min_coord,
                correction,
                folder,
                additional_files,
                queue,
                lock,
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
            queue_out.put(
                Result(
                    SegmentationData(),
                    metadata,
                    True,
                )
            )
        else:
            result, files, error = queue.get()
            queue.close()
            queue.join_thread()
            process.join()
            queue_out.put(Result(result, metadata, False, files, error))
    except Exception as e:
        queue_out.put(
            Result(
                SegmentationData(),
                metadata,
                error=str(e),
            )
        )


def fix_segmentation(
    spines_files: List[str],
    adjusted_spines_files: List[str],
    shape: tuple,
    scale: list,
    min_coord: list,
    folder: str,
    additional_files: dict,
    deleted_spines: list,
) -> tuple:
    with open(folder + additional_files["spines"]) as f:
        spines = load(f)

    deleted_spines_id = set()
    for deleted_spine in deleted_spines:
        id = spines["pos_to_id"][str(deleted_spine)]
        del spines["spines"][str(id)]
        deleted_spines_id.add(id)

    additional_deleted_spines_id = set()
    files = {}
    adjusted_files = {}
    for spine in spines["spines"].values():
        spine["parent_id"] = None
        if len(spine["child_spines"]) > 0 or adjusted_spines_files[spine["pos"]] == "":
            additional_deleted_spines_id.add(spine["id"])
        else:
            files[spine["id"]] = spines_files[spine["pos"]]
            adjusted_files[spine["id"]] = adjusted_spines_files[spine["pos"]]

            intersecting_spines_id = spine.get("intersecting_spines", [])
            new_intersecting_spines_id = []
            for intersecting_spine_id in intersecting_spines_id:
                if intersecting_spine_id not in deleted_spines_id:
                    new_intersecting_spines_id.append(intersecting_spine_id)
            spine["intersecting_spines"] = new_intersecting_spines_id
    for additional_deleted_spine in additional_deleted_spines_id:
        del spines["spines"][str(additional_deleted_spine)]

    new_spines_files = []
    new_adjusted_spines_files = []
    spine_meshes = []
    ids = []
    for id, file in adjusted_files.items():
        spine_meshes.append(Polyhedron_3(folder + file))
        ids.append(id)
    mesh_v_f, spines_indices, _, _ = _mesh_list_to_v_f(spine_meshes, ids, shape, scale, min_coord)

    spines["pos_to_id"] = {}
    spine_ids = list(spines["spines"].keys())
    spine_ids.sort(key=lambda x: int(x))
    for i, id in enumerate(spine_ids):
        int_id = int(id)
        spines["spines"][id]["indices"] = list(spines_indices[int_id])
        spines["spines"][id]["pos"] = i
        spines["pos_to_id"][i] = int_id
        new_spines_files.append(files[int_id])
        new_adjusted_spines_files.append(adjusted_files[int_id])

    additional_file = (
        TMP_AUXILIARY_PATH
        + "/spines_"
        + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        + ".json"
    )
    f = open(folder + additional_file, "w")
    dump(spines, f)
    f.close()

    return (
        SegmentationData(new_spines_files, new_adjusted_spines_files, mesh_v_f),
        {"spines": additional_file},
    )
