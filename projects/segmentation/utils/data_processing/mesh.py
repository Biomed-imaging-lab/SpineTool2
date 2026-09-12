import os
from datetime import datetime
from json import dump
from multiprocessing import Lock, Process, Queue
from time import sleep

import numpy as np
import point_cloud_utils as pcu
from meshlib import mrmeshpy as mm
from scipy.ndimage import binary_erosion, generate_binary_structure, binary_fill_holes, binary_closing, binary_dilation
from tifffile import imwrite
from skimage.measure import marching_cubes

from CGAL.CGAL_Polygon_mesh_processing import Polylines, stitch_borders, triangulate_and_refine_hole, triangulate_hole, triangulate_refine_and_fair_hole
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
from projects.segmentation.utils.constants import (
    DEFAULT_MESH_COMPLEXITY,
    MESH_COMPLEXITY_PROFILES,
    TMP_AUXILIARY_PATH,
    TMP_LAYERS_PATH,
)
from projects.segmentation.utils.data_processing.voxel_cleanup import (
    fill_small_enclosed_holes,
)
from projects.segmentation.utils.data_processing.result import (
    CHECK_QUEUE_TIME_INTERVAL,
    Result,
)
from projects.segmentation.utils.data_processing.segmentation_utils import (
    build_correspondence,
    build_graph,
    build_reverse_correpondence,
    find_longest_path,
    find_main_label,
    get_path_statistics,
    point_to_list,
)
from projects.segmentation.utils.project_info import SurfaceData


def _mesh_to_v_f(mesh: Polyhedron_3, shape, scale, min_coords=[0.0, 0.0, 0.0]) -> tuple:
    with np.errstate(all="ignore"):
        vertices = np.ndarray((mesh.size_of_vertices(), 3)).astype(np.uint16)
        for i, vertex in enumerate(mesh.vertices()):
            vertex.set_id(i)
            vertices[i, :] = point_to_list(vertex.point(), shape, scale, min_coords=min_coords)

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
        return vertices, facets


def _save_off(vertices: np.ndarray, facets: np.ndarray, file_name: str):
    with open(file_name, "w") as f:
        f.write("OFF\n")
        f.write(f"{vertices.shape[0]} {facets.shape[0]} 0\n\n")
        for i in range(vertices.shape[0]):
            f.write(f"{vertices[i][0]} {vertices[i][1]} {vertices[i][2]}\n")
        for i in range(facets.shape[0]):
            f.write(f"3  {facets[i][0]} {facets[i][1]} {facets[i][2]}\n")

def _try_repair_with_cgal(poly):
    try:
        if "stitch_borders" in globals():
            try:
                stitch_borders(poly)
            except Exception:
                pass
    except Exception:
        pass

    hole_fillers = []
    if "triangulate_refine_and_fair_hole" in globals():
        hole_fillers.append(triangulate_refine_and_fair_hole)
    if "triangulate_and_refine_hole" in globals():
        hole_fillers.append(triangulate_and_refine_hole)
    if "triangulate_hole" in globals():
        hole_fillers.append(triangulate_hole)

    if len(hole_fillers) == 0:
        return

    try:
        border_halfedges = []
        for h in poly.halfedges():
            is_border = False

            if hasattr(h, "is_border"):
                try:
                    is_border = h.is_border()
                except Exception:
                    is_border = False
            elif hasattr(poly, "is_border"):
                try:
                    is_border = poly.is_border(h)
                except Exception:
                    is_border = False

            if is_border:
                border_halfedges.append(h)

        for h in border_halfedges:
            for filler in hole_fillers:
                try:
                    filler(poly, h)
                    break
                except Exception:
                    continue
    except Exception:
        pass

def _marching_cubes_step_size(complexity: str) -> int:
    profile = MESH_COMPLEXITY_PROFILES.get(
        str(complexity).lower(), MESH_COMPLEXITY_PROFILES[DEFAULT_MESH_COMPLEXITY]
    )
    return max(1, int(profile["marching_cubes_step_size"]))


def voxel_to_mesh(
    data: np.ndarray,
    shape: tuple,
    scale: list,
    folder: str,
    mesh_complexity: str = DEFAULT_MESH_COMPLEXITY,
) -> tuple[Polyhedron_3, np.ndarray]:
    """returns surface_poly of mesh and updated voxel data"""
    data = np.asarray(data).copy()
    data[data > 0] = 1
    labels, dendrite_label = find_main_label(data)
    del data

    mesh_base = np.zeros_like(labels, dtype=np.uint8)
    mesh_base[labels == dendrite_label] = 1
    del labels

    # делаем erosion->dilation, но пересекаем его с исходной маской чтобы не уменьшать объект
    mesh_base = (binary_dilation(
        binary_erosion(
            mesh_base, structure=generate_binary_structure(3, 1), border_value=1
        ),
        structure=generate_binary_structure(3, 1),
    ) + mesh_base).astype(np.uint8) 

            # padding нужен, чтобы marching cubes корректно строил поверхность
    # у объектов, касающихся границы объема
    padded = np.pad(mesh_base, 1, mode="constant", constant_values=0)

    step_size = _marching_cubes_step_size(mesh_complexity)
    vertices, facets, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        spacing=(float(scale[0]), float(scale[1]), float(scale[2])),
        step_size=step_size,
    )
    facets = facets[:, [0, 2, 1]]

    # компенсируем padding: сдвигаем координаты обратно
    vertices -= np.array([scale[0], scale[1], scale[2]], dtype=np.float32)

    # A coarse grid naturally produces roughly step_size**2 fewer surface
    # vertices. Keep the effective small-object cutoff equivalent to high.
    min_vertices = max(1, int(np.ceil(10000 / (step_size ** 2))))
    if vertices.shape[0] < min_vertices:
        print(("Too small object"))
        return

    vertices = np.asarray(vertices, dtype=np.float32)
    facets = np.asarray(facets, dtype=np.int32)

    # Не вырезаем крупнейшую компоненту безусловно.
    # Делаем это только если она реально почти весь меш.
    _, _, cf, nf = pcu.connected_components(vertices, facets)
    cf = np.asarray(cf).reshape(-1)
    nf = np.asarray(nf).reshape(-1)

    if nf.size > 0:
        comp_max = int(np.argmax(nf))
        total_faces = int(np.sum(nf))
        max_faces = int(nf[comp_max])

        if total_faces > 0 and max_faces / total_faces > 0.98:
            vertices, facets, _, _ = pcu.remove_unreferenced_mesh_vertices(
                vertices, facets[cf == comp_max]
            )

    mesh_file = (
        TMP_AUXILIARY_PATH
        + "/mesh_tmp_"
        + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        + ".off"
    )
    _save_off(vertices, facets, folder + mesh_file)

    surface_poly = Polyhedron_3(folder + mesh_file)
    try:
        os.remove(folder + mesh_file)
    except:
        ...

    _try_repair_with_cgal(surface_poly)

    return surface_poly, mesh_base


def _build_surface(
    data: np.ndarray, shape: tuple, scale: list, folder: str, queue: Queue, lock,
    mesh_complexity: str, fill_holes_before_mesh: bool,
):
    try:
        if fill_holes_before_mesh:
            data = fill_small_enclosed_holes(data)
        surface_poly, mesh_base = voxel_to_mesh(
            data, shape, scale, folder, mesh_complexity
        )
        
        tif_file = (
            TMP_LAYERS_PATH
            + "/mesh_base_"
            + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
            + ".tif"
        )
        imwrite(folder + tif_file, data=mesh_base)

        mesh_file = (
            TMP_LAYERS_PATH
            + "/mesh_"
            + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
            + ".off"
        )
        surface_poly.write_to_file(folder + mesh_file)

        surface_poly = Polyhedron_3(folder + mesh_file)

        descriptions, error = create_mesh_descriptions(
            surface_poly, folder + TMP_AUXILIARY_PATH
        )

        lock.acquire()
        if error != "":
            queue.put(
                (
                    SurfaceData(),
                    {},
                    error,
                )
            )
        else:
            queue.put(
                (
                    SurfaceData(
                        mesh_file, _mesh_to_v_f(surface_poly, shape, scale), tif_file
                    ),
                    {"descriptions": TMP_AUXILIARY_PATH + descriptions},
                    error,
                )
            )
        lock.release()
    except Exception as e:
        queue.put((SurfaceData(), {}, str(e)))


def build_mesh(
    data: np.ndarray,
    shape: tuple,
    scale: list,
    folder: str,
    metadata: dict,
    queue_in: Queue,
    queue_out: Queue,
    mesh_complexity: str = DEFAULT_MESH_COMPLEXITY,
    fill_holes_before_mesh: bool = True,
):
    try:
        if np.max(data) == 0:
            queue_out.put(Result(SurfaceData(), metadata, error="Empty area"))
            return

        lock = Lock()
        queue = Queue()
        process = Process(
            target=_build_surface,
            args=(
                data, shape, scale, folder, queue, lock,
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
            queue_out.put(Result(SurfaceData(), metadata, True))
        else:
            result, files, error = queue.get()
            queue.close()
            queue.join_thread()
            process.join()
            queue_out.put(Result(result, metadata, False, files, error))
    except Exception as e:
        queue_out.put(Result(SurfaceData(), metadata, error=str(e)))


def create_mesh_descriptions(
    surface_poly: Polyhedron_3, folder: str
) -> tuple[str, str]:
    skeleton_polylines = Polylines()
    correspondence_polylines = Polylines()
    surface_mesh_skeletonization(
        surface_poly, skeleton_polylines, correspondence_polylines
    )

    skeleton_graph = build_graph(skeleton_polylines)
    correspondence = build_correspondence(correspondence_polylines)
    reverse_correspondence = build_reverse_correpondence(correspondence_polylines)

    try:
        dendrite_skeleton_points = find_longest_path(skeleton_graph)
    except:
        return (
            "",
            "Unable to reconstruct connected mesh. Try to eliminate too thin connections",
        )

    path_statistics = get_path_statistics(
        dendrite_skeleton_points, reverse_correspondence, 0
    )
    all_distance = np.concatenate([x for x in path_statistics.values()])

    descriptions = {
        "correspondence": correspondence,
        "dendrite_skeleton_points": dendrite_skeleton_points,
        "all_distance": all_distance.tolist(),
    }
    descriptions_file = (
        "/descriptions_"
        + str(datetime.now()).replace(".", "_").replace(" ", "_").replace(":", "_")
        + ".json"
    )

    f = open(folder + descriptions_file, "w")
    dump(descriptions, f)
    f.close()
    return descriptions_file, ""
