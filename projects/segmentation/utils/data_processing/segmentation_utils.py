import ast
from itertools import chain, product
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import shortest_path
from scipy import spatial
from skimage.measure import label
from skimage.morphology import skeletonize

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polygon_mesh_processing import (
    Polylines,
    keep_connected_components,
    remove_connected_components,
    volume,
)
from CGAL.CGAL_Polyhedron_3 import (
    Polyhedron_3,
    Polyhedron_3_Facet_handle,
    Polyhedron_3_Halfedge_around_facet_circulator,
    Polyhedron_3_Halfedge_around_vertex_circulator,
    Polyhedron_3_Halfedge_handle,
    Polyhedron_3_Vertex_handle,
)

Segmentation = Set[str]
Correspondence = Dict[str, str]
ReverseCorrespondence = Dict[str, List[Point_3]]


def _find_farthest(skel):
    pts = np.argwhere(skel)
    dist_mat = spatial.distance_matrix(pts, pts)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    return pts[i], pts[j]


def find_medial(skel):
    start, end = _find_farthest(skel)
    stack = [start]
    visited = np.zeros_like(skel)
    branches = [[]]
    box_coords = list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    box_coords.remove((0, 0, 0))
    while True:
        point = stack.pop()
        if np.all(point == end):
            break
        d, h, w = skel.shape
        neibs = []
        for coord in box_coords:
            k, i, j = point + coord
            if (
                0 <= i
                and i < h
                and 0 <= j
                and j < w
                and 0 <= k < d
                and not visited[k, i, j]
            ):
                neibs.append(np.array([k, i, j]))
        neibs.sort(key=lambda x: spatial.distance.cosine(x - point, end - point))
        real_neibs = [x for x in neibs if skel[tuple(x)]]
        if not real_neibs:
            if len(branches) > 0:
                branches.pop()
        elif len(real_neibs) == 1 and len(branches) > 0:
            branches[-1].append(point)
        else:
            branches.append([point])

        for n in real_neibs[::-1]:
            stack.append(n)

        visited[tuple(point)] = True
    return list(chain.from_iterable(branches))


def find_main_label(data: np.ndarray, data_as_labels=False, connectivity=1):
    if not data_as_labels:
        labels = label(data, connectivity=connectivity)
    else:
        labels = data.copy()

    unique_labels, counts = np.unique(labels, return_counts=True)
    unique_counts = {u: c for u, c in zip(unique_labels, counts)}
    unique_counts = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)

    main_label = None
    for unique, _ in unique_counts:
        if unique != 0:
            main_label = unique
            break
    return labels, main_label


def get_skelet(shaft: np.ndarray):
    skel_labeled, main_label = find_main_label(skeletonize(shaft), connectivity=3)
    skel_labeled[skel_labeled != main_label] = 0
    return skel_labeled


def list_2_point(coords: list) -> Point_3:
    return Point_3(float(coords[0]), float(coords[1]), float(coords[2]))


def point_to_list(point: Point_3, shape, scale, min_coords=[0.0, 0.0, 0.0]) -> List[float]:
    return [
        min(shape[0] - 1, max(round((point.x() - min_coords[0]) / scale[0]), 0)),
        min(shape[1] - 1, max(round((point.y() - min_coords[1]) / scale[1]), 0)),
        min(shape[2] - 1, max(round((point.z() - min_coords[2]) / scale[2]), 0)),
    ]


def point_2_list(point: Point_3) -> List[float]:
    return [point.x(), point.y(), point.z()]


def hash_point(point: Point_3) -> str:
    return str(np.asarray(point_2_list(point)).tolist())


def unhash_point(hashed_point: str) -> Point_3:
    return list_2_point(ast.literal_eval(hashed_point))


def get_distance(x: Point_3, y: Point_3) -> float:
    return np.sqrt((x - y).squared_length())


def build_graph(polylines: Polylines) -> nx.Graph:
    output = nx.Graph()

    for line in polylines:
        for i in range(len(line) - 1):
            u: Point_3 = line[i]
            v: Point_3 = line[i + 1]

            hashed_u = hash_point(u)
            hashed_v = hash_point(v)

            if i == 0:
                output.add_node(hashed_u)
            output.add_node(hashed_v)

            output.add_edge(hashed_u, hashed_v, weight=get_distance(u, v))

    return output


def build_correspondence(corr_polylines: Polylines) -> Correspondence:
    corr: Correspondence = {}
    for line in corr_polylines:
        corr[hash_point(line[1])] = hash_point(line[0])
    return corr


def build_reverse_correpondence(corr_polylines: Polylines) -> ReverseCorrespondence:
    corr: ReverseCorrespondence = {}
    for line in corr_polylines:
        key = hash_point(line[0])
        if not key in corr:
            corr[key] = []
        corr[key].append(line[1])
    return corr


def find_longest_path(graph: nx.Graph) -> List:
    if len(graph.nodes()) < 2:
        return []

    nodes_arr = [point_2_list(unhash_point(x)) for x in graph.nodes()]
    dist_mat = spatial.distance_matrix(nodes_arr, nodes_arr)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)  # the most remote points
    longest_path = shortest_path(
        graph,
        hash_point(list_2_point(nodes_arr[i])),
        hash_point(list_2_point(nodes_arr[j])),
    )
    return longest_path


def get_distance_statistic(
    longest_path: List[str],
    reverse_correspondence: ReverseCorrespondence,
    center_index: int,
    window_halfsize: int = 5,
) -> List[float]:
    output = []
    start_index = max(center_index - window_halfsize, 0)
    end_index = min(center_index + 1 + window_halfsize, len(longest_path))
    for i in range(start_index, end_index):
        hashed_skeleton_point: str = longest_path[i]
        if not hashed_skeleton_point in reverse_correspondence:
            continue
        for surface_point in reverse_correspondence[hashed_skeleton_point]:
            output.append(
                get_distance(surface_point, unhash_point(hashed_skeleton_point))
            )
    return output


def get_path_statistics(
    path: List[str],
    reverse_correspondence: ReverseCorrespondence,
    window_halfsize: int = 0,
) -> Dict:
    path_statistics = {}
    for i, hashed_skeleton_point in enumerate(path):
        path_statistics[hashed_skeleton_point] = get_distance_statistic(
            path, reverse_correspondence, i, window_halfsize
        )
    return path_statistics


def segmentation_by_distance(
    polyhedron: Polyhedron_3,
    correspondence: Correspondence,
    dendrite_skeleton_points: List,
    all_distance: List,
    distance_sensitivity: float = 0.75,
) -> Segmentation:
    all_distance = np.array(all_distance)
    distance_threshold = np.quantile(all_distance, distance_sensitivity)

    output_segmentation: Segmentation = set()
    dendrite_set: Set = set(dendrite_skeleton_points)
    for surface_point in polyhedron.points():
        hashed_surface_point = hash_point(surface_point)
        hashed_skeleton_point = correspondence[hashed_surface_point]
        if hashed_skeleton_point in dendrite_set:
            distance = get_distance(surface_point, unhash_point(hashed_skeleton_point))
            if distance > distance_threshold:
                output_segmentation.add(hashed_surface_point)
        else:
            output_segmentation.add(hashed_surface_point)

    return output_segmentation


def find_edge_vertices(
    segmentation: Segmentation, mesh: Polyhedron_3
) -> Set[Polyhedron_3_Vertex_handle]:
    edge_vertices = set()
    for v in mesh.vertices():
        if not hash_point(v.point()) in segmentation:
            continue
        h = v.halfedge()
        circulator: Polyhedron_3_Halfedge_around_vertex_circulator = h.vertex_begin()
        begin = h.vertex_begin()
        while circulator.hasNext():
            h1: Polyhedron_3_Halfedge_handle = circulator.next()
            v1 = h1.opposite().vertex()
            if not hash_point(v1.point()) in segmentation:
                edge_vertices.add(h.vertex())
                break
            if circulator == begin:
                break
    return edge_vertices


def expand_segmentation(
    segmentation: Segmentation, mesh: Polyhedron_3, wave_num: int
) -> Segmentation:
    out = segmentation.copy()

    edge_vertices = find_edge_vertices(out, mesh)

    for i in range(wave_num):
        points_to_add = set()
        new_edge_vertices = set()
        for v in edge_vertices:
            h = v.halfedge()
            circulator: Polyhedron_3_Halfedge_around_vertex_circulator = (
                h.vertex_begin()
            )
            begin = h.vertex_begin()
            while circulator.hasNext():
                h1: Polyhedron_3_Halfedge_handle = circulator.next()
                v1 = h1.opposite().vertex()
                if not hash_point(v1.point()) in out:
                    new_edge_vertices.add(v1)
                    points_to_add.add(hash_point(v1.point()))
                if circulator == begin:
                    break
        out = out.union(points_to_add)
        edge_vertices = new_edge_vertices

    return out


def shrink_segmentation(
    segmentation: Segmentation, mesh: Polyhedron_3, wave_num: int
) -> Segmentation:
    out = segmentation.copy()

    edge_vertices = find_edge_vertices(out, mesh)

    for i in range(wave_num):
        points_to_remove = set()
        new_edge_vertices = set()
        for v in edge_vertices:
            points_to_remove.add(hash_point(v.point()))
            h = v.halfedge()
            circulator: Polyhedron_3_Halfedge_around_vertex_circulator = (
                h.vertex_begin()
            )
            begin = h.vertex_begin()
            while circulator.hasNext():
                h1: Polyhedron_3_Halfedge_handle = circulator.next()
                v1 = h1.opposite().vertex()
                if hash_point(v1.point()) in out:
                    new_edge_vertices.add(v1)
                if circulator == begin:
                    break
        out = out.difference(points_to_remove)
        edge_vertices = new_edge_vertices

    return out


def correct_segmentation(
    segmentation: Segmentation, mesh: Polyhedron_3, delta: int
) -> Segmentation:
    if delta > 0:
        return expand_segmentation(segmentation, mesh, delta)
    return shrink_segmentation(segmentation, mesh, -delta)


def get_spine_meshes(
    in_mesh: Polyhedron_3, segmentation: Segmentation, min_volume: float | None = None
) -> List[Polyhedron_3]:
    mesh: Polyhedron_3 = erase_dendrite_facets(in_mesh, segmentation)

    for i, halfedge in enumerate(mesh.halfedges()):
        halfedge.set_id(i)

    component_halfedge_ids: List[int] = []
    reduced_mesh: Polyhedron_3 = mesh.deepcopy()
    while reduced_mesh.size_of_facets() > 0:
        facet: Polyhedron_3_Facet_handle = reduced_mesh.facets().next()
        component_halfedge_ids.append(facet.halfedge().id())
        remove_connected_components(reduced_mesh, [facet])

    output = []
    for halfedge_id in component_halfedge_ids:
        spine_mesh: Polyhedron_3 = mesh.deepcopy()
        component_facet: Polyhedron_3_Facet_handle = spine_mesh.facets().next()
        for halfedge in spine_mesh.halfedges():
            if halfedge.id() == halfedge_id:
                component_facet = halfedge.facet()
                break
        keep_connected_components(spine_mesh, [component_facet])
        if min_volume is not None:
            completed_spine_mesh: Polyhedron_3 = spine_mesh.deepcopy()
            for h in completed_spine_mesh.halfedges():
                if h.is_border():
                    completed_spine_mesh.fill_hole(h)
                    h.facet().set_id(0)
            for facet in completed_spine_mesh.facets():
                if not facet.is_triangle():
                    completed_spine_mesh.create_center_vertex(facet.halfedge())
            if volume(completed_spine_mesh) < min_volume:
                continue
        for facet in spine_mesh.facets():
            if not facet.is_triangle():
                spine_mesh.create_center_vertex(facet.halfedge())
        output.append(spine_mesh)

    return output


def erase_dendrite_facets(
    in_mesh: Polyhedron_3, segmentation: Segmentation
) -> Polyhedron_3:
    mesh: Polyhedron_3 = in_mesh.deepcopy()

    for facet in mesh.facets():
        circulator: Polyhedron_3_Halfedge_around_facet_circulator = facet.facet_begin()
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge: Polyhedron_3_Halfedge_handle = circulator.next()
            v: Polyhedron_3_Vertex_handle = halfedge.vertex()
            if not hash_point(v.point()) in segmentation:
                mesh.erase_facet(facet.halfedge())
                break
            if circulator == begin:
                break

    return mesh
