import os
import json
import numpy as np
import kimimaro
import networkx as nx
from tifffile import imread, imwrite
from scipy.ndimage import distance_transform_edt, find_objects, label
from skimage.segmentation import watershed

# RES_YXZ = (500,100, 100)#(24, 24, 30)
# RES_YXZ = (24, 24, 30)

def _farthest_node(tree, source):
    """Find the farthest tree node in linear time and retain its parent map."""
    parents = {source: None}
    distances = {source: 0.0}
    stack = [source]
    target = source
    while stack:
        node = stack.pop()
        node_distance = distances[node]
        if node_distance > distances[target]:
            target = node
        for neighbor, edge_data in tree[node].items():
            if neighbor in parents:
                continue
            parents[neighbor] = node
            distances[neighbor] = node_distance + edge_data["weight"]
            stack.append(neighbor)
    return target, parents


def _protected_main_paths(graph):
    """Return the weighted diameter of every connected skeleton component."""
    protected = set()
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes)
        start = next(iter(component_nodes))
        endpoint, _ = _farthest_node(component, start)
        opposite_endpoint, parents = _farthest_node(component, endpoint)
        diameter_path = []
        node = opposite_endpoint
        while node is not None:
            diameter_path.append(node)
            node = parents[node]
        protected.update(diameter_path)
    return protected


def prune_and_segment_skeleton(
    skel,
    res_zyx,
    volume_shape=None,
    min_length_nm=1000,
    border_margin_voxels=2,
    vertex_offset_nm=None,
):
    """Prune short side branches without eroding the dendrite main path."""
    vertices = np.asarray(skel.vertices)
    if vertex_offset_nm is not None:
        vertices = vertices + np.asarray(vertex_offset_nm, dtype=vertices.dtype)
    G = nx.Graph()
    G.add_nodes_from(range(len(vertices)))
    edges = np.asarray(skel.edges, dtype=np.int64)
    if edges.size:
        edge_lengths = np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
        G.add_weighted_edges_from(
            (int(edge[0]), int(edge[1]), float(weight))
            for edge, weight in zip(edges, edge_lengths)
        )

    # Break cycles by physical edge length, then protect the main geodesic path.
    if G.number_of_edges() and not nx.is_forest(G):
        G = nx.minimum_spanning_tree(G, weight="weight")
    protected_nodes = _protected_main_paths(G)

    # An endpoint at the acquisition boundary may be a truncated dendrite trunk,
    # not a short side branch. Never prune such endpoints.
    boundary_nodes = set()
    if volume_shape is not None:
        voxel_coords = vertices / np.asarray(res_zyx, dtype=np.float64)
        upper = np.asarray(volume_shape, dtype=np.float64) - 1
        near_boundary = np.any(
            (voxel_coords <= border_margin_voxels)
            | (voxel_coords >= upper - border_margin_voxels),
            axis=1,
        )
        boundary_nodes.update(np.flatnonzero(near_boundary).tolist())

    # Repeated pruning is safe now because the component diameter and boundary
    # endpoints are protected. Only short branches attached to that backbone go.
    while True:
        degrees = dict(G.degree())
        leaves = [n for n, d in degrees.items() if d == 1]
        nodes_to_remove = set()

        for leaf in leaves:
            if leaf in protected_nodes or leaf in boundary_nodes:
                continue
            path = [leaf]
            previous = None
            current = leaf
            length_nm = 0.0

            while True:
                next_nodes = [n for n in G.neighbors(current) if n != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                length_nm += G[current][next_node]["weight"]
                if next_node in protected_nodes or G.degree(next_node) > 2:
                    break
                path.append(next_node)
                previous = current
                current = next_node

            if length_nm < min_length_nm:
                nodes_to_remove.update(
                    node
                    for node in path
                    if node not in protected_nodes and node not in boundary_nodes
                )

        if not nodes_to_remove:
            break
        G.remove_nodes_from(nodes_to_remove)

    degrees = dict(G.degree())
    new_junctions = set(n for n, d in degrees.items() if d > 2)
    G.remove_nodes_from(new_junctions)
    segments = list(nx.connected_components(G))
    return segments, vertices


def _foreground_slices(mask, margin=4):
    objects = find_objects(mask)
    if not objects or objects[0] is None:
        return None
    return tuple(
        slice(
            max(0, axis_slice.start - margin),
            min(axis_size, axis_slice.stop + margin),
        )
        for axis_slice, axis_size in zip(objects[0], mask.shape)
    )

def scelete(json_path=None, base_path="../../", do_segmentation=True):
    with open(json_path, 'r') as f:
        config = json.load(f)

    def resolve(p): 
        if p is None: return None
        return os.path.normpath(os.path.join(base_path, p))
    
    for item in config["data"]:
        print(f"\n--- Обработка: {item['name']} ---", flush=True)
        # 1. Загрузка исходников
        area_path = resolve(item.get('area_of_interest'))
        area = None
        if area_path and os.path.exists(area_path):
            area = imread(area_path) > 0

        input_path = (
            item.get("stage3_mask")
            or item.get("shaft_mask")
            or item.get("general_mask")
            or item.get("image")
        )
        if not input_path:
            raise KeyError(
                "Stage 4 skeleton input is missing. Expected one of: "
                "stage3_mask, shaft_mask, general_mask, image"
            )
        input_file = resolve(input_path)
        skel_out = resolve(item["skeleton"])
        seg_out = resolve(item["segments_mask"])

        #расчет масштаба из JSON [Z, Y, X] мкм -> [Z, Y, X] нм
        scale_um = item["scale"]
        res_z = scale_um[0] * 1000
        res_y = scale_um[1] * 1000
        res_x = scale_um[2] * 1000
        res_zyx = (res_z, res_y, res_x)

        foreground = imread(input_file) > 0
        if area is not None:
            foreground &= area
        crop_slices = _foreground_slices(
            foreground, margin=int(item.get("skeleton_crop_margin_voxels", 4))
        )
        if crop_slices is None:
            print("Скелет не найден: входная маска пуста!", flush=True)
            continue
        crop_offset_voxels = np.asarray(
            [axis_slice.start for axis_slice in crop_slices], dtype=np.int64
        )
        labels_crop = np.empty(
            foreground[crop_slices].shape, dtype=np.uint32
        )
        component_count = label(
            foreground[crop_slices],
            structure=np.ones((3, 3, 3), dtype=np.uint8),
            output=labels_crop,
        )
        component_sizes = np.bincount(labels_crop.ravel())
        object_ids = np.flatnonzero(component_sizes >= 100)
        object_ids = object_ids[object_ids != 0]
        retained_components = int(object_ids.size)
        print(
            "Kimimaro skeletonization: full shape "
            + str(tuple(foreground.shape))
            + ", ROI shape "
            + str(tuple(labels_crop.shape))
            + ", foreground voxels "
            + str(int(np.count_nonzero(labels_crop)))
            + ", components "
            + str(component_count)
            + ", components >= 100 voxels "
            + str(retained_components),
            flush=True,
        )
        if retained_components == 0:
            print("Скелет не найден: все компоненты меньше 100 вокселей!", flush=True)
            continue

        skels = kimimaro.skeletonize(
            labels_crop,
            teasar_params={
                'scale': 4,
                'const': 1000, # physical units
                'pdrf_exponent': 4,
                'pdrf_scale': 100000,
                'soma_detection_threshold': 1100, # physical units
                'soma_acceptance_threshold': 3500, # physical units
                'soma_invalidation_scale': 1.0,
                'soma_invalidation_const': 300, # physical units
                'max_paths': None, # default None
            },
            object_ids=object_ids.tolist(),
            # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
            # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
            dust_threshold=100, # skip connected components with fewer than this many voxels
            anisotropy=res_zyx, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True/у гришы было False
            fill_holes=False, # default False
            fix_avocados=False, # default False
            progress=True, # default False, show progress bar
            parallel=int(item.get("skeleton_parallel", 10)),
            parallel_chunk_size=100, # how many skeletons to process before updating progress bar
        )
        print("Kimimaro finished; skeleton objects: " + str(len(skels)), flush=True)

        if not skels:
            print("Скелет не найден!")
            continue

        # 3. СОХРАНЕНИЕ ПОЛНОГО СКЕЛЕТА (ds_skeleton)
        binimg_skel = np.zeros(foreground.shape, dtype=np.uint8)
        if do_segmentation:
            prune_length_nm = float(item.get("skeleton_prune_length_nm", 1000))
            print(
                "Защита главного пути и фильтрация боковых ветвей < "
                f"{prune_length_nm / 1000:.2f} мкм..."
            )
            markers = np.zeros(foreground.shape, dtype=np.uint16)
            segment_id = 0
            res_zyx_array = np.asarray(res_zyx, dtype=np.float64)
            vertex_offset_nm = crop_offset_voxels * res_zyx_array
            for skel in skels.values():
                segments, vertices = prune_and_segment_skeleton(
                    skel,
                    res_zyx,
                    volume_shape=foreground.shape,
                    min_length_nm=prune_length_nm,
                    vertex_offset_nm=vertex_offset_nm,
                )
                for node_set in segments:
                    segment_id += 1
                    nodes = np.fromiter(node_set, dtype=np.int64)
                    coordinates = (vertices[nodes] / res_zyx_array).astype(np.int64)
                    valid = np.all(
                        (coordinates >= 0)
                        & (coordinates < np.asarray(markers.shape)),
                        axis=1,
                    )
                    marker_indices = tuple(coordinates[valid].T)
                    binimg_skel[marker_indices] = 255
                    markers[marker_indices] = segment_id
            imwrite(skel_out, binimg_skel, photometric='minisblack')
            print(f"Полный скелет сохранен: {skel_out}")
            print("Распределение вокселей по сегментам (Watershed)...")
        # если очень большой снимок, то памяти не хватает,выделяем только нужную часть и считаем на ней, потом вставляем обратно
#         # 1. Находим минимальную "коробочку" (Bounding Box), куда влезает весь дендрит
            mask_crop = foreground[crop_slices]
            markers_crop = markers[crop_slices]

            # 3. Считаем SciPy EDT и Watershed только на маленьком кусочке!
            distance_crop = distance_transform_edt(mask_crop, sampling=res_zyx)
            seg_crop = watershed(-distance_crop, markers_crop, mask=mask_crop)

            # 4. Создаем пустой черный объем оригинального размера и вставляем туда результат
            segmented_volume = np.zeros(foreground.shape, dtype=np.uint16)
            segmented_volume[crop_slices] = seg_crop

            imwrite(seg_out, segmented_volume.astype(np.uint16), photometric='minisblack')
            print(f"Готово! Результат сохранен в {seg_out}")
        else:
            print("Сохранение сырого полного скелета (без сегментации)...")
            res_zyx_array = np.asarray(res_zyx, dtype=np.float64)
            vertex_offset_nm = crop_offset_voxels * res_zyx_array
            for skel in skels.values():
                verts_all = (
                    (np.asarray(skel.vertices) + vertex_offset_nm) / res_zyx_array
                ).astype(np.int64)
                valid = np.all(
                    (verts_all >= 0)
                    & (verts_all < np.asarray(binimg_skel.shape)),
                    axis=1,
                )
                binimg_skel[tuple(verts_all[valid].T)] = 255
            imwrite(skel_out, binimg_skel, photometric='minisblack')
            print(f"Полный скелет сохранен: {skel_out}")        

if __name__ == "__main__":
    ##проверка получившегося скелета
    # skel = imread("C:/Users/Student/datasets/Vsot/dataset/D5_Branch_1/ds_skeleton.tif").astype(np.uint8)
    # print(skel.shape)
    # num_voxels = np.count_nonzero(skel)

    # print(f"Количество положительных вокселов: {num_voxels}")
    # max_val = skel.max()

    # if max_val > 0:
    #     print(f"Снимок не пустой. Максимальная яркость: {max_val}")
    # else:
    #     print("Снимок полностью черный.")
    # # skel = np.max(skel, axis=2)
    # skel[skel>0]=255
    # imwrite("skel_vsot3.tif", data=np.max(skel, axis=0)) 
    scelete("data_preprocessing/CVstage4/folds1.json")
    # scelete("data_preprocessing/description_stage4_vsot.json")#запускать без разбиения на сегменты, так как там и так один сегмент
