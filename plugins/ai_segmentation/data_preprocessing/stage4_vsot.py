import os
import json
import numpy as np
from tifffile import imread, imwrite
from scipy.ndimage import (
    binary_dilation,
    find_objects,
    distance_transform_edt,
    label as connected_components,
)


class ShaftRadiusEstimationError(RuntimeError):
    """The segment is too thin or degenerate for VSOT radius estimation."""


def _ball(radius):
    coordinates = np.ogrid[
        tuple(slice(-radius, radius + 1) for _ in range(3))
    ]
    squared_distance = sum(axis * axis for axis in coordinates)
    return squared_distance <= radius * radius


def _octave_compatible_shaft(mask, skeleton, distance, res_x, res_y, res_z):
    """Port of level_1_segmentation_function_fin for GNU Octave installations.

    VSOT's source is left untouched.  scipy supplies the MATLAB-only morphology,
    graph/maxflow and label2idx operations that GNU Octave does not provide.
    """
    mask = np.asarray(mask, dtype=bool)
    skeleton = binary_dilation(skeleton, structure=_ball(3)) & mask
    skeleton_distances = distance[skeleton]
    skeleton_distances = skeleton_distances[
        skeleton_distances > min(res_x, res_z)
    ]
    if skeleton_distances.size == 0:
        raise ShaftRadiusEstimationError(
            "VSOT cannot estimate shaft radius from this skeleton"
        )
    radius = float(np.min(skeleton_distances))

    distance_to_skeleton = distance_transform_edt(
        ~skeleton, sampling=(res_z, res_y, res_x)
    )
    source_mask = mask & (distance_to_skeleton <= radius)
    outer_boundary = binary_dilation(mask, structure=_ball(1)) ^ mask
    sink_mask = binary_dilation(outer_boundary, structure=_ball(1)) & mask
    sink_mask &= ~source_mask

    if not np.any(mask):
        return np.zeros_like(mask)

    try:
        import maxflow
    except ImportError as error:
        raise RuntimeError(
            "GNU Octave compatibility requires PyMaxflow. "
            "Install the ai_segmentation requirements again."
        ) from error

    distance_to_source = distance_transform_edt(
        ~source_mask, sampling=(res_z, res_y, res_x)
    )
    terminal_weight = (res_x * res_y * res_z) ** (2.0 / 3.0) / 32.0

    # MATLAB maxflow uses Boykov-Kolmogorov by default. PyMaxflow wraps the
    # reference C++ implementation of that algorithm and supports float
    # capacities, avoiding the rounding required by scipy.maximum_flow.
    graph = maxflow.Graph[float]()
    node_volume = graph.add_grid_nodes(mask.shape)

    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left_slice = tuple(left_slice)
        right_slice = tuple(right_slice)
        valid = mask[left_slice] & mask[right_slice]
        pair_weights = np.sqrt(
            distance_to_source[left_slice][valid]
            * distance_to_source[right_slice][valid]
        ) / 8.0
        edge_weights = np.zeros(mask.shape, dtype=np.float64)
        edge_weights[left_slice][valid] = pair_weights
        structure = np.zeros((3, 3, 3), dtype=np.int8)
        offset = [1, 1, 1]
        offset[axis] = 2
        structure[tuple(offset)] = 1
        graph.add_grid_edges(
            node_volume,
            weights=edge_weights,
            structure=structure,
            symmetric=True,
        )

    rest_mask = mask & ~source_mask & ~sink_mask
    finite_capacity_sum = float(terminal_weight * np.count_nonzero(rest_mask))
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left_slice = tuple(left_slice)
        right_slice = tuple(right_slice)
        valid = mask[left_slice] & mask[right_slice]
        finite_capacity_sum += float(
            2.0
            * np.sum(
                np.sqrt(
                    distance_to_source[left_slice][valid]
                    * distance_to_source[right_slice][valid]
                )
                / 8.0
            )
        )
    infinite_capacity = max(finite_capacity_sum + 1.0, 1e9)
    source_capacities = np.zeros(mask.shape, dtype=np.float64)
    sink_capacities = np.zeros(mask.shape, dtype=np.float64)
    source_capacities[source_mask] = infinite_capacity
    source_capacities[rest_mask] = terminal_weight
    sink_capacities[sink_mask] = infinite_capacity
    graph.add_grid_tedges(node_volume, source_capacities, sink_capacities)
    graph.maxflow()
    selected = ~graph.get_grid_segments(node_volume) & mask

    shaft = mask & (
        distance_transform_edt(~selected, sampling=(res_z, res_y, res_x))
        <= 2.0 * res_x
    )
    component_labels, component_count = connected_components(
        shaft, structure=np.ones((3, 3, 3), dtype=bool)
    )
    if component_count:
        sizes = np.bincount(component_labels.ravel())
        keep = np.flatnonzero(sizes >= 20000)
        keep = keep[keep != 0]
        shaft = np.isin(component_labels, keep)
    return shaft


def run_vsot_oct2py(json_path, matlab_dir, base_path="../../", vsot_root=None):
    if not vsot_root:
        raise ValueError("VSOT root is required")
    vsot_root = os.path.abspath(vsot_root)
    if not os.path.isdir(vsot_root):
        raise FileNotFoundError("VSOT root directory not found: " + vsot_root)

    required_directories = [
        matlab_dir,
        os.path.join(vsot_root, "resources", "edt_mex", "edt_mex"),
        os.path.join(
            vsot_root,
            "resources",
            "src_mex",
            "mex_EM_analysis",
            "mex_EM_analysis",
        ),
        os.path.join(vsot_root, "resources", "graph_related", "graph_mex"),
        os.path.join(vsot_root, "src", "misc"),
    ]
    missing_directories = [path for path in required_directories if not os.path.isdir(path)]
    if missing_directories:
        raise FileNotFoundError(
            "Required VSOT directories were not found: " + ", ".join(missing_directories)
        )

    # Import only after OCTAVE_EXECUTABLE has been configured by the caller.
    from oct2py import octave

    octave.addpath(matlab_dir)
    octave.eval("pkg load image")
    res_dir = os.path.join(vsot_root, "resources")
    octave.addpath(os.path.join(res_dir, "edt_mex", "edt_mex"))
    octave.addpath(
        os.path.join(res_dir, "src_mex", "mex_EM_analysis", "mex_EM_analysis")
    )
    octave.addpath(os.path.join(res_dir, "graph_related", "graph_mex"))
    octave.addpath(os.path.join(vsot_root, "src", "misc"))

    use_compatibility_backend = False

    def segment_with_vsot(mask_part, skeleton_part, distance_part, resolutions):
        nonlocal use_compatibility_backend
        res_x, res_y, res_z = resolutions
        if not use_compatibility_backend:
            try:
                result = octave.level_1_segmentation_function_fin(
                    np.transpose(mask_part, (2, 1, 0)).astype(float),
                    np.transpose(skeleton_part, (2, 1, 0)).astype(float),
                    np.transpose(distance_part, (2, 1, 0)).astype(float),
                    float(res_x),
                    float(res_y),
                    float(res_z),
                )
                return np.transpose(np.asarray(result) > 0, (2, 1, 0))
            except Exception as error:
                message = str(error).lower()
                octave_incompatibilities = (
                    "unknown shape `sphere'",
                    "'graph' undefined",
                    "'maxflow' undefined",
                    "'label2idx' undefined",
                    "not implemented",
                )
                if not any(token in message for token in octave_incompatibilities):
                    raise
                use_compatibility_backend = True
                print(
                    "GNU Octave lacks MATLAB APIs required by VSOT; "
                    "switching to the built-in compatible backend.",
                    flush=True,
                )
        return _octave_compatible_shaft(
            mask_part,
            skeleton_part,
            distance_part,
            res_x,
            res_y,
            res_z,
        )

    with open(json_path, 'r') as f:
        config = json.load(f)

    def resolve(p): 
        if p is None: return None
        return os.path.normpath(os.path.join(base_path, p))    
        
    for item in config["data"]:
        print(f"\n=== Запуск VSOT: {item['name']} ===")
        
        # 1. Читаем пути из JSON
        mask_path = resolve(item.get("stage3_mask", item.get("general_mask")))
        skel_path = resolve(item["skeleton"])
        seg_path = resolve(item["segments_mask"])
        
        out_shaft_path = resolve(item["shaft_vsot"])
        out_spine_path = resolve(item["spine_vsot"])
        
        # 2. Масштаб: [Z, Y, X] мкм -> [Y, X, Z] нм для MATLAB
        scale_um = item["scale"]
        res_z = scale_um[0] * 1000 # 0.1 -> 100.0
        res_y = scale_um[1] * 1000 # 0.022 -> 22.0
        res_x = scale_um[2] * 1000 # 0.022 -> 22.0
        
        # 3. Загружаем данные
        labels = imread(mask_path) > 0
        skeleton = imread(skel_path) > 0
        segments = imread(seg_path)
        
        # Считаем Distance Transform заранее в Python (быстрее и безопаснее)
        print("Предварительный расчет матрицы расстояний (EDT)...")
        dist_all = distance_transform_edt(labels, sampling=(res_z, res_y, res_x))
        
        # Готовим пустой объем для итогового ствола
        final_shaft = np.zeros(labels.shape, dtype=np.uint8)
        
        # 4. Нарезаем на сегменты
        slices_list = find_objects(segments)
        
        for i, slc in enumerate(slices_list):
            if slc is None: continue
                
            seg_id = i + 1
            mask_seg = (segments[slc] == seg_id)
            skel_seg = skeleton[slc] & mask_seg
            
            if not np.any(skel_seg):
                continue
                
            dist_seg = dist_all[slc]
            print(f"  -> Отправка сегмента {seg_id} в Octave...")
            # Подготавливаем пустой холст для результата этого сегмента
            out_seg_result = np.zeros_like(mask_seg, dtype=bool)
            
            # Размеры текущего Bounding Box (Z, Y, X)
            len_z, len_y, len_x = mask_seg.shape
            
            # Проверяем, нужно ли бить на чанки (как в оригинале > 600)
            if len_x > 600 or len_y > 600 or len_z > 600:
                print(f"     Сегмент слишком большой ({len_z}x{len_y}x{len_x}). Включаем Sliding Window с перекрытием...")
                
                # Базовый размер шага (как в MATLAB)
                lmz, lmy, lmx = 400, 400, 400
                
                num_z = int(np.floor(len_z / lmz))
                num_y = int(np.floor(len_y / lmy))
                num_x = int(np.floor(len_x / lmx))
                
                # Итерируемся по сетке
                for iz in range(num_z + 1):
                    for iy in range(num_y + 1):
                        for ix in range(num_x + 1):
                            
                            # Вычисляем границы с перекрытием (padding = lmx/2)
                            # Обратите внимание на координаты: в Python индексация с 0!
                            z_start = max(iz * lmz - int(lmz / 2), 0)
                            z_end   = min((iz + 1) * lmz + int(lmz / 2), len_z)
                            
                            y_start = max(iy * lmy - int(lmy / 2), 0)
                            y_end   = min((iy + 1) * lmy + int(lmy / 2), len_y)
                            
                            x_start = max(ix * lmx - int(lmx / 2), 0)
                            x_end   = min((ix + 1) * lmx + int(lmx / 2), len_x)
                            
                            # Вырезаем текущий чанк
                            chunk_slices = np.s_[z_start:z_end, y_start:y_end, x_start:x_end]
                            
                            mask_chunk = mask_seg[chunk_slices]
                            skel_chunk = skel_seg[chunk_slices]
                            dist_chunk = dist_seg[chunk_slices]
                            
                            # Если в этом конкретном чанке есть скелет - запускаем VSOT
                            # (в оригинале проверка: sum(tmp_skel(:)) > 100, мы сделаем чуть мягче)
                            if np.sum(skel_chunk) > 50: 
                                try:
                                    out_chunk_bool = segment_with_vsot(
                                        mask_chunk,
                                        skel_chunk,
                                        dist_chunk,
                                        (res_x, res_y, res_z),
                                    )
                                    # Логическое ИЛИ с тем, что уже лежит в out_seg_result
                                    out_seg_result[chunk_slices] = out_seg_result[chunk_slices] | out_chunk_bool
                                    print(f"Доля дендрита: {np.mean(mask_seg):.4f}")
                                    print(f"Доля результата Octave: {np.mean(out_chunk_bool):.4f}")
                                except ShaftRadiusEstimationError as e:
                                    print(
                                        "     [!] Пропуск вырожденного чанка "
                                        f"{ix},{iy},{iz} сегмента {seg_id}: {e}",
                                        flush=True,
                                    )
                                except Exception as e:
                                    raise RuntimeError(
                                        "VSOT failed in Octave for segment "
                                        f"{seg_id}, chunk {ix},{iy},{iz}: {e}"
                                    ) from e
                                    
            else:
                # Если сегмент маленький, передаем его целиком
                try:
                    out_seg_result = segment_with_vsot(
                        mask_seg,
                        skel_seg,
                        dist_seg,
                        (res_x, res_y, res_z),
                    )
                except ShaftRadiusEstimationError as e:
                    print(
                        f"  [!] Пропуск вырожденного сегмента {seg_id}: {e}",
                        flush=True,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"VSOT failed in Octave for segment {seg_id}: {e}"
                    ) from e
            
            # Вклеиваем собранный сегмент (из чанков или целый) в финальный объем
            final_shaft[slc][out_seg_result] = 255
                
        pixels_0 = np.sum(final_shaft == 0)
        pixels_255 = np.sum(final_shaft == 255)
        print(f"Статистика массива final_shaft (Ствол) перед вычислением шипиков:")
        print(f"  -> Фон (0): {pixels_0} вокселей")
        print(f"  -> Ствол (255): {pixels_255} вокселей")  
        # 6. Извлекаем шипики
        # Это аналог xor(mask_dendrite, maskRemoveAll) из оригинального кода
        final_spines = (labels & (final_shaft == 0)).astype(np.uint8) * 255
        pixels_spines = np.sum(final_spines == 255)
        print(f"  -> Итого шипиков (255 в final_spines): {pixels_spines} вокселей")
        # 7. Сохраняем результаты по путям из JSON
        imwrite(out_shaft_path, final_shaft, photometric='minisblack')
        imwrite(out_spine_path, final_spines, photometric='minisblack')
        print(f"Успешно сохранено:\n - {out_shaft_path}\n - {out_spine_path}")


if __name__ == "__main__":
    raise SystemExit("Run stage4_vsot through the segmentation project runtime")
