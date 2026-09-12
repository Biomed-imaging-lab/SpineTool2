import os
import json
import numpy as np
import argparse
import random
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

import edt
from tifffile import imread, imwrite
from scipy.ndimage import rotate
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from skimage.draw import line_nd
from skimage.graph import MCP_Geometric
from stage1 import normalize_minmax

# from data_preprocessing.stage1 import normalize_minmax

FRAGMENT_SHAPE = [64, 128, 128] 
FINAL_SHAPE = [64, 128, 128]
BASE_SCALE = [0.1, 0.1]

ANGLES_POOL = [i * 10 for i in range(36)]
ANGLES_COUNT_TRAIN = 3
POSITIONS = [[0, 0, 0], [-48, -96, -96]]

def generate_distance_channels(mask, skeleton, res_nm=(100.0, 22.0, 22.0)):
    """
    Генерирует нормализованные матрицы расстояний (D_dendr и D_spine).
    """
    # 1. D_dendr (расстояние до скелета)
    skeleton_inverted = np.ascontiguousarray(~skeleton)
    d_dendr = edt.edt(skeleton_inverted, anisotropy=res_nm).astype(np.float32)
    
    d_dendr_norm = np.zeros_like(d_dendr, dtype=np.float32)
    if np.any(mask):
        mask_dist = d_dendr[mask]
        d_dendr_norm[mask] = (d_dendr[mask] - mask_dist.min()) / (mask_dist.max() - mask_dist.min() + 1e-8)
    print(f"D_dendr")
    # 2. Поиск поверхности
    eroded_mask = ndi.binary_erosion(mask)
    surface = mask & ~eroded_mask
    
    surface_dist = np.zeros_like(d_dendr, dtype=np.float32)
    surface_dist[surface] = d_dendr[surface]

    if np.any(surface):
        baseline_radius_nm = np.median(surface_dist[surface])
    else:
        baseline_radius_nm = 500.0
    print(f"[i] Автоматически вычисленный радиус ствола: {baseline_radius_nm:.1f} нм")    

    min_peak_dist = (baseline_radius_nm * 1.0) +  res_nm[0] * 1.0

    # Адаптивное окно поиска (ищем в радиусе ствола, но не меньше 250 нм)
    search_radius_nm = max(1500.0, baseline_radius_nm * 0.9)
    search_radius_xy = max(400.0, min(900.0, baseline_radius_nm))
    search_radius_z = baseline_radius_nm * 1.0
    # # 3. Поиск осей шипиков (окрестность 500 нм)
    # radius_nm = 500.0
    win_z = max(1, int(np.round(search_radius_z / res_nm[0])))
    win_y = max(1, int(np.round(search_radius_xy / res_nm[1])))
    win_x = max(1, int(np.round(search_radius_xy / res_nm[2])))

    # Ищем пики на слегка сглаженной поверхности
    smoothed_dist = ndi.gaussian_filter(surface_dist, sigma=1.0)
    local_max = ndi.maximum_filter(smoothed_dist, size=(win_z, win_y, win_x))
    peaks_mask = surface & (smoothed_dist == local_max) & (smoothed_dist > min_peak_dist)
    # local_max = ndi.maximum_filter(surface_dist, size=(win_z, win_y, win_x))
    # peaks_mask = surface & (surface_dist == local_max) & (surface_dist > 0)
    peak_coords = np.argwhere(peaks_mask)
    print(f"[i] Найдено потенциальных макушек шипиков: {len(peak_coords)}")
    # 4. Построение предполагаемых осей шипиков
    spine_axes = np.zeros_like(mask, dtype=bool)
    skel_coords = np.argwhere(skeleton)
    
    if len(skel_coords) > 0 and len(peak_coords) > 0:
        print("[i] Вычисление геодезических путей (Dijkstra)...")
        # # Размеры под-блоков для Дейкстры и размеры перекрытий (оверлапов)
        # step_z, step_y, step_x = 128, 256, 256
        # overlap_z, overlap_y, overlap_x = 48, 64, 64  
        # nz, ny, nx = mask.shape
        
        # for z0 in range(0, nz, step_z):
        #     z1 = min(z0 + step_z, nz)
        #     for y0 in range(0, ny, step_y):
        #         y1 = min(y0 + step_y, ny)
        #         for x0 in range(0, nx, step_x):
        #             x1 = min(x0 + step_x, nx)
                    
        #             # Отбираем макушки, принадлежащие текущему блоку
        #             block_peaks = peak_coords[
        #                 (peak_coords[:, 0] >= z0) & (peak_coords[:, 0] < z1) &
        #                 (peak_coords[:, 1] >= y0) & (peak_coords[:, 1] < y1) &
        #                 (peak_coords[:, 2] >= x0) & (peak_coords[:, 2] < x1)
        #             ]
        #             if len(block_peaks) == 0:
        #                 continue
                        
        #             # Вычисляем расширенные границы куска с оверлапом
        #             z_start_pad = max(0, z0 - overlap_z)
        #             z_end_pad = min(nz, z1 + overlap_z)
        #             y_start_pad = max(0, y0 - overlap_y)
        #             y_end_pad = min(ny, y1 + overlap_y)
        #             x_start_pad = max(0, x0 - overlap_x)
        #             x_end_pad = min(nx, x1 + overlap_x)
                    
        #             chunk_slice = np.s_[z_start_pad:z_end_pad, y_start_pad:y_end_pad, x_start_pad:x_end_pad]
        #             local_mask = mask[chunk_slice]
        #             local_skeleton = skeleton[chunk_slice]
                    
        #             local_skel_coords = np.argwhere(local_skeleton)
        #             if len(local_skel_coords) == 0:
        #                 continue
                        
        #             # Строим ЛОКАЛЬНЫЙ MCP (выделяет всего ~200 МБ вместо 23 ГБ)
        #             local_cost = np.full(local_mask.shape, 10.0, dtype=np.float32)
        #             local_cost[local_mask] = 1.0

        #             mcp = MCP_Geometric(local_cost)
        #             mcp.find_costs(local_skel_coords)
                    
        #             for p_global in block_peaks:
        #                 p_local = (
        #                     p_global[0] - z_start_pad,
        #                     p_global[1] - y_start_pad,
        #                     p_global[2] - x_start_pad
        #                 )
        #                 try:
        #                     path = mcp.traceback(p_local)
        #                     for z_loc, y_loc, x_loc in path:
        #                         spine_axes[z_loc + z_start_pad, y_loc + y_start_pad, x_loc + x_start_pad] = True
        #                 except ValueError:
        #                     pass

        # print("[i] Прокладка осей завершена.")
        cost_surface = np.full(mask.shape, 10.0, dtype=np.float32)
        cost_surface[mask] = 1.0

        mcp = MCP_Geometric(cost_surface)
        mcp.find_costs(skel_coords)
        print("[i] Прокладка осей шипиков")
        successful_paths = 0
        for p_surf in peak_coords:
            try:
                # trace_back от макушки до скелета
                path = mcp.traceback(tuple(p_surf))
                for z, y, x in path:
                    spine_axes[z, y, x] = True
                successful_paths += 1
            except ValueError:
                # Если путь абсолютно невозможен
                pass
        print(f"[i] Успешно проложено осей: {successful_paths}")

    print("[i] Вычисление D_spine...")
    spine_axes_inverted = np.ascontiguousarray(~spine_axes)

    if not np.any(spine_axes):
        d_spine_norm = np.ones_like(mask, dtype=np.float32)
    else:
        d_spine = edt.edt(spine_axes_inverted, anisotropy=res_nm).astype(np.float32)  
        d_spine_norm = np.zeros_like(d_spine, dtype=np.float32)  
        if np.any(mask):
            mask_spine_dist = d_spine[mask]

            dist_range = mask_spine_dist.max() - mask_spine_dist.min()
            if dist_range > 0:
                d_spine_norm[mask] = (d_spine[mask] - mask_spine_dist.min()) / (mask_spine_dist.max() - mask_spine_dist.min() + 1e-8)
    # # Защита от пустых снимков
    # if len(skel_coords) > 0 and len(peak_coords) > 0:
    #     # Строим дерево поиска с учетом масштаба в нанометрах
    #     tree = KDTree(skel_coords * np.array(res_nm))
        
    #     for p_surf in peak_coords:
    #         z_surf, y_surf, x_surf = p_surf
    #     # Ищем ближайшую точку скелета к текущему пику
    #         dist, idx = tree.query(p_surf * np.array(res_nm))
    #         z_skel, y_skel, x_skel = skel_coords[idx]
    #         line_coords = line_nd((z_surf, y_surf, x_surf), (z_skel, y_skel, x_skel))
    #         spine_axes[line_coords] = True
    # print(f"D_spine")
    # # 5. D_spine (расстояние до осей)
    # spine_axes_inverted = np.ascontiguousarray(~spine_axes)
    # d_spine = edt.edt(spine_axes_inverted, anisotropy=res_nm)
    # d_spine_norm = np.zeros_like(d_spine, dtype=np.float32)
    # if np.any(mask):
    #     mask_spine_dist = d_spine[mask]
    #     d_spine_norm[mask] = (d_spine[mask] - mask_spine_dist.min()) / (mask_spine_dist.max() - mask_spine_dist.min() + 1e-8)

    print(f"[i] Генерация каналов завершена")
    return d_dendr_norm, d_spine_norm

def process_item(item, base_path):

    name = item['name']

    def resolve(p): 
        if p is None: return None
        return os.path.normpath(os.path.join(base_path, p))

        
    skel_path = resolve(item.get('skeleton'))
    vsot_shaft_path = resolve(item.get('shaft_vsot'))
    vsot_spine_path = resolve(item.get('spine_vsot'))
    shaft_distance_path = resolve(item.get('shaft_dist'))
    spine_distance_path = resolve(item.get('spine_dist'))
    
    scale = item.get('scale', None)
    res_nm = (scale[0]*1000, scale[1]*1000, scale[2]*1000)

    skeleton = imread(skel_path) > 0
    vsot_shaft = imread(vsot_shaft_path) > 0
    vsot_spine = imread(vsot_spine_path) > 0

    vsot_dendrite = vsot_shaft | vsot_spine

    coords = np.argwhere(vsot_dendrite)
    if len(coords) == 0:
        return 0  # Пустой снимок
        
    # Запас (padding) равен размеру патча, чтобы повороты не обрезали дендрит
    pad_z, pad_y, pad_x = FRAGMENT_SHAPE
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    orig_shape = vsot_dendrite.shape
    z_start = max(0, z_min - pad_z)
    z_end   = min(orig_shape[0], z_max + pad_z + 1)
    y_start = max(0, y_min - pad_y)
    y_end   = min(orig_shape[1], y_max + pad_y + 1)
    x_start = max(0, x_min - pad_x)
    x_end   = min(orig_shape[2], x_max + pad_x + 1)

    crop_slice = np.s_[z_start:z_end, y_start:y_end, x_start:x_end]

    skeleton = skeleton[crop_slice]

    vsot_dendrite = vsot_dendrite[crop_slice]


    d_dendr_ch2, d_spine_ch3 = generate_distance_channels(vsot_dendrite, skeleton, res_nm)

    d_dendr_full = np.zeros(orig_shape, dtype=np.float32)
    d_spine_full = np.zeros(orig_shape, dtype=np.float32)

    # Вставляем посчитанные кропнутые участки обратно на свои глобальные позиции
    d_dendr_full[crop_slice] = d_dendr_ch2
    d_spine_full[crop_slice] = d_spine_ch3

    os.makedirs(os.path.dirname(shaft_distance_path), exist_ok=True)
    os.makedirs(os.path.dirname(spine_distance_path), exist_ok=True)

    imwrite(shaft_distance_path, d_dendr_full)
    imwrite(spine_distance_path, d_spine_full)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--base_path", default="../../")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    
    with open(args.json_path) as f: 
        cfg = json.load(f)
        
    items = cfg['data'] # Считаем геометрию вообще для ВСЕХ снимков без деления на train/val
    print(f"подсчет матриц расстояний геометрии для {len(items)} снимков...")
    
    worker_func = partial(process_item, base_path=args.base_path)
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, items), total=len(items), desc="Общий прогресс"))