import os
import json
import numpy as np
import argparse
import random
import cv2
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

import edt
import networkx as nx
from tifffile import imread, imwrite
from scipy.ndimage import rotate
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from skimage.measure import label
from skimage.draw import line_nd
# from stage1 import normalize_minmax

from data_preprocessing.stage1 import normalize_minmax

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
    d_dendr = edt.edt(skeleton_inverted, anisotropy=res_nm)
    
    d_dendr_norm = np.zeros_like(d_dendr, dtype=np.float32)
    if np.any(mask):
        mask_dist = d_dendr[mask]
        d_dendr_norm[mask] = (d_dendr[mask] - mask_dist.min()) / (mask_dist.max() - mask_dist.min() + 1e-8)
    print(f"D_dendr")
    # 2. Поиск поверхности
    eroded_mask = ndi.binary_erosion(mask)
    surface = mask & ~eroded_mask

    # 3. Поиск осей шипиков (окрестность 500 нм)
    radius_nm = 500.0
    win_z = max(1, int(np.round(radius_nm / res_nm[0])))
    win_y = max(1, int(np.round(radius_nm / res_nm[1])))
    win_x = max(1, int(np.round(radius_nm / res_nm[2])))
    
    surface_dist = np.zeros_like(d_dendr)
    surface_dist[surface] = d_dendr[surface]

    local_max = ndi.maximum_filter(surface_dist, size=(win_z, win_y, win_x))
    peaks_mask = surface & (surface_dist == local_max) & (surface_dist > 0)
    peak_coords = np.argwhere(peaks_mask)

    # 4. Построение предполагаемых осей шипиков
    spine_axes = np.zeros_like(mask, dtype=bool)
    skel_coords = np.argwhere(skeleton)
    
    # Защита от пустых снимков
    if len(skel_coords) > 0 and len(peak_coords) > 0:
        # Строим дерево поиска с учетом масштаба в нанометрах
        tree = KDTree(skel_coords * np.array(res_nm))
        
        for p_surf in peak_coords:
            z_surf, y_surf, x_surf = p_surf
        # Ищем ближайшую точку скелета к текущему пику
            dist, idx = tree.query(p_surf * np.array(res_nm))
            z_skel, y_skel, x_skel = skel_coords[idx]
            line_coords = line_nd((z_surf, y_surf, x_surf), (z_skel, y_skel, x_skel))
            spine_axes[line_coords] = True
    print(f"D_spine")
    # 5. D_spine (расстояние до осей)
    spine_axes_inverted = np.ascontiguousarray(~spine_axes)
    d_spine = edt.edt(spine_axes_inverted, anisotropy=res_nm)
    d_spine_norm = np.zeros_like(d_spine, dtype=np.float32)
    if np.any(mask):
        mask_spine_dist = d_spine[mask]
        d_spine_norm[mask] = (d_spine[mask] - mask_spine_dist.min()) / (mask_spine_dist.max() - mask_spine_dist.min() + 1e-8)

    print(f"D_all")
    return d_dendr_norm, d_spine_norm

def pad_to_min_shape(volume):
    current = volume.shape
    min_z, min_y, min_x = FINAL_SHAPE
    pad_z, pad_y, pad_x = max(0, min_z - current[0]), max(0, min_y - current[1]), max(0, min_x - current[2])
    if pad_z == 0 and pad_y == 0 and pad_x == 0: return volume
    pads = [(0, pad_z), (0, pad_y), (0, pad_x)]
    return np.pad(volume, pads, mode='constant', constant_values=0)

def get_fragment(image, start_pos, shape=FINAL_SHAPE):
    """
    Вырезает патч 64x128x128. 
    Если выходит за границы или start_pos отрицательный -> заливает нулями.
    """
    d, h, w = image.shape

    z_min, y_min, x_min = start_pos
    z_max, y_max, x_max = z_min + shape[0], y_min + shape[1], x_min + shape[2]

    img_z_min, img_y_min, img_x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    img_z_max, img_y_max, img_x_max = min(d, z_max), min(h, y_max), min(w, x_max)

    patch = np.zeros(shape, dtype=image.dtype)

    out_z_min = img_z_min - z_min
    out_y_min = img_y_min - y_min
    out_x_min = img_x_min - x_min

    out_z_max = out_z_min + (img_z_max - img_z_min)
    out_y_max = out_y_min + (img_y_max - img_y_min)
    out_x_max = out_x_min + (img_x_max - img_x_min)

    # Если пересечения нет
    if out_z_max <= out_z_min or out_y_max <= out_y_min or out_x_max <= out_x_min:
        return patch, None
    
    patch[out_z_min:out_z_max, out_y_min:out_y_max, out_x_min:out_x_max] = \
        image[img_z_min:img_z_max, img_y_min:img_y_max, img_x_min:img_x_max]
    
    borders = np.array([
        [out_z_min, out_z_max], 
        [out_y_min, out_y_max], 
        [out_x_min, out_x_max]
    ], dtype=np.int32)

    return patch, borders

def split_and_save(i_vsot, d_dendr, d_spine, mask, name, angles, start_pos, global_offset, output_dir):
    frag_z, frag_y, frag_x = FRAGMENT_SHAPE
    z_count = max(1, int(np.ceil((i_vsot.shape[0] - start_pos[0]) / frag_z)))
    y_count = max(1, int(np.ceil((i_vsot.shape[1] - start_pos[1]) / frag_y)))
    x_count = max(1, int(np.ceil((i_vsot.shape[2] - start_pos[2]) / frag_x)))

    pad_y, pad_x = 28, 28  # (184 - 128) / 2 = 28
    large_shape = [frag_z, frag_y + 2 * pad_y, frag_x + 2 * pad_x]

    saved_count = 0
    for k in range(z_count):
        for j in range(y_count):
            for i in range(x_count):
                curr_pos = [start_pos[0] + k*frag_z, start_pos[1] + j*frag_y, start_pos[2] + i*frag_x]
                large_pos = [curr_pos[0], curr_pos[1] - pad_y, curr_pos[2] - pad_x]
                p_mask_crop, borders = get_fragment(mask, large_pos, shape=large_shape)
                if borders is None: continue # Invalid crop
                
                if np.sum(p_mask_crop) < 300: continue
                
                l_vsot, _ = get_fragment(i_vsot, large_pos, shape=large_shape)
                l_dendr, _ = get_fragment(d_dendr, large_pos, shape=large_shape)
                l_spine, _ = get_fragment(d_spine, large_pos, shape=large_shape)
                l_mask, _ = get_fragment(mask, large_pos, shape=large_shape)
                
                global_z = global_offset[0] + curr_pos[0]
                global_y = global_offset[1] + curr_pos[1]
                global_x = global_offset[2] + curr_pos[2]
                
                for angle in angles:
                    if angle == 0:
                        # Если не крутим — просто берем сердцевину 128x128
                        ch1 = l_vsot[:, pad_y:-pad_y, pad_x:-pad_x]
                        ch2 = l_dendr[:, pad_y:-pad_y, pad_x:-pad_x]
                        ch3 = l_spine[:, pad_y:-pad_y, pad_x:-pad_x]
                        p_mask_crop = l_mask[:, pad_y:-pad_y, pad_x:-pad_x]
                    else:
                        # Поворачиваем БОЛЬШИЕ патчи
                        ch1_r = rotate(l_vsot, angle, axes=(2, 1), reshape=False, order=1)
                        ch2_r = rotate(l_dendr, angle, axes=(2, 1), reshape=False, order=1)
                        ch3_r = rotate(l_spine, angle, axes=(2, 1), reshape=False, order=1)
                        mask_r = rotate(l_mask, angle, axes=(2, 1), reshape=False, order=0)
                        
                        # Отрезаем рамку, возвращаясь к 64x128x128 (черных углов не будет!)
                        ch1 = ch1_r[:, pad_y:-pad_y, pad_x:-pad_x]
                        ch2 = ch2_r[:, pad_y:-pad_y, pad_x:-pad_x]
                        ch3 = ch3_r[:, pad_y:-pad_y, pad_x:-pad_x]
                        p_mask_crop = mask_r[:, pad_y:-pad_y, pad_x:-pad_x]

                    full_tensor = np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)
                
                    fname = f"{name}_a{angle}_p{global_z}{global_y}{global_x}_idx{k}-{j}-{i}.npz"
                    save_path = os.path.join(output_dir, fname)
                    np.savez_compressed(save_path, image=full_tensor, mask=p_mask_crop, borders=borders)
                    saved_count += 1
    return saved_count

def process_item(item, output_dir, is_train_mode, base_path):
    np.random.seed(); random.seed()
    name = item['name']
    volume_output_dir = os.path.join(output_dir, name)
    os.makedirs(volume_output_dir, exist_ok=True)

    def resolve(p): 
        if p is None: return None
        return os.path.normpath(os.path.join(base_path, p))
    
    # 1. Загрузка исходников
    area_path = resolve(item.get('area_of_interest'))
    area = None
    if area_path and os.path.exists(area_path):
        area = imread(area_path).astype(np.uint8)
        area[area > 0] = 1
        
    skel_path = resolve(item.get('skeleton'))
    general_mask_path = resolve(item.get('image'))
    shaft_path = resolve(item.get('shaft_mask'))
    spine_path = resolve(item.get('spine_mask'))
    scale = item.get('scale', None)
    res_nm = (scale[0]*1000, scale[1]*1000, scale[2]*1000)

    skeleton = imread(skel_path) > 0
    general_mask = imread(general_mask_path) > 0
    shaft = imread(shaft_path) > 0
    spine = imread(spine_path) > 0

    vsot_dendrite = general_mask

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
    global_offset = [z_start, y_start, x_start] # Запоминаем для имен файлов

    skeleton = skeleton[crop_slice]
    shaft = shaft[crop_slice]
    spine = spine[crop_slice]
    vsot_dendrite = vsot_dendrite[crop_slice]
    if area is not None:
        area = area[crop_slice]


    # 2. Формирование I_VSOT (Канал 1)
    i_vsot = np.zeros_like(vsot_dendrite, dtype=np.float32)
    i_vsot[vsot_dendrite] = 1.0
    i_vsot_norm = normalize_minmax(i_vsot)

    d_dendr_ch2, d_spine_ch3 = generate_distance_channels(vsot_dendrite, skeleton, res_nm)

    gt_mask = np.zeros_like(vsot_dendrite, dtype=np.uint8)
    gt_mask[shaft] = 1
    gt_mask[spine] = 2


    if area is not None: 
        i_vsot_norm = i_vsot_norm * area
        d_dendr_ch2 = d_dendr_ch2 * area
        d_spine_ch3 = d_spine_ch3 * area
        gt_mask = gt_mask * area

    # 6. Паддинг
    i_vsot_norm = pad_to_min_shape(i_vsot_norm)
    d_dendr_ch2 = pad_to_min_shape(d_dendr_ch2)
    d_spine_ch3 = pad_to_min_shape(d_spine_ch3)
    gt_mask = pad_to_min_shape(gt_mask)

    # 7. Нарезка и Аугментация
    angles = random.sample(ANGLES_POOL, ANGLES_COUNT_TRAIN) if is_train_mode else [0]
    total_saved = 0
    
            
    for pos in POSITIONS:
        if not is_train_mode and pos != [0,0,0]: continue
        saved = split_and_save(i_vsot_norm, d_dendr_ch2, d_spine_ch3, gt_mask, name, angles, pos, global_offset, volume_output_dir)
        total_saved += saved
            
    return total_saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["train", "val", "test"])
    parser.add_argument("--base_path", default=".")
    parser.add_argument("--workers", type=int, default=cpu_count()) 
    args = parser.parse_args()
    
    final_output_dir = os.path.join(args.out_dir, args.mode)
    os.makedirs(final_output_dir, exist_ok=True)
    random.seed(42)
    
    with open(args.json_path) as f: cfg = json.load(f)
    items = []
    ts, vs = set(cfg.get('test_data', [])), set(cfg.get('validation_data', []))
    for x in cfg['data']:
        n = x['name']
        if args.mode == 'train' and n not in ts and n not in vs: items.append(x)
        elif args.mode == 'val' and n in vs: items.append(x)
        elif args.mode == 'test' and n in ts: items.append(x)
        
    print(f"Mode: {args.mode}. Items: {len(items)}")
    worker_func = partial(
        process_item, 
        output_dir=final_output_dir, 
        is_train_mode=(args.mode == 'train'), 
        base_path=args.base_path,
    )
    
    total_patches = 0
    with Pool(processes=args.workers) as pool:
        for count in tqdm(pool.imap_unordered(worker_func, items), total=len(items)):
            total_patches += count
    print(f" Generated {total_patches} patches (128x128x64) for Stage 4.")