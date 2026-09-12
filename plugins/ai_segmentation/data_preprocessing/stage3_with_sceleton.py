import os
import json
import numpy as np
import argparse
import random
import cv2
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from tifffile import imread
from scipy.ndimage import gaussian_filter, rotate
from skimage.measure import label

# from data_preprocessing.stage1 import normalize_minmax
# from data_preprocessing.stage2 import calculate_interim_image, find_optimal_j0, calculate_isi
from stage1 import normalize_minmax
from stage2 import calculate_interim_image, find_optimal_j0, calculate_isi
FRAGMENT_SHAPE = [64, 128, 128] 
FINAL_SHAPE = [64, 128, 128]
BASE_SCALE = [0.1, 0.1]

ANGLES_POOL = [i * 10 for i in range(36)]
ANGLES_COUNT_TRAIN = 3
POSITIONS = [[0, 0, 0], [-32, -64, -64]]

def scale_to_shape(image, target_shape):

    target_z, target_y, target_x = target_shape
    
    # Если размеры уже совпадают, возвращаем
    if image.shape[1] == target_y and image.shape[2] == target_x:
        return image
    interp = cv2.INTER_LINEAR 
    
    resized = np.zeros((image.shape[0], target_y, target_x), dtype=np.float32)
    for i in range(image.shape[0]):
        resized[i] = cv2.resize(image[i].astype(np.float32), (target_x, target_y), interpolation=interp)
    return resized

def apply_gaussian_3d_stack(image, sigma=2.0):
    """Применяет Гаусс сглаживание к каждому срезу стека (для предотвращения артефактов)"""
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        blurred[i] = gaussian_filter(image[i], sigma=sigma)
    return blurred

def pad_to_min_shape(volume):
    current = volume.shape
    min_z, min_y, min_x = FINAL_SHAPE
    pad_z, pad_y, pad_x = max(0, min_z - current[0]), max(0, min_y - current[1]), max(0, min_x - current[2])
    if pad_z == 0 and pad_y == 0 and pad_x == 0: return volume
    pads = [(0, pad_z), (0, pad_y), (0, pad_x)]
    return np.pad(volume, pads, mode='constant', constant_values=0)

def get_fragment(image, start_pos):
    """
    Вырезает патч 64x128x128. 
    Если выходит за границы или start_pos отрицательный -> заливает нулями.
    """
    shape = FINAL_SHAPE
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

def split_and_save(img_c, l1_s, l2_s, isi_s, mask, skeleton, name, angle, start_pos, output_dir):
    frag_z, frag_y, frag_x = FRAGMENT_SHAPE
    z_count = max(1, int(np.ceil((img_c.shape[0] - start_pos[0]) / frag_z)))
    y_count = max(1, int(np.ceil((img_c.shape[1] - start_pos[1]) / frag_y)))
    x_count = max(1, int(np.ceil((img_c.shape[2] - start_pos[2]) / frag_x)))

    saved_count = 0
    for k in range(z_count):
        for j in range(y_count):
            for i in range(x_count):
                curr_pos = [start_pos[0] + k*frag_z, start_pos[1] + j*frag_y, start_pos[2] + i*frag_x]
                
                p_mask_crop, borders = get_fragment(mask, curr_pos)
                if borders is None: continue 
                
                # Порог валидности патча (можно подкрутить, если шипики очень мелкие)
                if np.sum(p_mask_crop) < 100: continue 
                p_skeleton_crop, _ = get_fragment(skeleton, curr_pos)

                p_img_crop, _ = get_fragment(img_c, curr_pos)
                p_l1_crop, _ = get_fragment(l1_s, curr_pos)
                p_l2_crop, _ = get_fragment(l2_s, curr_pos)
                p_isi_crop, _ = get_fragment(isi_s, curr_pos)
                
                # --- Сборка 4 каналов по статье ---
                ch1 = p_img_crop                   # ||I||p'
                ch2 = p_isi_crop                   # I_si,s p'
                ch3 = p_l1_crop                    # L1,s p'
                ch4 = p_l2_crop                    # L2,s p'
                
                full_tensor = np.stack([ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
                
                combined_mask = np.stack([p_mask_crop, p_skeleton_crop], axis=0).astype(np.float32)

                fname = f"{name}_a{angle}_p{start_pos[0]}{start_pos[1]}{start_pos[2]}_idx{k}-{j}-{i}.npz"
                save_path = os.path.join(output_dir, fname)
                np.savez_compressed(save_path, image=full_tensor, mask=combined_mask, borders=borders)
                saved_count += 1
    return saved_count

def process_item(item, output_dir, is_train_mode, base_path, preds_stage1_dir, preds_stage2_dir):
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
        
    img_path = resolve(item.get('image'))
    if not img_path or not os.path.exists(img_path): return 0
    raw_img = imread(img_path)
    if area is not None: raw_img = raw_img * area

    # 2. Загрузка предсказаний L1 и L2
    l1_path = os.path.join(preds_stage1_dir, f"{name}_L_dendrite.tif")
    l2_path = os.path.join(preds_stage2_dir, f"{name}_necks.tif")
    
    if not os.path.exists(l1_path) or not os.path.exists(l2_path):
        print(f"Skipping {name}: L1 or L2 predictions missing.")
        return 0
        
    l1_raw = imread(l1_path)
    l2_raw = imread(l2_path)
    scale = item.get('scale', None)
    target_high_res_shape = raw_img.shape
    # Масштабируем L1 для расчета j0
    # if scale:
    #     l1_scaled = scale_image(l1_raw, scale)
    # else: l1_scaled = l1_raw

    # 3. Базовый снимок и Вычисление Isi
    i_interim = calculate_interim_image(raw_img, scale)
    l1_bin = (l1_raw > 0.5).astype(np.uint8)
    j0 = find_optimal_j0(i_interim, l1_bin)
    I_si = calculate_isi(i_interim, j0)

    l1_up = scale_to_shape(l1_raw, target_high_res_shape)
    l2_up = scale_to_shape(l2_raw, target_high_res_shape)
    isi_up = scale_to_shape(I_si, target_high_res_shape)
    l1_s = apply_gaussian_3d_stack(l1_up, sigma=2.0)
    l2_s = apply_gaussian_3d_stack(l2_up, sigma=2.0)
    isi_s = apply_gaussian_3d_stack(isi_up, sigma=2.0)

    I_norm = normalize_minmax(raw_img)

    # 5. Загрузка масок
    g_path = resolve(item.get('general_mask'))
    s_path = resolve(item.get('spines_mask'))
    h_path = resolve(item.get('shaft_mask'))

    skel_path = resolve(item.get('skeleton'))
    skeleton = imread(skel_path).astype(np.float32)

    mask = None
    if g_path and os.path.exists(g_path): mask = imread(g_path)
    elif s_path and h_path and os.path.exists(s_path) and os.path.exists(h_path):
        s_mask, h_mask = imread(s_path), imread(h_path)
        mask = np.maximum(s_mask, h_mask) 

    if area is not None: 
        mask = mask * area
        skeleton = skeleton * area
    mask[mask > 0] = 1
    skeleton[skeleton > 0] = 1

    # 6. Паддинг
    I_s = pad_to_min_shape(I_norm)
    l1_s = pad_to_min_shape(l1_s)
    l2_s = pad_to_min_shape(l2_s)
    isi_s = pad_to_min_shape(isi_s)
    mask = pad_to_min_shape(mask)
    skeleton = pad_to_min_shape(skeleton)

    # 7. Нарезка и Аугментация
    angles = random.sample(ANGLES_POOL, ANGLES_COUNT_TRAIN) if is_train_mode else [0]
    total_saved = 0
    
    for angle in angles:
        if angle != 0:
            img_rot = rotate(I_s, angle, axes=(2, 1), reshape=True, order=1)
            l1_rot = rotate(l1_s, angle, axes=(2, 1), reshape=True, order=1)
            l2_rot = rotate(l2_s, angle, axes=(2, 1), reshape=True, order=1)
            isi_rot = rotate(isi_s, angle, axes=(2, 1), reshape=True, order=1)
            mask_rot = rotate(mask, angle, axes=(2, 1), reshape=True, order=0)
            skeleton_rot = rotate(skeleton, angle, axes=(2, 1), reshape=True, order=0)
        else:
            img_rot, l1_rot, l2_rot, isi_rot, mask_rot, skeleton_rot = I_s, l1_s, l2_s, isi_s, mask, skeleton
            
        for pos in POSITIONS:
            if not is_train_mode and pos != [0,0,0]: continue
            saved = split_and_save(img_rot, l1_rot, l2_rot, isi_rot, mask_rot, skeleton_rot, name, angle, pos, volume_output_dir)
            total_saved += saved
            
    return total_saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--l1_dir", required=True, help="Путь к предсказаниям L1 (Дендриты)")
    parser.add_argument("--l2_dir", required=True, help="Путь к предсказаниям L2 (Шейки/Дендриты)")
    parser.add_argument("--mode", required=True, choices=["train", "val", "test"])
    parser.add_argument("--base_path", default="../../")
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
        preds_stage1_dir=args.l1_dir,
        preds_stage2_dir=args.l2_dir
    )
    
    total_patches = 0
    with Pool(processes=args.workers) as pool:
        for count in tqdm(pool.imap_unordered(worker_func, items), total=len(items)):
            total_patches += count
    print(f" Generated {total_patches} patches (128x128x64) for Stage 3.")