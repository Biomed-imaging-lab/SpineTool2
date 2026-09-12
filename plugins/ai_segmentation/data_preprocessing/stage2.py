import os
import json
import numpy as np
import argparse
import random
import cv2
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy import ndimage
from tifffile import imread
from scipy.ndimage import gaussian_filter, rotate
from skimage.measure import label
from data_preprocessing.stage1 import normalize_minmax, normalize_image, pad_to_min_shape, get_fragment, scale_image
# from stage1 import normalize_minmax, normalize_image, pad_to_min_shape, get_fragment, scale_image
# --- КОНСТАНТЫ ---
FRAGMENT_SHAPE = [64, 64, 64]
FINAL_SHAPE = [64, 64, 64]
BASE_SCALE = [0.1, 0.1]

ANGLES_POOL = [i * 10 for i in range(36)]
ANGLES_COUNT_TRAIN = 3
POSITIONS = [[0, 0, 0], [-16, -32, -32]]

def calculate_interim_image(image, scale):
    """
    Вычисляет I_interim:
    1. Гаусс
    2. масштабирование
    3. Уменьшение интенсивности на 70% от среднего.
    4. Обрезка отрицательных.
    5. MinMax нормализация -> * 255.
    """
    for i in range(image.shape[0]):
        image[i] = gaussian_filter(image[i], sigma=2)

    if scale:
        image = scale_image(image, scale)

    mean_val = np.mean(image)
    img_reduced = image - (0.7 * mean_val)
    img_clipped = np.maximum(img_reduced, 0)

    img_clipped = normalize_minmax(img_clipped)
        
    return img_clipped * 255.0

def find_optimal_j0(i_interim, l_bin):
    """
    Поиск j0 на основе анализа связности групп.
    """

    labeled_dendrite, num_features = label(l_bin, return_num=True, connectivity=3)
    print(f"   [-] Базовое количество компонент (L_bin): {num_features}")
    if num_features == 0:
        print("num_features=0")
        return 2
    
    group_counts = {}

    for j in range(2, 27):
        thresh_val = j * 8
        m_j = (i_interim >= thresh_val).astype(np.int32)
        
        # если M_j пустая, связности нет, количество групп = количеству кусков L_bin
        if np.sum(m_j) == 0:
            group_counts[j] = num_features
            continue
        
        union_mask = np.logical_or(l_bin, m_j)
        labeled_union, _ = label(union_mask, return_num=True, connectivity=3)

        unique_labels = np.unique(labeled_union[l_bin > 0])
        
        group_counts[j] = len(unique_labels)

    min_groups = min(group_counts.values())

    best_j = 2
    for j in range(2, 27):
        if group_counts[j] == min_groups:
            best_j = j
        else:
            pass
            
    return best_j

def calculate_isi(i_interim, j0):
    """
    Нелинейное масштабирование интенсивности на основе j0.
    """
    val_j0_minus = (j0 - 1) * 8
    val_j0_plus = (j0 + 1) * 8
    
    i_si = np.zeros_like(i_interim, dtype=np.float32)
    
    # Условие 1: 0 .. (j0-1)*8 -> 0.0 .. 0.4
    mask1 = i_interim < val_j0_minus
    if val_j0_minus > 0:
        i_si[mask1] = (i_interim[mask1] / val_j0_minus) * 0.4
    
    # Условие 2: (j0-1)*8 .. (j0+1)*8 -> 0.4 .. 0.8
    mask2 = (i_interim >= val_j0_minus) & (i_interim <= val_j0_plus)
    range2 = val_j0_plus - val_j0_minus
    if range2 > 0:
        i_si[mask2] = 0.4 + ((i_interim[mask2] - val_j0_minus) / range2) * 0.4
        
    # Условие 3: > (j0+1)*8 -> 0.8 .. 1.0
    mask3 = i_interim > val_j0_plus

    denom = 255.0 - val_j0_plus
    if denom > 0:
        i_si[mask3] = 0.8 + ((i_interim[mask3] - val_j0_plus) / denom) * 0.2
    else:
        i_si[mask3] = 1.0
        
    return i_si

def split_and_save(img_c, l_dendrite, isi, mask, g_norm, name, angle, start_pos, output_dir):
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
                if np.sum(p_mask_crop) < 300: continue # Пропускаем пустые куски
                
                p_img_crop, _ = get_fragment(img_c, curr_pos)
                p_gnorm_crop, _ = get_fragment(g_norm, curr_pos)
                p_ldendrite_crop, _ = get_fragment(l_dendrite, curr_pos)
                p_isi_crop, _ = get_fragment(isi, curr_pos)
                
                # --- Сборка 4 каналов ---
                ch1 = p_gnorm_crop # ||Ic||p
                ch2 = normalize_minmax(p_img_crop) # ||p||
                ch3 = p_ldendrite_crop             # L_dendrite_p
                ch4 = p_isi_crop                   # I_si_p
                
                full_tensor = np.stack([ch1, ch2, ch3, ch4], axis=0).astype(np.float32)
                
                fname = f"{name}_a{angle}_p{start_pos[0]}{start_pos[1]}{start_pos[2]}_idx{k}-{j}-{i}.npz"
                save_path = os.path.join(output_dir, fname)
                
                np.savez_compressed(save_path, image=full_tensor, mask=p_mask_crop, borders=borders)
                saved_count += 1
    return saved_count

def process_item(item, output_dir, is_train_mode, base_path, preds_dir):
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
    if area is not None:
        raw_img = raw_img * area

    # 2. Загрузка предсказаний 1-го этапа
    l_dendrite_path = os.path.join(preds_dir, f"{name}_L_dendrite.tif")
    if not os.path.exists(l_dendrite_path):
        print(f"Skipping {name}: L_dendrite missing at {l_dendrite_path}")
        return 0
    l_dendrite_raw = imread(l_dendrite_path)

    scale = item.get('scale', None)

    # 3. Вычисление математики 2-го этапа
    i_interim = calculate_interim_image(raw_img, scale)
    # if scale:
        # l_dendrite_raw = scale_image(l_dendrite_raw, scale)
    l_bin = (l_dendrite_raw > 0.5).astype(np.uint8)
    j0 = find_optimal_j0(i_interim, l_bin)
    print(f"j {j0}")
    i_si_raw = calculate_isi(i_interim, j0)

    # 4. Гаусс+масштабирование+усреднение+нормализация I_c
    img_c_raw = normalize_image(raw_img, scale)

    # 5. Загрузка масок
    g_path = resolve(item.get('general_mask'))
    s_path = resolve(item.get('spines_mask'))
    h_path = resolve(item.get('shaft_mask'))

    mask = None
    if g_path and os.path.exists(g_path): mask = imread(g_path)
    elif s_path and h_path and os.path.exists(s_path) and os.path.exists(h_path):
        s_mask, h_mask = imread(s_path), imread(h_path)
        mask = np.maximum(s_mask, h_mask)    

    if area is not None: mask = mask * area
    mask[mask > 0] = 1


    if scale:
        mask = scale_image(mask.astype(np.uint8), scale)

    # 7. Паддинг всех слоев
    img_c_raw = pad_to_min_shape(img_c_raw)
    l_dendrite_raw = pad_to_min_shape(l_dendrite_raw)
    i_si_raw = pad_to_min_shape(i_si_raw)
    mask = pad_to_min_shape(mask)

    # 8. Нарезка и Аугментация
    angles = random.sample(ANGLES_POOL, ANGLES_COUNT_TRAIN) if is_train_mode else [0]
    total_saved = 0
    
    for angle in angles:
        if angle != 0:
            img_rot = rotate(img_c_raw, angle, axes=(2, 1), reshape=True, order=1)
            l_dendrite_rot = rotate(l_dendrite_raw, angle, axes=(2, 1), reshape=True, order=1)
            isi_rot = rotate(i_si_raw, angle, axes=(2, 1), reshape=True, order=1)
            mask_rot = rotate(mask, angle, axes=(2, 1), reshape=True, order=0)
        else:
            img_rot, l_dendrite_rot, isi_rot, mask_rot = img_c_raw, l_dendrite_raw, i_si_raw, mask
            
        g_norm = normalize_minmax(img_rot)    
        for pos in POSITIONS:
            if not is_train_mode and pos != [0,0,0]: continue
            saved = split_and_save(img_rot, l_dendrite_rot, isi_rot, mask_rot, g_norm, name, angle, pos, volume_output_dir)
            total_saved += saved
            
    return total_saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--preds_dir", required=True, help="Путь к папке с L_dendrite_preds")
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
        preds_dir=args.preds_dir # Передаем путь к предсказаниям
    )
    
    total_patches = 0
    with Pool(processes=args.workers) as pool:
        for count in tqdm(pool.imap_unordered(worker_func, items), total=len(items)):
            total_patches += count
    print(f" Generated {total_patches} 4-channel patches for Stage 2.")