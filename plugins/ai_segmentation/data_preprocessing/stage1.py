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
from skimage.filters import threshold_multiotsu

# --- КОНСТАНТЫ ---
FRAGMENT_SHAPE = [64, 64, 64]
PAD = 16
FINAL_SHAPE = [64, 64, 64]
OTSU_CLASSES = 4
BASE_SCALE = [0.1, 0.1]

ANGLES_POOL = [i * 10 for i in range(36)]
ANGLES_COUNT_TRAIN = 3
POSITIONS = [[0, 0, 0], [-32, -32, -32]]

def normalize_minmax(data):
    d_min, d_max = np.min(data), np.max(data)
    if d_max - d_min > 0:
        return (data - d_min) / (d_max - d_min)
    return np.zeros_like(data)

def get_otsu_map(image):
    try:
        thresholds = threshold_multiotsu(image, classes=OTSU_CLASSES)
        return np.digitize(image, thresholds).astype(np.float32)
    except:
        return np.zeros_like(image, dtype=np.float32)

def normalize_image(image, scale):
    image = image.astype(np.float32)
    for i in range(image.shape[0]):
        image[i] = gaussian_filter(image[i], sigma=2)

    if scale:
        image = scale_image(image, scale)
    image = image - np.mean(image)
    image[image < 0] = 0
    mx = np.max(image)
    if mx > 0: image /= mx
    return image

def scale_image(image, current_scale):
    if current_scale is None: return image
    scale_y = current_scale[1] / BASE_SCALE[0]
    scale_x = current_scale[2] / BASE_SCALE[1]
    if abs(scale_y - 1.0) < 0.01 and abs(scale_x - 1.0) < 0.01: return image
    new_y = int(np.round(image.shape[1] * scale_y))
    new_x = int(np.round(image.shape[2] * scale_x))
    interp = cv2.INTER_NEAREST if np.issubdtype(image.dtype, np.integer) else cv2.INTER_AREA
    resized = np.zeros((image.shape[0], new_y, new_x), dtype=image.dtype)
    for i in range(image.shape[0]):
        resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=interp)
    return resized

def pad_to_min_shape(volume):
    current = volume.shape
    min_z, min_y, min_x = FINAL_SHAPE
    pad_z, pad_y, pad_x = max(0, min_z - current[0]), max(0, min_y - current[1]), max(0, min_x - current[2])
    if pad_z == 0 and pad_y == 0 and pad_x == 0: return volume
    pads = [(0, pad_z), (0, pad_y), (0, pad_x)]
    return np.pad(volume, pads, mode='constant', constant_values=0)

def get_fragment(image, start_pos):
    """
    Вырезает патч 64x64x64. 
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

def split_and_save(image, mask, g_norm, g_otsu, name, angle, start_pos, output_dir):
    
    frag_z, frag_y, frag_x = FRAGMENT_SHAPE
    z_count = max(1, int(np.ceil((image.shape[0] - start_pos[0]) / frag_z)))
    y_count = max(1, int(np.ceil((image.shape[1] - start_pos[1]) / frag_y)))
    x_count = max(1, int(np.ceil((image.shape[2] - start_pos[2]) / frag_x)))

    saved_count = 0
    for k in range(z_count):
        for j in range(y_count):
            for i in range(x_count):
                curr_pos = [start_pos[0] + k*frag_z, start_pos[1] + j*frag_y, start_pos[2] + i*frag_x]
                
                # Теперь получаем еще и borders
                p_mask_crop, borders = get_fragment(mask, curr_pos)
                if borders is None: continue # Invalid crop
                
                if np.sum(p_mask_crop) < 300: continue
                
                p_img_crop, _ = get_fragment(image, curr_pos)
                p_gnorm_crop, _ = get_fragment(g_norm, curr_pos)
                p_gotsu_crop, _ = get_fragment(g_otsu, curr_pos)
                
                if p_img_crop is None: continue

                p_lnorm = normalize_minmax(p_img_crop)
                p_lotsu = get_otsu_map(p_img_crop)
                
                ch1 = p_gnorm_crop
                ch2 = p_lnorm
                ch3 = (p_gotsu_crop + p_lotsu) / 6.0
                full_tensor = np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)
                
                fname = f"{name}_a{angle}_p{start_pos[0]}{start_pos[1]}{start_pos[2]}_idx{k}-{j}-{i}.npz"
                save_path = os.path.join(output_dir, fname)
                
                np.savez_compressed(save_path, 
                                    image=full_tensor, 
                                    mask=p_mask_crop, 
                                    borders=borders)
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
    
    area_path = resolve(item.get('area_of_interest'))
    area = None
    if area_path and os.path.exists(area_path):
        try:
            area = imread(area_path).astype(np.uint8)
            area[area > 0] = 1
        except Exception as e:
            print(f"Error loading area for {name}: {e}")
            return 0    
        
    img_path = resolve(item.get('image'))
    if not img_path or not os.path.exists(img_path): return 0
    try:
        raw_img = imread(img_path)
    except: return 0

    if area is not None:
        # Проверка размеров
        if raw_img.shape != area.shape:
             print(f"Shape: img {raw_img.shape} vs area {area.shape} for {name}")
             return 0
        raw_img = raw_img * area

    g_path = resolve(item.get('general_mask'))
    s_path = resolve(item.get('spines_mask'))
    h_path = resolve(item.get('shaft_mask'))

    mask = None
    if g_path and os.path.exists(g_path):
        mask = imread(g_path)
    elif s_path and h_path and os.path.exists(s_path) and os.path.exists(h_path):
        s_mask = imread(s_path)
        h_mask = imread(h_path)
        mask = np.maximum(s_mask, h_mask)    

    if area is not None:
        mask = mask * area
        
    mask[mask > 0] = 1

    scale = item.get('scale', None)
    img = normalize_image(raw_img, scale)

    if scale:
        mask = scale_image(mask.astype(np.uint8), scale)

    if PAD > 0:
        pad_tuple = ((PAD, PAD), (PAD, PAD), (PAD, PAD))
        img = np.pad(img, pad_tuple, mode='reflect')
        mask = np.pad(mask, pad_tuple, mode='reflect')

    img = pad_to_min_shape(img)
    mask = pad_to_min_shape(mask)

    angles = random.sample(ANGLES_POOL, ANGLES_COUNT_TRAIN) if is_train_mode else [0]
    total_saved_for_file = 0
    
    for angle in angles:
        if angle != 0:
            i_rot = rotate(img, angle, axes=(2, 1), reshape=True, order=1)
            m_rot = rotate(mask, angle, axes=(2, 1), reshape=True, order=0)
        else: i_rot, m_rot = img, mask
            
        g_norm = normalize_minmax(i_rot)
        g_otsu = get_otsu_map(i_rot)
        
        for pos in POSITIONS:
            if not is_train_mode and pos != [0,0,0]: continue
            saved = split_and_save(i_rot, m_rot, g_norm, g_otsu, name, angle, pos, volume_output_dir)
            total_saved_for_file += saved
            
    return total_saved_for_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--out_dir", required=True)
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
    worker_func = partial(process_item, output_dir=final_output_dir, is_train_mode=(args.mode == 'train'), base_path=args.base_path)
    
    total_patches = 0
    with Pool(processes=args.workers) as pool:
        for count in tqdm(pool.imap_unordered(worker_func, items), total=len(items)):
            total_patches += count
    print(f" Generated {total_patches} patches.")