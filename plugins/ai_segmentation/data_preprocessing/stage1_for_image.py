import os
import json
import numpy as np
import argparse
import cv2
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu

# --- КОНСТАНТЫ (из stage1.py) ---
FRAGMENT_SHAPE = [64, 64, 64]
FINAL_SHAPE = [64, 64, 64]
OTSU_CLASSES = 4
BASE_SCALE = [0.1, 0.1]

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

def normalize_image(image, scale):
    image = image.astype(np.float32)
    for i in range(image.shape[0]):
        image[i] = gaussian_filter(image[i], sigma=1)
    if scale:
        image = scale_image(image, scale)
    image = image - np.mean(image)
    image[image < 0] = 0
    mx = np.max(image)
    if mx > 0: image /= mx
    return image

def pad_to_min_shape(volume):
    current = volume.shape
    min_z, min_y, min_x = FINAL_SHAPE
    pad_z = max(0, min_z - current[0])
    pad_y = max(0, min_y - current[1])
    pad_x = max(0, min_x - current[2])
    
    if pad_z == 0 and pad_y == 0 and pad_x == 0: 
        return volume, (0, 0, 0)
        
    pads = [(0, pad_z), (0, pad_y), (0, pad_x)]
    return np.pad(volume, pads, mode='constant', constant_values=0), (pad_z, pad_y, pad_x)

def get_fragment_and_coords(image, start_pos):
    """
    Модифицированная функция вырезки. Возвращает патч и абсолютные координаты 
    для обратной вставки в полное изображение.
    """
    shape = FINAL_SHAPE
    d, h, w = image.shape
    z_min, y_min, x_min = start_pos
    z_max, y_max, x_max = z_min + shape[0], y_min + shape[1], x_min + shape[2]

    img_z_min, img_y_min, img_x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    img_z_max, img_y_max, img_x_max = min(d, z_max), min(h, y_max), min(w, x_max)

    patch = np.zeros(shape, dtype=image.dtype)

    out_z_min, out_y_min, out_x_min = img_z_min - z_min, img_y_min - y_min, img_x_min - x_min
    out_z_max, out_y_max, out_x_max = out_z_min + (img_z_max - img_z_min), out_y_min + (img_y_max - img_y_min), out_x_min + (img_x_max - img_x_min)

    if out_z_max <= out_z_min or out_y_max <= out_y_min or out_x_max <= out_x_min:
        return patch, None, None

    patch[out_z_min:out_z_max, out_y_min:out_y_max, out_x_min:out_x_max] = \
        image[img_z_min:img_z_max, img_y_min:img_y_max, img_x_min:img_x_max]
    
    patch_borders = (out_z_min, out_z_max, out_y_min, out_y_max, out_x_min, out_x_max)
    abs_borders = (img_z_min, img_z_max, img_y_min, img_y_max, img_x_min, img_x_max)

    return patch, patch_borders, abs_borders

def visualize_channels(item, base_path, output_dir):
    name = item['name']
    print(f"\nНачало обработки снимка: {name}")

    def resolve(p): 
        if p is None: return None
        return os.path.normpath(os.path.join(base_path, p))
    
    # 1. Загрузка данных
    area_path = resolve(item.get('area_of_interest'))
    area = None
    if area_path and os.path.exists(area_path):
        area = imread(area_path).astype(np.uint8)
        area[area > 0] = 1

    img_path = resolve(item.get('image'))
    if not img_path or not os.path.exists(img_path):
        print(f"Ошибка: Изображение не найдено -> {img_path}")
        return

    raw_img = imread(img_path)
    if area is not None:
        raw_img = raw_img * area

    scale = item.get('scale', None)
    
    # 2. Базовая предобработка
    print("Применение фильтров и нормализации...")
    img = normalize_image(raw_img, scale)
    
    original_shape = img.shape
    img, pads = pad_to_min_shape(img)

    # 3. Подготовка глобальных каналов
    print("Вычисление глобальных параметров...")
    g_norm = normalize_minmax(img)
    g_otsu = get_otsu_map(img)

    # Инициализация пустых холстов для склейки Каналов 2 и 3
    ch2_full = np.zeros_like(img, dtype=np.float32)
    ch3_full = np.zeros_like(img, dtype=np.float32)

    frag_z, frag_y, frag_x = FRAGMENT_SHAPE
    z_count = max(1, int(np.ceil(img.shape[0] / frag_z)))
    y_count = max(1, int(np.ceil(img.shape[1] / frag_y)))
    x_count = max(1, int(np.ceil(img.shape[2] / frag_x)))

    print("Нарезка, вычисление локальных параметров и склейка...")
    for k in range(z_count):
        for j in range(y_count):
            for i in range(x_count):
                curr_pos = [k*frag_z, j*frag_y, i*frag_x]
                
                # Вырезаем патч
                p_img_crop, p_borders, a_borders = get_fragment_and_coords(img, curr_pos)
                if p_borders is None: continue
                
                p_gotsu_crop, _, _ = get_fragment_and_coords(g_otsu, curr_pos)
                
                # Вычисляем локальные параметры (как в Stage 1)
                p_lnorm = normalize_minmax(p_img_crop)
                p_lotsu = get_otsu_map(p_img_crop)
                
                # Формируем каналы для данного патча
                ch2_patch = p_lnorm
                ch3_patch = (p_gotsu_crop + p_lotsu) / 6.0
                
                # Распаковываем координаты
                oz_min, oz_max, oy_min, oy_max, ox_min, ox_max = p_borders
                iz_min, iz_max, iy_min, iy_max, ix_min, ix_max = a_borders
                
                # Вклеиваем локально обработанные куски обратно в полное изображение
                ch2_full[iz_min:iz_max, iy_min:iy_max, ix_min:ix_max] = \
                    ch2_patch[oz_min:oz_max, oy_min:oy_max, ox_min:ox_max]
                    
                ch3_full[iz_min:iz_max, iy_min:iy_max, ix_min:ix_max] = \
                    ch3_patch[oz_min:oz_max, oy_min:oy_max, ox_min:ox_max]

    # 4. Удаление паддинга (чтобы размер совпал с оригиналом после масштабирования)
    g_norm = g_norm[:original_shape[0], :original_shape[1], :original_shape[2]]
    ch2_full = ch2_full[:original_shape[0], :original_shape[1], :original_shape[2]]
    ch3_full = ch3_full[:original_shape[0], :original_shape[1], :original_shape[2]]

    # 5. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    
    path_ch1 = os.path.join(output_dir, f"{name}_CH1_Global_Norm.tif")
    path_ch2 = os.path.join(output_dir, f"{name}_CH2_Local_Norm.tif")
    path_ch3 = os.path.join(output_dir, f"{name}_CH3_Combo_Otsu.tif")

    print("Сохранение TIF файлов...")
    imwrite(path_ch1, g_norm)
    imwrite(path_ch2, ch2_full)
    imwrite(path_ch3, ch3_full)
    
    print(f"Успешно сохранено в папку: {output_dir}")
    print("Готово!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка целых снимков по каналам из Stage 1 для визуализации.")
    parser.add_argument("--json_path", required=True, help="Путь к JSON датасета")
    parser.add_argument("--out_dir", required=True, help="Папка для сохранения")
    parser.add_argument("--target_name", required=True, help="Имя снимка (name из JSON), который нужно визуализировать")
    parser.add_argument("--base_path", default="../../", help="Базовый путь к файлам")
    
    args = parser.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f: 
        cfg = json.load(f)
        
    target_item = next((x for x in cfg['data'] if x['name'] == args.target_name), None)
    
    if target_item:
        visualize_channels(target_item, args.base_path, args.out_dir)
    else:
        print(f"Ошибка: Снимок с именем '{args.target_name}' не найден в JSON файле!")

        #  --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage1/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage1/statia" --target_name in_vitro_3