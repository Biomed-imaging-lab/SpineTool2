import os
import json
import numpy as np
import argparse
import cv2
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from stage1 import normalize_minmax, normalize_image, scale_image
from stage2 import calculate_interim_image, find_optimal_j0, calculate_isi
# --- КОНСТАНТЫ ---
BASE_SCALE = [0.1, 0.1]


import os
import json
import numpy as np
import argparse
import cv2
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter
from skimage.measure import label

# --- КОНСТАНТЫ ---
BASE_SCALE = [0.1, 0.1]

def scale_to_shape(image, target_shape):
    target_z, target_y, target_x = target_shape
    if image.shape[1] == target_y and image.shape[2] == target_x:
        return image
    interp = cv2.INTER_LINEAR 
    
    resized = np.zeros((image.shape[0], target_y, target_x), dtype=np.float32)
    for i in range(image.shape[0]):
        resized[i] = cv2.resize(image[i].astype(np.float32), (target_x, target_y), interpolation=interp)
    return resized

def apply_gaussian_3d_stack(image, sigma=1.0):
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        blurred[i] = gaussian_filter(image[i], sigma=sigma)
    return blurred


def visualize_stage3_channels(item, base_path, l1_dir, l2_dir, output_dir):
    name = item['name']
    print(f"\n======================================")
    print(f"Обработка снимка: {name}")
    print(f"======================================")

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
    if not img_path or not os.path.exists(img_path): 
        print(f"   [!] Ошибка: Изображение не найдено -> {img_path}")
        return
    raw_img = imread(img_path)
    if area is not None: raw_img = raw_img * area

    # 2. Загрузка предсказаний L1 и L2
    l1_path = os.path.join(l1_dir, f"{name}_L_dendrite.tif")
    l2_path = os.path.join(l2_dir, f"{name}_necks.tif") # В вашем коде L2 ищется как necks.tif
    
    if not os.path.exists(l1_path):
        print(f"   [!] Ошибка: Не найдены предсказания L1 или L2.")
        return
        
    l1_raw = imread(l1_path)
    l2_raw = imread(l2_path)
    scale = item.get('scale', None)
    target_high_res_shape = raw_img.shape

    # 3. Вычисление промежуточных карт (j0, I_si)
    print("   [1] Вычисление i_interim и I_si...")
    i_interim = calculate_interim_image(raw_img, scale)
    
    # Защита от падения размерностей при поиске j0 (масштабируем l1_raw если нужно)
    # l1_for_j0 = l1_raw
    # if scale and l1_raw.shape != i_interim.shape:
        # l1_for_j0 = scale_image(l1_raw, scale)
        
    # l1_bin = (l1_for_j0 > 0.5).astype(np.uint8)
    # j0 = find_optimal_j0(i_interim, l1_bin)
    # I_si = calculate_isi(i_interim, j0)

    # 4. Апскейл до высокого разрешения и сглаживание (логика Stage 3)
    print("   [2] Масштабирование (scale_to_shape) и сглаживание по Гауссу...")
    # l1_up = scale_to_shape(l1_raw, target_high_res_shape)
    l2_up = scale_to_shape(l2_raw, target_high_res_shape)
    # isi_up = scale_to_shape(I_si, target_high_res_shape)
    
    # l1_s = apply_gaussian_3d_stack(l1_up, sigma=1.0)
    l2_s = apply_gaussian_3d_stack(l2_up, sigma=1.0)
    # isi_s = apply_gaussian_3d_stack(isi_up, sigma=1.0)

    # 5. Вычисление I_norm
    print("   [3] Вычисление глобальной нормализации I_norm...")
    # I_norm = normalize_minmax(raw_img)

    # 6. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    print(f"   [4] Сохранение TIF файлов в {output_dir}...")
    
    # imwrite(os.path.join(output_dir, f"{name}_Stage3_CH1_I_norm.tif"), I_norm.astype(np.float32))
    # imwrite(os.path.join(output_dir, f"{name}_Stage3_CH2_Isi_s.tif"), isi_s.astype(np.float32))
    # imwrite(os.path.join(output_dir, f"{name}_Stage3_CH3_L1_s.tif"), l1_s.astype(np.float32))
    imwrite(os.path.join(output_dir, f"{name}_Stage3_CH4_L2_s.tif"), l2_s.astype(np.float32))
    
    print("   [-] Готово!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка 4 каналов из Stage 3 для визуализации")
    parser.add_argument("--json_path", required=True, help="Путь к JSON датасета")
    parser.add_argument("--out_dir", required=True, help="Папка для сохранения")
    parser.add_argument("--target_name", required=True, help="Имя снимка")
    parser.add_argument("--l1_dir", required=True, help="Путь к папке с предсказаниями Stage 1 (L_dendrite)")
    parser.add_argument("--l2_dir", required=True, help="Путь к папке с предсказаниями Stage 2 (necks)")
    parser.add_argument("--base_path", default="../../", help="Базовый путь")
    
    args = parser.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f: 
        cfg = json.load(f)
        
    target_item = next((x for x in cfg['data'] if x['name'] == args.target_name), None)
    
    if target_item:
        visualize_stage3_channels(target_item, args.base_path, args.l1_dir,  args.l2_dir, args.out_dir)########вставить l2
    else:
        print(f"Ошибка: Снимок '{args.target_name}' не найден в JSON файле!")
#   & C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage3_for_image.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage3/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage3/statia" --target_name in_vitro_3 --l1_dir "C:/Users/Student/datasets/CVstage2/L_dendrite"         
# & C:/Users/Student/.conda/envs/ai_segm/python.exe -m tensorboard.main --logdir="C:/Users/Student/datasets/CVstage3/tensorboard_logs/stage3_model"