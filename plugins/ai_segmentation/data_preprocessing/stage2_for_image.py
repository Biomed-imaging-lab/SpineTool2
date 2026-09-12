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


def visualize_stage2(item, base_path, preds_dir, output_dir):
    name = item['name']
    print(f"\n======================================")
    print(f"Обработка снимка: {name}")
    print(f"======================================")

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
        print(f"   [!] Ошибка: Изображение не найдено -> {img_path}")
        return
        
    raw_img = imread(img_path)
    if area is not None:
        raw_img = raw_img * area

    # 2. Загрузка предсказаний (L_dendrite)
    l_dendrite_path = os.path.join(preds_dir, f"{name}_L_dendrite.tif")
    if not os.path.exists(l_dendrite_path):
        print(f"   [!] Ошибка: Предсказания не найдены -> {l_dendrite_path}")
        return
    l_dendrite_raw = imread(l_dendrite_path)

    scale = item.get('scale', None)

    # 3. Вычисление I_interim
    print("   [1] Вычисление I_interim...")
    i_interim = calculate_interim_image(raw_img, scale)
    
    # ПРИМЕЧАНИЕ: В вашем stage2.py масштабирование l_dendrite закомментировано. 
    # Если i_interim отмасштабирован, а l_dendrite - нет, это может вызвать ошибку размерностей 
    # в find_optimal_j0. Для безопасности я раскомментировал масштабирование здесь:
    # if scale and l_dendrite_raw.shape != i_interim.shape:
    #     print("   [*] Масштабирование L_dendrite для совпадения размерностей...")
    #     l_dendrite_raw = scale_image(l_dendrite_raw, scale)

    l_bin = (l_dendrite_raw > 0.5).astype(np.uint8)

    # 4. Поиск j0
    print("   [2] Поиск оптимального j0...")
    j0 = find_optimal_j0(i_interim, l_bin)
    print(f"   [+] Найден оптимальный j0 = {j0}")
    # j0=10
    # 5. Вычисление I_si
    print("   [3] Вычисление I_si (нелинейное масштабирование)...")
    i_si_raw = calculate_isi(i_interim, j0)

    # 6. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    
    out_path_isi = os.path.join(output_dir, f"{name}_Stage2_Isi.tif")
    out_path_interim = os.path.join(output_dir, f"{name}_Stage2_I_interim.tif") # Дополнительно сохраним interim
    
    print(f"   [4] Сохранение результатов в {output_dir}...")
    
    # Сохраняем в формате float32
    imwrite(out_path_isi, i_si_raw)
    imwrite(out_path_interim, i_interim.astype(np.float32))
    
    print("   [-] Готово!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сохранение полных тензоров I_si из Stage 2")
    parser.add_argument("--json_path", required=True, help="Путь к JSON датасета")
    parser.add_argument("--out_dir", required=True, help="Папка для сохранения TIF-файлов")
    parser.add_argument("--target_name", required=True, help="Имя снимка (name из JSON) для обработки")
    parser.add_argument("--preds_dir", required=True, help="Путь к предсказаниям L_dendrite")
    parser.add_argument("--base_path", default="../../", help="Базовый путь к датасету")
    
    args = parser.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f: 
        cfg = json.load(f)
        
    target_item = next((x for x in cfg['data'] if x['name'] == args.target_name), None)
    
    if target_item:
        visualize_stage2(target_item, args.base_path, args.preds_dir, args.out_dir)
    else:
        print(f"Ошибка: Снимок с именем '{args.target_name}' не найден в JSON файле!")