import os
import json
import numpy as np
import argparse
import edt
import scipy.ndimage as ndi
from tifffile import imread, imwrite
from scipy.spatial import KDTree
from skimage.draw import line_nd
from stage1 import normalize_minmax
from stage4 import generate_distance_channels



def visualize_stage4_channels(item, base_path, output_dir):
    name = item['name']
    print(f"\n======================================")
    print(f"Обработка снимка (Stage 4): {name}")
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
        
    skel_path = resolve(item.get('skeleton'))
    vsot_shaft_path = resolve(item.get('shaft_vsot'))
    vsot_spine_path = resolve(item.get('spine_vsot'))
    
    if not all([skel_path, vsot_shaft_path, vsot_spine_path]):
        print(f"   [!] Ошибка: Не указаны пути к skeleton или vsot маскам в JSON.")
        return

    print("   [1] Загрузка масок и скелета...")
    skeleton = imread(skel_path) > 0
    vsot_shaft = imread(vsot_shaft_path) > 0
    vsot_spine = imread(vsot_spine_path) > 0

    vsot_dendrite = vsot_shaft | vsot_spine

    scale = item.get('scale', None) # Фолбэк, если scale не указан
    res_nm = (scale[0]*1000, scale[1]*1000, scale[2]*1000)

    # 2. Формирование I_VSOT (Канал 1)
    print("   [2] Формирование канала I_VSOT_norm...")
    i_vsot = np.zeros_like(vsot_dendrite, dtype=np.float32)
    i_vsot[vsot_shaft] = 1.0  # ствол
    i_vsot[vsot_spine] = 2.0  # шипики
    i_vsot_norm = normalize_minmax(i_vsot)

    # 3. Вычисление карт расстояний (Каналы 2 и 3)
    print("   [3] Генерация карт расстояний (Distance Transforms)...")
    d_dendr_ch2, d_spine_ch3 = generate_distance_channels(vsot_dendrite, skeleton, res_nm)

    # 4. Применение маски Area of Interest
    if area is not None:
        print("   [4] Применение маски Area of Interest...")
        i_vsot_norm = i_vsot_norm * area
        d_dendr_ch2 = d_dendr_ch2 * area
        d_spine_ch3 = d_spine_ch3 * area

    # 5. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    print(f"   [5] Сохранение TIF файлов в {output_dir}...")
    
    imwrite(os.path.join(output_dir, f"{name}_Stage4_CH1_I_vsot.tif"), i_vsot_norm.astype(np.float32))
    imwrite(os.path.join(output_dir, f"{name}_Stage4_CH2_D_dendr.tif"), d_dendr_ch2.astype(np.float32))
    imwrite(os.path.join(output_dir, f"{name}_Stage4_CH3_D_spine.tif"), d_spine_ch3.astype(np.float32))
    
    print("   [-] Готово!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка 3 каналов из Stage 4 для визуализации")
    parser.add_argument("--json_path", required=True, help="Путь к JSON датасета")
    parser.add_argument("--out_dir", required=True, help="Папка для сохранения визуализаций")
    parser.add_argument("--target_name", required=True, help="Имя снимка из JSON")
    parser.add_argument("--base_path", default="../../", help="Базовый путь к файлам")
    
    args = parser.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f: 
        cfg = json.load(f)
        
    target_item = next((x for x in cfg['data'] if x['name'] == args.target_name), None)
    
    if target_item:
        visualize_stage4_channels(target_item, args.base_path, args.out_dir)
    else:
        print(f"Ошибка: Снимок '{args.target_name}' не найден в JSON файле!")