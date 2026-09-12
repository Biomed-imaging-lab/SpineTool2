import os
import json
import argparse
import numpy as np
import cv2
from tifffile import imread, imwrite
from tqdm import tqdm

# Базовый масштаб
BASE_SCALE = [0.1, 0.1]

def scale_image(image, current_scale):
    if current_scale is None: return image
    scale_y = current_scale[1] / BASE_SCALE[0]
    scale_x = current_scale[2] / BASE_SCALE[1]
    
    if abs(scale_y - 1.0) < 0.01 and abs(scale_x - 1.0) < 0.01: return image
    
    new_y = int(np.round(image.shape[1] * scale_y))
    new_x = int(np.round(image.shape[2] * scale_x))
    
    # Для масок (целочисленных типов) используем INTER_NEAREST, чтобы не смазать метки 1 и 2
    interp = cv2.INTER_NEAREST if np.issubdtype(image.dtype, np.integer) else cv2.INTER_AREA
    
    resized = np.zeros((image.shape[0], new_y, new_x), dtype=image.dtype)
    for i in range(image.shape[0]):
        resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=interp)
        
    return resized

def resolve_path(base_path, p):
    if p is None: return None
    return os.path.normpath(os.path.join(base_path, p))

def main():
    parser = argparse.ArgumentParser(description="Масштабирование масок для подсчета метрик")
    parser.add_argument("--json_path", required=True, help="Путь к исходному JSON файлу (из stage2)")
    parser.add_argument("--out_dir", required=True, help="Папка, куда сохранить отмасштабированные маски")
    parser.add_argument("--base_path", default="../../", help="Базовый путь для относительных путей в JSON")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    new_cfg = cfg.copy()
    new_data = []

    mask_keys_to_check = ['area_of_interest']
    print(f"Найдено {len(cfg.get('data', []))} записей. Начинаем масштабирование...")

    for item in tqdm(cfg.get('data', [])):
        new_item = item.copy()
        name = item.get('name', item.get('id', 'unknown'))
        scale = item.get('scale')

        for key in mask_keys_to_check:
            mask_rel_path = item.get(key)
            if not mask_rel_path:
                continue

            full_mask_path = resolve_path(args.base_path, mask_rel_path)
            
            if os.path.exists(full_mask_path):
                mask_img = imread(full_mask_path)
                
                scaled_mask = scale_image(mask_img, scale)
                
                out_filename = f"{name}_scaled.tif"
                out_path = os.path.join(args.out_dir, out_filename)
                
                imwrite(out_path, scaled_mask)
                
                #Обновляем путь в новом JSON (используем абсолютный путь для надежности)
                new_item[key] = os.path.abspath(out_path)
            else:
                print(f"\n[Внимание] Файл не найден: {full_mask_path}")

        new_data.append(new_item)

    # Сохраняем новый JSON-конфиг
    new_cfg['data'] = new_data
    new_json_path = os.path.join(args.out_dir, "scaled_masks_config.json")
    
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_cfg, f, indent=4, ensure_ascii=False)

    print(f"\nГотово! Все маски отмасштабированы и сохранены в: {args.out_dir}")
    print(f"Новый конфигурационный файл создан здесь: {new_json_path}")

if __name__ == "__main__":
    main()