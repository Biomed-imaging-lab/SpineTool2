import os
import json
import cv2
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm

def scale_and_pad_to_target(image, target_shape, is_binary=False):
    """
    Масштабирует оси Y и X к исходному разрешению, а также симметрично 
    дописывает нули по оси Z, если объём был урезан паддингом при инференсе.
    """
    target_z, target_y, target_x = target_shape
    source_z, source_y, source_x = image.shape

    # 1. Шаг по оси Y и X: Послойно меняем разрешение
    interp = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
    dtype = np.uint8 if is_binary else np.float32

    # Создаем временный массив с правильными Y и X, но пока со старым Z
    yx_scaled = np.zeros((source_z, target_y, target_x), dtype=dtype)
    for i in range(source_z):
        yx_scaled[i] = cv2.resize(
            image[i].astype(dtype), 
            (target_x, target_y), 
            interpolation=interp
        )

    # 2. Шаг по оси Z: Если слоёв меньше, чем в GT (из-за pad=16), восстанавливаем паддинг нулями
    if source_z == target_z:
        return yx_scaled
    
    print(f"      [i] Восстановление слоев Z: {source_z} -> {target_z} (добавление пустых слоев)")
    final_volume = np.zeros((target_z, target_y, target_x), dtype=dtype)
    
    # Вычисляем отступ сверху (обычно равен значению pad, т.е. 16 слоёв)
    pad_top = (target_z - source_z) // 2
    final_volume[pad_top : pad_top + source_z, :, :] = yx_scaled
    
    return final_volume


def main():
    # =========================================================================
    # НАСТРОЙКИ ПУТЕЙ
    # =========================================================================
    JSON_PATH = "C:/Users/Student/source/repos/ai_spines_segmentation/src/utils/stage1_finetune.json" # Путь к вашему новому JSON
    INPUT_DIR = "C:/Users/Student/datasets/CVstage1/bin_finetune/finetune_all" # Папка с результатами инференса
    OUTPUT_DIR = "C:/Users/Student/datasets/CVstage1/bin_fin_scaled/finetune_all" # Куда сохранить подогнанные файлы
    # =========================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Загружаем конфигурационный файл
    with open(JSON_PATH, 'r') as f:
        config = json.load(f)

    # Строим карту соответствия: базовое имя файла -> путь к его ground truth
    gt_mapping = {}
    for item in config['data']:
        gt_id = item["id"] # например, "endogenous_cell1_C1_bin"
        gt_path = item['ground truth']
        gt_mapping[gt_id] = gt_path

    # Сканируем входную папку со снимками инференса
    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.tif', '.tiff'))]
    
    if not files_to_process:
        print(f"В папке {INPUT_DIR} не найдено TIFF файлов.")
        return

    print(f"Найдено файлов для масштабирования: {len(files_to_process)}")

    for fname in tqdm(files_to_process, desc="Общая обработка папки"):
        input_file_path = os.path.join(INPUT_DIR, fname)
        
        # Выделяем ID из имени файла для сопоставления с конфигом
        # Метод убирает расширение и ищет совпадение ключа в нашей карте gt_mapping
        base_name_no_ext = os.path.splitext(fname)[0]
        
        # Ищем, какому id из JSON принадлежит этот файл
        matched_id = None
        for gt_id in gt_mapping.keys():
            if gt_id in base_name_no_ext or base_name_no_ext in gt_id:
                matched_id = gt_id
                break

        if not matched_id:
            print(f"\n[Предупреждение] Пропуск {fname}: не найден соответствующий id в JSON.")
            continue

        gt_file_path = gt_mapping[matched_id]
        if not os.path.exists(gt_file_path):
            print(f"\n[Ошибка] Файл Ground Truth не найден по пути: {gt_file_path}")
            continue

        # Читаем целевые размеры из оригинальной маски (ground truth)
        gt_shape = imread(gt_file_path).shape
        
        # Читаем наш снимок, полученный после инференса
        img = imread(input_file_path)

        # Автоматически определяем, бинарный ли это файл (по имени или типу данных)
        is_binary = "bin" in fname.lower() or np.issubdtype(img.dtype, np.integer)

        # Запускаем комбинированное масштабирование и Z-выравнивание
        scaled_img = scale_and_pad_to_target(img, gt_shape, is_binary=is_binary)

        # Сохраняем итоговый подогнанный файл
        out_file_path = os.path.join(OUTPUT_DIR, f"scaled_{fname}")
        imwrite(out_file_path, scaled_img, imagej=True)

    print(f"\n[+] Скрипт успешно завершил работу. Все подогнанные файлы сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()