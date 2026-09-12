import os
import glob
import numpy as np
from tifffile import imwrite
import re
from collections import defaultdict
# ==========================================
# НАСТРОЙКИ
# ==========================================
INPUT_DIR = "C:/Users/Student/datasets/CVstage4/f_s2_08s/folds1/val"
OUTPUT_DIR = "C:/Users/Student/datasets/CVstage4/visualvivo/f_s2_08s/folds1_new_3"

# Фильтры для поиска нужных патчей
# TARGET_ANGLE = "150"    # Число после 'a', например "0", "90"
# TARGET_POS = "000"    # Числа после 'p', например "000" (если start_pos было 0,0,0)

# Цвет фона для пропущенных патчей. 
GREY_VALUE = 0.5
#for stage 1, 2
# FRAG_Z, FRAG_Y, FRAG_X = 64, 64, 64
#for stage 3, 4
FRAG_Z, FRAG_Y, FRAG_X = 64, 128, 128
# ==========================================

def unpack_patches_to_tiff(input_dir, output_dir, angle, pos):
    os.makedirs(output_dir, exist_ok=True)
    
    # Ищем файлы по шаблону (например: *_a0_p000_*.npz)
    search_pattern = os.path.join(input_dir, f"*_a{angle}_p{pos}_*.npz")
    npz_files = glob.glob(search_pattern)
    
    if not npz_files:
        print(f"Файлы с углом a{angle} и позицией p{pos} не найдены!")
        return
        
    print(f"Найдено файлов для распаковки: {len(npz_files)}")
    
    for file_path in npz_files:
        # Получаем имя файла без расширения
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Обработка: {base_name}")
        
        # Загружаем архив
        data = np.load(file_path)
        image = data['image'] # Ожидаемая форма: (3, Z, Y, X)
        mask = data['mask']   # Ожидаемая форма: (Z, Y, X) или (1, Z, Y, X)
        
        # ----------------------------------------------------
        # Подготовка изображения для ImageJ
        # ImageJ предпочитает порядок осей: (Z, C, Y, X)
        # ----------------------------------------------------
        if len(image.shape) == 4 and image.shape[0] == 3:
            # Перемещаем ось каналов (0) на место оси Z (1), чтобы стало (Z, C, Y, X)
            image_ij = np.transpose(image, (1, 0, 2, 3))
        else:
            image_ij = image # Если форма другая, оставляем как есть
            
        # Формируем пути для сохранения
        img_out_path = os.path.join(output_dir, f"{base_name}_img.tif")
        mask_out_path = os.path.join(output_dir, f"{base_name}_mask.tif")
        
        # Сохраняем изображение как ImageJ Hyperstack
        imwrite(
            img_out_path, 
            image_ij, 
            imagej=True,      # Включает метаданные для ImageJ
            metadata={'axes': 'ZCYX'} # Явно указываем оси: Z-срезы, Каналы, Y, X
        )
        
        # Сохраняем маску
        imwrite(
            mask_out_path, 
            mask.astype(np.uint8), # Маску лучше сохранять в 8-бит для удобства
            imagej=True,
            compression=None
        )

def reconstruct_and_save(input_dir, output_dir, angle, pos):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ищем все кубики, принадлежащие одному снимку/углу/позиции
    search_pattern = os.path.join(input_dir, f"*_a{angle}_p{pos}_idx*-*-*.npz")
    npz_files = glob.glob(search_pattern)
    
    if not npz_files:
        print(f"Файлы по шаблону {search_pattern} не найдены!")
        return
        
    print(f"Найдено патчей для склейки: {len(npz_files)}")
    valid_patches = []
    max_z, max_y, max_x = 0, 0, 0
    
    for file_path in npz_files:
        # Ищем в названии кусок вроде idx1-2-3 и достаем цифры k, j, i
        match = re.search(r'_idx(\d+)-(\d+)-(\d+)\.npz', file_path)
        if not match:
            continue
            
        k = int(match.group(1)) # индекс по Z
        j = int(match.group(2)) # индекс по Y
        i = int(match.group(3)) # индекс по X
        
        # Подгружаем только маску, чтобы узнать точный размер патча
        # (вдруг патч на краю снимка меньше, чем 64)
        data = np.load(file_path)
        z_shape, y_shape, x_shape = data['mask'].shape
        
        # Вычисляем конечную координату, где заканчивается этот патч
        z_end = (k * FRAG_Z) + z_shape
        y_end = (j * FRAG_Y) + y_shape
        x_end = (i * FRAG_X) + x_shape
        
        # Расширяем границы виртуального холста, если нужно
        max_z = max(max_z, z_end)
        max_y = max(max_y, y_end)
        max_x = max(max_x, x_end)
        
        # Запоминаем данные, чтобы не парсить имя файла второй раз
        valid_patches.append((file_path, k, j, i, z_shape, y_shape, x_shape))

    print(f"Размер склеенного снимка составит: Z={max_z}, Y={max_y}, X={max_x}")

    # -----------------------------------------------------------------
    # ЭТАП 2: Создаем холсты и собираем пазл
    # -----------------------------------------------------------------
    full_image = np.full((3, max_z, max_y, max_x), GREY_VALUE, dtype=np.float32)
    full_mask = np.zeros((max_z, max_y, max_x), dtype=np.uint8)

    print("Наложение патчей на холст...")
    for file_path, k, j, i, z_shape, y_shape, x_shape in valid_patches:
        data = np.load(file_path)
        img_patch = data['image']
        mask_patch = data['mask']
        
        # Вычисляем абсолютные координаты начала
        z0 = k * FRAG_Z
        y0 = j * FRAG_Y
        x0 = i * FRAG_X
        
        # Вставляем!
        full_image[:, z0:z0+z_shape, y0:y0+y_shape, x0:x0+x_shape] = img_patch
        full_mask[z0:z0+z_shape, y0:y0+y_shape, x0:x0+x_shape] = mask_patch

    # -----------------------------------------------------------------
    # ЭТАП 3: Сохранение
    # -----------------------------------------------------------------
    print("Сохранение результатов для ImageJ...")
    full_image_ij = np.transpose(full_image, (1, 0, 2, 3))
    
    out_name = f"a{angle}_p{pos}"
    imwrite(os.path.join(output_dir, f"{out_name}_img.tif"), full_image_ij, imagej=True, metadata={'axes': 'ZCYX'})
    imwrite(os.path.join(output_dir, f"{out_name}_mask.tif"), full_mask, imagej=True)

def process_all_folders(input_dir, output_dir):
    # os.walk рекурсивно обходит все папки и подпапки
    for root, dirs, files in os.walk(input_dir):
        # Ищем только файлы .npz в текущей папке
        npz_files = [f for f in files if f.endswith('.npz')]
        if not npz_files:
            continue # Если npz нет, идем в следующую папку

        # Воссоздаем структуру папок в выходной директории
        rel_path = os.path.relpath(root, input_dir)
        out_folder = os.path.join(output_dir, rel_path)
        os.makedirs(out_folder, exist_ok=True)
        
        print(f"\n--- Обработка папки: {rel_path} ---")

        # Группируем файлы по имени, углу и позиции
        # Ключ словаря: (name, angle, pos)
        # Значение: список кортежей (путь_к_файлу, k, j, i)
        groups = defaultdict(list)
        
        for file in npz_files:
            # Парсим название файла с помощью регулярного выражения
            # Ожидаем формат: name_aAngle_pPos_idxK-J-I.npz
            match = re.search(r'(.+)_a(\d+)_p([-\d]+)_idx(\d+)-(\d+)-(\d+)\.npz', file)
            if match:
                name = match.group(1)
                angle = match.group(2)
                # pos = match.group(3)
                k, j, i = int(match.group(4)), int(match.group(5)), int(match.group(6))
                
                full_path = os.path.join(root, file)
                groups[(name, angle)].append((full_path, k, j, i))

        # Склеиваем каждую найденную группу
        for (name, angle), file_data in groups.items():
            print(f"[*] Сборка снимка: {name} (Угол: {angle}), Патчей: {len(file_data)}")
            
            # 1. Вычисляем глобальный размер снимка
            max_z, max_y, max_x = 0, 0, 0
            valid_patches = []
            
            for filepath, k, j, i in file_data:
                data = np.load(filepath)
                mask = data['mask']
                zs, ys, xs = mask.shape
                
                max_z = max(max_z, k * FRAG_Z + zs)
                max_y = max(max_y, j * FRAG_Y + ys)
                max_x = max(max_x, i * FRAG_X + xs)
                
                valid_patches.append((filepath, k, j, i, zs, ys, xs))

            # 2. Создаем холсты
            full_image = np.full((3, max_z, max_y, max_x), GREY_VALUE, dtype=np.float32)
            full_mask = np.zeros((max_z, max_y, max_x), dtype=np.uint8)
            full_valid = np.zeros((max_z, max_y, max_x), dtype=np.uint8)
            # 3. Накладываем патчи
            for filepath, k, j, i, zs, ys, xs in valid_patches:
                data = np.load(filepath)
                z0, y0, x0 = k * FRAG_Z, j * FRAG_Y, i * FRAG_X
                
                full_image[:, z0:z0+zs, y0:y0+ys, x0:x0+xs] = data['image']
                full_mask[z0:z0+zs, y0:y0+ys, x0:x0+xs] = data['mask']
                if 'valid_mask' in data:
                    full_valid[z0:z0+zs, y0:y0+ys, x0:x0+xs] = data['valid_mask'].astype(np.uint8)
            # 4. Сохранение каждого канала в отдельный файл
            base_out_name = os.path.join(out_folder, f"{name}_a{angle}")
            
            # Поскольку мы сохраняем одиночные каналы (форма Z, Y, X), 
            # ImageJ автоматически откроет их правильно, только с ползунком Z!
            
            # Канал 1: Глобальная нормализация (индекс 0)
            imwrite(f"{base_out_name}_ch1.tif", full_image[0], imagej=True)
            
            # Канал 2: Локальная нормализация (индекс 1)
            imwrite(f"{base_out_name}_ch2.tif", full_image[1], imagej=True)
            
            # Канал 3: Комбинация Otsu (индекс 2)
            imwrite(f"{base_out_name}_ch3.tif", full_image[2], imagej=True)

            # # Канал 3: Комбинация Otsu (индекс 3)
            # imwrite(f"{base_out_name}_ch4_necks.tif", full_image[3], imagej=True)
            
            # Маска (исправляем на 255 для видимости)
            if full_mask.max() == 2:
                full_mask = full_mask * 127
            imwrite(f"{base_out_name}_mask.tif", full_mask, imagej=True)
            if full_valid.max() == 1:
                full_valid = full_valid * 255
            imwrite(f"{base_out_name}_valid_mask.tif", full_valid, imagej=True)


if __name__ == "__main__":
    # unpack_patches_to_tiff(INPUT_DIR, OUTPUT_DIR, TARGET_ANGLE, TARGET_POS)
    # reconstruct_and_save(INPUT_DIR, OUTPUT_DIR, TARGET_ANGLE, TARGET_POS)
    process_all_folders(INPUT_DIR, OUTPUT_DIR)
    print("\n[+] Готово! Все патчи распакованы в TIF.")