import os
import re
import numpy as np
from tifffile import imread, imwrite
from collections import defaultdict

def stack_2d_to_3d(input_dir, output_dir):
    print(f"Поиск файлов в папке: {input_dir}")
    
    # Словарь для группировки файлов: {базовое_имя: [(z_индекс, имя_файла), ...]}
    stacks = defaultdict(list)
    
    # Регулярное выражение, которое ищет файлы, заканчивающиеся на _Z и цифры
    # Например: "имя_файла_C0_Z001.tif" -> Группа 1: "имя_файла_C0", Группа 2: "001"
    pattern = re.compile(r"^(.*_Z)(\d+)(\.tiff?)$", re.IGNORECASE)
    
    # 1. Сканируем папку и группируем файлы
    found_files = 0
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            base_name = match.group(1)   # всё до номера слоя включительно (с _Z)
            z_index = int(match.group(2)) # номер слоя как число (000 -> 0)
            
            stacks[base_name].append((z_index, filename))
            found_files += 1

    if found_files == 0:
        print("Внимание: Не найдено файлов, подходящих под шаблон (заканчивающихся на _Z000.tif и т.д.)")
        return

    print(f"Найдено {found_files} слоев. Начинаем сборку 3D снимков...\n")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Собираем каждый стек
    for base_name, files in stacks.items():
        # Сортируем файлы строго по Z-индексу (чтобы слой 2 не оказался после слоя 10)
        files.sort(key=lambda x: x[0])
        
        # Убираем "_Z" на конце для красивого имени итогового файла
        clean_base_name = base_name[:-2] if base_name.upper().endswith('_Z') else base_name
        
        print(f"Сборка снимка: {clean_base_name} (Слоев: {len(files)})")
        
        layers = []
        for z_idx, fname in files:
            fpath = os.path.join(input_dir, fname)
            # Читаем слой
            img = imread(fpath)
            layers.append(img)
            
        # Склеиваем список слоев в один 3D-массив
        # Если слои (Y, X), то после stack получится (Z, Y, X)
        stack_3d = np.stack(layers, axis=0)
        
        # Формируем итоговое имя
        out_filename = f"5_{clean_base_name}.tif"
        out_filepath = os.path.join(output_dir, out_filename)
        
        # Сохраняем (с базовой компрессией, чтобы сэкономить место)
        imwrite(out_filepath, stack_3d, compression='zlib')
        
        print(f"  -> Успешно сохранено: {out_filename} | Размерность: {stack_3d.shape}")

    print("\nГотово! Все слои склеены.")

if __name__ == "__main__":
    # ==========================================
    # НАСТРОЙКИ ПУТЕЙ
    # ==========================================
    # Папка, куда вы выгрузили свои сотни одиночных 2D срезов:
    INPUT_FOLDER = r"C:\Users\Student\datasets\finetune_exogenous\TIF Images, cell 5, endogenous PRG 555, exogenous DISC1 647, GFP 488" 
    
    # Папка, куда скрипт положит готовые 3D тензоры:
    OUTPUT_FOLDER = r"C:\Users\Student\datasets\finetune_exogenous" 
    
    stack_2d_to_3d(INPUT_FOLDER, OUTPUT_FOLDER)