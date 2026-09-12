# import json
# import statistics

# with open("C:/Users/Student/datasets/CVstage1/metrics_ft_cv/pre/metrics_report.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# dice_values = [
#     item["binarization_metrics"]["dice"]
#     for item in data["files_metrics"]
# ]

# iou_values = [
#     item["binarization_metrics"]["iou"]
#     for item in data["files_metrics"]
# ]

# print("Dice:", statistics.mean(dice_values), statistics.stdev(dice_values))
# print("IoU:", statistics.mean(iou_values), statistics.stdev(iou_values))
import json
import os
import statistics

# 1. Путь к твоему файлу метрик
JSON_PATH = "C:/Users/Student/datasets/metrices_2/res_r_2/metrics_report.json"

if not os.path.exists(JSON_PATH):
    print(f"Ошибка: Файл {JSON_PATH} не найден.")
    exit()

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

files_metrics = data.get("files_metrics", [])

# 2. Карта путей ко всем метрикам (одинаковая и для пофайловых, и для general_metrics)
metrics_map = {
    "Бинаризация: Dice": ["binarization_metrics", "dice"],
    "Бинаризация: IoU": ["binarization_metrics", "iou"],
    
    "Ствол (Dendrite): Dice": ["dendrite_metrics", "dice"],
    "Ствол (Dendrite): IoU": ["dendrite_metrics", "iou"],
    
    "Шипики (Воксели): Dice": ["spines_metrics", "voxels_metrics", "dice"],
    "Шипики (Воксели): IoU": ["spines_metrics", "voxels_metrics", "iou"],
    
    "Шипики (Инстансы): Precision": ["spines_metrics", "instances_metrics", "precision"],
    "Шипики (Инстансы): Recall": ["spines_metrics", "instances_metrics", "recall"],
    "Шипики (Инстансы): F1-Score": ["spines_metrics", "instances_metrics", "f1"],
}

print(f"\n{'Название метрики':<30} | {'Среднее (Mean)':<14} | {'Разброс (Std Dev)':<14}")
print("-" * 66)

# 3. Сбор данных, расчёт статистики и инжекция в JSON
for display_name, path in metrics_map.items():
    values = []
    
    # Собираем значения со всех файлов тестового набора
    for item in files_metrics:
        try:
            val = item
            for key in path:
                val = val[key]
            values.append(val)
        except KeyError:
            # Если в каком-то файле вдруг нет этой метрики — пропускаем
            continue

    if len(values) > 1:
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        # Выводим строчку в консоль
        print(f"{display_name:<30} | {mean_val:<14.6f} | {std_val:<14.6f}")
        
        # Инжектируем значение std в структуру general_metrics
        try:
            target_node = data["general_metrics"]
            for key in path:
                target_node = target_node[key]
            
            # Дописываем ключ "std" прямо в словарь этой метрики
            target_node["std"] = std_val
        except KeyError:
            print(f"      [!] Не удалось записать std для {display_name} в general_metrics")

# 4. Сохраняем обновленный JSON с метаданными std
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"\n[+] Расчет окончен. Значения 'std' успешно внедрены в файл: {JSON_PATH}")