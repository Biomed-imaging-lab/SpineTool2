import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc as auc_metric
from tifffile import imread
import os

def build_roc_from_json(json_path):
    print("Загрузка конфигурации из JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    target_class = 2 # 2 - шипики, 1 - ствол
    
    y_true_all = []
    y_scores_all = []

    print(f"Сбор данных для класса {target_class}...")
    
    for item in cfg.get("data", []):
        gt_path = item.get("ground truth")
        prob_path = item.get("pred_prob") 
        area_path = item.get("area of interest", "")
        
        if not gt_path or not os.path.exists(gt_path) or not os.path.exists(prob_path):
            print(f"[!] Пропускаем, не найдены основные файлы: {item.get('id', 'Unknown')}")
            continue
            
        print(f" Обработка: {item.get('id', 'Unknown')}")
        
        # 1. Загрузка зоны интересов (Area of Interest)
        area = None
        if area_path and os.path.exists(area_path):
            area = imread(area_path).astype(np.uint8)
            area[area > 0] = 1
        else:
            print("    [-] Area of interest не найдена, используем весь снимок.")
        
        # 2. Загружаем Ground Truth
        gt = imread(gt_path)
        y_true_img = (gt == target_class).astype(np.uint8)
        
        # 3. Загружаем предсказания нейросети (вероятности)
        prob_img = imread(prob_path).astype(np.float32)
        
        # Защита от сырых логитов: если значения вне [0, 1], применяем сигмоиду
        if prob_img.min() < 0 or prob_img.max() > 1.0:
            prob_img = 1 / (1 + np.exp(-prob_img))

        if target_class == 1:#  ONLY 4 STAGE
            # Превращаем диапазон [0.25, 0.75] в высокие вероятности, 
            # а фон (<0.25) и шипики (>0.75) в низкие вероятности.
            # Пик вероятности ствола находится на 0.5. Мы считаем расстояние до 0.5
            distance_to_center = np.abs(prob_img - 0.5)
            
            # Инвертируем: 0 расстояние -> 1.0 вероятность, 0.5 расстояние -> 0.0 вероятность
            prob_img = 1.0 - (distance_to_center / 0.5)
            
            # Гарантируем, что значения не выйдут за пределы [0, 1]
            prob_img = np.clip(prob_img, 0, 1)    
            
        # 4. Применяем зону интереса и переводим в 1D
        if area is not None:
            # ОГРОМНАЯ ЭКОНОМИЯ ПАМЯТИ: берем только те пиксели, где area == 1
            y_true_flat = y_true_img[area > 0]
            y_prob_flat = prob_img[area > 0]
        else:
            y_true_flat = y_true_img.flatten()
            y_prob_flat = prob_img.flatten()
        
        # 5. SUBSAMPLING (чтобы не переполнить оперативную память)
        # Берем каждый 50-й пиксель внутри зоны интереса.
        step = 1
        y_true_all.append(y_true_flat[::step])
        y_scores_all.append(y_prob_flat[::step])

    # 6. Объединяем данные со всех снимков
    print("Объединение массивов...")
    if not y_true_all:
        print("Ошибка: Нет данных для построения графика.")
        return
        
    y_true_final = np.concatenate(y_true_all)
    y_scores_final = np.concatenate(y_scores_all)

    # 7. Вычисляем ROC и AUC
    print("Расчет ROC-кривой...")
    fpr, tpr, thresholds = roc_curve(y_true_final, y_scores_final)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds_pr = precision_recall_curve(y_true_final, y_scores_final)
    pr_auc = auc_metric(recall, precision) # Площадь под PR-кривой
    print(f"Готово! Площадь под кривой (AUC) = {roc_auc:.4f}")
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_index_pr = np.argmax(f1_scores)
    best_thresh_pr = thresholds_pr[best_index_pr]
    best_f1 = f1_scores[best_index_pr]
    
    print(f"---")
    print(f"Лучший порог по F1-score (Dice): {best_thresh_pr:.3f}")
    print(f"Максимальный F1-score составит: {best_f1:.3f}")
    print(f"---")
    print(f"fpr{len(fpr)}")
    print(f"tpr{len(tpr)}")
    print(f"recall{len(recall)}")
    print(f"precision{len(precision)}")
    
    indices_fpr = np.linspace(0, len(fpr) - 1, num=len(fpr) // 5000, dtype=int)
    indices_recall = np.linspace(0, len(recall) - 1, num=len(recall) // 5000, dtype=int)
    # 1. Сохраняем данные ROC-кривой
# X = False Positive Rate, Y = True Positive Rate
    roc_df = pd.DataFrame({
        'FPR (X)': fpr[indices_fpr],
        'TPR (Y)': tpr[indices_fpr]
    })
    roc_df.to_csv('roc_curve_stage4_spine_5000.csv', index=False, float_format='%.5f')
    print("  [+] Данные ROC сохранены в roc_curve_prism.csv")

# 2. Сохраняем данные PR-кривой
# X = Recall, Y = Precision
    pr_df = pd.DataFrame({
        'Recall (X)': recall[indices_recall],
        'Precision (Y)': precision[indices_recall]
    })
    pr_df.to_csv('pr_curve_stage4_spine_5000.csv', index=False, float_format='%.5f')
    print("  [+] Данные PR сохранены в pr_curve_prism.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: ROC-кривая
    axes[0].plot(fpr[indices_fpr], tpr[indices_fpr], color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_title('ROC-кривая сегментации на ствол и шипики (шипики)')
    axes[0].set_xlabel('False Positive Rate (FPR)')
    axes[0].set_ylabel('True Positive Rate (TPR)')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    # График 2: PR-кривая
    axes[1].plot(recall[indices_recall], precision[indices_recall], color='green', lw=2, label=f'PR AUC = {pr_auc:.4f}')
    axes[1].set_title('PR-кривая сегментации на ствол и шипики (шипики)')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ROC_vs_PR_Curve.png", dpi=300)
    plt.show()
    # Сохраняем картинку
    # plt.savefig("ROC_Curve_with_Area.png", dpi=300, bbox_inches='tight')
    # print("График сохранен как ROC_Curve_with_Area.png")
    # plt.show()

if __name__ == "__main__":
    # Укажите путь к вашему json-файлу
    JSON_PATH = "C:/Users/Student/source/repos/ai_spines_segmentation/src/utils/stage4.json" 
    build_roc_from_json(JSON_PATH)

    #threshold stage1 0.465  st2 0.532   st3 0.583   st4 shaft 0.442(перевести в норм масштаб)    st4 spine 0.739