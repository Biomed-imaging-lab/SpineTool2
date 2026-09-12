from tifffile import imread
from scipy.ndimage import gaussian_filter, rotate
from skimage.filters import threshold_multiotsu
import os
import os
import numpy as np
from tifffile import imread, imwrite

def create_diff_mask_binary(pred_bin, label_bin):
    """
    Создает цветную карту ошибок.
    TP (True Positive) = Белый (Совпадение)
    FN (False Negative) = Красный (Нейросеть пропустила)
    FP (False Positive) = Синий (Нейросеть придумала лишнее)
    """
    # Убеждаемся, что на входе булевые массивы
    pred_bin = pred_bin.astype(bool)
    label_bin = label_bin.astype(bool)

    # Создаем пустой RGB объем
    rgb = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)

    # Intersection (White) - Истино положительные
    rgb[pred_bin & label_bin] = [255, 255, 255]
    
    # False Negative (Red) - Ложно отрицательные (есть в разметке, но нет в предсказании)
    rgb[label_bin & ~pred_bin] = [250, 0, 0]
    
    # False Positive (Blue) - Ложно положительные (есть в предсказании, но нет в разметке)
    rgb[pred_bin & ~label_bin] = [0, 0, 245]

    return rgb

if __name__ == "__main__":
    # === НАСТРОЙКА ПУТЕЙ ===
    # 1. Файл предсказания (где 1 - ствол, 2 - шипики)
    predicted_path = "C:/Users/Student/datasets/final_segm_preds2version/in_vivo_14_mask_with_vsot.tif"
    
    # 2. Файлы оригинальной разметки (Ground Truth)
    label_shaft_path = "C:/Users/Student/datasets/in_vivo/14/dendrite.tif"
    label_spine_path = "C:/Users/Student/datasets/in_vivo/14/spines.tif"
    
    # 3. Зона интереса (опционально)
    area_path = "C:/Users/Student/datasets/in_vivo/14/areas.tif" 
    
    # 4. Куда сохранять цветные карты
    out_dir = "C:/Users/Student/datasets/diff_maps/"
    os.makedirs(out_dir, exist_ok=True)
    out_shaft_diff = os.path.join(out_dir, "vivo14diff_shaft_with_vsot.tif")
    out_spine_diff = os.path.join(out_dir, "vivo14diff_spine_with_vsot.tif")

    print("Загрузка файлов...")
    # --- ЗАГРУЗКА ---
    predicted_img = imread(predicted_path)
    label_shaft_img = imread(label_shaft_path)
    label_spine_img = imread(label_spine_path)

    print("Разделение классов и бинаризация...")
    # --- ВЫДЕЛЕНИЕ ПРЕДСКАЗАНИЙ ---
    # Вытаскиваем ствол (где пиксели равны 1) и шипики (где равны 2)
    pred_shaft = (predicted_img == 1)
    pred_spine = (predicted_img == 2)

    # --- БИНАРИЗАЦИЯ РАЗМЕТКИ ---
    gt_shaft = (label_shaft_img > 0)
    gt_spine = (label_spine_img > 0)

    # --- ЗОНА ИНТЕРЕСА (Area of Interest) ---
    if os.path.exists(area_path):
        print("Применение маски Area of Interest...")
        area = imread(area_path) > 0
        pred_shaft = pred_shaft & area
        pred_spine = pred_spine & area
        gt_shaft = gt_shaft & area
        gt_spine = gt_spine & area

    print("Генерация цветных RGB масок...")
    # --- СОЗДАНИЕ КАРТ ОШИБОК ---
    diff_shaft = create_diff_mask_binary(pred_shaft, gt_shaft)
    diff_spine = create_diff_mask_binary(pred_spine, gt_spine)

    print("Сохранение результатов...")
    # --- СОХРАНЕНИЕ ---
    imwrite(out_shaft_diff, diff_shaft, photometric='rgb')
    imwrite(out_spine_diff, diff_spine, photometric='rgb')
    
    print(f"Готово! Карта ствола сохранена в: {out_shaft_diff}")
    print(f"Готово! Карта шипиков сохранена в: {out_spine_diff}")

# if __name__ == "__main__":
#     from tifffile import imread
#     N="2"
#     stage3_path = "C:/Users/Student/datasets/origin_scale_preds/in_vitro_2_final.tif"
#     stage2_path = "C:/Users/Student/datasets/necks_scale_preds/in_vitro_2_necks.tif"
#     stage1_path = "C:/Users/Student/datasets/L_dendrite_scale_preds/in_vitro_2_L_dendrite.tif"
#     label_path = "C:/Users/Student/datasets/in_vitro/2/binarization.tif"
#     necks_path = "C:/Users/Student/datasets/in_vitro/2/necks.tif"
#     area_path = "C:/Users/Student/datasets/in_vitro/2/areas.tif"
#     st3_img = imread(stage3_path)
#     st2_img = imread(stage2_path)
#     st1_img = imread(stage1_path)
#     label_img = imread(label_path)
#     necks_img = imread(necks_path)

#     labels_img = scale_image(label_img, [ 0.1, 0.022, 0.022 ])
#     necks_imgs = scale_image(necks_img, [ 0.1, 0.022, 0.022 ])

#     if os.path.exists(area_path):
#         area = imread(area_path)
#         area[area > 0] = 1
#         necks_img = necks_img * area
#         areas = scale_image(area, [ 0.1, 0.022, 0.022 ])
#         labels_img = labels_img * areas
#         st1_img = st1_img * areas
#         st2_img = st2_img * areas
#         st3_img = st3_img * area
#         necks_imgs = necks_imgs * areas


#     diff_1label = create_diff_mask_binary(st1_img, labels_img)
#     diff_2label = create_diff_mask_binary(st2_img, necks_imgs)
#     diff_3label = create_diff_mask_binary(st3_img, necks_img)
#     diff_21 = create_diff_mask_binary(st2_img, st1_img)
    
#     out_1label = "2/1Sdiff_1label_vitro2.tif"
#     out_2label = "2/1Sdiff_2label_vitro2.tif"
#     out_3label = "2/Sdiff_3label_vitro2.tif"
#     out_21 = "2/1Sdiff_21_vitro2.tif"
#     imwrite(out_1label, diff_1label, photometric='rgb')
#     imwrite(out_2label, diff_2label, photometric='rgb')
#     imwrite(out_3label, diff_3label, photometric='rgb')
#     imwrite(out_21, diff_21, photometric='rgb')