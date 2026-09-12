import argparse
import sys
import json
from json import dump, load
from typing import Dict, List, Tuple
import time
import numpy as np
from scipy import ndimage as ndi
from tifffile import imread, imwrite

class NumpyEncoder(json.JSONEncoder):
    """Специальный класс, который учит JSON понимать типы NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# входные данные - json файл следующего вида
"""
{
    "need_images": <true/false, определяет будут ли генерироваться цветные маски с размеченными TP,FP,FN,TN вокселами; по умолчанию true>,
    "iou_threshold": <число от 0.5 до 1.0, определяет при каком значении IOU шипик считается отмеченным верно; по умолчанию 0.5>,
    "data": [
        {
            "id": <идентификатор, чтобы выходные данные можно было сопоставить потом; по умолчанию объединение строк из area of interest и ground truth>,
            "area of interest": <путь к маске, определяющей какие области изображения учитывать>,
            "ground truth": <путь к ручной разметке>,
            "result dendrites": <путь к бинарной маске сегментированного нейросетью дендрита>,
            "result spines": <путь к бинарной маске сегментированных нейросетью шипиков>
        },
        ...
    ]
}
"""

# выходные данные - набор цветных масок, если они запрошены, и json файл следующего вида
"""
{
    "general_metrics": {
        "binarization_metrics": { <только метрики вычисленные по маске целиком, на основе вокселов>
            "dice": {
                "min": <минимальное значение из всех обработанных фалов>,
                "max": <максимальное значение из всех обработанных фалов>,
                "mean": <среднее значение из всех обработанных фалов>,
                "median": <медианное значение из всех обработанных фалов>,
                "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
            },
            "iou": {
                "min": <минимальное значение из всех обработанных фалов>,
                "max": <максимальное значение из всех обработанных фалов>,
                "mean": <среднее значение из всех обработанных фалов>,
                "median": <медианное значение из всех обработанных фалов>,
                "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
            }
        },
        "dendrite_metrics": { <только метрики вычисленные по маске дендрита, на основе вокселов>
            "dice": {
                "min": <минимальное значение из всех обработанных фалов>,
                "max": <максимальное значение из всех обработанных фалов>,
                "mean": <среднее значение из всех обработанных фалов>,
                "median": <медианное значение из всех обработанных фалов>,
                "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
            },
            "iou": {
                "min": <минимальное значение из всех обработанных фалов>,
                "max": <максимальное значение из всех обработанных фалов>,
                "mean": <среднее значение из всех обработанных фалов>,
                "median": <медианное значение из всех обработанных фалов>,
                "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
            }
        },
        "spines_metrics": {
            "voxels_metrics": { <метрики вычисленные по маске шипиков, на основе вокселов>
                "dice": {
                    "min": <минимальное значение из всех обработанных фалов>,
                    "max": <максимальное значение из всех обработанных фалов>,
                    "mean": <среднее значение из всех обработанных фалов>,
                    "median": <медианное значение из всех обработанных фалов>,
                    "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
                },
                "iou": {
                    "min": <минимальное значение из всех обработанных фалов>,
                    "max": <максимальное значение из всех обработанных фалов>,
                    "mean": <среднее значение из всех обработанных фалов>,
                    "median": <медианное значение из всех обработанных фалов>,
                    "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
                }
            },
            "instances_metrics": { <метрики вычисленные по числу распознанных шипиков>
                "precision": {
                    "min": <минимальное значение из всех обработанных фалов>,
                    "max": <максимальное значение из всех обработанных фалов>,
                    "mean": <среднее значение из всех обработанных фалов>,
                    "median": <медианное значение из всех обработанных фалов>,
                    "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
                },
                "recall": {
                    "min": <минимальное значение из всех обработанных фалов>,
                    "max": <максимальное значение из всех обработанных фалов>,
                    "mean": <среднее значение из всех обработанных фалов>,
                    "median": <медианное значение из всех обработанных фалов>,
                    "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
                },
                "f1": {
                    "min": <минимальное значение из всех обработанных фалов>,
                    "max": <максимальное значение из всех обработанных фалов>,
                    "mean": <среднее значение из всех обработанных фалов>,
                    "median": <медианное значение из всех обработанных фалов>,
                    "global": <значение вычисленное по всем файлам сразу, путем объединения tp,fp,fn>
                }
            }
        }
    },
    "files_metrics": [
        {
            "id": <идентификатор, такой же как во входных данных, чтобы можно было устанавливать соответствие>,
            "colored_mask": <путь к цветной маске; присутствует, если need_images=true>,
            "binarization_metrics": {
                "tp": <>,
                "fp": <>,
                "fn": <>,
                "dice": <>,
                "iou": <>
            },
            "dendrite_metrics": {
                "tp": <>,
                "fp": <>,
                "fn": <>,
                "dice": <>,
                "iou": <>
            },
            "spines_metrics": {
                "voxels_metrics": {
                    "tp": <>,
                    "fp": <>,
                    "fn": <>,
                    "dice": <>,
                    "iou": <>
                },
                "instances_metrics": {
                    "tp": <>,
                    "fp": <>,
                    "fn": <>,
                    "precision": <>,
                    "recall": <>,
                    "f1": <>
                }
            }
        },
        ...
    ]
}
"""


def dice(tp, fp, fn) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def iou(tp, fp, fn) -> float:
    return tp / (tp + fp + fn)


def precision(tp, fp, fn) -> float:
    return tp / (tp + fp)


def recall(tp, fp, fn) -> float:
    return tp / (tp + fn)


def compare_mask(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[int, int, int]:
    mask1[mask1 > 0] = 1
    mask2[mask2 > 0] = 1
    intersection = mask1 * mask2
    fp = mask2 - intersection
    fn = mask1 - intersection
    return np.sum(intersection), np.sum(fp), np.sum(fn)

def compare_spines(
    spines1: np.ndarray, spines2: np.ndarray, iou_threshold: float
) -> Tuple[int, int, int]:
    print("      [+] compare_spines: Начало...")
    t0 = time.time()
    spines1 = (spines1 > 0).astype(np.uint8)
    spines2 = (spines2 > 0).astype(np.uint8)

    structure = ndi.generate_binary_structure(3, 3)

    print("      [+] compare_spines: ndi.label (маркировка связных компонент)...")
    t1 = time.time()

    labeled_spines1, num_spines1 = ndi.label(spines1, structure=structure)
    labeled_spines2, num_spines2 = ndi.label(spines2, structure=structure)

    print(f"      [-] ndi.label завершен за {time.time() - t1:.2f} сек. Найдено GT: {num_spines1}, Pred: {num_spines2}")
    if num_spines1 == 0 and num_spines2 == 0:
        return 0, 0, 0
    elif num_spines1 == 0:
        return 0, num_spines2, 0 # Все предсказанные - ложные (FP)
    elif num_spines2 == 0:
        return 0, 0, num_spines1 # Все эталонные - пропущены (FN)

    print("      [+] compare_spines: Векторизованный подсчет пересечений...")
    t2 = time.time()
    sizes1 = np.bincount(labeled_spines1.ravel())
    sizes2 = np.bincount(labeled_spines2.ravel())

    mask_intersect = (labeled_spines1 > 0) & (labeled_spines2 > 0)
    
    MAX_PRED_ID = num_spines2 + 1
    combined_ids = labeled_spines1[mask_intersect].astype(np.uint64) * MAX_PRED_ID + labeled_spines2[mask_intersect].astype(np.uint64)
    
    unique_ids, counts = np.unique(combined_ids, return_counts=True)

    tp = 0
    matched_gt = set()
    matched_pred = set()

    for comb_id, intersect_vol in zip(unique_ids, counts):
        i = int(comb_id // MAX_PRED_ID)
        j = int(comb_id % MAX_PRED_ID)
        
        vol_gt = sizes1[i]
        vol_pred = sizes2[j]
        
        union_vol = vol_gt + vol_pred - intersect_vol
        
        if (intersect_vol / union_vol) >= iou_threshold:
            tp += 1
            matched_gt.add(i)
            matched_pred.add(j)

    fn = num_spines1 - len(matched_gt)
    fp = num_spines2 - len(matched_pred)
    print(f"      [-] Подсчет пересечений завершен за {time.time() - t2:.2f} сек.")
    print(f"      [-] compare_spines: ОБЩЕЕ время {time.time() - t0:.2f} сек.")
    return tp, fp, fn

# def compare_spines(
#     spines1: np.ndarray, spines2: np.ndarray, iou_threshold: float
# ) -> Tuple[int, int, int]:
#     spines1[spines1 > 0] = 1
#     spines2[spines2 > 0] = 1

#     structure = ndi.generate_binary_structure(3, 3)
#     labeled_spines1, num_spines1 = ndi.label(spines1, structure=structure)
#     labeled_spines2, num_spines2 = ndi.label(spines2, structure=structure)

#     first_to_second: Dict[int, list] = {}
#     second_to_first: Dict[int, list] = {}
#     for i in range(num_spines1):
#         for j in range(num_spines2):
#             if (
#                 first_to_second.get(i, None)
#                 and iou(
#                     *(
#                         compare_mask(
#                             labeled_spines1[labeled_spines1 == i],
#                             labeled_spines2[labeled_spines2 == j],
#                         )
#                     )
#                 )
#                 >= iou_threshold
#             ):
#                 first_to_second[i] = j
#                 second_to_first[j] = i

#     tp = 0
#     fp = 0
#     fn = 0
#     for i in range(num_spines1):
#         if first_to_second.get(i, None) is None:
#             fn += 1
#         else:
#             tp += 1
#     for j in range(num_spines2):
#         if second_to_first.get(j, None) is None:
#             fp += 1

#     return tp, fp, fn


def create_colored_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    print("      [+] Запуск генерации RGB маски...")
    t0 = time.time()
    m1_bool = mask1 > 0
    m2_bool = mask2 > 0

    intersection = m1_bool & m2_bool
    fn = m1_bool & ~m2_bool
    fp = m2_bool & ~m1_bool
    # mask1[mask1 > 0] = 1
    # mask2[mask2 > 0] = 1

    # intersection = mask1 * mask2
    # fn = mask1 - intersection
    # fp = mask2 - intersection

    red = np.array([250, 0, 0], dtype=np.uint8)
    black = np.array([0, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    blue = np.array([0, 0, 245], dtype=np.uint8)

    rgb_image = np.zeros((*mask1.shape, 3), dtype=np.uint8) + black

    rgb_image[intersection] = white
    rgb_image[fn] = red
    rgb_image[fp] = blue
    print(f"      [-] RGB маска сгенерирована за {time.time() - t0:.2f} сек.")
    return rgb_image


def get_min_max_mean_median_global(
    files_metrics: List[dict], metric_name: str, group: List[str], global_cb
) -> dict:
    global_tp = 0
    global_fp = 0
    global_fn = 0
    metrics = []

    for obj in files_metrics:
        obj_ = obj
        for g in group:
            obj_ = obj_[g]
        metrics.append(obj_[metric_name])
        global_tp += obj_["tp"]
        global_fp += obj_["fp"]
        global_fn += obj_["fn"]

    return {
        "min": np.min(metrics),
        "max": np.max(metrics),
        "mean": np.mean(metrics),
        "median": np.median(metrics),
        "global": global_cb(global_tp, global_fp, global_fn),
    }


def make_report(json: dict, output_folder: str) -> dict:
    files_metrics = []

    need_images = json.get("need_images", True)
    iou_threshold = json.get("iou_threshold", 0.5)
    if iou_threshold < 0.5:
        sys.exit(
            "Invalid data description file: iou_threshold mast be >= 0.5 and <= 1.0"
        )

    for obj in json["data"]:
        area_path = obj.get("area of interest", "")
        gt_path = obj["ground truth"]
        metric = {
            "id": obj.get("id", f"{gt_path}_{area_path}"),
            "binarization_metrics": {},
            "dendrite_metrics": {},
            "spines_metrics": {"voxels_metrics": {}, "instances_metrics": {}},
        }
        item_id = obj.get("id", f"{gt_path}_{area_path}")
        print(f"\n========================================")
        print(f"Обработка снимка: {item_id}")
        print(f"========================================")
        
        t_start = time.time()
        
        print("   [1] Загрузка TIF файлов...")
        t_load = time.time()

        area = None
        if area_path:
            area = imread(area_path)
            area[area > 0] = 1
        gt = imread(gt_path)
        if area is not None:
            gt = gt * area

        gt_dendr = np.zeros_like(gt)
        gt_dendr[gt == 1] = 1
        gt_spines = np.zeros_like(gt)
        gt_spines[gt == 2] = 1
        
        
        pred_img = imread(obj["result mask"])
        if area is not None:
            pred_img = pred_img * area

        # Разделяем: 1 - ствол, 2 - шипики
        segm_dendr = np.zeros_like(pred_img)
        segm_dendr[pred_img == 255] = 1
        
        segm_spines = np.zeros_like(pred_img)
        segm_spines[pred_img == 2] = 1

        binarization = segm_dendr + segm_spines


        # segm_dendr = imread(obj["result dendrites"])
        # segm_spines = imread(obj["result spines"])
        # binarization = segm_dendr + segm_spines

        print(f"   [-] Файлы загружены за {time.time() - t_load:.2f} сек.")

        print("   [2] Подсчет пиксельных метрик (compare_mask)...")

        structure = ndi.generate_binary_structure(3, 3)
        spines11 = (gt > 0).astype(np.uint8)
        spines21 = (binarization > 0).astype(np.uint8)

        print("      [+] compare_spines: ndi.label (маркировка связных компонент)...")

        labeled_spines11, num_spines11 = ndi.label(spines11, structure=structure)
        labeled_spines21, num_spines21 = ndi.label(spines21, structure=structure)
        metric["binarization_metrics"]["num_components_gt"] =  num_spines11
        metric["binarization_metrics"]["num_components"] =  num_spines21
        tp, fp, fn = compare_mask(gt, binarization)
        metric["binarization_metrics"]["tp"] = tp
        metric["binarization_metrics"]["fp"] = fp
        metric["binarization_metrics"]["fn"] = fn
        metric["binarization_metrics"]["dice"] = dice(tp, fp, fn)
        metric["binarization_metrics"]["iou"] = iou(tp, fp, fn)

        tp, fp, fn = compare_mask(gt_dendr, segm_dendr)
        metric["dendrite_metrics"]["tp"] = tp
        metric["dendrite_metrics"]["fp"] = fp
        metric["dendrite_metrics"]["fn"] = fn
        metric["dendrite_metrics"]["dice"] = dice(tp, fp, fn)
        metric["dendrite_metrics"]["iou"] = iou(tp, fp, fn)

        # tp, fp, fn = compare_mask(gt_spines, segm_spines)
        # metric["spines_metrics"]["voxels_metrics"]["tp"] = tp
        # metric["spines_metrics"]["voxels_metrics"]["fp"] = fp
        # metric["spines_metrics"]["voxels_metrics"]["fn"] = fn
        # metric["spines_metrics"]["voxels_metrics"]["dice"] = dice(tp, fp, fn)
        # metric["spines_metrics"]["voxels_metrics"]["iou"] = iou(tp, fp, fn)

        # print("   [3] Подсчет инстанс-метрик шипиков (compare_spines)...")

        # tp, fp, fn = compare_spines(gt_spines, segm_spines, iou_threshold)
        # metric["spines_metrics"]["instances_metrics"]["tp"] = tp
        # metric["spines_metrics"]["instances_metrics"]["fp"] = fp
        # metric["spines_metrics"]["instances_metrics"]["fn"] = fn
        # metric["spines_metrics"]["instances_metrics"]["precision"] = precision(
        #     tp, fp, fn
        # )
        # metric["spines_metrics"]["instances_metrics"]["recall"] = recall(tp, fp, fn)
        # metric["spines_metrics"]["instances_metrics"]["f1"] = dice(tp, fp, fn)

        if need_images:
            print("   [4] Сохранение цветной маски на диск...")
            t_save = time.time()
            colored_mask = create_colored_mask(gt, binarization)
            id_str = metric["id"].replace("/", "_").replace(".", "_").replace("\\", "_")
            path = f"{output_folder}/{id_str}.tif"
            imwrite(path, data=colored_mask)
            metric["colored_mask"] = path
            print(f"   [-] Маска сохранена за {time.time() - t_save:.2f} сек.")
            
        print(f"Снимок {item_id} полностью обработан за {time.time() - t_start:.2f} сек.")

        files_metrics.append(metric)

    general_metrics = {
        "binarization_metrics": {
            "dice": get_min_max_mean_median_global(
                files_metrics, "dice", ["binarization_metrics"], dice
            ),
            "iou": get_min_max_mean_median_global(
                files_metrics, "iou", ["binarization_metrics"], iou
            ),
        },
        "dendrite_metrics": {
            "dice": get_min_max_mean_median_global(
                files_metrics, "dice", ["dendrite_metrics"], dice
            ),
            "iou": get_min_max_mean_median_global(
                files_metrics, "iou", ["dendrite_metrics"], iou
            ),
        }
        # "spines_metrics": {
        #     "voxels_metrics": {
        #         "dice": get_min_max_mean_median_global(
        #             files_metrics, "dice", ["spines_metrics", "voxels_metrics"], dice
        #         ),
        #         "iou": get_min_max_mean_median_global(
        #             files_metrics, "iou", ["spines_metrics", "voxels_metrics"], iou
        #         ),
        #     },
            # "instances_metrics": {
            #     "precision": get_min_max_mean_median_global(
            #         files_metrics,
            #         "precision",
            #         ["spines_metrics", "instances_metrics"],
            #         precision,
            #     ),
            #     "recall": get_min_max_mean_median_global(
            #         files_metrics,
            #         "recall",
            #         ["spines_metrics", "instances_metrics"],
            #         recall,
            #     ),
            #     "f1": get_min_max_mean_median_global(
            #         files_metrics, "f1", ["spines_metrics", "instances_metrics"], dice
            #     ),
            # },
        # },
    }

    report = {}
    report["general_metrics"] = general_metrics
    report["files_metrics"] = files_metrics
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_file", type=str, default="", help="Path to the data description file"
    )
    parser.add_argument(
        "-output_folder", type=str, default="", help="Path to save the result"
    )

    args = parser.parse_args()

    if args.input_file == "":
        sys.exit("Invalid args. Path to the data description file is not specified")

    if args.output_folder == "":
        sys.exit("Invalid args. Path to save the result is not specified")

    with open(args.input_file) as f:
        data_description = load(f)
    if not isinstance(data_description, dict):
        sys.exit("Invalid data description file")

    with open(f"{args.output_folder}/metrics_report.json", "w") as f:
        dump(make_report(data_description, args.output_folder), f, cls=NumpyEncoder, indent=4)
