import hydra
import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from tifffile import imread, imwrite
from tqdm import tqdm

from src.models.stage4_module import Stage4SegmentationModule


import cc3d
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter


def postprocess_patch_boundary_spines_voxels(
    prob_shaft: np.ndarray,
    prob_spine: np.ndarray,
    patch_size=(64, 128, 128),
    overlap=(48, 96, 96),
    neighborhood_size=9,
    uncertainty_threshold=0.15,
    spine_advantage=2.0,
):
    """
    Исправляет артефакты на границах патчей.

    Меняет только:
        - текущие shaft-воксели
        - лежащие на границе патча
        - с низкой уверенностью сети
        - окружённые преимущественно spine

    Возвращает:
        corrected_prob_shaft,
        corrected_prob_spine
    """

    prob_shaft = prob_shaft.copy()
    prob_spine = prob_spine.copy()

    shape = prob_shaft.shape

    stride = tuple(
        p - o
        for p, o in zip(patch_size, overlap)
    )

    #
    # Точная маска границ патчей
    #
    patch_boundary = np.zeros(shape, dtype=bool)

    # z-плоскости
    for z in range(stride[0], shape[0], stride[0]):
        patch_boundary[z, :, :] = True

    # y-плоскости
    for y in range(stride[1], shape[1], stride[1]):
        patch_boundary[:, y, :] = True

    # x-плоскости
    for x in range(stride[2], shape[2], stride[2]):
        patch_boundary[:, :, x] = True

    #
    # Текущий класс
    #
    shaft_prediction = prob_shaft > prob_spine

    #
    # Неуверенные решения сети
    #
    uncertainty = np.abs(prob_shaft - prob_spine)

    uncertain = uncertainty < uncertainty_threshold

    #
    # Локальное окружение
    #
    spine_density = uniform_filter(
        prob_spine.astype(np.float32),
        size=neighborhood_size,
        mode="nearest",
    )

    shaft_density = uniform_filter(
        prob_shaft.astype(np.float32),
        size=neighborhood_size,
        mode="nearest",
    )

    #
    # Кандидаты на исправление
    #
    fix_mask = (
        patch_boundary
        & shaft_prediction
        & uncertain
        & (spine_density > shaft_density * spine_advantage)
    )

    n_fixed = int(np.count_nonzero(fix_mask))

    if n_fixed > 0:
        transferred = prob_shaft[fix_mask]

        prob_spine[fix_mask] = np.maximum(
            prob_spine[fix_mask],
            transferred,
        )

        prob_shaft[fix_mask] = 0.0

    print(
        f"Patch-boundary correction: "
        f"{n_fixed:,} voxels corrected"
    )

    return prob_shaft, prob_spine


def convert_small_shaft_components_to_spines(
    prob_shaft: np.ndarray,
    prob_spine: np.ndarray,
    max_component_size: int = 50_000,
):
    """
    Все маленькие компоненты shaft,
    не являющиеся главным дендритом,
    переводятся в spine.
    """

    prob_shaft = prob_shaft.copy()
    prob_spine = prob_spine.copy()

    shaft_mask = prob_shaft > prob_spine

    labels, n_labels = cc3d.connected_components(
        shaft_mask.astype(np.uint8),
        connectivity=26,
        return_N=True,
    )

    if n_labels == 0:
        return prob_shaft, prob_spine

    sizes = np.bincount(labels.ravel())

    #
    # самая большая компонента = главный дендрит
    #
    main_label = np.argmax(sizes[1:]) + 1

    candidate_labels = np.where(
        sizes <= max_component_size
    )[0]

    candidate_labels = candidate_labels[
        candidate_labels != 0
    ]

    candidate_labels = candidate_labels[
        candidate_labels != main_label
    ]

    if len(candidate_labels) == 0:
        return prob_shaft, prob_spine

    fix_mask = np.isin(
        labels,
        candidate_labels,
    )

    transferred = prob_shaft[fix_mask]

    prob_spine[fix_mask] = np.maximum(
        prob_spine[fix_mask],
        transferred,
    )

    prob_shaft[fix_mask] = 0.0

    print(
        f"Corrected {len(candidate_labels)} components "
        f"({fix_mask.sum():,} voxels)"
    )

    return prob_shaft, prob_spine


def predict_and_stitch_on_the_fly(model, datamodule, device="cuda"):
    """
    Делает предсказания и СРАЗУ сшивает их в итоговый объем,
    чтобы не копить патчи в оперативной памяти (решает проблему OOM).
    """
    # 1. Получаем загрузчик данных и оригинальный размер снимка
    dl = datamodule.predict_dataloader()
    original_shape = datamodule.predict_ds.shape
    
    # 2. Выделяем память под два независимых холста (ствол и шипики) и один общий счетчик
    full_shaft = np.zeros(original_shape, dtype=np.float32)
    full_spine = np.zeros(original_shape, dtype=np.float32)
    counts = np.zeros(original_shape, dtype=np.uint8) # uint8 весит в 4 раза меньше
    
    # Переводим модель в режим предсказания и кидаем на видеокарту
    model.eval()
    model.to(device)
    i = 0
    print("Начинаем инференс со сшиванием на лету...")
    with torch.no_grad(): # Обязательно выключаем градиенты
        for batch in tqdm(dl, desc="Predicting & Stitching"):
            
            # А. Кидаем картинку на GPU и получаем прогноз
            imgs = batch["image"].to(device)
            
            preds = model(imgs)
            preds = torch.sigmoid(preds)
            # Б. Сразу стягиваем результаты обратно на процессор (в numpy)
            preds = preds.cpu().numpy()
            coords = batch["coords"].numpy()
            valid_shapes = batch["valid_shape"].numpy()
            vsot = batch["vsot_segm"].numpy()
            
            # В. Вшиваем патчи в гигантский массив
            for b in range(preds.shape[0]):
                z, y, x = coords[b]
                vz, vy, vx = valid_shapes[b]
            
                shaft_patch = preds[b, 0, :vz, :vy, :vx]
                spine_patch = preds[b, 1, :vz, :vy, :vx]

                vsot_patch = vsot[b, :vz, :vy, :vx]

                has_shaft = np.any(vsot_patch == 1)

                # Если в патче нет ни одного пикселя ствола,
                # считаем весь патч шипиком
                if not has_shaft:
                    spine_patch += shaft_patch
                    shaft_patch.fill(0)

                full_shaft[
                    z:z+vz,
                    y:y+vy,
                    x:x+vx
                ] += shaft_patch

                full_spine[
                    z:z+vz,
                    y:y+vy,
                    x:x+vx
                ] += spine_patch

                counts[
                    z:z+vz,
                    y:y+vy,
                    x:x+vx
                ] += 1
            i += 1
                
    # 3. Усредняем вероятности на перекрытиях
    counts[counts == 0] = 1
    prob_map_shaft = full_shaft / counts
    prob_map_spine = full_spine / counts
    
    return prob_map_shaft, prob_map_spine


@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage4")
def main(cfg):
    JSON_PATH = "data_preprocessing/CVstage1/9009_d_disk.json"
    BASE_PATH = "" # базовый путь к картинкам
    OUT_DIR = "D:/shteinberg/9009/ai_segm/stage4/result_gt_ch2"

    os.makedirs(OUT_DIR, exist_ok=True)

    # ckpt = "checkpoints_stage4_new/epoch=7-step=22144.ckpt"
    ckpt = "../../../datasets/CVstage4/checkpoints/gt_ch2/folds3/epoch=25-val_loss=0.0089.ckpt"
    #gt_ch2/folds1/epoch=21-val_loss=0.0045.ckpt
    #gt_ch2/folds2/epoch=26-val_loss=0.0059.ckpt
    #gt_ch2/folds3/epoch=25-val_loss=0.0089.ckpt
    try:
        model = Stage4SegmentationModule.load_from_checkpoint(ckpt, weights_only=False)
    except:
        print(f"Checkpoint {ckpt} not found.")
        return
    
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)


    with open(JSON_PATH, 'r') as f:
        config = json.load(f)

    def resolve(p): 
        if not p: return None
        return os.path.normpath(os.path.join(BASE_PATH, p))
    ts = set(config.get('test_data', []))
    for item in config['data']:
        name = item['name']
        if name not in ts: continue

        area_path = resolve(item.get("area_of_interest"))

        # general_mask_path = resolve(item.get("general_mask"))#labid
        # # general_mask_path = resolve(item.get("image"))#vsot
        vsot_shaft_path = resolve(item.get("shaft_vsot"))
        vsot_spine_path = resolve(item.get("spine_vsot"))
        shaft_dist_path = resolve(item.get("shaft_dist"))
        spine_dist_path = resolve(item.get("spine_dist"))
        if area_path:
            area_path = os.path.normpath(os.path.join(BASE_PATH, area_path))

        if not os.path.exists(vsot_shaft_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {vsot_shaft_path}")
            continue
        if not os.path.exists(vsot_spine_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {vsot_spine_path}")
            continue
        if not os.path.exists(shaft_dist_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {shaft_dist_path}")
            continue
        if not os.path.exists(spine_dist_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {spine_dist_path}")
            continue
        # if not os.path.exists(general_mask_path):
        #     raise ValueError("Stage 4 inference requires general_mask_path!")
        
        print(f"\nProcessing {name}...")


        cfg.datamodule.vsot_shaft_path = vsot_shaft_path
        cfg.datamodule.vsot_spine_path = vsot_spine_path
        cfg.datamodule.shaft_dist_path = shaft_dist_path
        cfg.datamodule.spine_dist_path = spine_dist_path
        # cfg.datamodule.general_mask_path = general_mask_path
        cfg.datamodule.area_path = area_path
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup("predict")

        # preds = trainer.predict(model, datamodule=dm)

        # prob_map = stitch_volume(preds, dm.predict_ds.shape)
        prob_shaft, prob_spine = predict_and_stitch_on_the_fly(model, dm, device="cuda")
        prob_shaft, prob_spine = convert_small_shaft_components_to_spines(prob_shaft, prob_spine)
        # Сценарий 1: Двухканальный цветной TIFF-композит (Hyperstack для ImageJ)
        composite_volume = np.stack([prob_shaft, prob_spine], axis=1).astype(np.float32)
        out_composite_file = os.path.join(OUT_DIR, f"{name}_final_2ch.tif")
        
        # 'axes': 'CZYX' сообщает ImageJ, что первое измерение массива — это Каналы цвета
        imwrite(
            out_composite_file, 
            composite_volume, 
            imagej=True, 
            metadata={'axes': 'ZCYX'}
        )
        print(f"    [+] Сохранен 2-канальный цветной: {out_composite_file}")

        # Сценарий 2: Одноканальная жесткая маска классов (0, 1, 2) через argmax
        # Создаем фиксированный слой фона (порог уверенности 0.5)
        bg_threshold = np.ones_like(prob_shaft) * 0.1
        
        # Собираем 4D-массив вероятностей формы [3, Depth, Height, Width]
        # Индекс 0 = Фон, Индекс 1 = Ствол, Индекс 2 = Шипик
        stacked_probs = np.stack([bg_threshold, prob_shaft, prob_spine], axis=0)
        
        # argmax находит индекс слоя с максимальной вероятностью для каждого вокселя.
        final_integer_mask = np.argmax(stacked_probs, axis=0).astype(np.uint8)
        
        # Умножаем маску на 127, чтобы классы (0, 1, 2) превратились в контрастные (0, 127, 254)
        out_mask_file = os.path.join(OUT_DIR, f"{name}_final.tif")
        imwrite(out_mask_file, (final_integer_mask * 127), imagej=True)
        print(f"    [+] Сохранена жесткая маска классов (0,1,2): {out_mask_file}")
        
        print(f"[+] Снимок {name} успешно обработан.\n")
        # out_file = os.path.join(OUT_DIR, f"{name}_result.tif")
        # imwrite(out_file, prob_map.astype(np.float32))
        # print(f"Saved probabilities to {out_file}")

if __name__ == "__main__":
    main()