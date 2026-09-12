import hydra
import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from tifffile import imread, imwrite
from tqdm import tqdm

from src.models.stage4_module import Stage4SegmentationModule

def stitch_volume(predictions, original_shape):
    full = torch.zeros((1, *original_shape), dtype=torch.float32)
    counts = torch.zeros((1, *original_shape), dtype=torch.float32)

    for batch in tqdm(predictions):
        preds = batch["preds"].cpu() 
        coords = batch["coords"].cpu() 
        valid_shape = batch["valid_shape"].cpu()
        
        p = preds[0, 0] 
        z, y, x = coords[0]
        vz, vy, vx = valid_shape[0]
        p_crop = p[:vz, :vy, :vx]
        
        full[:, z:z+vz, y:y+vy, x:x+vx] += p_crop
        counts[:, z:z+vz, y:y+vy, x:x+vx] += 1

    counts[counts == 0] = 1
    return (full / counts).squeeze().numpy()


def predict_and_stitch_on_the_fly(model, datamodule, device="cuda"):
    """
    Делает предсказания и СРАЗУ сшивает их в итоговый объем,
    чтобы не копить патчи в оперативной памяти (решает проблему OOM).
    """
    # 1. Получаем загрузчик данных и оригинальный размер снимка
    dl = datamodule.predict_dataloader()
    original_shape = datamodule.predict_ds.shape
    
    # 2. Выделяем память под итоговый объем один раз
    full = np.zeros(original_shape, dtype=np.float32)
    counts = np.zeros(original_shape, dtype=np.uint8) # uint8 весит в 4 раза меньше
    
    # Переводим модель в режим предсказания и кидаем на видеокарту
    model.eval()
    model.to(device)
    
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
            
            # В. Вшиваем патчи в гигантский массив
            for b in range(preds.shape[0]):
                p = preds[b, 0]  # Берем единственный канал (0.5 ствол, 1.0 шипики)
                z, y, x = coords[b]
                vz, vy, vx = valid_shapes[b]
                
                # Прибавляем только валидную (без паддинга) часть
                full[z:z+vz, y:y+vy, x:x+vx] += p[:vz, :vy, :vx]
                counts[z:z+vz, y:y+vy, x:x+vx] += 1
                
    # 3. Усредняем вероятности на перекрытиях
    counts[counts == 0] = 1
    prob_map = full / counts
    
    return prob_map


@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage4")
def main(cfg):
    JSON_PATH = "data_preprocessing/CVstage4/folds5.json"
    BASE_PATH = "../../" # базовый путь к картинкам
    OUT_DIR = "C:/Users/Student/datasets/CVstage4/result"

    os.makedirs(OUT_DIR, exist_ok=True)

    # ckpt = "checkpoints_stage4_new/epoch=7-step=22144.ckpt"
    ckpt = "../../../datasets/CVstage4/checkpoints/folds5/epoch=27-val_loss=0.0008.ckpt"
    try:
        model = Stage4SegmentationModule.load_from_checkpoint(ckpt, weights_only=False)
    except:
        print(f"Checkpoint {ckpt} not found, using random weights for test.")
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
        scale = item.get("scale")

        area_path = resolve(item.get("area_of_interest"))
        skel_path = resolve(item.get("skeleton"))
        # general_mask_path = resolve(item.get("general_mask"))#labid
        # # general_mask_path = resolve(item.get("image"))#vsot
        vsot_shaft_path = resolve(item.get("shaft_vsot"))
        vsot_spine_path = resolve(item.get("spine_vsot"))
        if area_path:
            area_path = os.path.normpath(os.path.join(BASE_PATH, area_path))


        if not os.path.exists(skel_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {skel_path}")
            continue
        if not os.path.exists(vsot_shaft_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {vsot_shaft_path}")
            continue
        if not os.path.exists(vsot_spine_path):
            print(f"Skipping {name}, Stage 4 prediction missing: {vsot_spine_path}")
            continue
        # if not os.path.exists(general_mask_path):
        #     raise ValueError("Stage 4 inference requires general_mask_path!")
        
        print(f"\nProcessing {name}...")

        cfg.datamodule.skeleton_path = skel_path
        cfg.datamodule.vsot_shaft_path = vsot_shaft_path
        cfg.datamodule.vsot_spine_path = vsot_spine_path
        # cfg.datamodule.general_mask_path = general_mask_path
        cfg.datamodule.area_path = area_path
        cfg.datamodule.current_scale = scale
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup("predict")

        # preds = trainer.predict(model, datamodule=dm)

        # prob_map = stitch_volume(preds, dm.predict_ds.shape)
        prob_map = predict_and_stitch_on_the_fly(model, dm, device="cuda")

        out_file = os.path.join(OUT_DIR, f"{name}_result.tif")
        imwrite(out_file, prob_map.astype(np.float32))
        print(f"Saved probabilities to {out_file}")

if __name__ == "__main__":
    main()