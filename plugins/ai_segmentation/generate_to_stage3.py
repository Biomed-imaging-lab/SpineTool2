import hydra
import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from tifffile import imread, imwrite
from tqdm import tqdm

from src.models.segmentation_module_stage2_skeleton import Stage2SegmentationModule

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

@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage2")
def main(cfg):
    JSON_PATH = "data_preprocessing/CVstage2/folds5.json"
    L1_DIR = "C:/Users/Student/datasets/CVstage2/L_dendrite_2"
    BASE_PATH = "../../" # базовый путь к картинкам
    # OUT_DIR = "C:/Users/Student/datasets/CVstage3/necks_03skel"
    # OUT_DIR_bin = "C:/Users/Student/datasets/CVstage2/bin_neck_03skel"
    OUT_DIR = "C:/Users/Student/datasets/CVstage3/necks_08skel"
    OUT_DIR_bin = "C:/Users/Student/datasets/CVstage2/bin_neck_08skel"

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_bin, exist_ok=True)

    # ckpt = "../../../datasets/CVstage2/checkpoints_03skel/folds5/last-v1.ckpt"#checkpoints_1/folds1/epoch=27-val_loss=0.1253   checkpoints_2/folds1/last-v1.ckpt
    ckpt = "../../../datasets/CVstage2/checkpoints_08skel/folds5/last-v1.ckpt"
    try:
        model = Stage2SegmentationModule.load_from_checkpoint(ckpt, weights_only=False)
    except:
        print(f"Checkpoint {ckpt} not found, using random weights for test.")
        return
    
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)


    with open(JSON_PATH, 'r') as f:
        config = json.load(f)

    ts = set(config.get('test_data', []))
    for item in config['data']:
        name = item['name']
        if name not in ts: continue
        
        scale = item.get("scale")
        img_path = os.path.normpath(os.path.join(BASE_PATH, item['image']))
        area_path = item.get("area_of_interest")
        if area_path:
            area_path = os.path.normpath(os.path.join(BASE_PATH, area_path))

        l_dendrite_path = os.path.join(L1_DIR, f"{name}_L_dendrite.tif")

        if not os.path.exists(img_path):
            print(f"skipping {name}, file not found: {img_path}")
            continue
        if not os.path.exists(l_dendrite_path):
            print(f"Skipping {name}, Stage 1 prediction missing: {l_dendrite_path}")
            continue

        print(f"\nProcessing {name}...")

        cfg.datamodule.predict_tiff_path = img_path
        cfg.datamodule.predict_l_dendrite_path = l_dendrite_path
        cfg.datamodule.area_path = area_path
        cfg.datamodule.current_scale = scale
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup("predict")

        preds = trainer.predict(model, datamodule=dm)

        prob_map = stitch_volume(preds, dm.predict_ds.shape)
        binary = (prob_map > 0.5).astype(np.uint8) * 255

        out_file = os.path.join(OUT_DIR, f"{name}_necks.tif")
        imwrite(out_file, prob_map.astype(np.float32))
        out_file_bin = os.path.join(OUT_DIR_bin, f"{name}_bin_neck.tif")
        imwrite(out_file_bin, binary)
        print(f"Saved probabilities to {out_file}")

if __name__ == "__main__":
    main()