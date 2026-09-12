import hydra
import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from tifffile import imread, imwrite
from tqdm import tqdm
import cv2

from src.models.segmentation_module import DendriteSegmentationModule

def scale_binary_to_original(binary_mask, target_shape):
    """
    Масштабирует Y и X бинарной маски к исходному разрешению снимка,
    используя метод ближайших соседей (INTER_NEAREST).
    """
    _, target_y, target_x = target_shape
    
    if binary_mask.shape[1] == target_y and binary_mask.shape[2] == target_x:
        return binary_mask
        
    resized = np.zeros((binary_mask.shape[0], target_y, target_x), dtype=np.uint8)
    
    for i in range(binary_mask.shape[0]):
        resized[i] = cv2.resize(
            binary_mask[i], 
            (target_x, target_y), 
            interpolation=cv2.INTER_NEAREST
        )
    return resized

def stitch_volume(predictions, original_shape, pad=16):
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

    stitched_volume = (full / counts).squeeze().numpy()
    if pad > 0:
        stitched_volume = stitched_volume[pad : -pad, pad : -pad, pad : -pad]
    return stitched_volume

@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage1_lora")
def main(cfg):
    JSON_PATH = "data_preprocessing/CVstage1/finetune_5im.json"
    BASE_PATH = "../../" # базовый путь к картинкам
    OUT_DIR = "C:/Users/Student/datasets/CVstage2/L_dendrite_2/prefinetune"
    OUT_DIR_bin = "C:/Users/Student/datasets/CVstage1/bin_fin_scaled/prefinetune"
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_bin, exist_ok=True)

    ckpt = "../../../datasets/CVstage1/checkpoints_2/folds1/epoch=14-val_loss=0.0093.ckpt"#epoch=13-val_loss=0.0065#epoch=11-val_loss=0.0066
    try:
        model = DendriteSegmentationModule.load_from_checkpoint(ckpt, weights_only=False)
    except:
        print(f"Checkpoint {ckpt} not found, using random weights for test.")

    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)


    with open(JSON_PATH, 'r') as f: config = json.load(f)

    ts = set(config.get('test_data', []))
    for item in config['data']:
        name = item['name']
        if name not in ts: continue
        
        scale = item.get("scale")
        img_path = os.path.normpath(os.path.join(BASE_PATH, item['image']))
        area_path = item.get("area_of_interest")
        if area_path:
            area_path = os.path.normpath(os.path.join(BASE_PATH, area_path))

        if not os.path.exists(img_path):
            print(f"skipping {name}, file not found: {img_path}")
            continue
            
        print(f"\nProcessing {name}...")

        cfg.datamodule.predict_tiff_path = img_path
        cfg.datamodule.area_path = area_path
        cfg.datamodule.current_scale = scale
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup("predict")

        preds = trainer.predict(model, datamodule=dm)
        prob_map = stitch_volume(preds, dm.predict_ds.shape)
        out_file = os.path.join(OUT_DIR, f"{name}_L_dendrite.tif")
        imwrite(out_file, prob_map.astype(np.float32))

        binary_scaled = (prob_map > 0.5).astype(np.uint8) * 255
        
        original_shape = imread(img_path).shape
        binary = scale_binary_to_original(binary_scaled, original_shape)
        out_file_bin = os.path.join(OUT_DIR_bin, f"{name}_bin.tif")
        imwrite(out_file_bin, binary)
        print(f"Saved probabilities to {out_file}")

if __name__ == "__main__":
    main()