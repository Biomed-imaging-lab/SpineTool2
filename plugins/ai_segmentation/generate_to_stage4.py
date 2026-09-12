import hydra
import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
from tifffile import imwrite
from tqdm import tqdm

from src.models.stage3_module import Stage3SegmentationModule

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

@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage3")
def main(cfg):
    JSON_PATH = "data_preprocessing/CVstage1/9009_d_disk.json"
    L1_DIR = "D:/shteinberg/9009/ai_segm/stage2/prefinetune"
    L2_DIR = "D:/shteinberg/9009/ai_segm/stage3/necks_08skel"
    BASE_PATH = "" # базовый путь к картинкам
    OUT_DIR = "D:/shteinberg/9009/ai_segm/stage4/pred_image_f_s2_08s"
    OUT_DIR_bin = "D:/shteinberg/9009/ai_segm/stage4/bin_pred_image_f_s2_08s"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_bin, exist_ok=True)
#1 epoch=16-val_loss=0.1691
#2 epoch=02-val_loss=0.2779
#3 epoch=05-val_loss=0.1634
#4 epoch=06-val_loss=0.2857
#5 epoch=11-val_loss=0.2087
    ckpt = "D:/shteinberg/CVstage3/checkpoints/f_s2_08s/folds5/epoch=11-val_loss=0.2087.ckpt"# -1folds
    try:#epoch=11-val_loss=0.2600
        model = Stage3SegmentationModule.load_from_checkpoint(ckpt, weights_only=False)
    except:
        print(f"Checkpoint {ckpt} not found, using random weights for test.")
        return
    
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=True, strategy="auto", precision="16-mixed")


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
        necks_path = os.path.join(L2_DIR, f"{name}_necks.tif")
        if not os.path.exists(img_path):
            print(f"skipping {name}, file not found: {img_path}")
            continue
        if not os.path.exists(l_dendrite_path):
            print(f"Skipping {name}, Stage 1 prediction missing: {l_dendrite_path}")
            continue
        if not os.path.exists(necks_path):
            print(f"Skipping {name}, Stage 1 prediction missing: {necks_path}")
            continue

        print(f"\nProcessing {name}...")

        cfg.datamodule.predict_tiff_path = img_path
        cfg.datamodule.predict_l_dendrite_path = l_dendrite_path
        cfg.datamodule.predict_necks_path = necks_path
        cfg.datamodule.area_path = area_path
        cfg.datamodule.current_scale = scale
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup("predict")

        loader = dm.predict_dataloader()

        full = torch.zeros(
            (1, *dm.predict_ds.shape),
            dtype=torch.float32,
        )

        counts = torch.zeros(
            (1, *dm.predict_ds.shape),
            dtype=torch.float32,
        )

        model.eval()
        model.cuda()

        with torch.inference_mode():
            for batch in tqdm(loader):

                x = batch["image"].cuda(non_blocking=True)

                coords = batch["coords"]
                valid_shape = batch["valid_shape"]

                logits = model(x)
                preds = torch.sigmoid(logits).cpu()

                batch_size = preds.shape[0]

                for i in range(batch_size):

                    p = preds[i, 0]

                    z, y, x_coord = coords[i]
                    vz, vy, vx = valid_shape[i]

                    z = int(z)
                    y = int(y)
                    x_coord = int(x_coord)

                    vz = int(vz)
                    vy = int(vy)
                    vx = int(vx)

                    p_crop = p[:vz, :vy, :vx]

                    full[
                        :,
                        z:z+vz,
                        y:y+vy,
                        x_coord:x_coord+vx,
                    ] += p_crop

                    counts[
                        :,
                        z:z+vz,
                        y:y+vy,
                        x_coord:x_coord+vx,
                    ] += 1

                del x
                del logits
                del preds

        counts[counts == 0] = 1

        prob_map = (full / counts).squeeze().numpy()

        # from skimage.filters import threshold_local
        # block_size = (11, 33, 33) 
        # local_thresh = threshold_local(prob_map, block_size=block_size, offset=0.01)
        # binary = prob_map > local_thresh 
        binary = (prob_map > 0.3).astype(np.uint8) * 255

        out_file = os.path.join(OUT_DIR, f"{name}_final.tif")
        imwrite(out_file, prob_map.astype(np.float32))
        out_file_bin = os.path.join(OUT_DIR_bin, f"{name}_final_bin.tif")
        imwrite(out_file_bin, binary)
        print(f"Saved probabilities to {out_file}")

        del prob_map
        del binary

        if hasattr(dm, "predict_ds"):
            del dm.predict_ds

        del dm

        import gc
        gc.collect()

        torch.cuda.empty_cache()

        print(
            f"GPU allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB"
        )

if __name__ == "__main__":
    main()