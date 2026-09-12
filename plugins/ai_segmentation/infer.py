import hydra
import torch
import numpy as np
import pytorch_lightning as pl
from tifffile import imwrite
from tqdm import tqdm
from src.models.segmentation_module import DendriteSegmentationModule


def stitch_volume(predictions, original_shape):
    # original_shape: (Z, Y, X) исходного файла
    full = torch.zeros((1, *original_shape), dtype=torch.float32)
    counts = torch.zeros((1, *original_shape), dtype=torch.float32)

    print("Stitching...")
    for batch in tqdm(predictions):
        preds = batch["preds"].cpu() # [1, D, H, W]
        coords = batch["coords"].cpu() # [1, 3]
        valid_shape = batch["valid_shape"].cpu()
		
        p = preds[0, 0] # (64, 64, 64)
        z, y, x = coords[0]
        vz, vy, vx = valid_shape[0]
        p_crop = p[:vz, :vy, :vx]
        
        # Складываем
        full[:, z:z+vz, y:y+vy, x:x+vx] += p_crop
        counts[:, z:z+vz, y:y+vy, x:x+vx] += 1

    counts[counts == 0] = 1
    return (full / counts).squeeze().numpy()


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg):
	if not cfg.datamodule.predict_tiff_path:
		print("Error: Specify datamodule.predict_tiff_path")
		return

	# Загрузка
	dm = hydra.utils.instantiate(cfg.datamodule)
	dm.setup("predict")

	ckpt = "checkpoints/epoch=10-step=2167.ckpt"
	try:
		model = DendriteSegmentationModule.load_from_checkpoint(ckpt, weights_only=False)
	except:
		print(f"Checkpoint {ckpt} not found, using random weights for test.")
		# model = hydra.utils.instantiate(cfg.model)

	trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)

	# Предикт
	preds = trainer.predict(model, dm)

	# Сборка
	result = stitch_volume(preds, dm.predict_ds.shape)

	# Бинаризация и сохранение
	# binary = (result > 0.5).astype(np.uint8) * 255
	# imwrite("prediction_result30_bin_8.tif", binary)
	# binary = ((result - np.min(result)) / ((np.max(result) - np.min(result))) * 255).astype(np.uint8)
	imwrite("stage1_vivo14.tif", result)
	print("Saved prediction_result.tif")


if __name__ == "__main__":
	main()