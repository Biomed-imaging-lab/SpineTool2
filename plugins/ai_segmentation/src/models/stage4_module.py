import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from src.models.components.loss import Dice
from src.models.components.unet3d import U_Net

# Функция безопасного кропа по батчу границ
def crop_by_borders(tensor, borders):
    # tensor: [B, C, D, H, W]
    # borders: [B, 3, 2]
    cropped_list = []
    for i in range(tensor.shape[0]):
        z0, z1 = borders[i, 0]
        y0, y1 = borders[i, 1]
        x0, x1 = borders[i, 2]
        # Вырезаем только валидную часть
        cropped_list.append(tensor[i, :, z0:z1, y0:y1, x0:x1].reshape(-1))
    
    # Собираем в один длинный вектор (так проще для лосса)
    return torch.cat(cropped_list)

class Stage4SegmentationModule(pl.LightningModule):
    def __init__(
			self,
			in_ch: int = 3,
			out_ch: int = 2,
			init_features: int = 64,
			lr: float = 0.005,
	):
        super().__init__()
        self.save_hyperparameters()

        self.net = U_Net(in_ch=self.hparams.in_ch, out_ch=self.hparams.out_ch, init_features=self.hparams.init_features)

        # Метрики для Шипиков (значение близко к 1.0, порог отсечения 0.75)
        self.val_spine_f1 = BinaryF1Score(threshold=0.5)
        self.val_spine_iou = BinaryJaccardIndex(threshold=0.5)

        # Метрики для всего Дендрита (Ствол 0.5 + Шипики 1.0, порог отсечения 0.25)
        self.val_dendrite_f1 = BinaryF1Score(threshold=0.5)

    def forward(self, x):
        return self.net(x)

    def calculate_loss(self, logits, targets, valid_mask):
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss_elementwise = criterion(logits, targets) # Форма: [B, 2, D, H, W]
        loss_masked = loss_elementwise * valid_mask
        # Усредняем ошибку по валидным вокселям
        # Количество элементов = количество True в маске * 2 канала
        denom = valid_mask.sum() * logits.shape[1]
        
        return loss_masked.sum() / (denom + 1e-8)
        # # 1. Обрезаем предсказания и маски по границам
        # # Результат - длинный 1D вектор всех валидных пикселей батча
        # preds_valid = crop_by_borders(preds, borders)
        # targets_valid = crop_by_borders(targets, borders)
        
        # # 2. Считаем MSE на валидных пикселях
        # loss = torch.nn.functional.mse_loss(preds_valid, targets_valid)
        # return loss, preds_valid, targets_valid

    
    def training_step(self, batch, batch_idx):
        x, y, valid_mask = batch["image"], batch["mask"], batch["valid_mask"]

        logits = self(x)

        loss = self.calculate_loss(logits, y, valid_mask)
        # preds = torch.sigmoid(logits)
		# Считаем лосс с учетом границ
        # loss, _, _ = self.calculate_loss(preds, y, borders)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, valid_mask = batch["image"], batch["mask"], batch["valid_mask"]

        logits = self(x)
        loss = self.calculate_loss(logits, y, valid_mask)
        self.log("val/loss", loss, prog_bar=True)
        preds = torch.sigmoid(logits)
        valid_mask_bool = (valid_mask.squeeze(1) > 0) # Форма: [B, D, H, W]

        if valid_mask_bool.any():
            # 1. Метрики для ШИПИКОВ (Канал 1)
            p_spine = preds[:, 1][valid_mask_bool]
            t_spine = y[:, 1][valid_mask_bool].int()
            
            self.log("val/spine_f1", self.val_spine_f1(p_spine, t_spine), prog_bar=True, on_epoch=True)
            self.log("val/spine_iou", self.val_spine_iou(p_spine, t_spine), prog_bar=True, on_epoch=True)

            # 2. Метрики для ДЕНДРИТА целиком 
            p_dendrite = (preds[:, 0][valid_mask_bool] > 0.5) | (preds[:, 1][valid_mask_bool] > 0.5)
            t_dendrite = (y[:, 0][valid_mask_bool] > 0.5) | (y[:, 1][valid_mask_bool] > 0.5)

            self.log("val/dendrite_f1", self.val_dendrite_f1(p_dendrite.float(), t_dendrite.int()), prog_bar=False, on_epoch=True)
#         loss, p_valid, t_valid = self.calculate_loss(preds, y, borders)

#         self.log("val/loss", loss, prog_bar=True)
# # --- Специфичные метрики качества ---
#         # 1. Для шипиков: истина там, где таргет равен 1.0 (задаем порог > 0.75)
#         spine_targets = (t_valid > 0.75).int()
#         self.log("val/spine_f1", self.val_spine_f1(p_valid, spine_targets), prog_bar=True)
#         self.log("val/spine_iou", self.val_spine_iou(p_valid, spine_targets), prog_bar=True)

#         # 2. Для дендрита целиком: истина там, где таргет 0.5 или 1.0 (задаем порог > 0.25)
#         dendrite_targets = (t_valid > 0.25).int()
#         self.log("val/dendrite_f1", self.val_dendrite_f1(p_valid, dendrite_targets), prog_bar=False)

    def predict_step(self, batch, batch_idx):

        x, coords, valid_shape = batch["image"], batch["coords"], batch["valid_shape"]

		# Инференс
        logits = self(x)
        preds = torch.sigmoid(logits)

		# Возвращаем предсказание и координаты, чтобы собрать их обратно
        return {"preds": preds, "coords": coords, "valid_shape": valid_shape}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
			self.parameters(),
			lr=self.hparams.lr,
			betas=(0.9, 0.999),
			eps=1e-8
		)
        return optimizer