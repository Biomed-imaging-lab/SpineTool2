import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from src.models.components.loss_stage2 import ProjectedDiceLoss, SkeletonRecallLoss
from src.models.components.unet3d import U_Net

class Stage2SegmentationModule(pl.LightningModule):
    def __init__(
			self,
			in_ch: int = 4,
			out_ch: int = 1,
			init_features: int = 64,
			lr: float = 0.005,
	):
        super().__init__()
        self.save_hyperparameters()

        self.net = U_Net(in_ch=self.hparams.in_ch, out_ch=self.hparams.out_ch, init_features=self.hparams.init_features)

        self.loss_projected_dice = ProjectedDiceLoss()
        self.loss_skel_recall = SkeletonRecallLoss()

        self.val_dice = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()

    def forward(self, x):
        return self.net(x)

    def calculate_loss(self, preds, targets_combined, borders):

        batch_size = preds.shape[0]
        total_loss = 0.0
        
        # Списки для сохранения 1D векторов (только для метрик F1 и IoU)
        p_1d_list = []
        t_1d_list = []

        # targets_combined имеет размерность [Batch, 2, D, H, W]
        # Берем канал 0 с сохранением размерности (срез 0:1) -> [Batch, 1, D, H, W]
        targets_vol = targets_combined[:, 0, ...].unsqueeze(1) 
        # Берем канал 1 с сохранением размерности (срез 1:2) -> [Batch, 1, D, H, W]
        targets_skel = targets_combined[:, 1, ...].unsqueeze(1)

        # Проходим по каждому элементу батча отдельно
        for i in range(batch_size):
            z0, z1 = borders[i, 0]
            y0, y1 = borders[i, 1]
            x0, x1 = borders[i, 2]
            
            # Вырезаем валидный 3D куб. 
            # Оставляем dim=0 (батч), чтобы размерность осталась [1, 1, D, H, W]
            p_valid_3d = preds[i:i+1, :, z0:z1, y0:y1, x0:x1]
            
            t_valid_vol = targets_vol[i:i+1, :, z0:z1, y0:y1, x0:x1]
            t_valid_skel = targets_skel[i:i+1, :, z0:z1, y0:y1, x0:x1]
            
            # Считаем лосс с проекциями для конкретного кубика
            loss_vol = self.loss_projected_dice(p_valid_3d, t_valid_vol)
            # Считаем лосс на связность
            loss_topo = self.loss_skel_recall(p_valid_3d, t_valid_skel)

            # item_loss = 0.7 * loss_vol + 0.3 * loss_topo
            item_loss = 0.2 * loss_vol + 0.8 * loss_topo
            total_loss += item_loss
            
            # Сплющиваем для стандартных Lightning-метрик
            p_1d_list.append(p_valid_3d.reshape(-1))
            t_1d_list.append(t_valid_vol.reshape(-1))
            
        # Усредняем лосс по батчу
        final_loss = total_loss / batch_size
        
        # Собираем валидные пиксели для логирования
        preds_valid_1d = torch.cat(p_1d_list)
        targets_valid_1d = torch.cat(t_1d_list)

        return final_loss, preds_valid_1d, targets_valid_1d
    
    def training_step(self, batch, batch_idx):
        x, y, borders = batch["image"], batch["mask"], batch["borders"]

        logits = self(x)
        preds = torch.sigmoid(logits)
		# Считаем лосс с учетом границ
        loss, _, _ = self.calculate_loss(preds, y, borders)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, borders = batch["image"], batch["mask"], batch["borders"]

        logits = self(x)
        preds = torch.sigmoid(logits)

        loss, p_valid, t_valid = self.calculate_loss(preds, y, borders)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dice", self.val_dice(p_valid, t_valid.int()))
        self.log("val/iou", self.val_iou(p_valid, t_valid.int()))

    def predict_step(self, batch, batch_idx):

        x, coords, valid_shape = batch["image"], batch["coords"],  batch["valid_shape"]

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',        
            factor=0.5,        
            patience=3,        # Ждать 3 валидационных эпох без улучшений перед снижением
            min_lr=1e-6,       
            verbose=True       
        )
        
        # 3. Возвращаем словарь, понятный для PyTorch Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  
                "frequency": 1          
            }
        }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
	# 		self.parameters(),
	# 		lr=self.hparams.lr,
	# 		betas=(0.9, 0.999),
	# 		eps=1e-8
	# 	)
    #     return optimizer