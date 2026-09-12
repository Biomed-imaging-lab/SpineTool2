import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from src.models.components.unet3d import U_Net
from src.models.components.loss import FocalTverskyLoss

# Функция безопасного кропа по батчу границ
def crop_by_borders(tensor, borders):
    # tensor: [B, C, D, H, W]
    # borders: [B, 3, 2]
    cropped_list = []
    for i in range(tensor.shape[0]):
        z0, z1 = borders[i, 0]
        y0, y1 = borders[i, 1]
        x0, x1 = borders[i, 2]
        cropped_list.append(tensor[i, :, z0:z1, y0:y1, x0:x1].reshape(-1))
    
    return torch.cat(cropped_list)

class DendriteSegmentationModule(pl.LightningModule):
	def __init__(
			self,
			in_ch: int = 3,
			out_ch: int = 1,
			init_features: int = 32,
			lr: float = 0.005,
			use_lora: bool = False,
        	lora_rank: int = 8,
        	lora_alpha: float = 16.0,
			lora_adapter: str = "lora",
			lora_scope: str = "all",
        	pretrained_ckpt: str = None,
        	freeze_base: bool = False,
        	train_head: bool = True,
        	train_batchnorm: bool = False
	):
		super().__init__()
		self.save_hyperparameters()

		effective_lora_rank = self.hparams.lora_rank if self.hparams.use_lora else 0

		self.net = U_Net(
			in_ch=self.hparams.in_ch, 
			out_ch=self.hparams.out_ch, 
			init_features=self.hparams.init_features,
			lora_rank=effective_lora_rank,
            lora_alpha=self.hparams.lora_alpha,
			lora_adapter=self.hparams.lora_adapter,
		)

		if self.hparams.pretrained_ckpt:
			self._load_pretrained_checkpoint(self.hparams.pretrained_ckpt)
    
		if self.hparams.freeze_base:
			if self.hparams.use_lora:
				self._freeze_for_lora(
                	train_head=self.hparams.train_head,
                	train_batchnorm=self.hparams.train_batchnorm,
            	)
			else:
        # Если LoRA выключен — размораживаем ТЕ ЖЕ области, но в БАЗОВОЙ модели
				self._freeze_for_traditional_finetuning(
            		train_head=self.hparams.train_head,
            		train_batchnorm=self.hparams.train_batchnorm,
        		)	

		# Метрики
		self.val_dice = BinaryF1Score()
		self.val_iou = BinaryJaccardIndex()

	def forward(self, x):
		return self.net(x)
	

	def train(self, mode: bool = True):
		super().train(mode)

		if not mode:
			return self

		freeze_base = bool(self.hparams.get("freeze_base", False))
		train_batchnorm = bool(self.hparams.get("train_batchnorm", False))

		if freeze_base and not train_batchnorm:
			for module in self.net.modules():
				if isinstance(module, torch.nn.BatchNorm3d):
					module.eval()
		return self
	
	def _load_pretrained_checkpoint(self, checkpoint_path: str):
		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
		state_dict = checkpoint.get("state_dict", checkpoint)

		current_state = self.state_dict()
		compatible_state = {}

		for key, value in state_dict.items():
			if key in current_state and current_state[key].shape == value.shape:
				compatible_state[key] = value

		missing, unexpected = self.load_state_dict(compatible_state, strict=False)
		for module in self.net.modules():
			if hasattr(module, "reset_magnitude_from_weight"):
				module.reset_magnitude_from_weight()
		print(
            f"Loaded {len(compatible_state)} tensors from {checkpoint_path}. "
            f"Missing: {len(missing)}. Unexpected: {len(unexpected)}."
        )

	def _freeze_for_traditional_finetuning(self, train_head: bool, train_batchnorm: bool):
		"""
        Классическое дообучение без LoRA. Замораживает базу и полностью
        размораживает выбранный scope исходных весов модели.
        """
        # 1. Сначала жестко замораживаем абсолютно все веса сети
		for param in self.net.parameters():
			param.requires_grad = False

		scope = self.hparams.get("lora_scope", "all")

        # 2. Размораживаем базовые слои в зависимости от выбранного scope
		for name, param in self.net.named_parameters():
            
			if scope == "all":
				param.requires_grad = True
				continue

			if scope == "decoder" and name.startswith((
                "Up5.", "Up_conv5.", "Up4.", "Up_conv4.",
                "Up3.", "Up_conv3.", "Up2.", "Up_conv2.", "Conv.",
            )):
				param.requires_grad = True

			elif scope == "encoder" and name.startswith((
                "Conv1.", "Conv2.", "Conv3.", "Conv4.", "Conv5."
            )):
				param.requires_grad = True

			elif scope == "shallow" and name.startswith((
                "Conv1.", "Conv2.", "Up2.", "Up_conv2.", "Conv."
            )):
				param.requires_grad = True
                
			elif scope == "shallow_no_head" and name.startswith((
                "Conv1.", "Conv2.", "Up2.", "Up_conv2."
            )):
				param.requires_grad = True

        # 3. Принудительно открываем голову, если флаг True (или если она не попала в scope)
		if train_head:
			for param in self.net.Conv.parameters():
				param.requires_grad = True

        # 4. Принудительно открываем слои BatchNorm, если это необходимо
		if train_batchnorm:
			for module in self.net.modules():
				if isinstance(module, torch.nn.BatchNorm3d):
					for param in module.parameters():
						param.requires_grad = True	
	
	def _freeze_for_lora(self, train_head: bool, train_batchnorm: bool):
		for param in self.net.parameters():
			param.requires_grad = False

		lora_scope = self.hparams.get("lora_scope", "all")

		def is_adapter_param(name):
			return "lora_" in name or "magnitude" in name

		def should_train_adapter(name):
			if not is_adapter_param(name):
				return False

			if lora_scope == "all":
				return True

			if lora_scope == "decoder":
				return name.startswith((
					"Up5.", "Up_conv5.",
					"Up4.", "Up_conv4.",
					"Up3.", "Up_conv3.",
					"Up2.", "Up_conv2.",
					"Conv.",
				))

			if lora_scope == "encoder":
				return name.startswith(("Conv1.", "Conv2.", "Conv3.", "Conv4.", "Conv5."))

			if lora_scope == "shallow":
				return name.startswith(("Conv1.", "Conv2.", "Up2.", "Up_conv2.", "Conv."))
			
			if lora_scope == "shallow_no_head":
				return name.startswith(("Conv1.", "Conv2.", "Up2.", "Up_conv2."))

			return False

		for name, param in self.net.named_parameters():
			if should_train_adapter(name):
				param.requires_grad = True

		if train_head:
			for param in self.net.Conv.parameters():
				param.requires_grad = True

		if train_batchnorm:
			for module in self.net.modules():
				if isinstance(module, torch.nn.BatchNorm3d):
					for param in module.parameters():
						param.requires_grad = True

	def calculate_loss(self, preds, targets, borders):
        # 1. Обрезаем предсказания и маски по границам
        # Результат - длинный 1D вектор всех валидных пикселей батча
		preds_valid = crop_by_borders(preds, borders)
		targets_valid = crop_by_borders(targets, borders)
        
        # 2. Считаем MSE на валидных пикселях
		loss = torch.nn.functional.mse_loss(preds_valid, targets_valid)
		return loss, preds_valid, targets_valid

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
		self.log("val/dice", self.val_dice(p_valid, t_valid.int()), prog_bar=True)
		self.log("val/iou", self.val_iou(p_valid, t_valid.int()), prog_bar=True)

	def predict_step(self, batch, batch_idx):
		x, coords, valid_shape = batch["image"], batch["coords"],  batch["valid_shape"]

		# Инференс
		logits = self(x)
		preds = torch.sigmoid(logits)

		# Возвращаем предсказание и координаты, чтобы собрать их обратно
		return {"preds": preds, "coords": coords, "valid_shape": valid_shape}

	def configure_optimizers(self):
		trainable_params = [param for param in self.parameters() if param.requires_grad]
		if not trainable_params:
			raise ValueError("No trainable parameters found. Check freeze_base and LoRA settings.")

		optimizer = torch.optim.Adam(
			trainable_params,
			# self.parameters(),
			lr=self.hparams.lr,
			betas=(0.9, 0.999),
			eps=1e-8
		)
		return optimizer