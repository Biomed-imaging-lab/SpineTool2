import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import json
import os
from typing import Optional

from src.data.datasets import DendriteTrainDataset, DendriteInferenceDataset


class DendriteDataModule(pl.LightningDataModule):
	def __init__(
			self,
			train_patches_dir: str = None,  # Путь к папке с train .npz
			val_patches_dir: str = None,  # Путь к папке с val .npz
			predict_tiff_path: str = None,  # Путь к .tif для инференса
			area_path: str = None,
            current_scale: list = None,
			batch_size: int = 4,
			num_workers: int = 0,
			# Параметры для инференса:
			patch_size: list = [64, 64, 64],
			overlap: int = 16,
	):
		super().__init__()
		self.save_hyperparameters()
		self.train_ds = None
		self.val_ds = None
		self.predict_ds = None

	def setup(self, stage: Optional[str] = None):
		# --- ОБУЧЕНИЕ ---
		if stage == "fit" or stage is None:
			if not self.hparams.train_patches_dir or not self.hparams.val_patches_dir:
				raise ValueError("Please provide train_patches_dir and val_patches_dir")

			# Просто инициализируем датасеты из папок
			self.train_ds = DendriteTrainDataset(self.hparams.train_patches_dir)
			self.val_ds = DendriteTrainDataset(self.hparams.val_patches_dir)
			
		if stage == "validate" or stage == "test":
			if not self.hparams.val_patches_dir:
				raise ValueError("Please provide val_patches_dir for validation/testing")
			
			self.val_ds = DendriteTrainDataset(self.hparams.val_patches_dir)	


		# --- ПРЕДСКАЗАНИЕ ---
		if stage == "predict":
			if not self.hparams.predict_tiff_path:
				raise ValueError("Please provide predict_tiff_path for inference")
			if self.hparams.current_scale is None:
				raise ValueError("current_scale must be provided for Stage 1 inference")
			# Здесь используется датасет, который режет TIFF на лету
			self.predict_ds = DendriteInferenceDataset(
				tiff_path=self.hparams.predict_tiff_path,
				current_scale=self.hparams.current_scale,
                area_path=self.hparams.area_path,
				patch_size=self.hparams.patch_size,
				overlap=self.hparams.overlap,
				sigma=2.0,  # Хардкод параметров эксперимента
				otsu_classes=4
			)

	def train_dataloader(self):
		return DataLoader(
			self.train_ds,
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.num_workers,
			shuffle=True,  # Обязательно шаффл для обучения
			pin_memory=True,
			persistent_workers=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_ds,
			batch_size=self.hparams.batch_size,
			num_workers=self.hparams.num_workers,
			shuffle=False,
			pin_memory=True,
			persistent_workers=True
		)

	def predict_dataloader(self):
		return DataLoader(
			self.predict_ds,
			batch_size=1,  # Для инференса по одному патчу (или больше, если влезет)
			num_workers=self.hparams.num_workers,
			shuffle=False,
			persistent_workers=self.hparams.num_workers > 0
		)