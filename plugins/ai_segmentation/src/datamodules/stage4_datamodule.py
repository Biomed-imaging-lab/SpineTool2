import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import json
import os
from typing import Optional

from src.data.datasets_stage4 import Stage4TrainDataset, Stage4InferenceDataset, Stage4WithoutVSOTInferenceDataset


class Stage4DataModule(pl.LightningDataModule):
	def __init__(
			self,
			train_patches_dir: str = None,  # Путь к папке с train .npz
			val_patches_dir: str = None,  # Путь к папке с val .npz
			vsot_shaft_path: str = None,  # Путь к .tif для инференса
			vsot_spine_path: str = None,
			general_mask_path: str = None,
			shaft_dist_path: str = None,
			spine_dist_path: str = None,
			area_path: str = None,
			batch_size: int = 4,
			num_workers: int = 0,
			# Параметры для инференса:
			patch_size: list = [64, 128, 128],
			overlap: list = [48, 96, 96],
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
			self.train_ds = Stage4TrainDataset(self.hparams.train_patches_dir)
			self.val_ds = Stage4TrainDataset(self.hparams.val_patches_dir)
			
		if stage == "validate" or stage == "test":
			if not self.hparams.val_patches_dir:
				raise ValueError("Please provide val_patches_dir for validation/testing")
			
			self.val_ds = Stage4TrainDataset(self.hparams.val_patches_dir)	


		# --- ПРЕДСКАЗАНИЕ ---
		if stage == "predict":
			if not self.hparams.vsot_shaft_path:
				raise ValueError("Please provide vsot_shaft_path for inference")
			if not self.hparams.vsot_spine_path:
				raise ValueError("Stage 4 inference requires vsot_shaft_path!")
			# if not self.hparams.general_mask_path:
			# 	raise ValueError("Stage 4 inference requires general_mask_path!")

			# Здесь используется датасет, который режет TIFF на лету
			self.predict_ds = Stage4InferenceDataset(
				vsot_shaft_path=self.hparams.vsot_shaft_path,
				vsot_spine_path=self.hparams.vsot_spine_path,
				shaft_dist_path=self.hparams.shaft_dist_path, 
				spine_dist_path=self.hparams.spine_dist_path,
				area_path=self.hparams.area_path,
				patch_size=self.hparams.patch_size,
				overlap=self.hparams.overlap,
				sigma=1.0
			)
			# # Здесь используется датасет, который режет TIFF на лету
			# self.predict_ds = Stage4WithoutVSOTInferenceDataset(
			# 	current_scale=self.hparams.current_scale,
			# 	general_mask_path=self.hparams.general_mask_path,
			# 	skeleton_path=self.hparams.skeleton_path,
			# 	area_path=self.hparams.area_path,
			# 	patch_size=self.hparams.patch_size,
			# 	overlap=self.hparams.overlap,
			# 	sigma=1.0
			# )

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
			batch_size=4,  # Для инференса по одному патчу (или больше, если влезет)
			# num_workers=self.hparams.num_workers,
			num_workers=4,
			shuffle=False,
			persistent_workers=True
		)