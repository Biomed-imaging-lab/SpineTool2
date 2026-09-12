import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tifffile import imread
import os
import gc
from data_preprocessing.stage1 import normalize_minmax

class Stage4TrainDataset(Dataset):
	"""
	Датасет для обучения. Ожидает список словарей с путями к файлам.
	"""

	def __init__(self, data_dir):
		search_path = os.path.join(data_dir, "**", "*.npz")
		self.files = glob.glob(search_path, recursive=True)
		if not self.files:
			print(f"Warning: No files found in {data_dir}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		# Загружаем сжатый файл
		# allow_pickle=True нужен для безопасности numpy, для npz это ок
		data = np.load(self.files[idx], allow_pickle=True)

		# Данные уже имеют форму [3, 64, 128, 128] и тип float32
		image = torch.from_numpy(data['image'])

		# Маска имеет форму [64, 128, 128], добавляем канал -> [1, 64, 128, 128]
		gt_mask = data['mask']

        # Расщепляем на 2 бинарных канала для BCEWithLogitsLoss
		target_shaft = (gt_mask == 1).astype(np.float32) # Маска ствола
		target_spine = (gt_mask == 2).astype(np.float32) # Маска шипиков
		target_2ch = np.stack([target_shaft, target_spine], axis=0) # Форма [2, 64, 128, 128]
		mask_tensor = torch.from_numpy(target_2ch)

        # Загружаем 3D маску валидности вместо старого borders
		valid_mask = torch.from_numpy(data['valid_mask']).float().unsqueeze(0) # Форма [1, 64, 128, 128]

		return {
            "image": image,
            "mask": mask_tensor,       # Теперь это [2, 64, 128, 128]
            "valid_mask": valid_mask   # Теперь это [1, 64, 128, 128]
        }


class Stage4InferenceDataset(Dataset):
	"""
	Датасет "Скользящее окно" для инференса больших Tiff файлов.
	"""

	def __init__(self, vsot_shaft_path, vsot_spine_path, shaft_dist_path, spine_dist_path, area_path=None, patch_size=(64, 128, 128), overlap=(48, 96, 96), sigma=1.0):
		self.patch_size = np.array(patch_size)
		self.overlap = np.array(overlap)
		self.stride = (self.patch_size - self.overlap).astype(int)

		# 1. Загрузка
		vsot_shaft = imread(vsot_shaft_path)>0
		vsot_spine = imread(vsot_spine_path)>0
		d_dendr_norm = imread(shaft_dist_path)
		d_spine_norm = imread(spine_dist_path)

		self.shape = vsot_shaft.shape
		i_vsot = np.zeros_like(vsot_shaft, dtype=np.uint8)
		i_vsot[vsot_shaft] = 1  
		i_vsot[vsot_spine] = 2
		i_vsot_norm = normalize_minmax(i_vsot).astype(np.float16)

	
		if area_path and os.path.exists(area_path):
			area = imread(area_path) > 0
			area[area > 0] = 1.0
			if i_vsot_norm.shape == area.shape:
				i_vsot_norm = (i_vsot_norm * area).astype(np.float16)
				d_dendr_norm = d_dendr_norm * area
				d_spine_norm = d_spine_norm * area
			else:
				print(f"Shape mismatch {i_vsot_norm.shape} vs {area.shape}")

		# Сохраняем предобработанные карты в памяти
		self.ch1 = i_vsot_norm
		self.ch2 = d_dendr_norm.astype(np.float16)
		self.ch3 = d_spine_norm.astype(np.float16)

		# Генерация сетки координат
		self.coords = self._generate_grid()
		self.vsot_orig = i_vsot

	def _generate_grid(self):
		coords = []
		z_steps = range(0, self.shape[0], self.stride[0])
		y_steps = range(0, self.shape[1], self.stride[1])
		x_steps = range(0, self.shape[2], self.stride[2])
		for z in z_steps:
			for y in y_steps:
				for x in x_steps:
					coords.append((z, y, x))
		return coords

	def __pad_chunk(self, chunk):
		"""
		Дополняет кусок нулями до self.patch_size, если он меньше.
		"""
        # Текущий размер куска
		cz, cy, cx = chunk.shape
		dz, dy, dx = self.patch_size
        
        # Если размер совпадает, возвращаем как есть
		if cz == dz and cy == dy and cx == dx:
			return chunk
            
        # Вычисляем сколько не хватает
		pad_z = dz - cz
		pad_y = dy - cy
		pad_x = dx - cx
        
        # Делаем паддинг константой (0) справа/снизу
        # format: ((before, after), ...)
		return np.pad(chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

	def __len__(self):
		return len(self.coords)

	def __getitem__(self, idx):
		z, y, x = self.coords[idx]
		dz, dy, dx = self.patch_size

		z_end = min(z + dz, self.shape[0])
		y_end = min(y + dy, self.shape[1])
		x_end = min(x + dx, self.shape[2])

		valid_z = z_end - z
		valid_y = y_end - y
		valid_x = x_end - x	

		# Вырезаем куски
		p_ch1 = self.ch1[z:z_end, y:y_end, x:x_end]
		p_ch2 = self.ch2[z:z_end, y:y_end, x:x_end]
		p_ch3 = self.ch3[z:z_end, y:y_end, x:x_end]
		vsot = self.vsot_orig[z:z_end, y:y_end, x:x_end]

		# Дополняем до 64x64x64, если кусок обрезан
		p_ch1 = self.__pad_chunk(p_ch1)
		p_ch2 = self.__pad_chunk(p_ch2)
		p_ch3 = self.__pad_chunk(p_ch3)
		vsot = self.__pad_chunk(vsot)

		# Сборка каналов

		input_tensor = np.stack([p_ch1, p_ch2, p_ch3], axis=0).astype(np.float32)

		return {
			"image": torch.from_numpy(input_tensor),
			"coords": torch.tensor([z, y, x]),
			"valid_shape": torch.tensor([valid_z, valid_y, valid_x]),
			"vsot_segm": torch.from_numpy(vsot)
		}
	

class Stage4WithoutVSOTInferenceDataset(Dataset):
	"""
	Датасет "Скользящее окно" для инференса больших Tiff файлов.
	"""

	def __init__(self, current_scale, general_mask_path, skeleton_path, area_path=None, patch_size=(64, 128, 128), overlap=(48, 96, 96), sigma=1.0):
		self.patch_size = np.array(patch_size)
		self.overlap = np.array(overlap)
		self.stride = (self.patch_size - self.overlap).astype(int)

		scale_nm = (current_scale[0]*1000, current_scale[1]*1000, current_scale[2]*1000)
		# 1. Загрузка
		general_mask = imread(general_mask_path) > 0
		skeleton = imread(skeleton_path)>0

		self.shape = general_mask.shape
		vsot_dendrite = general_mask
		i_vsot = np.zeros(self.shape, dtype=np.uint8)
		i_vsot[vsot_dendrite] = 1  
		i_vsot_norm = normalize_minmax(i_vsot).astype(np.float16)
		del i_vsot
		gc.collect()

		d_dendr_norm, d_spine_norm = generate_distance_channels(vsot_dendrite, skeleton, scale_nm)
		del vsot_dendrite, skeleton
		gc.collect()
	
		if area_path and os.path.exists(area_path):
			area = imread(area_path) > 0
			area[area > 0] = 1.0
			if i_vsot_norm.shape == area.shape:
				i_vsot_norm = (i_vsot_norm * area).astype(np.float16)
				d_dendr_norm = d_dendr_norm * area
				d_spine_norm = d_spine_norm * area
			else:
				print(f"Shape mismatch {i_vsot_norm.shape} vs {area.shape}")
			del area
			gc.collect()
		# Сохраняем предобработанные карты в памяти
		self.ch1 = i_vsot_norm
		self.ch2 = d_dendr_norm.astype(np.float16)
		self.ch3 = d_spine_norm.astype(np.float16)
		del d_dendr_norm, d_spine_norm
		gc.collect()
		# Генерация сетки координат
		self.coords = self._generate_grid()

	def _generate_grid(self):
		coords = []
		z_steps = range(0, self.shape[0], self.stride[0])
		y_steps = range(0, self.shape[1], self.stride[1])
		x_steps = range(0, self.shape[2], self.stride[2])
		for z in z_steps:
			for y in y_steps:
				for x in x_steps:
					coords.append((z, y, x))
		return coords

	def __pad_chunk(self, chunk):
		"""
		Дополняет кусок нулями до self.patch_size, если он меньше.
		"""
        # Текущий размер куска
		cz, cy, cx = chunk.shape
		dz, dy, dx = self.patch_size
        
        # Если размер совпадает, возвращаем как есть
		if cz == dz and cy == dy and cx == dx:
			return chunk
            
        # Вычисляем сколько не хватает
		pad_z = dz - cz
		pad_y = dy - cy
		pad_x = dx - cx
        
        # Делаем паддинг константой (0) справа/снизу
        # format: ((before, after), ...)
		return np.pad(chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

	def __len__(self):
		return len(self.coords)

	def __getitem__(self, idx):
		z, y, x = self.coords[idx]
		dz, dy, dx = self.patch_size

		z_end = min(z + dz, self.shape[0])
		y_end = min(y + dy, self.shape[1])
		x_end = min(x + dx, self.shape[2])

		valid_z = z_end - z
		valid_y = y_end - y
		valid_x = x_end - x	

		# Вырезаем куски
		p_ch1 = self.ch1[z:z_end, y:y_end, x:x_end]
		p_ch2 = self.ch2[z:z_end, y:y_end, x:x_end]
		p_ch3 = self.ch3[z:z_end, y:y_end, x:x_end]

		# Дополняем до 64x64x64, если кусок обрезан
		p_ch1 = self.__pad_chunk(p_ch1)
		p_ch2 = self.__pad_chunk(p_ch2)
		p_ch3 = self.__pad_chunk(p_ch3)

		# Сборка каналов

		input_tensor = np.stack([p_ch1, p_ch2, p_ch3], axis=0).astype(np.float32)

		return {
			"image": torch.from_numpy(input_tensor),
			"coords": torch.tensor([z, y, x]),
			"valid_shape": torch.tensor([valid_z, valid_y, valid_x])
		}	